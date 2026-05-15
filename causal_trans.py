"""Treatment Learning Transformer (TLT) training code.

This file is intentionally self-contained.  The original repository only kept a
partial model definition; the implementation below adds the missing residual
blocks, attention encoder, decoder heads, variational objective, data loading,
training/validation/testing loops, reproducibility controls, checkpointing and
loss-curve export described by the WACV 2023 TLT paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re

import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF



@dataclass
class BatchMetrics:
    loss: float
    elbo: float
    recon_y: float
    aux_y: float
    aux_t: float
    kl_z: float
    acc: float
    t_acc: float
    ate: float


def set_seed(seed: int) -> None:
    """Set Python and PyTorch seeds for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)


class ResBlock(nn.Module):
    """Two-convolution residual block used by the TLT encoder heads."""

    def __init__(self, channels: int, use_bias: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResBlocks(nn.Module):
    """Stack of residual blocks; replaces the missing ``.networks.ResBlocks``."""

    def __init__(self, num_blocks: int, channels: int, use_bias: bool = True) -> None:
        super().__init__()
        self.model = nn.Sequential(*[ResBlock(channels, use_bias) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvHead(nn.Module):
    """Adaptive-average-pooling classifier/regressor head."""

    def __init__(self, in_channels: int, out_dim: int, use_sigmoid: bool = False, use_bias: bool = True) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, out_dim, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.sigmoid(x) if self.use_sigmoid else x


class ResnetEncoder(nn.Module):
    """ResNet-34 feature extractor that returns the final 512-channel feature map."""

    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        if in_channels != 3:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if weights is not None and in_channels == 1:
                conv1.weight.data.copy_(backbone.conv1.weight.data.mean(dim=1, keepdim=True))
            backbone.conv1 = conv1
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


def create_attn_fc(in_channels: int, out_channels: int, spectral_norm: bool) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.xavier_uniform_(conv.weight)
    return nn.utils.spectral_norm(conv, eps=1e-12) if spectral_norm else conv


class Attention(nn.Module):
    """Single-head convolutional attention used to approximate q(z|x,t,y)."""

    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        value_channels: int,
        d_k: int = 128,
        d_v: int = 128,
        out_channels: int = 512,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.scale = math.sqrt(d_k)
        self.query_conv = create_attn_fc(query_channels, d_k, spectral_norm)
        self.key_conv = create_attn_fc(key_channels, d_k, spectral_norm)
        self.value_conv = create_attn_fc(value_channels, d_v, spectral_norm)
        self.output_conv = create_attn_fc(d_v, out_channels, spectral_norm)

    @staticmethod
    def _flatten_spatial(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, c, h * w)

    def forward(self, query_in: torch.Tensor, key_in: torch.Tensor, value_in: torch.Tensor) -> torch.Tensor:
        _, _, qh, qw = query_in.shape
        queries = self._flatten_spatial(self.query_conv(query_in))
        keys = self._flatten_spatial(self.key_conv(key_in))
        values = self._flatten_spatial(self.value_conv(value_in))
        attn = torch.softmax(torch.bmm(keys.transpose(1, 2), queries) / self.scale, dim=1)
        attended = torch.bmm(values, attn).view(-1, self.d_v, qh, qw)
        return self.output_conv(attended)


class TLTEncoder(nn.Module):
    """Inference network for q(t|x), q(y|x,t), and q(z|x,t,y)."""

    def __init__(
        self,
        pretrained_backbone: bool = True,
        latent_channels: int = 512,
        use_attention: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.rgb_emb = ResnetEncoder(pretrained=pretrained_backbone, in_channels=3)
        self.depth_emb = ResnetEncoder(pretrained=False, in_channels=1)
        channels = self.rgb_emb.out_channels
        self.logits_t = ConvHead(channels, 1)
        self.hqy_t0 = ResBlocks(1, channels)
        self.hqy_t1 = ResBlocks(1, channels)
        self.qy = ConvHead(channels, num_classes)
        self.attn = Attention(channels, channels, channels, out_channels=channels) if use_attention else None
        self.hqz = ResBlocks(1, channels)
        self.residual_z = nn.Conv2d(channels, latent_channels, kernel_size=1, bias=False)
        self.prior_mu = nn.Sequential(ResBlocks(1, latent_channels, use_bias=False), nn.Conv2d(latent_channels, latent_channels, 1))
        self.prior_logvar = nn.Sequential(ResBlocks(1, latent_channels), nn.Conv2d(latent_channels, latent_channels, 1))
        self.muq_t0 = nn.Sequential(ResBlocks(1, latent_channels, use_bias=False), nn.Conv2d(latent_channels, latent_channels, 1))
        self.logvarq_t0 = nn.Sequential(ResBlocks(1, latent_channels), nn.Conv2d(latent_channels, latent_channels, 1))
        self.muq_t1 = nn.Sequential(ResBlocks(1, latent_channels, use_bias=False), nn.Conv2d(latent_channels, latent_channels, 1))
        self.logvarq_t1 = nn.Sequential(ResBlocks(1, latent_channels), nn.Conv2d(latent_channels, latent_channels, 1))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, depth: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.rgb_emb(x)
        depth_h = self.depth_emb(depth)
        qt_logits = self.logits_t(h)
        qt_prob = torch.sigmoid(qt_logits)
        h_t0 = self.hqy_t0(h)
        h_t1 = self.hqy_t1(h)
        if t is None:
            t_gate = (qt_prob > 0.5).float().view(-1, 1, 1, 1)
        else:
            t_gate = t.float().view(-1, 1, 1, 1)
        query = t_gate * h_t1 + (1.0 - t_gate) * h_t0
        qy_logits = self.qy(query)
        qy_prob = torch.softmax(qy_logits, dim=1)
        z_features = self.attn(query, depth_h, depth_h) if self.attn is not None else query
        z_features = self.hqz(z_features) + self.residual_z(depth_h)
        prior_mu = self.prior_mu(self.residual_z(depth_h))
        prior_logvar = self.prior_logvar(self.residual_z(depth_h)).clamp(-10.0, 10.0)
        mu0, logvar0 = self.muq_t0(z_features), self.logvarq_t0(z_features).clamp(-10.0, 10.0)
        mu1, logvar1 = self.muq_t1(z_features), self.logvarq_t1(z_features).clamp(-10.0, 10.0)
        mu = t_gate * mu1 + (1.0 - t_gate) * mu0
        logvar = t_gate * logvar1 + (1.0 - t_gate) * logvar0
        z = self.reparameterize(mu, logvar) if self.training else mu
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "qt_logits": qt_logits,
            "qt_prob": qt_prob,
            "qy_logits": qy_logits,
            "qy_prob": qy_prob,
        }


class TLTDecoder(nn.Module):
    """Model network for p(x|z), p(t|z), p(y|z,t=0) and p(y|z,t=1)."""

    def __init__(self, latent_channels: int = 512, num_classes: int = 2) -> None:
        super().__init__()
        self.px = nn.Sequential(
            ResBlocks(1, latent_channels),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.pt = ConvHead(latent_channels, 1)
        self.py_t0 = ConvHead(latent_channels, num_classes)
        self.py_t1 = ConvHead(latent_channels, num_classes)

    def forward(self, z: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        px_features = self.px(z)
        pt_logits = self.pt(z)
        y0_logits = self.py_t0(z)
        y1_logits = self.py_t1(z)
        if t is None:
            y_logits = 0.5 * (y0_logits + y1_logits)
        else:
            t_col = t.float().view(-1, 1)
            y_logits = t_col * y1_logits + (1.0 - t_col) * y0_logits
        return {
            "px_features": px_features,
            "pt_logits": pt_logits,
            "py_t0_logits": y0_logits,
            "py_t1_logits": y1_logits,
            "y_logits": y_logits,
            "y_prob": torch.softmax(y_logits, dim=1),
            "y0_prob": torch.softmax(y0_logits, dim=1),
            "y1_prob": torch.softmax(y1_logits, dim=1),
        }


class CausalTransformer(nn.Module):
    """End-to-end Treatment Learning Transformer."""

    def __init__(self, pretrained_backbone: bool = True, use_attention: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = TLTEncoder(pretrained_backbone=pretrained_backbone, use_attention=use_attention, num_classes=num_classes)
        self.decoder = TLTDecoder(num_classes=num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self.encoder(batch["x"], batch["depth"], batch.get("t"))
        dec = self.decoder(enc["z"], batch.get("t"))
        return {**enc, **dec}


class CEVAE(CausalTransformer):
    """Compatibility wrapper for the original constructor signature."""

    def __init__(self, *args: Any, pretrained_backbone: bool = False, use_attention: bool = False, **kwargs: Any) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone, use_attention=use_attention, num_classes=kwargs.pop("num_classes", 2))


class CEVAE_Att(CausalTransformer):
    """Compatibility wrapper that enables the attention path by default."""

    def __init__(self, *args: Any, pretrained_backbone: bool = False, use_attention: bool = True, **kwargs: Any) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone, use_attention=use_attention, num_classes=kwargs.pop("num_classes", 2))


class Causal_Transformer(CausalTransformer):
    """Compatibility wrapper matching the paper-code class name."""

    def __init__(self, *args: Any, pretrained_backbone: bool = False, use_attention: bool = True, **kwargs: Any) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone, use_attention=use_attention, num_classes=kwargs.pop("num_classes", 2))


class Encoder(TLTEncoder):
    def __init__(self, *args: Any, pretrained_backbone: bool = False, use_attention: bool = False, **kwargs: Any) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone, use_attention=use_attention, num_classes=kwargs.pop("num_classes", 2))


class Encoder_v2(TLTEncoder):
    def __init__(self, *args: Any, pretrained_backbone: bool = False, use_attention: bool = True, **kwargs: Any) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone, use_attention=use_attention, num_classes=kwargs.pop("num_classes", 2))


class Encoder_v3(TLTEncoder):
    def __init__(self, *args: Any, pretrained_backbone: bool = False, use_attention: bool = True, **kwargs: Any) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone, use_attention=use_attention, num_classes=kwargs.pop("num_classes", 2))


class Decoder(TLTDecoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(num_classes=kwargs.pop("num_classes", 2))


def tlt_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    kl_weight: float = 1e-4,
    aux_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Negative variational objective plus supervised auxiliary losses.

    The implementation follows the paper's Eq. (5) with practical image
    classification terms: decoder likelihood p(y|z,t), decoder p(t|z), encoder
    q(y|x,t), encoder q(t|x), and a depth-conditioned Gaussian prior KL for z.
    """
    y = batch["y"].long().view(-1)
    t = batch["t"].float().view(-1, 1)
    recon_y = F.cross_entropy(outputs["y_logits"], y)
    model_t = F.binary_cross_entropy_with_logits(outputs["pt_logits"], t)
    aux_y = F.cross_entropy(outputs["qy_logits"], y)
    aux_t = F.binary_cross_entropy_with_logits(outputs["qt_logits"], t)
    posterior_var = outputs["logvar"].exp()
    prior_var = outputs["prior_logvar"].exp()
    kl_z = 0.5 * torch.mean(
        outputs["prior_logvar"]
        - outputs["logvar"]
        + (posterior_var + (outputs["mu"] - outputs["prior_mu"]).pow(2)) / prior_var
        - 1.0
    )
    elbo = recon_y + model_t + kl_weight * kl_z
    aux = aux_weight * (aux_y + aux_t)
    loss = elbo + aux
    parts = {
        "loss": loss.detach(),
        "elbo": elbo.detach(),
        "recon_y": recon_y.detach(),
        "model_t": model_t.detach(),
        "aux_y": aux_y.detach(),
        "aux_t": aux_t.detach(),
        "kl_z": kl_z.detach(),
    }
    return loss, parts


class RGBDepthImageFolder(Dataset):
    """ImageFolder-style RGB dataset with a paired ``.npy`` depth map per image.

    Class folders provide the outcome label ``y``.  The depth path is resolved by
    replacing the RGB suffix with ``.npy`` next to the RGB image, or by mirroring
    the RGB relative path under ``depth_root`` when that argument is provided.
    """

    treatment_pattern = re.compile(r"(?:^|[_\-/])(?:t|treatment)[=_-]?([01])(?:$|[_\-./])", re.IGNORECASE)

    def __init__(
        self,
        root: str,
        transform: "PairedRGBDepthTransform",
        depth_root: Optional[str] = None,
        treatment_mode: str = "filename",
    ) -> None:
        self.root = Path(root)
        self.depth_root = Path(depth_root) if depth_root else None
        self.dataset = ImageFolder(root=root)
        self.transform = transform
        self.treatment_mode = treatment_mode
        self.classes = self.dataset.classes

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def treatment_from_path(self, path: str, y: int) -> int:
        if self.treatment_mode == "label-parity":
            return int(y) % 2
        if self.treatment_mode == "none":
            return 0
        match = self.treatment_pattern.search(path.replace(os.sep, "/"))
        return int(match.group(1)) if match else 0

    def treatment_values(self) -> List[int]:
        return [self.treatment_from_path(path, y) for path, y in self.dataset.samples]

    def depth_path_for_rgb(self, rgb_path: str) -> Path:
        rgb = Path(rgb_path)
        if self.depth_root is None:
            candidates = [rgb.with_suffix(".npy")]
        else:
            rel = rgb.relative_to(self.root)
            candidates = [self.depth_root / rel.with_suffix(".npy"), self.depth_root / rel.name.replace(rgb.suffix, ".npy")]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"No paired depth .npy found for RGB image {rgb_path}. Searched: {searched}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, y = self.dataset.samples[idx]
        image = self.dataset.loader(rgb_path)
        depth = np.load(self.depth_path_for_rgb(rgb_path))
        x, depth_tensor = self.transform(image, depth)
        t = self.treatment_from_path(rgb_path, y)
        return {
            "x": x,
            "depth": depth_tensor,
            "y": torch.tensor(y, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.float32),
            "path": rgb_path,
        }


# Backward-compatible name for callers that imported the old dataset class.
CausalImageFolder = RGBDepthImageFolder


class PairedRGBDepthTransform:
    """Apply aligned spatial preprocessing to RGB images and depth arrays."""

    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = image_size
        self.train = train
        self.rgb_mean = models.ResNet34_Weights.IMAGENET1K_V1.transforms().mean
        self.rgb_std = models.ResNet34_Weights.IMAGENET1K_V1.transforms().std
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    @staticmethod
    def depth_to_tensor(depth: np.ndarray) -> torch.Tensor:
        depth = np.nan_to_num(depth.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 2:
            tensor = torch.from_numpy(depth).unsqueeze(0)
        elif depth.ndim == 3:
            tensor = torch.from_numpy(depth)
            if tensor.shape[0] not in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor[:1]
        else:
            raise ValueError(f"Depth array must be 2D or 3D, got shape {depth.shape}")
        min_val, max_val = tensor.amin(), tensor.amax()
        if torch.isfinite(max_val - min_val) and (max_val - min_val) > 0:
            tensor = (tensor - min_val) / (max_val - min_val)
        return tensor.float()

    def __call__(self, image: Any, depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_tensor = self.depth_to_tensor(depth)
        if self.train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
            image = TF.resized_crop(image, i, j, h, w, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            depth_tensor = TF.resized_crop(depth_tensor, i, j, h, w, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            if random.random() < 0.5:
                image = TF.hflip(image)
                depth_tensor = TF.hflip(depth_tensor)
            image = self.color_jitter(image)
        else:
            resize_size = int(self.image_size * 1.15)
            image = TF.resize(image, resize_size, interpolation=TF.InterpolationMode.BILINEAR)
            depth_tensor = TF.resize(depth_tensor, resize_size, interpolation=TF.InterpolationMode.BILINEAR)
            image = TF.center_crop(image, [self.image_size, self.image_size])
            depth_tensor = TF.center_crop(depth_tensor, [self.image_size, self.image_size])
        rgb_tensor = TF.to_tensor(image)
        rgb_tensor = TF.normalize(rgb_tensor, mean=self.rgb_mean, std=self.rgb_std)
        return rgb_tensor, depth_tensor


def build_transforms(image_size: int, train: bool) -> PairedRGBDepthTransform:
    return PairedRGBDepthTransform(image_size=image_size, train=train)


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "depth": torch.stack([item["depth"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "t": torch.stack([item["t"] for item in batch]),
        "path": [item["path"] for item in batch],
    }

def split_indices_by_ratio(indices: List[int], train_ratio: float, val_ratio: float, generator: torch.Generator) -> Tuple[List[int], List[int], List[int]]:
    """Shuffle indices and split them into train/val/test according to ratios."""
    if len(indices) < 3:
        raise ValueError("At least three samples are required for train/val/test splits.")
    shuffled = torch.tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
    n_train = max(1, int(len(shuffled) * train_ratio))
    n_val = max(1, int(len(shuffled) * val_ratio))
    n_test = len(shuffled) - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = len(shuffled) - n_val - n_test
    if n_train < 1:
        raise ValueError("Split ratios leave no training samples; adjust --train-ratio/--val-ratio.")
    train_indices = shuffled[:n_train]
    val_indices = shuffled[n_train:n_train + n_val]
    test_indices = shuffled[n_train + n_val:]
    return train_indices, val_indices, test_indices


def build_dataloaders(args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], int]:
    if not args.data_root:
        raise ValueError("--data-root is required because FakeData support has been removed.")
    train_tf = build_transforms(args.image_size, train=True)
    eval_tf = build_transforms(args.image_size, train=False)
    full_for_split = RGBDepthImageFolder(
        args.data_root,
        transform=train_tf,
        depth_root=args.depth_root,
        treatment_mode=args.treatment_mode,
    )
    eval_source = RGBDepthImageFolder(
        args.data_root,
        transform=eval_tf,
        depth_root=args.depth_root,
        treatment_mode=args.treatment_mode,
    )
    num_classes = full_for_split.num_classes
    treatment_values = full_for_split.treatment_values()
    if len(set(treatment_values)) < 2:
        print(
            "WARNING: all inferred treatment labels are identical. "
            "Your class folders supply outcome classes, not treatment labels; "
            "add t0/t1 markers to filenames or use --treatment-mode label-parity only for debugging."
        )

    n_total = len(full_for_split)
    if n_total < 3:
        raise ValueError("Dataset must contain at least three samples for train/val/test splits.")

    generator = torch.Generator().manual_seed(args.seed)
    all_indices = list(range(n_total))
    if args.split_t1_across_splits:
        train_indices, val_indices, test_indices = split_indices_by_ratio(all_indices, args.train_ratio, args.val_ratio, generator)
    else:
        t0_indices = [idx for idx, treatment in enumerate(treatment_values) if int(treatment) == 0]
        t1_indices = [idx for idx, treatment in enumerate(treatment_values) if int(treatment) == 1]
        if len(t0_indices) < 3:
            raise ValueError(
                "--no-split-t1-across-splits requires at least three t0 samples because "
                "train/val/test ratios are applied to t0 images only."
            )
        if not t1_indices:
            print("WARNING: --no-split-t1-across-splits was set, but no t1 samples were found.")
        train_t0, val_indices, test_indices = split_indices_by_ratio(t0_indices, args.train_ratio, args.val_ratio, generator)
        train_indices = train_t0 + t1_indices
        train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices), generator=generator)].tolist()
        print(
            "Split policy: train/val/test ratios were applied to t0 only; "
            f"all {len(t1_indices)} t1 samples were added to train."
        )

    print(
        f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)} "
        f"(split_t1_across_splits={args.split_t1_across_splits})"
    )
    loaders = {
        "train": DataLoader(Subset(full_for_split, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=_collate),
        "val": DataLoader(Subset(eval_source, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=_collate),
        "test": DataLoader(Subset(eval_source, test_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=_collate),
    }
    return loaders, num_classes


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_parts: Dict[str, torch.Tensor]) -> BatchMetrics:
    y = batch["y"].long().view(-1)
    t = batch["t"].float().view(-1, 1)
    y_pred = torch.argmax(outputs["y_prob"], dim=1)
    t_pred = (outputs["qt_prob"] >= 0.5).float()
    ate = torch.mean(torch.abs(outputs["y1_prob"] - outputs["y0_prob"]))
    return BatchMetrics(
        loss=float(loss_parts["loss"].item()),
        elbo=float(loss_parts["elbo"].item()),
        recon_y=float(loss_parts["recon_y"].item()),
        aux_y=float(loss_parts["aux_y"].item()),
        aux_t=float(loss_parts["aux_t"].item()),
        kl_z=float(loss_parts["kl_z"].item()),
        acc=float((y_pred == y).float().mean().item()),
        t_acc=float((t_pred == t).float().mean().item()),
        ate=float(ate.item()),
    )


def average_metrics(metrics: Iterable[BatchMetrics]) -> Dict[str, float]:
    rows = [asdict(m) for m in metrics]
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]} if rows else {}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train(optimizer is not None)
    rows: List[BatchMetrics] = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(batch)
            loss, loss_parts = tlt_loss(outputs, batch, kl_weight=args.kl_weight, aux_weight=args.aux_weight)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        rows.append(compute_metrics(outputs, batch, loss_parts))
    return average_metrics(rows)


def save_history(history: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    fieldnames = sorted({key for row in history for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib.pyplot as plt

        for metric in ["loss", "recon_y", "aux_y", "aux_t", "kl_z", "acc", "ate"]:
            plt.figure()
            for split in ["train", "val", "test"]:
                xs = [row["epoch"] for row in history if row["split"] == split and metric in row]
                ys = [row[metric] for row in history if row["split"] == split and metric in row]
                if xs:
                    plt.plot(xs, ys, label=split)
            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric}_curve.png", dpi=150)
            plt.close()
    except Exception as exc:  # plotting is a convenience; metrics.csv is the canonical result
        (out_dir / "plot_warning.txt").write_text(f"Could not generate plots: {exc}\n")


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics}, path)


def train(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loaders, num_classes = build_dataloaders(args)
    print(f"Detected/using {num_classes} outcome classes. Treatment mode: {args.treatment_mode}")
    model = CausalTransformer(pretrained_backbone=args.pretrained_backbone, use_attention=not args.no_attention, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, optimizer, args)
        val_metrics = run_epoch(model, loaders["val"], device, None, args)
        scheduler.step()
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            row = {"epoch": epoch, "split": split, **metrics, "lr": scheduler.get_last_lr()[0]}
            history.append(row)
        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")
        save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "best.pt")
        save_history(history, out_dir)

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(model, loaders["test"], device, None, args)
    history.append({"epoch": args.epochs, "split": "test", **test_metrics, "lr": scheduler.get_last_lr()[0]})
    save_history(history, out_dir)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print("test", json.dumps(test_metrics, indent=2))
    return test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a Treatment Learning Transformer for noisy image classification.")
    parser.add_argument("--data-root", type=str, required=True, help="RGB ImageFolder root containing class subdirectories.")
    parser.add_argument("--depth-root", type=str, default="", help="Optional root for paired .npy depth maps. If omitted, each depth file is expected next to its RGB image with the same stem.")
    parser.add_argument("--output-dir", type=str, default="runs/tlt", help="Directory for checkpoints, metrics and loss curves.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--treatment-mode", choices=["filename", "label-parity", "none"], default="filename", help="How to derive binary treatment t. Use filename for t0/t1 markers, label-parity only for debugging when no treatment labels exist, or none to set all t=0.")
    parser.add_argument("--split-t1-across-splits", action=argparse.BooleanOptionalAction, default=True, help="If true, split all samples normally so train/val/test may all contain t1. If false, apply train/val/test ratios to t0 samples only and put every t1 sample into train.")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use ImageNet ResNet-34 weights; requires cached/downloadable weights.")
    parser.add_argument("--no-attention", action="store_true", help="Disable the TLT attention block for ablation/debugging.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
