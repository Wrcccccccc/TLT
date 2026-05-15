# Treatment Learning Transformer (TLT)

This repository contains a cleaned-up, runnable implementation of **Treatment Learning Causal Transformer for Noisy Image Classification** (WACV 2023), extended for paired RGB + depth inputs:

- paper: <https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Treatment_Learning_Causal_Transformer_for_Noisy_Image_Classification_WACV_2023_paper.pdf>
- local copy: `水边的猫Yang_Treatment_Learning_Causal_Transformer_for_Noisy_Image_Classification_WACV_2023_paper.pdf`

`causal_trans.py` is self-contained and includes:

- RGB and depth ResNet-34 branches.
- A cross-modal TLT encoder: RGB-derived treatment/outcome features remain the Query, while depth features provide Key and Value for attention.
- A depth-conditioned latent prior `p(z|depth)=N(mu(f_depth), var(f_depth))` instead of a fixed standard Normal prior.
- Multi-class decoder heads for `p(t|z)`, `p(y|z,t=0)` and `p(y|z,t=1)`.
- Variational/auxiliary loss terms following the paper objective; outcome labels use cross-entropy, so folders such as `class0` ... `class4` are handled correctly.
- RGB/depth ImageFolder-style data loading, paired spatial preprocessing, train-time augmentation and evaluation preprocessing.
- Random seed control, AdamW optimizer, cosine learning-rate schedule, gradient clipping, train/validation/test loops and checkpoint/metric export.

## Dataset layout

FakeData support has been removed; `--data-root` is required.  The RGB directory follows `torchvision.datasets.ImageFolder` and supplies the multi-class outcome label `y`:

```text
MTL/
  class0/
    sample001_t0.png
    sample002_t1.png
  class1/
    sample003_t0.png
    sample004_t1.png
```

Each RGB image must have a paired `.npy` depth map.  Two layouts are supported:

1. **Depth next to RGB**: `MTL/class0/sample001_t0.png` pairs with `MTL/class0/sample001_t0.npy`.
2. **Separate depth root**: pass `--depth-root /path/to/depth`; the code first tries the mirrored path, e.g. `/path/to/depth/class0/sample001_t0.npy`, then `/path/to/depth/sample001_t0.npy`.

Filenames or parent paths supply the binary treatment `t` when they contain `t0`, `t1`, `t=0`, `t=1`, `treatment0` or `treatment1`.  If no treatment marker is found, `t` defaults to `0`; use `--treatment-mode label-parity` only as a debugging fallback when your dataset has no treatment labels at all.

## Training

```bash
python causal_trans.py \
  --data-root /path/to/MTL \
  --depth-root /path/to/depth \
  --epochs 50 \
  --batch-size 32 \
  --image-size 128 \
  --pretrained-backbone \
  --output-dir runs/tlt_rgb_depth
```

If `.npy` depth files are stored next to the RGB images, omit `--depth-root`.

## Treatment-aware split policy

By default, `--split-t1-across-splits` is `True`, so the split is performed over all images and `train`, `val` and `test` may each contain `t1` samples, matching the original simple split behavior.

If you want only the training set to contain `t1`, pass `--no-split-t1-across-splits`.  In that mode the `--train-ratio` and `--val-ratio` are applied **only to t0 images**; after the t0 split is created, every `t1` image is appended to the training set, and `val`/`test` contain only `t0`.

```bash
python causal_trans.py \
  --data-root /path/to/MTL \
  --depth-root /path/to/depth \
  --no-split-t1-across-splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --output-dir runs/tlt_t1_train_only
```

## Outputs

Each run writes the following files under `--output-dir`:

- `config.json`: command-line configuration.
- `best.pt` and `last.pt`: checkpoints containing model, optimizer, epoch and validation metrics.
- `metrics.csv`: per-epoch train/validation/test metrics.
- `test_metrics.json`: final test metrics from the best validation checkpoint.
- `*_curve.png`: loss, component loss, accuracy and ATE curves when `matplotlib` is installed.

## Method correspondence

The implementation follows the paper method section, with the requested RGB-depth extension:

- the RGB branch estimates treatment and outcome auxiliary distributions before posterior inference;
- attention uses RGB treatment-query features as Query and depth features as Key/Value;
- latent `z` is sampled by the reparameterization trick;
- the prior is conditioned on depth features, so the KL term is `KL(q(z|x,t,y,depth) || p(z|depth))`;
- the decoder predicts potential outcomes for both treatment arms, enabling batch-level ATE estimates with `mean(y1 - y0)`.

## Citation

```bibtex
@inproceedings{yang2023treatment,
  title={Treatment Learning Causal Transformer for Noisy Image Classification},
  author={Yang, Chao-Han Huck and Hung, I-Te and Liu, Yi-Chieh and Chen, Pin-Yu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6139--6150},
  year={2023}
}
```
