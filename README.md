# Treatment Learning Transformer (TLT)

This repository contains a cleaned-up, runnable implementation of **Treatment Learning Causal Transformer for Noisy Image Classification** (WACV 2023):

- paper: <https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Treatment_Learning_Causal_Transformer_for_Noisy_Image_Classification_WACV_2023_paper.pdf>
- local copy: `水边的猫Yang_Treatment_Learning_Causal_Transformer_for_Noisy_Image_Classification_WACV_2023_paper.pdf`

The original code skeleton omitted several components.  `causal_trans.py` is now self-contained and includes:

- ResNet-34 feature encoder and residual blocks.
- TLT attention-based inference network for `q(t|x)`, `q(y|x,t)` and `q(z|x,t,y)`.
- Multi-class decoder heads for `p(t|z)`, `p(y|z,t=0)` and `p(y|z,t=1)`.
- Variational/auxiliary loss terms following the paper objective; outcome labels use cross-entropy, so folders such as `class0` ... `class4` are handled correctly.
- ImageFolder/FakeData data loaders, train-time augmentation and evaluation preprocessing.
- Random seed control.
- AdamW optimizer, cosine learning-rate schedule and gradient clipping.
- Train/validation/test loops.
- Checkpoint saving (`best.pt`, `last.pt`), `metrics.csv`, `test_metrics.json` and loss/accuracy/ATE curve images.

## Quick smoke test

If you do not have a real dataset ready, run one epoch on synthetic binary data:

```bash
python causal_trans.py \
  --epochs 1 \
  --fake-size 32 \
  --num-classes 5 \
  --image-size 64 \
  --batch-size 4 \
  --workers 0 \
  --device cpu \
  --no-attention \
  --output-dir runs/smoke
```

## Training on an ImageFolder dataset

Expected layout follows `torchvision.datasets.ImageFolder`:

```text
data_root/
  class0/
    image_t0_0001.jpg
    image_t1_0002.jpg
  class1/
    image_t0_0003.jpg
    image_t1_0004.jpg
```

The parent directory supplies the multi-class outcome label `y`; for your `MTL/class0` ... `MTL/class4` layout, the code automatically detects `num_classes=5` and trains with multi-class cross-entropy.  Filenames or parent paths supply the binary treatment `t` when they contain `t0`, `t1`, `t=0`, `t=1`, `treatment0` or `treatment1`.  If no treatment marker is found, `t` defaults to `0`; use `--treatment-mode label-parity` only as a debugging fallback when your dataset has no treatment labels at all.

```bash
python causal_trans.py \
  --data-root /path/to/MTL \
  --epochs 50 \
  --batch-size 32 \
  --image-size 128 \
  --pretrained-backbone \
  --output-dir runs/tlt_experiment
```

## Outputs

Each run writes the following files under `--output-dir`:

- `config.json`: command-line configuration.
- `best.pt` and `last.pt`: checkpoints containing model, optimizer, epoch and validation metrics.
- `metrics.csv`: per-epoch train/validation/test metrics.
- `test_metrics.json`: final test metrics from the best validation checkpoint.
- `*_curve.png`: loss, component loss, accuracy and ATE curves when `matplotlib` is installed.

## Method correspondence

The implementation follows the paper method section:

- the encoder estimates treatment and outcome auxiliary distributions before posterior inference;
- attention uses query features from the selected treatment branch and key/value features from image features;
- latent `z` is sampled by the reparameterization trick;
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
