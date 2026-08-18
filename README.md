# ScaleErasure

> Official Code of ICML 2026 paper *ScaleErasure: Inference-Time Minimal
> Intervention for Precise Concept Erasure in Next-Scale Autoregressive Image
> Generation*.

[![Paper](https://img.shields.io/badge/arXiv-2606.29282-b31b1b.svg)](https://arxiv.org/abs/2606.29282)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ScaleErasure is an inference-time concept-erasure method for Infinity. It keeps
the pretrained weights frozen and applies selective logits guidance over the
scale, token, and bit-channel dimensions. This release contains the public
inference path and the minimal Infinity runtime required to reproduce the
paper experiments.

## Repository layout

```text
.
├── scaleerasure/                  # Configuration, data, backend, and sampler
├── configs/                       # Paper I2P and MS-COCO configurations
├── scripts/run_scaleerasure.py    # YAML launcher
├── scripts/export_token_masks.py  # Token-mask inspection utility
├── tools/run_infinity.py          # Infinity model-loading utilities
├── infinity/                      # Minimal Infinity inference runtime
├── DockerFile                     # CUDA 11.8 container recipe
└── metrics/                       # Optional evaluation utilities
```

The vendored Infinity runtime is derived from the official
[Infinity implementation](https://github.com/FoundationVision/Infinity) and
is retained here so the ScaleErasure inference path is self-contained.

## Installation

The implementation was tested with Python 3.10, PyTorch 2.5.1, CUDA 11.8,
FlashAttention, and an NVIDIA GPU. Install a CUDA-enabled PyTorch build first,
then install the dependencies:

```bash
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## Model weights

Download the following files from the official
[Infinity weights](https://huggingface.co/FoundationVision/Infinity) and
[Flan-T5-XL](https://huggingface.co/google/flan-t5-xl) repositories:

```text
infinity_2b_reg.pth
infinity_vae_d32reg.pth
flan-t5-xl/
```

Set the local paths before inference:

```bash
export SCALEERASURE_MODEL_ROOT=/path/to/infinity-weights
export SCALEERASURE_TEXT_ENCODER=/path/to/flan-t5-xl
export SCALEERASURE_DATASET_CACHE=/path/to/huggingface-datasets
```

Weights, datasets, caches, and generated images are intentionally excluded
from Git.

## I2P inference

Run the full I2P configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_scaleerasure.py \
  --config configs/scaleerasure_i2p.yaml
```

For a quick local verification, limit the same paper configuration to a few
samples at runtime:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_scaleerasure.py \
  --config configs/scaleerasure_i2p.yaml \
  --max_samples 3
```

Outputs are saved under `outputs/` and are ignored by Git.

## Token-mask inspection

Export the last active image-token mask for one sexual I2P prompt and one
normal COCO caption:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_token_masks.py \
  --config configs/scaleerasure_i2p.yaml \
  --i2p_index 0 \
  --coco_prompt "A bicycle replica with a clock as the front wheel."
```

The output directory contains the generated image, the binary mask, a
relevance heatmap, and a red overlay. The PNGs show the first temporal plane
of scale 7 (the last scale before the default early exit at scale 8); the
compressed `.npz` file keeps all flattened image tokens for exact inspection.

## Evaluation

Install optional metric dependencies with:

```bash
pip install -r requirements-metrics.txt
```

Then use `metrics/compute_metrics.py` for CLIP, FID, LPIPS, and paper-compatible
NudeNet body-part evaluation. NudeNet filters exposed body-part detections at
threshold `0.6` and reports `common`, `female`, `male`, `total`, and image-level
`ASR`.

For example:

```bash
python metrics/compute_metrics.py \
  --gen_dir outputs/i2p/images \
  --metrics nudenet \
  --nudenet_threshold 0.6 \
  --output_json outputs/i2p/nudenet.json
```

## Citation

```bibtex
@article{wang2026scaleerasure,
  title={ScaleErasure: Inference-Time Minimal Intervention for Precise Concept Erasure in Next-Scale Autoregressive Image Generation},
  author={Wang, Cong and Wu, Haiyu and Jiang, Zhiwei and Cheng, Zifeng and Shen, Fei and Yin, Yafeng and Gu, Qing},
  journal={arXiv preprint arXiv:2606.29282},
  year={2026}
}
```

## License

ScaleErasure is released under the MIT License. Please also review the
licenses and usage terms of Infinity, Flan-T5, FlashAttention, the datasets,
and the optional evaluation models.
