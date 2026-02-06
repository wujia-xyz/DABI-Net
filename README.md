# DABI-Net: Depth-Aware Bidirectional Interaction Network for ROI-Free Breast Ultrasound Diagnosis

Official code release for the paper:

**DABI-Net: Depth-Aware Bidirectional Interaction Network for Region-of-Interest-Free Breast Ultrasound Diagnosis**

## Abstract
Breast ultrasound imaging is widely used for cancer screening because it is noninvasive and accessible. However, many deep learning methods either rely on pre-defined regions of interest (ROI) or treat ultrasound images as isotropic two-dimensional grids, despite the anisotropic nature of acoustic imaging where the vertical axis corresponds to propagation depth. We show that a self-supervised vision transformer, Distillation with No Labels v2 (DINOv2), produces patch features with depth-dependent organization. We refer to this phenomenon as latent depth encoding. We further find that global average pooling causes information collapse by destroying this organization and discarding diagnostically relevant depth cues. Based on these observations, we propose DABI-Net, an ROI-free architecture that preserves and exploits the latent depth structure through depth sequence modeling. DABI-Net aggregates DINOv2 patch features along horizontal rows to form a one-dimensional depth sequence, augments tokens with hybrid depth encoding that combines learnable embeddings with sinusoidal positional encoding, and applies bidirectional interaction layers in which top-down and bottom-up pathways exchange information through cross-attention. We evaluate DABI-Net on three public breast ultrasound datasets, including the UDIAT diagnostic centre dataset (UDIAT), the Breast Ultrasound Images dataset (BUSI), and a three-modality breast ultrasound dataset (ARC), using five-fold cross-validation. Ablation experiments show that row aggregation alone improves recall and F1 score over global pooling by 0.100 and 0.053, respectively, supporting that the main gain comes from leveraging pre-existing depth structure rather than learning new representations. DABI-Net achieves a recall of 0.929 on BUSI without requiring lesion segmentation masks.

## Highlights
- ROI-free pipeline for DABI-Net (no lesion mask required).
- Unified launcher for all released methods (`scripts/train.py` / `scripts/train.sh`).
- YAML-only configuration system for normalized and archival configs.
- Includes single-modal and multimodal comparison methods used in the paper.
- Code-only release: no pretrained checkpoints are bundled.

## Implemented Methods
Single-modal methods (BUSI):
- DABI-Net, ResNet-18, ViT-B, ConvNeXt-B, SW-ForkNet, HoVerTrans
- ROI-based: MB-DCNN, CAM-QUS, MsGoF

Multimodal methods (paper comparisons):
- MUP-Net (ARC multimodal split files)
- MVMM (ARC multimodal split files)
- TDF-Net (US3M/BD3M-style patient-level split)

## Repository Layout
```text
github_upload_busi_code/
├── configs/
│   ├── dataset/
│   ├── methods/
│   ├── original/
│   └── PAPER_CONFIGS_FOR_RELEASE.yaml
├── scripts/
│   ├── train.py
│   └── train.sh
├── src/
│   └── methods/
├── docs/
├── data/
├── requirements.txt
└── .gitignore
```

## Installation
```bash
pip install -r requirements.txt
```

## Data Preparation
No datasets are shipped with this repository. Place your data under `./data/`.

### 1) BUSI (single-modal methods)
```text
./data/busi/
├── benign/
├── malignant/
└── normal/            # optional in binary setting
```
For ROI-based methods (MB-DCNN / CAM-QUS / MsGoF), mask files (`*_mask.png`) are required.

### 2) ARC multimodal (MUP-Net / MVMM)
```text
./data/arc_multimodal/
├── train_fold0.xlsx
├── ...
├── val_fold4.xlsx
├── 1/
│   ├── ROI_1.png   # BUS
│   ├── ROI_2.png   # CDFI
│   └── ROI_3.png   # UE
└── ...
```

### 3) US3M/BD3M multimodal (TDF-Net)
```text
./data/us3m/
├── BD3M.xlsx
└── BD3M/
    ├── 1/
    │   ├── BUS_*.jpg
    │   ├── DUS_*.jpg
    │   └── EUS_*.jpg
    └── ...
```
Default label columns are `序号` and `病理结果（0：良性，1：恶性）`, configurable via CLI.

## Unified Training
List all available methods:
```bash
bash scripts/train.sh --list
```

Train by method key:
```bash
bash scripts/train.sh --method dabi_net
bash scripts/train.sh --method resnet18
bash scripts/train.sh --method mup_net
bash scripts/train.sh --method mvmm
bash scripts/train.sh --method tdf_net
```

Train with explicit config:
```bash
bash scripts/train.sh --config configs/methods/dabi_net.yaml
bash scripts/train.sh --config configs/methods/tdf_net.yaml
```

Override defaults when needed (example):
```bash
bash scripts/train.sh --method tdf_net --data_dir ./data/us3m/BD3M --label_file ./data/us3m/BD3M.xlsx
```

If needed, pin Python interpreter:
```bash
PYTHON_BIN=/path/to/python bash scripts/train.sh --method dabi_net
```

## Config System
All experiment configs use **YAML** format.

- `configs/methods/`: normalized run configs used by the unified launcher.
- `configs/original/`: archival configs aligned to the same schema for reproducibility/reference (`metadata.config_role: archived_original`).
- `configs/PAPER_CONFIGS_FOR_RELEASE.yaml`: central method registry.

In short: use `configs/methods/` for routine runs, and `configs/original/` for archival matching.

## Reproducibility Notes
- Main registry: `configs/PAPER_CONFIGS_FOR_RELEASE.yaml`
- Default single-modal dataset meta: `configs/dataset/breast_ultrasound_binary.yaml`
- DABI-Net keeps DINOv2 frozen (`freeze_backbone: true`) to keep trainable parameters lightweight.

