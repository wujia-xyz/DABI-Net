# Data Placeholder

Datasets are not included in this repository.

Place data under the following paths:

## 1) BUSI (single-modal)
- `data/busi/benign/`
- `data/busi/malignant/`
- `data/busi/normal/` (optional in binary setting)

ROI-based methods on BUSI additionally require mask files (`*_mask.png`).

## 2) ARC multimodal (MUP-Net / MVMM)
- `data/arc_multimodal/train_fold0.xlsx` ... `train_fold4.xlsx`
- `data/arc_multimodal/val_fold0.xlsx` ... `val_fold4.xlsx`
- `data/arc_multimodal/<patient_id>/ROI_1.png`
- `data/arc_multimodal/<patient_id>/ROI_2.png`
- `data/arc_multimodal/<patient_id>/ROI_3.png`

## 3) US3M/BD3M multimodal (TDF-Net)
- `data/us3m/BD3M.xlsx`
- `data/us3m/BD3M/<patient_id>/BUS_*.jpg`
- `data/us3m/BD3M/<patient_id>/DUS_*.jpg`
- `data/us3m/BD3M/<patient_id>/EUS_*.jpg`
