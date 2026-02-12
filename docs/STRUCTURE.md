# Repository Structure

- `src/methods/`: method implementations and local training scripts.
  - Single-modal: `dabi_net`, `resnet18`, `vit_b`, `convnext_b`, `sw_forknet`, `hovertrans`
  - ROI-based: `mb_dcnn`, `cam_qus`, `msgof`
  - Multimodal: `mup_net`, `mvmm`, `tdf_net`
- `scripts/train.py`: unified launcher for all methods.
- `scripts/train.sh`: shell wrapper for launcher.
- `configs/methods/`: normalized release configs used directly by `scripts/train.py`.
- `configs/original/`: archival configs in the same YAML schema, marked with `metadata.config_role: archived_original`.
- `configs/dataset/`: dataset-level metadata.
- `configs/PAPER_CONFIGS_FOR_RELEASE.yaml`: method registry (method key -> method config).
- `data/`: placeholder only (datasets not bundled).
- `outputs/`: generated automatically at runtime (not committed).

No pretrained weights/checkpoints are included.
