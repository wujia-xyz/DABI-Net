# Unified Training

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) List methods

```bash
bash scripts/train.sh --list
```

## 3) Start training

By method key:

```bash
bash scripts/train.sh --method dabi_net
bash scripts/train.sh --method resnet18
bash scripts/train.sh --method vit_b
bash scripts/train.sh --method convnext_b
bash scripts/train.sh --method sw_forknet
bash scripts/train.sh --method hovertrans
bash scripts/train.sh --method mb_dcnn
bash scripts/train.sh --method cam_qus
bash scripts/train.sh --method msgof
bash scripts/train.sh --method mup_net
bash scripts/train.sh --method mvmm
bash scripts/train.sh --method tdf_net
```

By config file (normalized run set):

```bash
bash scripts/train.sh --config configs/methods/dabi_net.yaml
bash scripts/train.sh --config configs/methods/tdf_net.yaml
```

By config file (archival aligned set):

```bash
bash scripts/train.sh --config configs/original/dabi_net_paper_config.yaml
```

Optional extra args are passed to the underlying method trainer:

```bash
bash scripts/train.sh --method resnet18 --epochs 50 --batch_size 8
bash scripts/train.sh --method tdf_net --data_dir ./data/us3m/BD3M --label_file ./data/us3m/BD3M.xlsx
```

Optional Python interpreter override:

```bash
PYTHON_BIN=/path/to/python bash scripts/train.sh --method dabi_net
```

## 4) Dataset reminders

- BUSI methods expect `./data/busi/...`
- MUP-Net and MVMM expect ARC fold files in `./data/arc_multimodal/`
- TDF-Net default is US3M/BD3M-style input:
  - image root: `./data/us3m/BD3M`
  - labels: `./data/us3m/BD3M.xlsx`
