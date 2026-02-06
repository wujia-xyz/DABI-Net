"""Train TDF-Net with patient-level stratified K-fold cross-validation."""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Keep local module imports stable when launched from repository root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import MultimodalBUSDataset, get_train_transform, get_val_transform
from utils_model.TDFNet import TDF_Net


def calculate_metrics(labels, preds, probs):
    """Compute ACC/AUC/SEN/SPE/PRE/F1 for binary classification."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    acc = accuracy_score(labels, preds)
    sen = recall_score(labels, preds, zero_division=0)
    pre = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    if len(np.unique(preds)) == 1:
        tn, fp, fn, tp = 0, 0, 0, 0
        if preds[0] == 0:
            tn = int(np.sum(labels == 0))
            fn = int(np.sum(labels == 1))
        else:
            fp = int(np.sum(labels == 0))
            tp = int(np.sum(labels == 1))
    else:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5

    return {
        "acc": acc,
        "auc": auc,
        "sen": sen,
        "spe": spe,
        "pre": pre,
        "f1": f1,
    }


def train_one_epoch(model, loader, optimizer, epoch, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        img1, img2, img3 = images[0].to(device), images[1].to(device), images[2].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        evidence, loss_ct = model(img1, img2, img3, labels, epoch)
        loss = criterion(evidence, labels.long()) + loss_ct

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(evidence, dim=1)
        pred = evidence.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def validate(model, loader, epoch, device):
    """Run validation for one epoch."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            img1, img2, img3 = images[0].to(device), images[1].to(device), images[2].to(device)
            labels = labels.to(device)

            evidence, loss_ct = model(img1, img2, img3, labels, epoch)
            loss = criterion(evidence, labels.long()) + loss_ct
            total_loss += loss.item()

            probs = torch.softmax(evidence, dim=1)
            pred = evidence.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def train_fold(args, fold, train_ids, val_ids, device, log_file):
    """Train one fold and save best-F1 model."""
    print("\n" + "=" * 50)
    print(f"Fold {fold + 1}/{args.n_folds}")
    print("=" * 50)

    train_dataset = MultimodalBUSDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        patient_ids=train_ids,
        transform=get_train_transform(args.img_size),
        img_size=args.img_size,
        id_col=args.id_col,
        label_col=args.label_col,
    )
    val_dataset = MultimodalBUSDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        patient_ids=val_ids,
        transform=get_val_transform(args.img_size),
        img_size=args.img_size,
        id_col=args.id_col,
        label_col=args.label_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = TDF_Net(
        num_class=2,
        depth=-4,
        pretrain=args.pretrained,
        backbone=args.backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_f1 = -1.0
    best_metrics = None

    fold_dir = os.path.join(args.save_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, epoch, device)
        scheduler.step()

        val_metrics = validate(model, val_loader, epoch, device)

        if (epoch + 1) % args.log_step == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_metrics[loss]:.4f}, Train F1: {train_metrics[f1]:.4f} | "
                f"Val F1: {val_metrics[f1]:.4f}, Val AUC: {val_metrics[auc]:.4f}"
            )

        log_info = (
            f"[Fold {fold}, Epoch {epoch + 1}]\n"
            f"train_loss: {train_metrics[loss]:.6f}\n"
            f"train_acc: {train_metrics[acc]:.4f}\n"
            f"train_auc: {train_metrics[auc]:.4f}\n"
            f"train_sen: {train_metrics[sen]:.4f}\n"
            f"train_spe: {train_metrics[spe]:.4f}\n"
            f"train_pre: {train_metrics[pre]:.4f}\n"
            f"train_f1: {train_metrics[f1]:.4f}\n"
            f"valid_loss: {val_metrics[loss]:.6f}\n"
            f"valid_acc: {val_metrics[acc]:.4f}\n"
            f"valid_auc: {val_metrics[auc]:.4f}\n"
            f"valid_sen: {val_metrics[sen]:.4f}\n"
            f"valid_spe: {val_metrics[spe]:.4f}\n"
            f"valid_pre: {val_metrics[pre]:.4f}\n"
            f"valid_f1: {val_metrics[f1]:.4f}\n\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_info)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics.copy()
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
            print(f"  => Saved best model (F1: {best_f1:.4f})")

    with open(os.path.join(fold_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best Results for Fold {fold} (based on F1):\n")
        f.write("=" * 40 + "\n")
        f.write(f"ACC: {best_metrics[acc]:.4f}\n")
        f.write(f"AUC: {best_metrics[auc]:.4f}\n")
        f.write(f"SEN: {best_metrics[sen]:.4f}\n")
        f.write(f"SPE: {best_metrics[spe]:.4f}\n")
        f.write(f"PRE: {best_metrics[pre]:.4f}\n")
        f.write(f"F1:  {best_metrics[f1]:.4f}\n")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="TDF-Net training on US3M/BD3M multimodal dataset")
    parser.add_argument("--data_dir", type=str, default="./data/us3m/BD3M")
    parser.add_argument("--label_file", type=str, default="./data/us3m/BD3M.xlsx")
    parser.add_argument("--save_dir", type=str, default="./outputs/ARC/multimodal/TDF_Net")
    parser.add_argument("--id_col", type=str, default="序号", help="Patient ID column name in label_file")
    parser.add_argument(
        "--label_col",
        type=str,
        default="病理结果（0：良性，1：恶性）",
        help="Binary label column name in label_file",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--log_step", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="vit", choices=["vit", "resnet"])
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use pretrained backbone")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false", help="Disable pretrained backbone")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_excel(args.label_file)
    if args.id_col not in df.columns or args.label_col not in df.columns:
        raise KeyError(
            f"Columns not found in {args.label_file}. "
            f"Expected id_col={args.id_col}, label_col={args.label_col}."
        )

    patient_ids = df[args.id_col].values
    labels = df[args.label_col].values

    valid_mask = [os.path.exists(os.path.join(args.data_dir, str(pid))) for pid in patient_ids]
    patient_ids = patient_ids[valid_mask]
    labels = labels[valid_mask]

    print(f"Total patients: {len(patient_ids)}")
    print(f"Benign: {(labels == 0).sum()}, Malignant: {(labels == 1).sum()}")

    os.makedirs(args.save_dir, exist_ok=True)

    config_payload = {
        "method": {
            "name": "TDF-Net",
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "depth": -4,
        },
        "dataset": {
            "name": "US3M/BD3M",
            "modalities": ["BUS", "CDFI", "UE"],
            "data_dir": args.data_dir,
            "label_file": args.label_file,
            "id_col": args.id_col,
            "label_col": args.label_col,
        },
        "training": {
            "image_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=20,T_mult=2)",
            "n_folds": args.n_folds,
            "log_step": args.log_step,
            "seed": args.seed,
        },
    }
    with open(os.path.join(args.save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.save_dir, f"training_log_{timestamp}.txt")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        train_ids = patient_ids[train_idx].tolist()
        val_ids = patient_ids[val_idx].tolist()

        fold_results = train_fold(args, fold, train_ids, val_ids, device, log_file)
        all_results.append(fold_results)

    print("\n" + "=" * 60)
    print("5-Fold Cross Validation Results (Patient-Level Split)")
    print("Best model selected by F1 score")
    print("=" * 60)

    metrics_names = ["acc", "auc", "sen", "spe", "pre", "f1"]
    summary = {}
    for name in metrics_names:
        values = [r[name] for r in all_results]
        mean = np.mean(values)
        std = np.std(values)
        summary[name] = {"mean": mean, "std": std}
        print(f"{name.upper():12s}: {mean:.4f} ± {std:.4f}")

    summary_file = os.path.join(args.save_dir, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("5-Fold Cross Validation Results (Patient-Level Split)\n")
        f.write("Best model selected by F1 score\n")
        f.write("=" * 50 + "\n")
        for name in metrics_names:
            f.write(f"{name.upper()}: {summary[name][mean]:.4f} ± {summary[name][std]:.4f}\n")

    print(f"\nResults saved to: {args.save_dir}")
    print(f"Training log: {log_file}")


if __name__ == "__main__":
    main()
