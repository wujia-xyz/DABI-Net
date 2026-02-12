"""Train MVMM on ARC multimodal data with cross-validation."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from dataset import get_dataloaders
from model import create_mvmm


def evaluate_metrics(model, dataloader, device: str) -> dict:
    """Evaluate model and return binary classification metrics."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if len(np.unique(all_preds)) == 1:
        tn, fp, fn, tp = 0, 0, 0, 0
        if all_preds[0] == 0:
            tn = int(np.sum(all_labels == 0))
            fn = int(np.sum(all_labels == 1))
        else:
            fp = int(np.sum(all_labels == 0))
            tp = int(np.sum(all_labels == 1))
    else:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    return {
        "acc": accuracy_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        "sen": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "spe": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "pre": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


def train_one_fold(fold: int, args: argparse.Namespace, log_file, save_dir: str) -> dict:
    """Train one fold and keep the checkpoint with best F1."""
    print("\n" + "=" * 50 + f"\nTraining Fold {fold}\n" + "=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = create_mvmm(num_classes=2, num_modalities=3, pretrained=args.pretrained).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        fold=fold,
        batch_size=args.batch_size,
        target_size=args.img_size,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    fold_dir = os.path.join(save_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    best_f1 = -1.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        metrics = evaluate_metrics(model, val_loader, device)
        log_msg = (
            f"Epoch {epoch}: Loss={train_loss / len(train_loader):.4f}, "
            f"ACC={metrics[acc]:.4f}, AUC={metrics[auc]:.4f}, F1={metrics[f1]:.4f}"
        )
        print(log_msg)
        log_file.write(f"Fold {fold} - {log_msg}\n")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, os.path.join(fold_dir, "best_model.pth"))
            print(f"  New best F1: {best_f1:.4f}")

    with open(os.path.join(fold_dir, "result.txt"), "w", encoding="utf-8") as f:
        for key, val in best_metrics.items():
            f.write(f"{key}: {val:.4f}\n")

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="MVMM training on ARC multimodal dataset")
    parser.add_argument("--data_dir", default="./data/arc_multimodal", help="ARC multimodal data directory")
    parser.add_argument("--save_dir", default="./outputs/ARC/multimodal/MVMM", help="Output directory")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--img_size", default=224, type=int, help="Input image size")
    parser.add_argument("--folds", default=5, type=int, help="Number of folds")
    parser.add_argument("--num_workers", default=4, type=int, help="Data loader workers")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false", help="Disable pretrained weights")
    parser.set_defaults(pretrained=True)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    config_payload = {
        "method": {
            "name": "MVMM",
            "backbone": "ResNet-18 + SE",
            "pretrained": args.pretrained,
        },
        "dataset": {
            "name": "ARC multimodal",
            "modalities": ["BUS", "CDFI", "UE"],
            "data_dir": args.data_dir,
            "split_files": "train_fold{k}.xlsx / val_fold{k}.xlsx",
        },
        "training": {
            "image_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "scheduler": "StepLR(step_size=50,gamma=0.5)",
            "n_folds": args.folds,
            "seed": args.seed,
        },
    }
    with open(os.path.join(args.save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    log_file = open(os.path.join(args.save_dir, "training_log.txt"), "w", encoding="utf-8")

    print(f"Output directory: {args.save_dir}")
    print(f"Data directory: {args.data_dir}")

    all_metrics = []
    for fold in range(args.folds):
        metrics = train_one_fold(fold, args, log_file, args.save_dir)
        all_metrics.append(metrics)

    log_file.close()

    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    with open(os.path.join(args.save_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("5-Fold Cross Validation Results (Patient-Level Split)\n")
        f.write("Best model selected by F1 score\n")
        f.write("=" * 50 + "\n")
        for key in ["acc", "auc", "sen", "spe", "pre", "f1"]:
            f.write(f"{key.upper()}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")

    print("\n" + "=" * 50)
    print("Final Results:")
    for key in ["acc", "auc", "sen", "spe", "pre", "f1"]:
        print(f"{key.upper()}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
