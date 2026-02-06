"""Train MUP-Net on ARC multimodal data with cross-validation."""

import argparse
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

import tnt
from dataHelper import MyDataset
from helpers import makedir
from model import PPNet3, resnet18_features

TOPK_K = 5
PROTOTYPE_ACTIVATION = "log"
ADD_ON_LAYERS_TYPE = "regular"
CLASS_SPECIFIC = True
NUM_CLASSES = 2
COEFS = {"crs_ent": 1, "clst": 0.8, "sep": -0.08, "l1": 1e-4}


def construct_ppnet(n_protos: int, img_size: int, pretrained: bool = True) -> PPNet3:
    """Build the 3-branch prototype network used in MUP-Net."""
    prototype_shape = (n_protos, 512, 1, 1)
    features1 = resnet18_features(pretrained=pretrained)
    features2 = resnet18_features(pretrained=pretrained)
    features3 = resnet18_features(pretrained=pretrained)

    return PPNet3(
        features=[features1, features2, features3],
        img_size=img_size,
        prototype_shape=prototype_shape,
        topk_k=TOPK_K,
        num_classes=NUM_CLASSES,
        init_weights=True,
        prototype_activation_function=PROTOTYPE_ACTIVATION,
        add_on_layers_type=ADD_ON_LAYERS_TYPE,
        class_specific=CLASS_SPECIFIC,
    )


def evaluate_metrics(model, dataloader, device: str) -> dict:
    """Evaluate and return binary classification metrics."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)[0]
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
    """Train one fold and save the fold-best checkpoint by F1."""
    print("\n" + "=" * 50 + f"\nTraining Fold {fold}\n" + "=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ppnet = construct_ppnet(args.nprotos, args.img_size, pretrained=args.pretrained_backbone).to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    def merge_params(params):
        return [p for ps in params for p in ps]

    warm_optimizer = torch.optim.Adam(
        [
            {
                "params": merge_params([p.add_on_layers.parameters() for p in ppnet.pnet123]),
                "lr": 2e-3,
                "weight_decay": 1e-3,
            },
            {"params": [p.prototype_vectors for p in ppnet.pnet123], "lr": 3e-3},
        ]
    )
    joint_optimizer = torch.optim.Adam(
        [
            {
                "params": merge_params([p.features.parameters() for p in ppnet.pnet123]),
                "lr": 2e-4,
                "weight_decay": 1e-3,
            },
            {
                "params": merge_params([p.add_on_layers.parameters() for p in ppnet.pnet123]),
                "lr": 3e-3,
                "weight_decay": 1e-3,
            },
            {"params": [p.prototype_vectors for p in ppnet.pnet123], "lr": 3e-3},
        ]
    )
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)

    train_file = os.path.join(args.data_dir, f"train_fold{fold}.xlsx")
    val_file = os.path.join(args.data_dir, f"val_fold{fold}.xlsx")
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(
            f"Missing fold files for fold {fold}. Expected: {train_file} and {val_file}"
        )

    target_size = [args.img_size, args.img_size]
    train_loader = torch.utils.data.DataLoader(
        MyDataset(file_list=train_file, root_dir=args.data_dir, is_train=True, target_size=target_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        MyDataset(file_list=val_file, root_dir=args.data_dir, is_train=False, target_size=target_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    fold_dir = os.path.join(save_dir, f"fold{fold}")
    makedir(fold_dir)

    best_f1 = -1.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        log_file.write(f"Fold {fold} - Epoch {epoch}\n")

        if epoch <= args.warm_epochs:
            tnt.warm_only(model123=ppnet_multi, log=print)
            tnt.train(
                model123=ppnet_multi,
                dataloader=train_loader,
                optimizer=warm_optimizer,
                dev=device,
                class_specific=CLASS_SPECIFIC,
                coefs=COEFS,
                log=print,
            )
        else:
            tnt.joint(model123=ppnet_multi, log=print)
            tnt.train(
                model123=ppnet_multi,
                dataloader=train_loader,
                optimizer=joint_optimizer,
                dev=device,
                class_specific=CLASS_SPECIFIC,
                coefs=COEFS,
                log=print,
            )
            joint_lr_scheduler.step()

        metrics = evaluate_metrics(ppnet_multi, val_loader, device)
        log_msg = (
            f"  Val - ACC: {metrics[acc]:.4f}, AUC: {metrics[auc]:.4f}, "
            f"F1: {metrics[f1]:.4f}, SEN: {metrics[sen]:.4f}, "
            f"SPE: {metrics[spe]:.4f}, PRE: {metrics[pre]:.4f}"
        )
        print(log_msg)
        log_file.write(log_msg + "\n")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            torch.save(ppnet.state_dict(), os.path.join(fold_dir, "best_model.pth"))
            print(f"  New best F1: {best_f1:.4f}")

    with open(os.path.join(fold_dir, "result.txt"), "w", encoding="utf-8") as f:
        for key, val in best_metrics.items():
            f.write(f"{key}: {val:.4f}\n")

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="MUP-Net training on ARC multimodal dataset")
    parser.add_argument("--data_dir", default="./data/arc_multimodal", help="ARC multimodal data directory")
    parser.add_argument("--save_dir", default="./outputs/ARC/multimodal/MUP_Net", help="Output directory")
    parser.add_argument("--nprotos", default=6, type=int, help="Prototypes per class")
    parser.add_argument("--folds", default=5, type=int, help="Number of folds")
    parser.add_argument("--epochs", default=40, type=int, help="Training epochs")
    parser.add_argument("--warm_epochs", default=10, type=int, help="Warm-up epochs")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
    parser.add_argument("--img_size", default=224, type=int, help="Input image size")
    parser.add_argument("--num_workers", default=4, type=int, help="Data loader workers")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--pretrained_backbone",
        dest="pretrained_backbone",
        action="store_true",
        help="Use ImageNet-pretrained ResNet-18 backbones",
    )
    parser.add_argument(
        "--no_pretrained_backbone",
        dest="pretrained_backbone",
        action="store_false",
        help="Disable ImageNet pretrained backbones",
    )
    parser.set_defaults(pretrained_backbone=True)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    makedir(args.save_dir)

    config_payload = {
        "method": {
            "name": "MUP-Net",
            "architecture": "PPNet3",
            "prototypes_per_class": args.nprotos,
            "pretrained_backbone": args.pretrained_backbone,
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
            "warm_epochs": args.warm_epochs,
            "optimizer_warm": "Adam",
            "optimizer_joint": "Adam",
            "joint_scheduler": "StepLR(step_size=5,gamma=0.1)",
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
