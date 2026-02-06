"""
Train ConvNeXt-B on BUSI Dataset with 5-fold cross-validation.
Binary classification: benign vs malignant (excluding normal samples).
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import create_convnext_b

# Fixed output directory
EXP_DIR = './outputs/BUSI/single_modal/ConvNeXt_B'


class BUSIDataset(Dataset):
    """BUSI Dataset for breast ultrasound classification (benign vs malignant)."""

    def __init__(self, image_paths, labels, is_train=True, target_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size

        # Transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_busi_data(data_dir):
    """Load BUSI dataset (benign and malignant only, no masks)."""
    image_paths = []
    labels = []

    # Load benign samples (label=0)
    benign_dir = os.path.join(data_dir, 'benign')
    for fname in os.listdir(benign_dir):
        if fname.endswith('.png') and 'mask' not in fname:
            image_paths.append(os.path.join(benign_dir, fname))
            labels.append(0)

    # Load malignant samples (label=1)
    malignant_dir = os.path.join(data_dir, 'malignant')
    for fname in os.listdir(malignant_dir):
        if fname.endswith('.png') and 'mask' not in fname:
            image_paths.append(os.path.join(malignant_dir, fname))
            labels.append(1)

    return np.array(image_paths), np.array(labels)


def evaluate_metrics(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    # Handle edge case
    if len(np.unique(all_preds)) == 1:
        tn, fp, fn, tp = 0, 0, 0, 0
        if all_preds[0] == 0:
            tn = np.sum(all_labels == 0)
            fn = np.sum(all_labels == 1)
        else:
            fp = np.sum(all_labels == 0)
            tp = np.sum(all_labels == 1)
    else:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    return {
        'acc': accuracy_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        'sen': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'spe': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'pre': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }


def train_one_fold(fold, train_paths, train_labels, val_paths, val_labels, args, log_file):
    """Train one fold."""
    print(f"\n{'='*50}\nTraining Fold {fold}\n{'='*50}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Model
    model = create_convnext_b(
        num_classes=2,
        pretrained=True,
        dropout=0.5
    ).to(device)

    # Data loaders
    train_dataset = BUSIDataset(
        image_paths=train_paths,
        labels=train_labels,
        is_train=True,
        target_size=args.img_size
    )

    val_dataset = BUSIDataset(
        image_paths=val_paths,
        labels=val_labels,
        is_train=False,
        target_size=args.img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Output directory
    fold_dir = os.path.join(EXP_DIR, f'fold{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # Training
    best_f1 = -1
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Evaluate
        metrics = evaluate_metrics(model, val_loader, device)
        log_msg = f'Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, ACC={metrics["acc"]:.4f}, AUC={metrics["auc"]:.4f}, F1={metrics["f1"]:.4f}'
        print(log_msg)
        log_file.write(f'Fold {fold} - {log_msg}\n')

        # Save best model by F1
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics.copy()
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))
            print(f'  New best F1: {best_f1:.4f}')

    # Save fold results
    with open(os.path.join(fold_dir, 'result.txt'), 'w') as f:
        for k, v in best_metrics.items():
            f.write(f'{k}: {v:.4f}\n')

    return best_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/busi',
                        help='Data directory')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--img_size', default=224, type=int, help='Image size')
    parser.add_argument('--n_folds', default=5, type=int, help='Number of folds')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(EXP_DIR, exist_ok=True)

    # Load data
    image_paths, labels = load_busi_data(args.data_dir)

    print(f"Total samples: {len(image_paths)}")
    print(f"Class distribution: benign={np.sum(labels==0)}, malignant={np.sum(labels==1)}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    # Save config
    config_payload = {
        'method': {
            'name': 'ConvNeXt-B',
            'pretrained': True,
            'dropout': 0.5,
        },
        'dataset': {
            'name': 'BUSI',
            'task': 'binary_classification',
            'classes': ['benign', 'malignant'],
            'excluded_class': 'normal',
            'data_dir': args.data_dir,
        },
        'training': {
            'image_size': args.img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            'scheduler': 'CosineAnnealingLR',
            'n_folds': args.n_folds,
            'seed': args.seed,
        },
    }
    with open(os.path.join(EXP_DIR, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    # Training log
    log_file = open(os.path.join(EXP_DIR, 'training_log.txt'), 'w')

    print(f"Output directory: {EXP_DIR}")
    print(f"Data directory: {args.data_dir}")

    # Train all folds
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_paths = image_paths[train_idx]
        train_labels = labels[train_idx]
        val_paths = image_paths[val_idx]
        val_labels = labels[val_idx]

        metrics = train_one_fold(fold, train_paths, train_labels, val_paths, val_labels, args, log_file)
        all_metrics.append(metrics)

    log_file.close()

    # Compute average metrics
    avg_metrics, std_metrics = {}, {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    # Save summary
    with open(os.path.join(EXP_DIR, 'summary.txt'), 'w') as f:
        f.write("5-Fold Cross Validation Results\n")
        f.write("Best model selected by F1 score\n")
        f.write("=" * 50 + "\n")
        for key in ['acc', 'auc', 'sen', 'spe', 'pre', 'f1']:
            f.write(f"{key.upper()}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")

    print("\n" + "=" * 50)
    print("Final Results:")
    for key in ['acc', 'auc', 'sen', 'spe', 'pre', 'f1']:
        print(f"{key.upper()}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")


if __name__ == '__main__':
    main()
