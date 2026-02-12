"""
Train DABI-Net v2 on BUSI dataset with 5-fold cross-validation.
Binary classification: benign vs malignant (no normal samples, no masks).
Using configuration from configs/methods/dabi_net.yaml
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
import math
warnings.filterwarnings('ignore')

# Add model path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.dabi_net_v2_model import DINOv2DABINetV2Model


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


class BUSIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class CosineWarmupScheduler:
    """Cosine annealing with warmup scheduler."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), all_preds, all_labels


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), all_preds, all_probs, all_labels


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # sensitivity

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        'acc': acc,
        'auc': auc,
        'sen': recall,
        'spe': specificity,
        'pre': precision,
        'f1': f1
    }


def main():
    # Configuration from config.yaml
    data_dir = './data/busi'
    output_dir = './outputs/BUSI/single_modal/DABI_Net_v2'

    # Training config
    num_epochs = 100
    batch_size = 16
    learning_rate = 1e-5
    min_lr = 1e-6
    weight_decay = 0.01  # From config
    label_smoothing = 0.1  # From config
    warmup_epochs = 5
    num_folds = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    image_paths, labels = load_busi_data(data_dir)
    print(f"Total samples: {len(image_paths)}")
    print(f"Class distribution: benign={sum(labels==0)}, malignant={sum(labels==1)}")
    print(f"Output directory: {output_dir}")
    print(f"Data directory: {data_dir}")

    # Compute class weights
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold}")
        print(f"{'='*50}")

        # Create fold directory
        fold_dir = os.path.join(output_dir, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # Split data
        train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # Create datasets
        train_dataset = BUSIDataset(train_paths, train_labels, train_transform)
        val_dataset = BUSIDataset(val_paths, val_labels, val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"Using device: {device}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Create model - DABI-Net v2 with config settings
        model = DINOv2DABINetV2Model(
            num_classes=2,
            dinov2_model='dinov2_vitb14',
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            dropout=0.3,
            freeze_backbone=False,
            load_method='hf',  # Use HuggingFace for public release
            pooling='mean',
            row_pooling='attention',
            depth_embed_dim=64,
            simple_classifier=True,
            bidirectional_mode='true_bidirectional'
        )
        model = model.to(device)

        # Loss with label smoothing and class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        # Optimizer - all parameters since backbone is not frozen
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Cosine scheduler with warmup
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=warmup_epochs, total_epochs=num_epochs,
            min_lr=min_lr, base_lr=learning_rate
        )

        # Training loop
        best_f1 = -1
        best_metrics = None

        for epoch in range(num_epochs):
            train_loss, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_probs, val_labels_epoch = evaluate(model, val_loader, criterion, device)
            current_lr = scheduler.step()

            metrics = compute_metrics(val_labels_epoch, val_preds, val_probs)

            print(f"Epoch {epoch+1}: LR={current_lr:.2e}, Loss={train_loss:.4f}, ACC={metrics['acc']:.4f}, AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_metrics = metrics.copy()
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))
                print(f"  New best F1: {best_f1:.4f}")

        # Save fold results
        all_fold_results.append(best_metrics)

        with open(os.path.join(fold_dir, 'summary.txt'), 'w') as f:
            f.write(f"Fold {fold} Results:\n")
            for k, v in best_metrics.items():
                f.write(f"{k.upper()}: {v:.4f}\n")

    # Compute mean and std across folds
    print(f"\n{'='*50}")
    print("Final Results:")
    metric_names = ['acc', 'auc', 'sen', 'spe', 'pre', 'f1']
    final_results = {}

    for metric in metric_names:
        values = [r[metric] for r in all_fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_results[metric] = (mean_val, std_val)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    # Save final results
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("DABI-Net v2 on BUSI Dataset - 5-Fold Cross-Validation Results\n")
        f.write("="*50 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  - freeze_backbone: False\n")
        f.write(f"  - lr: {learning_rate}\n")
        f.write(f"  - scheduler: cosine (warmup={warmup_epochs}, min_lr={min_lr})\n")
        f.write(f"  - label_smoothing: {label_smoothing}\n")
        f.write(f"  - weight_decay: {weight_decay}\n")
        f.write(f"  - use_class_weights: True\n")
        f.write(f"  - bidirectional_mode: true_bidirectional\n\n")
        f.write("Results:\n")
        for metric in metric_names:
            mean_val, std_val = final_results[metric]
            f.write(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}\n")


if __name__ == '__main__':
    main()
