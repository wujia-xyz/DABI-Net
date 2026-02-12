"""
Train HoVerTrans on BUSI dataset with 5-fold cross-validation.
Binary classification: benign vs malignant (excluding normal samples).
Uses original HoVerTrans configuration from the paper.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import random
import math
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from hovertrans import create_model

# Fixed output directory
EXP_DIR = './outputs/BUSI/single_modal/HoVerTrans'

# Original HoVerTrans configuration from paper
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 250
LR = 0.0001
WARMUP_EPOCHS = 10
MIN_LR = 1e-6
LOG_STEP = 5


class AddGaussianNoise:
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            if len(img.shape) == 2:
                h, w = img.shape
            else:
                h, w, _ = img.shape
                img = img[:, :, 0]  # Convert to grayscale
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255
            img[img < 0] = 0
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img


class AddBlur:
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        import cv2
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img


class BUSIDataset(Dataset):
    """BUSI dataset for HoVerTrans (grayscale, 256x256)."""

    def __init__(self, image_paths, labels, is_train=True):
        self.is_train = is_train
        self.image_paths = image_paths
        self.labels = labels

        # Original HoVerTrans transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.Grayscale(),
                AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
                AddBlur(kernel=3, p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Expand grayscale to 3 channels for HoVerTrans
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)

        return {'imgs': image, 'labels': label, 'names': img_path}


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
    """Evaluate model."""
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for pack in dataloader:
            images = pack['imgs'].to(device)
            labels = pack['labels']

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

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


def train_one_fold(fold, train_paths, train_labels, val_paths, val_labels, log_file):
    """Train one fold with original HoVerTrans config."""
    print(f"\n{'='*50}\nTraining Fold {fold}\n{'='*50}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create model with original HoVerTrans config
    model = create_model(
        img_size=IMG_SIZE,
        num_classes=2,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        patch_size=[2, 2, 2, 2],
        dim=[4, 8, 16, 32],
        depth=[2, 4, 4, 2],
        num_heads=[2, 4, 8, 16],
        num_inner_head=[2, 4, 8, 16]
    ).to(device)

    # Data loaders
    train_dataset = BUSIDataset(train_paths, train_labels, is_train=True)
    val_dataset = BUSIDataset(val_paths, val_labels, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Loss and optimizer (original config)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    # Cosine scheduler with warmup (original config)
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return epoch * (1 - 0.01) / WARMUP_EPOCHS + 0.01
        else:
            return (1 - MIN_LR / LR) * 0.5 * (math.cos((epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS) * math.pi) + 1) + MIN_LR / LR

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # Output directory
    fold_dir = os.path.join(EXP_DIR, f'fold{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # Training
    best_f1 = -1
    best_metrics = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for pack in train_loader:
            images = pack['imgs'].to(device)
            labels = pack['labels'].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Evaluate every LOG_STEP epochs
        if epoch % LOG_STEP == 0:
            metrics = evaluate_metrics(model, val_loader, device)
            log_msg = f'Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, ACC={metrics["acc"]:.4f}, AUC={metrics["auc"]:.4f}, F1={metrics["f1"]:.4f}'
            print(log_msg)
            log_file.write(f'Fold {fold} - {log_msg}\n')

            # Save best model by F1 (after 1/4 epochs as in original)
            if epoch > EPOCHS // 4 and metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_metrics = metrics.copy()
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))
                print(f'  New best F1: {best_f1:.4f}')

    # If no best model saved, use last epoch
    if best_metrics is None:
        best_metrics = evaluate_metrics(model, val_loader, device)
        torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))

    # Save fold results
    with open(os.path.join(fold_dir, 'result.txt'), 'w') as f:
        for k, v in best_metrics.items():
            f.write(f'{k}: {v:.4f}\n')

    return best_metrics


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_torch(42)

    data_dir = './data/busi'

    # Create output directory
    os.makedirs(EXP_DIR, exist_ok=True)

    # Load data
    image_paths, labels = load_busi_data(data_dir)

    print(f"Total samples: {len(image_paths)}")
    print(f"Class distribution: benign={np.sum(labels==0)}, malignant={np.sum(labels==1)}")

    # 5-fold stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Save config
    config_payload = {
        'method': {
            'name': 'HoVerTrans',
            'variant': 'paper_default',
            'drop_rate': 0.1,
        },
        'dataset': {
            'name': 'BUSI',
            'task': 'binary_classification',
            'classes': ['benign', 'malignant'],
            'excluded_class': 'normal',
            'data_dir': data_dir,
            'total_samples': len(image_paths),
        },
        'training': {
            'image_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'optimizer': 'AdamW',
            'weight_decay': 0.1,
            'scheduler': 'CosineWithWarmup',
            'warmup_epochs': WARMUP_EPOCHS,
            'min_lr': MIN_LR,
            'n_folds': 5,
            'seed': 42,
        },
    }
    with open(os.path.join(EXP_DIR, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    # Training log
    log_file = open(os.path.join(EXP_DIR, 'training_log.txt'), 'w')

    print(f"Output directory: {EXP_DIR}")
    print(f"Data directory: {data_dir}")

    # Train all folds
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_paths = image_paths[train_idx]
        train_labels = labels[train_idx]
        val_paths = image_paths[val_idx]
        val_labels = labels[val_idx]

        metrics = train_one_fold(fold, train_paths, train_labels, val_paths, val_labels, log_file)
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
        f.write("HoVerTrans on BUSI Dataset\n")
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
