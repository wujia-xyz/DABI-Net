"""
Train MsGoF on BUSI Dataset with 5-fold cross-validation.
ROI-based method using multi-scale gradational-order fusion.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MsGoFWithROI

# Configuration
RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

DATA_ROOT = './data/busi'
SAVE_DIR = './outputs/BUSI/with_roi/MsGoF'

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        label = self.labels[idx]
        image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask, label


def load_busi_data():
    image_paths, mask_paths, labels = [], [], []
    for label, folder in [(0, 'benign'), (1, 'malignant')]:
        folder_path = os.path.join(DATA_ROOT, folder)
        for fname in os.listdir(folder_path):
            if fname.endswith('.png') and 'mask' not in fname:
                img_path = os.path.join(folder_path, fname)
                base_name = fname.replace('.png', '')
                mask_path = os.path.join(folder_path, f'{base_name}_mask.png')
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    labels.append(label)
    return np.array(image_paths), np.array(mask_paths), np.array(labels)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, masks, labels in dataloader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images, masks)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds, all_probs, all_labels = np.array(all_preds), np.array(all_probs), np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    return {'acc': acc, 'auc': auc, 'sen': sen, 'spe': spe, 'pre': pre, 'f1': f1}


def train_fold(fold, train_idx, val_idx, image_paths, mask_paths, labels, device, log_file):
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"\n{'='*50}\nFold {fold}/5\n{'='*50}")

    train_dataset = BUSIDataset(image_paths[train_idx], mask_paths[train_idx], labels[train_idx], is_train=True)
    val_dataset = BUSIDataset(image_paths[val_idx], mask_paths[val_idx], labels[val_idx], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = MsGoFWithROI(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    best_f1, best_metrics, best_epoch = -1, None, 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for images, masks, labels_batch in train_loader:
            images, masks, labels_batch = images.to(device), masks.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(images, masks)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics['f1'])

        log(f"Epoch {epoch+1}: Loss={train_loss:.4f}, ACC={metrics['acc']:.4f}, AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics.copy()
            best_epoch = epoch + 1
            fold_dir = os.path.join(SAVE_DIR, f'fold{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))
            log(f"  -> New best F1: {best_f1:.4f}")

    fold_dir = os.path.join(SAVE_DIR, f'fold{fold}')
    with open(os.path.join(fold_dir, 'summary.txt'), 'w') as f:
        f.write(f"Fold {fold} Results (Best Epoch: {best_epoch}):\n")
        for k, v in best_metrics.items():
            f.write(f'{k.upper()}: {v:.4f}\n')

    return best_metrics, best_epoch


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Save config
    config = {
        'model': 'MsGoF', 'dataset': 'BUSI', 'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE, 'epochs': NUM_EPOCHS, 'lr': LEARNING_RATE, 'seed': RANDOM_SEED
    }
    with open(os.path.join(SAVE_DIR, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    log_file = open(os.path.join(SAVE_DIR, 'training.log'), 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Device: {device}")

    image_paths, mask_paths, labels = load_busi_data()
    log(f"Total: {len(labels)}, benign={sum(labels==0)}, malignant={sum(labels==1)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        metrics, best_epoch = train_fold(fold, train_idx, val_idx, image_paths, mask_paths, labels, device, log_file)
        all_results.append({'metrics': metrics, 'best_epoch': best_epoch})

    log(f"\n{'='*50}\nFinal Results\n{'='*50}")
    metric_names = ['acc', 'auc', 'sen', 'spe', 'pre', 'f1']

    with open(os.path.join(SAVE_DIR, 'summary.txt'), 'w') as f:
        f.write("MsGoF on BUSI - 5-Fold Results\n" + "="*50 + "\n\n")
        f.write("Per-Fold Results:\n" + "-"*60 + "\n")
        f.write("Fold   ACC      AUC      SEN      SPE      PRE      F1       Epoch\n")
        for i, r in enumerate(all_results):
            m = r['metrics']
            f.write(f"{i}      {m['acc']:.4f}   {m['auc']:.4f}   {m['sen']:.4f}   {m['spe']:.4f}   {m['pre']:.4f}   {m['f1']:.4f}   {r['best_epoch']}\n")
        f.write("\nResults (Mean +/- Std):\n")
        for metric in metric_names:
            values = [r['metrics'][metric] for r in all_results]
            mean_val, std_val = np.mean(values), np.std(values)
            f.write(f"{metric.upper()}: {mean_val:.4f} +/- {std_val:.4f}\n")
            log(f"{metric.upper()}: {mean_val:.4f} +/- {std_val:.4f}")

    log_file.close()
    print(f"\nResults saved to {SAVE_DIR}")


if __name__ == '__main__':
    main()
