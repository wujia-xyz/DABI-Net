"""
Train CAM-QUS on BUSI Dataset with 5-fold cross-validation.
ROI-based method using CAM loss with ground truth masks.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import yaml

# Configuration
RANDOM_SEED = 42
IMAGE_SIZE = 299
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

DATA_ROOT = './data/busi'
SAVE_DIR = './outputs/BUSI/with_roi/CAM-QUS'

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class SEBlock(nn.Module):
    def __init__(self, n_features, reduction=8):
        super(SEBlock, self).__init__()
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.permute(0, 2, 3, 1)
        y = self.relu(self.linear1(y))
        y = self.sigmoid(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output * x


class PrepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.conv2 = nn.Conv2d(64, 1, 1)
        self.seb = SEBlock(64)
        self.sp = SpatialAttention(7)

    def forward(self, x):
        identity = x[:, 2:3, :, :]
        b, c, h, w = x.shape
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1[0].conv1(x)
        x = self.resnet.layer1[0].bn1(x)
        x = self.resnet.layer1[0].conv2(x)
        x = self.resnet.layer1[0].bn2(x)
        x = self.resnet.layer1[1].conv1(x)
        x = self.resnet.layer1[1].bn1(x)
        x = self.resnet.layer1[1].conv2(x)
        x = self.resnet.layer1[1].bn2(x)
        spa = self.sp(x)
        x = x + spa
        x = self.seb(x)
        x = F.interpolate(self.conv2(x), (h, w))
        x = identity + x
        return x


class CAMQUSModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.inception = models.inception_v3(pretrained=True)
        self.inception.aux_logits = False
        n_features = self.inception.fc.in_features
        self.seb = SEBlock(2048)
        self.preprocessor = PrepModel()
        self.mean = torch.Tensor([0.275]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.Tensor([0.197]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.inception.fc = nn.Linear(n_features, NUM_CLASSES)

    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        prep = self.preprocessor(x)
        prep = (prep - mean) / std
        prep_out = torch.cat([x[:, 0:2, :, :], prep], 1)

        x = self.inception.Conv2d_1a_3x3(prep_out)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)

        p = self.seb(x)
        x = self.inception.avgpool(p)
        x = x.view(x.size(0), -1)
        x = self.inception.dropout(x)
        logits = self.inception.fc(x)

        _, target_class = torch.max(logits, dim=1)
        b, c, h, w = p.shape
        fc_weights_class = self.inception.fc.weight[target_class]
        cam = fc_weights_class.unsqueeze(1) @ p.view((b, c, h * w))
        cam = cam.view(b, h, w)
        cam_resized = F.interpolate(cam.unsqueeze(1), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        cam_resized = cam_resized.squeeze(1)
        cam_min = cam_resized.view(b, -1).min(dim=1, keepdim=True)[0].unsqueeze(2)
        cam_resized = cam_resized - cam_min
        cam_max = cam_resized.view(b, -1).max(dim=1, keepdim=True)[0].unsqueeze(2)
        cam_resized = cam_resized / (cam_max + 1e-8)

        return logits, cam_resized


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
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize([0.275, 0.275, 0.275], [0.197, 0.197, 0.197])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.275, 0.275, 0.275], [0.197, 0.197, 0.197])
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
        return image, label, mask.squeeze(0)


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
        for images, labels, masks in dataloader:
            images = images.to(device)
            logits, cam = model(images)
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

    model = CAMQUSModel(device).to(device)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    best_f1, best_metrics, best_epoch = -1, None, 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for images, labels_batch, masks in train_loader:
            images, labels_batch, masks = images.to(device), labels_batch.to(device), masks.to(device)
            optimizer.zero_grad()
            logits, cam = model(images)
            loss_cls = ce_loss(logits, labels_batch)
            loss_cam = mse_loss(cam, masks)
            loss = loss_cls + 0.5 * loss_cam
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

    config = {'model': 'CAM-QUS', 'dataset': 'BUSI', 'image_size': IMAGE_SIZE,
              'batch_size': BATCH_SIZE, 'epochs': NUM_EPOCHS, 'lr': LEARNING_RATE, 'seed': RANDOM_SEED}
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
        f.write("CAM-QUS on BUSI - 5-Fold Results\n" + "="*50 + "\n\n")
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
