"""
Dataset loader for MVMM on ARC dataset.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ARCDataset(Dataset):
    """
    ARC dataset loader for multimodal breast ultrasound.

    Loads three modalities: B-mode (ROI_1), Doppler (ROI_2), Elastography (ROI_3)
    """

    def __init__(self, excel_file, root_dir, is_train=True, target_size=224):
        """
        Args:
            excel_file: Path to excel file with sample numbers and labels
            root_dir: Root directory of images
            is_train: Whether this is training set (for augmentation)
            target_size: Target image size
        """
        self.data = pd.read_excel(excel_file)
        self.root_dir = root_dir
        self.is_train = is_train
        self.target_size = target_size

        # Transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(),
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
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_num = str(int(row['num']))

        # Load three modalities
        # ROI_1: B-mode, ROI_2: Doppler, ROI_3: Elastography
        bus_path = os.path.join(self.root_dir, sample_num, 'ROI_1.png')
        dus_path = os.path.join(self.root_dir, sample_num, 'ROI_2.png')
        eus_path = os.path.join(self.root_dir, sample_num, 'ROI_3.png')

        bus_img = Image.open(bus_path).convert('RGB')
        dus_img = Image.open(dus_path).convert('RGB')
        eus_img = Image.open(eus_path).convert('RGB')

        # Apply transforms
        bus_tensor = self.transform(bus_img)
        dus_tensor = self.transform(dus_img)
        eus_tensor = self.transform(eus_img)

        # Concatenate along channel dimension
        # Shape: (9, H, W) = 3 modalities * 3 channels
        combined = torch.cat([bus_tensor, dus_tensor, eus_tensor], dim=0)

        # Get label
        label = int(row['label'])

        return combined, label


def get_dataloaders(data_dir, fold, batch_size=20, target_size=224, num_workers=4):
    """Create train and validation dataloaders for a specific fold."""

    train_dataset = ARCDataset(
        excel_file=os.path.join(data_dir, f'train_fold{fold}.xlsx'),
        root_dir=data_dir,
        is_train=True,
        target_size=target_size
    )

    val_dataset = ARCDataset(
        excel_file=os.path.join(data_dir, f'val_fold{fold}.xlsx'),
        root_dir=data_dir,
        is_train=False,
        target_size=target_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
