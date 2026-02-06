"""Multimodal breast ultrasound dataset utilities for TDF-Net."""

import os
from glob import glob

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalBUSDataset(Dataset):
    """Dataset loader for BUS/CDFI/UE multimodal ultrasound data."""

    def __init__(
        self,
        data_dir,
        label_file,
        patient_ids=None,
        transform=None,
        img_size=224,
        id_col="序号",
        label_col="病理结果（0：良性，1：恶性）",
    ):
        self.data_dir = data_dir
        self.img_size = img_size

        df = pd.read_excel(label_file)
        if id_col not in df.columns or label_col not in df.columns:
            raise KeyError(
                f"Missing required columns in {label_file}. "
                f"Expected id_col={id_col}, label_col={label_col}."
            )

        self.labels = dict(zip(df[id_col], df[label_col]))

        if patient_ids is not None:
            self.patient_ids = [pid for pid in patient_ids if pid in self.labels]
        else:
            self.patient_ids = list(self.labels.keys())

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def _load_modality(self, patient_folder, prefix):
        pattern = os.path.join(patient_folder, f"{prefix}_*.jpg")
        files = sorted(glob(pattern))
        if files:
            img = Image.open(files[0]).convert("RGB")
            return self.transform(img)
        return torch.zeros(3, self.img_size, self.img_size)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_folder = os.path.join(self.data_dir, str(patient_id))
        label = int(self.labels[patient_id])

        bus = self._load_modality(patient_folder, "BUS")
        dus = self._load_modality(patient_folder, "DUS")
        eus = self._load_modality(patient_folder, "EUS")

        return (bus, dus, eus), label


def get_train_transform(img_size=224):
    """Training transforms with augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(img_size=224):
    """Validation transforms without augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
