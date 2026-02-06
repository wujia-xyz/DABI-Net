import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import pandas as pd

tar_size = [300, 300]

def read_dataset_excel(fname):
    # read all data from the first N rows
    df = pd.read_excel(fname)  # read the 1st sheet of .xlsx file
    data2 = df.values          # forms 'list'，read all data in the excel
    rows = len(data2)
    ret = []
    for i in range(rows):
        data = data2[i]
        r = []
        for e in data:
            if pd.isnull(e):
                r.append(None)
            else:
                r.append(e)
        ret.append(r)
    return ret

# Inheritance of Dataset class，used for data loading
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_list='./labelTrain_test.xlsx',
                 root_dir='./dataset',
                 is_train=False,
                 target_size=(128,128)):  # default filename and location
        self.file_list = file_list
        self.root_dir = root_dir
        self.is_train = is_train
        self.target_size = target_size
        self.data_sets = read_dataset_excel(self.file_list)

    # length of the data
    def __len__(self):
        return len(self.data_sets)

    def get_data_sets(self):
        return self.data_sets

    # re-define how to get the data
    def __getitem__(self, item):
        row = self.data_sets[item]
        num, age, label = row[:3]

        img1 = Image.open(os.path.join(self.root_dir, str(int(num)), 'ROI_1.png')).convert('RGB')
        img2 = Image.open(os.path.join(self.root_dir, str(int(num)), 'ROI_2.png')).convert('RGB')
        img3 = Image.open(os.path.join(self.root_dir, str(int(num)), 'ROI_3.png')).convert('RGB')

        transform1 = transforms.Compose([
                     transforms.Resize(self.target_size),
                     ])
        img1 = transform1(img1)
        img2 = transform1(img2)
        img3 = transform1(img3)

        # data augmentation
        if self.is_train:
            img1, img2, img3 = aug_image([img1, img2, img3])

        # change PIL(image) to tensor (1,C,H,W)
        def img_to_tensor(image):
            loader = transforms.Compose([transforms.ToTensor()])
            image = loader(image).unsqueeze(0)
            return image
        # the following metadata will be used for model development
        images = torch.vstack([img_to_tensor(img1),
            img_to_tensor(img2), img_to_tensor(img3)])
        return images.float(), label.astype(np.int64)

# the detailed processing for data augmentation
def aug_image(imgs):
    ret = []
    for im in imgs:
        im = transforms.RandomHorizontalFlip(p=0.5)(im)
        im = transforms.ColorJitter(brightness=0.1, contrast=0.1)(im)
        im = transforms.RandomResizedCrop(im.size, scale=(0.9,1), ratio=(1,1))(im)
        im = transforms.RandomRotation(10)(im)
        ret.append(im)
    return ret

if __name__ == '__main__':
    train_dataset = MyDataset(
        file_list='./labelTrain_test.xlsx',
        root_dir='./sample',
        target_size=tar_size,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=2, pin_memory=False)
    for i, (image, label) in enumerate(train_loader):
        print(image.shape, label.shape) # torch.Size([2, 3, 3, 128, 128]) torch.Size([2])
        print(image.dtype, label.dtype) # torch.float32 torch.int64
        break