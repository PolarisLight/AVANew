"""
file - dataset.py
Customized dataset class to loop through the AVA dataset and apply needed image augmentations for training.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os
import time
import random

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import numpy as np


compare_fastsam = False

mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
    transforms.RandomCrop(448),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def dis_2_score(dis: torch.Tensor):
    """
    convert distance to score
    :param dis: distance
    :return: score, tensor
    """
    w = torch.linspace(1, 10, 10).to(dis.device)
    w_batch = w.repeat(dis.shape[0], 1)
    score = (dis * w_batch).sum(dim=1)
    return score
class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, imgsz=512, device='cpu', train=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.device = device
        self.imgsz = imgsz
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsz, imgsz), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsz, imgsz), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform = train_transform if train else val_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        # resize img
        image = image.resize((self.imgsz, self.imgsz))
        annotations = self.annotations.iloc[idx, 1:11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, )
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}
        sample['score'] = dis_2_score(torch.tensor(sample['annotations']))

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # print(sample['image'].shape)

        return sample

class NimaAVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, imgsz=512, device='cpu', train=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.device = device
        self.imgsz = imgsz
        
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsz, imgsz), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsz, imgsz), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform = train_transform if train else val_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        # resize img
        image = image.resize((self.imgsz, self.imgsz))
        annotations = self.annotations.iloc[idx, 1:11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, )
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}
        sample['score'] = dis_2_score(torch.tensor(sample['annotations']))

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # print(sample['image'].shape)

        return sample

if __name__ == "__main__":
    train_dataset = AVADataset(csv_file='D:\\Dataset\\AVA\\labels\\train_labels.csv',
                               root_dir='D:\\Dataset\\AVA\\images', imgsz=224, device='cuda', train=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0)

    for i, data in enumerate(train_loader):
        print(data['image'].shape)
        print(data['score'].shape)
        break
    print("Done!")