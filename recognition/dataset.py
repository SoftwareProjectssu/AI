import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
from PIL import Image

import torchvision.transforms as transforms

def parse_pts(pts_path):
    with open(pts_path, 'r') as f:
        lines = f.readlines()
        coords = []
        reading = False
        for line in lines:
            if line.strip() == '{':
                reading = True
                continue
            if line.strip() == '}':
                break
            if reading:
                x, y = map(float, line.strip().split())
                coords.append([x, y])
        return torch.tensor(coords, dtype=torch.float32)

def get_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

class ThreeHundredWDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_transform(is_train)
        self.is_train = is_train
        
        self.samples = []
        for subdir in ['01_Indoor', '02_Outdoor']:
            folder = os.path.join(root_dir, subdir)
            images = sorted(glob.glob(os.path.join(folder, '*.png')))
            for img_path in images:
                pts_path = img_path.replace('.png', '.pts')
                if os.path.exists(pts_path):
                    self.samples.append((img_path, pts_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pts_path = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        landmarks = parse_pts(pts_path)
        
        image = self.transform(image)
        
        landmarks[:, 0] = landmarks[:, 0] / original_size[0]
        landmarks[:, 1] = landmarks[:, 1] / original_size[1]
        
        if self.is_train:
            if np.random.random() > 0.5:
                landmarks[:, 0] = 1 - landmarks[:, 0]
            
            angle = np.random.uniform(-30, 30)
            center = torch.tensor([0.5, 0.5])
            angle_rad = np.radians(angle)
            rotation_matrix = torch.tensor([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ], dtype=torch.float32)
            landmarks = (landmarks - center) @ rotation_matrix + center
            landmarks = landmarks.float()
        
        return {
            'image': image,
            'landmarks': landmarks,
            'image_path': img_path
        }
