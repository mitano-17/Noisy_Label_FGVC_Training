# vanilla_resnet50.py
import os       
import warnings
warnings.filterwarnings('ignore', message='The `srun` command is available')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from torchvision.datasets import ImageFolder
import PIL
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from pil_loader import pil_loader
from gaussian_blur import CV2GaussianBlur

def augment_data(aug_strength):
    # Define your transforms (keeping your original augmentation strengths)
    size1 = 512
    size = 448
    s = 1
    kernel_size = int(0.1 * size)

    print("Using resolution: ", size, size1)
    
    if 'strong' in aug_strength:
        print("Using strong augmentations")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                CV2GaussianBlur(kernel_size=kernel_size)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    elif 'moderate' in aug_strength:
        print("Using moderate augmentations")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4*s,0.4*s,0.4*s,0.1*s)], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    elif 'weak' in aug_strength:
        print("Using weak augmentations")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2*s,0.2*s,0.2*s,0.05*s)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    else:  # 'none'
        print("Using little to no augmentation")
        train_transform = transforms.Compose([
            transforms.Resize(size1, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Test/validation transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(size1),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

def load_dataset(args):
    # Select the appropriate directory based on your existing logic

    if "web-bird" in args.dataset:
        num_classes = 200
        train_root = 'datasets/web-bird/train'
        val_root = 'datasets/web-bird/val'

    elif "web-aircraft" in args.dataset:
        num_classes = 100
        train_root = 'datasets/web-aircraft/train'
        val_root = 'datasets/web-aircraft/val'

    elif "web-car" in args.dataset:
        num_classes = 196
        train_root = 'datasets/web-car/train'
        val_root = 'datasets/web-car/val'

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    train_transform, val_transform = augment_data(args.aug_strength)
    
    # Training Data
    train_dataset = ImageFolder(root=train_root, transform=train_transform, loader=pil_loader)
    val_dataset = ImageFolder(root=val_root, transform=val_transform, loader=pil_loader)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=os.cpu_count(), pin_memory=True)  

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=os.cpu_count(), pin_memory=True)  


    return train_loader, val_loader, num_classes
