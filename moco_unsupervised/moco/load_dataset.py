from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torchvision.datasets import ImageFolder
import numpy as np
import random
import cv2
from PIL import Image
from moco.pil_loader import pil_loader
from train_moco import train
from moco.CV2GaussianBlur import CV2GaussianBlur
from moco.loader import TwoCropsTransform
import torch.utils.data.distributed


def augment_data(aug_strength):
    # Define your transforms (keeping your original augmentation strengths)
    size1 = 521
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
        train_root = 'datasets/web-bird/train'
        val_root = 'datasets/web-bird/val'

    elif "web-aircraft" in args.dataset:
        train_root = 'datasets/web-aircraft/train'
        val_root = 'datasets/web-aircraft/val'

    elif "web-car" in args.dataset:
        train_root = 'datasets/web-car/train'
        val_root = 'datasets/web-car/val'

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    train_transform, test_transform = augment_data(args.aug_strength)
    
    # Training Data (The Pair Dataset)
    train_dataset = ImageFolder(root=train_root, transform=TwoCropsTransform(train_transform), loader=pil_loader)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=os.cpu_count(), pin_memory=True, sampler=train_sampler, drop_last=True)    


    return train_loader, train_sampler