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
from lincls.pil_loader import pil_loader
import torch.utils.data.distributed


def augment_data():
    # Define your transforms (keeping your original augmentation strengths)
    size1 = 521
    size = 448
    s = 1
    kernel_size = int(0.1 * size)

    print("Using resolution: ", size, size1)
    
    # Training transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
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


    train_transform, val_transform = augment_data()
    
    # Training Data
    train_dataset = ImageFolder(root=train_root, transform=train_transform, loader=pil_loader)
    val_dataset = ImageFolder(root=val_root, transform=val_transform, loader=pil_loader)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=os.cpu_count(), pin_memory=True, sampler=train_sampler)  

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=os.cpu_count(), pin_memory=True)  

    return train_loader, train_sampler, val_loader
