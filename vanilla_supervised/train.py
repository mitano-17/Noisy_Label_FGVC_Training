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
import PIL
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def train(epoch, model, train_loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):  # Mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    train_acc = 100. * correct / total
    train_loss = running_loss / len(train_loader)

    return train_loss, train_acc, optimizer