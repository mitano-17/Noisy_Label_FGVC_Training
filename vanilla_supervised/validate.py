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


def validate(epoch, model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Reset tracking arrays
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad(), torch.amp.autocast('cuda'):  # Add mixed precision
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} Validation')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for analysis
            probabilities = torch.softmax(outputs, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    val_acc = 100. * correct / total
    val_loss = running_loss / len(val_loader)

    
    return val_loss, val_acc