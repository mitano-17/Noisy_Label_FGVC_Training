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

from train import train
from validate import validate
from utils import save_checkpoint, log_metrics


def run(args, current_epoch, results_dir, best_acc, model, 
    train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, csv_log_path):

    print(f"Results will be saved to: {results_dir}")
    
    for epoch in range(current_epoch, args.epochs):          
        print(f'\nEpoch: {epoch}')
        #print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        # Train
        train_loss, train_acc, optimizer = train(epoch, model, train_loader, optimizer, criterion, scaler, device)
        
        # Validate
        val_loss, val_acc = validate(epoch, model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save checkpoint
        is_best = val_acc > best_acc

        if is_best:
            best_acc = val_acc
            
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, best_acc, val_acc, scheduler, results_dir, is_best)
        
        # Log metrics to CSV
        log_metrics(csv_log_path, epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_acc:.2f}%')