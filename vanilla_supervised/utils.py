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


def set_device():
	# Determine device
    if torch.cuda.is_available():
        print("Using CUDA device")
        return "cuda:0"

    else:
        print("Using CPU device")
        return "cpu" 

def get_model(num_classes, device, args):
    """Load pretrained ResNet50 and modify final layer for target dataset"""
    
    #print("Training ResNet50 from scratch...")
    #model = torchvision.models.resnet50(weights=None)

    print("Training ResNet50 to fine-tune...")
    model = torchvision.models.resnet50(weights=None)

    # 1. Load the MoCo checkpoint
    if args.moco_path:
        print(f"Loading MoCo weights from {args.moco_path}")
        checkpoint = torch.load(args.moco_path, weights_only=True, map_location="cpu")
        state_dict = checkpoint['state_dict']
        
        # 2. Rename keys to match standard ResNet50
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        
        # 3. Load weights (strict=False because fc is missing)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded. Missing keys: {msg.missing_keys}")
    
    # Modify final fully connected layer for target dataset
    if num_classes != 1000:  # ImageNet has 1000 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model.to(device)


def set_super_func(model, lr, epochs):
    criterion = nn.CrossEntropyLoss()

    # Separate parameters into two groups
    head_params = model.fc.parameters()
    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]

    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': lr * 0.1}, # 0.00125 (10x smaller)
        {'params': head_params, 'lr': lr}           # 0.0125
    ], momentum=0.9, weight_decay=0.001)

    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    return criterion, optimizer, scheduler


def init_csv_log(csv_log_path):
    """Initialize CSV file with headers"""
    headers = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
    
    df = pd.DataFrame(columns=headers)
    df.to_csv(csv_log_path, index=False)


def log_metrics(csv_log_path, epoch, train_loss, train_acc, val_loss, val_acc, lr):
    """Log metrics to CSV file"""
    log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'learning_rate': lr
    }
    
    df = pd.DataFrame([log_data])
    df.to_csv(csv_log_path, mode='a', header=False, index=False)


def resume(args, trainer):
	print(f"Loading checkpoint from {args.resume}")

	checkpoint = torch.load(args.resume, weights_only=False, map_location=torch.device('cuda'))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	if trainer.scheduler is not None and 'scheduler_state_dict' in checkpoint:
	    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

	best_acc = checkpoint['best_acc']

	print(f"Resumed from epoch {checkpoint['epoch']}, best acc: {checkpoint['best_acc']:.2f}%")

	current_epoch = checkpoint['epoch']

	return current_epoch

def save_checkpoint(epoch, model, optimizer, best_acc, val_acc, scheduler, results_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'val_acc': val_acc
    }
    
    # FIX: Only save scheduler state if it exists
    if scheduler is not None:
        try:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        except:
            print("Warning: Could not save scheduler state")
    
    if is_best:
        torch.save(checkpoint, os.path.join(results_dir, 'best_model.pth'))

    torch.save(checkpoint, os.path.join(results_dir, f'checkpoint_epoch_{epoch}.pth'))