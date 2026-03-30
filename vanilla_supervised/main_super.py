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

from utils import set_device, get_model, set_super_func, init_csv_log, resume
from run import run
from load_dataset import load_dataset
    
    
def main():
    parser = argparse.ArgumentParser(description='Vanilla ResNet50 Training')
    parser.add_argument('--dataset', type=str, default='web-aircraft', choices=['web-bird', 'web-car', 'web-aircraft'])
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--conv', default="3", type=int)
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ImageNet weights')
    parser.add_argument('--aug-strength', type=str, default='moderate',
                   choices=['strong', 'moderate', 'weak', 'none'])
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--save-freq', default=10, type=int)
 
    parser.add_argument('--save-dir', type=str, default='./super_baseline_aug')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--moco-path', type=str, default=None,
                       help='Path to MoCo checkpoint to resume from')

    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        print("CUDA not available.")
    
    args = args
    device = set_device()
    scaler = torch.amp.GradScaler('cuda')
    aug_strength = args.aug_strength
    warmup_epochs = 5

    # Load Dataset
    train_loader, val_loader, num_classes = load_dataset(args)

    # Model
    model = get_model(num_classes, device, args)
    criterion, optimizer, scheduler = set_super_func(model, args.lr, args.epochs)

    # Metrics tracking
    best_acc = 0.0
    confusion_matrix = None
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.save_dir, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize CSV log
    csv_log_path = os.path.join(results_dir, 'training_metrics.csv')
    init_csv_log(csv_log_path)

    current_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume and os.path.isfile(args.resume):
        current_epoch = resume(args, trainer)
    
    run(args, current_epoch, results_dir, best_acc, model, train_loader, val_loader,
        criterion, optimizer, scheduler, scaler, device, csv_log_path)


if __name__ == '__main__':
    main()