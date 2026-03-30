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


def pil_loader(path):
    """Load image with error handling and proper transparency handling"""
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            # Handle palette images with transparency
            if img.mode == 'P':
                # Convert palette images to RGB, handling transparency properly
                img = img.convert('RGBA')
                img = img.convert('RGB')
            else:
                # For other modes, just convert to RGB
                img = img.convert('RGB')
                
            return img
            
    except (OSError, IOError) as e:
        #print(f"Error loading image {path}: {e}")
        return Image.new('RGB', (224, 224), color='gray')