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

# Replace your current pil_loader with this improved version
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