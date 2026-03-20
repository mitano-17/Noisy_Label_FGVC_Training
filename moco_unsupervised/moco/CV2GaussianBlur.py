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


class CV2GaussianBlur:
    def __init__(self, kernel_size):
        # Ensure kernel size is odd (OpenCV requirement)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
    def __call__(self, img):
        # img comes in as PIL Image
        # Convert PIL to numpy array (OpenCV format)
        img_np = np.array(img)
        
        # Random sigma between 0.1 and 2.0
        sigma = random.uniform(0.1, 2.0)
        
        # Apply Gaussian blur (OpenCV is FAST)
        blurred = cv2.GaussianBlur(
            img_np, 
            (self.kernel_size, self.kernel_size), 
            sigma
        )
        
        # Convert back to PIL Image
        return Image.fromarray(blurred)