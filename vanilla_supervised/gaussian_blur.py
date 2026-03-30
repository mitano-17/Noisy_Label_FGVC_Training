import cv2
import numpy as np
from PIL import Image
import random

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