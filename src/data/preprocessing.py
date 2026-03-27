import cv2
import numpy as np
from pathlib import Path
from PIL import Image


class ImagePreprocessor:
    """Preprocess medical images for denoising"""

    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def add_gaussian_noise(self, image, noise_std=25):
        """Add Gaussian noise to simulate degraded medical images"""
        noise = np.random.normal(0, noise_std, image.shape)
        noisy_image = np.clip(image + noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def normalize(self, image):
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0

    def denormalize(self, image):
        """Convert from [0, 1] back to [0, 255]"""
        return np.clip(image * 255, 0, 255).astype(np.uint8)

    def resize_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, self.target_size,
                          interpolation=cv2.INTER_CUBIC)

    def preprocess(self, image_path):
        """Complete preprocessing pipeline"""
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Resize
        img = self.resize_image(img)

        # Add noise to create input
        noisy_img = self.add_gaussian_noise(img, noise_std=25)

        # Normalize
        clean_normalized = self.normalize(img)
        noisy_normalized = self.normalize(noisy_img)

        return clean_normalized, noisy_normalized
