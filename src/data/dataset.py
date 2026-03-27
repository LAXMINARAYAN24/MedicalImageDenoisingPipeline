import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from .preprocessing import ImagePreprocessor


class MedicalImageDataset(Dataset):
    """
    Dataset for medical image denoising.
    Creates clean-noisy pairs from image directory.
    """

    def __init__(self, image_dir, transform=None, noise_std=25):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.png')) + \
                           list(self.image_dir.glob('*.jpg'))
        self.preprocessor = ImagePreprocessor()
        self.noise_std = noise_std
        self.transform = transform

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Preprocess: get clean and noisy pairs
        clean, noisy = self.preprocessor.preprocess(image_path)

        # Convert to tensors
        clean_tensor = torch.from_numpy(clean).unsqueeze(0).float()
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).float()

        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'filename': image_path.name
        }
