import cv2
import numpy as np
from pathlib import Path
import random

def create_synthetic_medical_dataset(num_images=500):
    """
    Create synthetic medical-like images for denoising task
    """
    output_dir = Path('datasets/raw/synthetic_medical')
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Create synthetic medical-like image
        # Simulate cell/tissue texture with patterns
        img = np.random.rand(256, 256) * 255

        # Add medical-like patterns (circles) — mask applied directly on [0,255] grid
        y, x = np.ogrid[0:256, 0:256]
        mask = (x - 128)**2 + (y - 128)**2 <= 80**2
        img[mask] = 200

        # Add some Gaussian blur for smoothness
        img = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)

        # Save image
        cv2.imwrite(
            str(output_dir / f'image_{i:04d}.png'),
            img
        )

        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num_images} images")

    print(f"Dataset created at {output_dir}")

if __name__ == "__main__":
    create_synthetic_medical_dataset()
