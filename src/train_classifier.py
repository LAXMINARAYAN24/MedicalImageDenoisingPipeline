"""
train_classifier.py — Train the CNN classifier (Normal vs. Abnormal).

The denoiser is run first to clean images; the classifier is then trained
on the denoised output.  Labels are derived from the filename:
  * filenames containing 'abnormal' → class 1
  * everything else                 → class 0 (Normal)

Usage (from project root, inside venv):
    python src/train_classifier.py
    python src/train_classifier.py --denoiser-checkpoint logs/autoencoder_best_psnr_XX.XX.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import yaml
from tqdm import tqdm

from models.autoencoder import DenoisingAutoEncoder
from models.classifier import MedicalImageClassifier
from data.preprocessing import ImagePreprocessor
from utils.metrics import calculate_psnr


# ── Dataset ───────────────────────────────────────────────────────────────────

class ClassificationDataset(Dataset):
    """
    Wraps the raw image directory and produces (denoised_tensor, label) pairs.
    Labels: 'abnormal' in filename → 1, otherwise → 0.
    Denoising is performed on-the-fly using a pre-trained (or random) AutoEncoder.
    """

    def __init__(self, image_dir: str, denoiser: nn.Module,
                 device: torch.device, noise_std: int = 25):
        self.image_paths = (
            list(Path(image_dir).glob('*.png')) +
            list(Path(image_dir).glob('*.jpg'))
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.denoiser    = denoiser
        self.device      = device
        self.noise_std   = noise_std
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        label = 1 if 'abnormal' in path.stem.lower() else 0

        clean, noisy = self.preprocessor.preprocess(str(path))
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            denoised = self.denoiser(noisy_t).squeeze(0).cpu()   # [1, H, W]

        return denoised, torch.tensor(label, dtype=torch.long)


# ── Trainer ───────────────────────────────────────────────────────────────────

class ClassifierTrainer:
    def __init__(self, config: dict, denoiser: nn.Module, device: torch.device):
        self.config  = config
        self.device  = device

        self.model = MedicalImageClassifier(
            in_channels=1,
            num_classes=config['model']['classifier']['num_classes']
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=0.5
        )

        self.logs_dir = Path(config['paths']['logs_dir'])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0

    # ── Epoch ──────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for images, labels in tqdm(loader, desc="Train" if train else "Val  "):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)

        return total_loss / len(loader), correct / total

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, num_epochs: int):
        history = {'train_loss': [], 'train_acc': [],
                   'val_loss':   [], 'val_acc':   []}

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            va_loss, va_acc = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(va_loss)
            history['val_acc'].append(va_acc)

            print(f"Train  Loss: {tr_loss:.4f} | Acc: {tr_acc*100:.1f}%")
            print(f"Val    Loss: {va_loss:.4f} | Acc: {va_acc*100:.1f}%")

            if va_acc > self.best_acc:
                self.best_acc = va_acc
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': va_acc
                }
                ckpt_path = self.logs_dir / f'classifier_best_acc_{va_acc*100:.1f}.pth'
                torch.save(ckpt, ckpt_path)
                print(f"✓ Classifier saved! Best Acc: {self.best_acc*100:.1f}%  → {ckpt_path}")

        return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CNN Classifier")
    parser.add_argument('--config',               default='config.yaml')
    parser.add_argument('--denoiser-checkpoint',  default=None,
                        help='Path to pre-trained denoiser .pth')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load (or init) denoiser ───────────────────────────────────────────────
    denoiser = DenoisingAutoEncoder(
        in_channels=1,
        initial_filters=config['model']['initial_filters']
    ).to(device)

    checkpoint = args.denoiser_checkpoint
    if checkpoint is None:
        logs_dir = Path(config['paths'].get('logs_dir', 'logs'))
        found = sorted(logs_dir.glob('autoencoder_best_psnr_*.pth'))
        if found:
            checkpoint = str(found[-1])
            print(f"Auto-selected denoiser checkpoint: {checkpoint}")

    if checkpoint:
        ckpt  = torch.load(checkpoint, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        denoiser.load_state_dict(state)
        print("✓ Denoiser weights loaded.")
    else:
        print("⚠  No denoiser checkpoint — using random weights.")

    denoiser.eval()

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = ClassificationDataset(
        config['paths']['train_data_dir'], denoiser, device,
        noise_std=config.get('data', {}).get('noise_std', 25)
    )
    val_ds = ClassificationDataset(
        config['paths']['val_data_dir'], denoiser, device,
        noise_std=config.get('data', {}).get('noise_std', 25)
    )

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = ClassifierTrainer(config, denoiser, device)
    history = trainer.train(train_loader, val_loader,
                            num_epochs=config['training']['num_epochs'])

    print(f"\n{'='*60}")
    print(f"Classifier training complete!")
    print(f"Best Val Accuracy: {trainer.best_acc*100:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
