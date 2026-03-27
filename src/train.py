import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

from models.autoencoder import DenoisingAutoEncoder
from models.classifier import MedicalImageClassifier
from data.dataset import MedicalImageDataset
from utils.metrics import calculate_psnr, calculate_ssim


class DenoisingTrainer:
    """Trainer for medical image denoising AutoEncoder"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        print(f"Using device: {self.device}")

        # Model
        self.model = DenoisingAutoEncoder(
            in_channels=1,
            initial_filters=self.config['model']['initial_filters']
        ).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['training']['lr_step_size'],
            gamma=0.5
        )

        self.best_psnr = 0
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        psnr_values = []

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(noisy)
            loss = self.criterion(output, clean)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate PSNR
            psnr = calculate_psnr(
                (output.detach() * 255).cpu().numpy(),
                (clean * 255).cpu().numpy()
            )
            psnr_values.append(psnr)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'psnr': f'{psnr:.2f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_psnr = float(np.mean(psnr_values))

        return avg_loss, avg_psnr

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        psnr_values = []
        ssim_values = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                noisy = batch['noisy'].to(self.device)
                clean = batch['clean'].to(self.device)

                output = self.model(noisy)
                loss = self.criterion(output, clean)

                total_loss += loss.item()

                # Metrics
                output_np = (output.detach() * 255).cpu().numpy()
                clean_np = (clean * 255).cpu().numpy()

                psnr = calculate_psnr(output_np, clean_np)
                ssim = calculate_ssim(output_np, clean_np)

                psnr_values.append(psnr)
                ssim_values.append(ssim)

        avg_loss = total_loss / len(val_loader)
        avg_psnr = float(np.mean(psnr_values))
        avg_ssim = float(np.mean(ssim_values))

        return avg_loss, avg_psnr, avg_ssim

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""
        training_history = {
            'train_loss': [],
            'train_psnr': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': []
        }

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss, train_psnr = self.train_epoch(train_loader)

            # Validate
            val_loss, val_psnr, val_ssim = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Store results
            training_history['train_loss'].append(train_loss)
            training_history['train_psnr'].append(train_psnr)
            training_history['val_loss'].append(val_loss)
            training_history['val_psnr'].append(val_psnr)
            training_history['val_ssim'].append(val_ssim)

            print(f"Train Loss: {train_loss:.6f} | Train PSNR: {train_psnr:.2f}")
            print(f"Val Loss:   {val_loss:.6f} | Val PSNR:   {val_psnr:.2f} | "
                  f"Val SSIM: {val_ssim:.4f}")

            # Save best model
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                self.save_checkpoint(epoch, val_psnr)
                print(f"✓ Model saved! Best PSNR: {self.best_psnr:.2f}")

        return training_history

    def save_checkpoint(self, epoch, psnr):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'psnr': psnr
        }
        checkpoint_path = self.logs_dir / f'autoencoder_best_psnr_{psnr:.2f}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")


def main():
    """Main training function"""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create datasets
    print("Loading datasets...")
    train_dataset = MedicalImageDataset(
        config['paths']['train_data_dir'],
        noise_std=25
    )
    val_dataset = MedicalImageDataset(
        config['paths']['val_data_dir'],
        noise_std=25
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # Initialize trainer
    trainer = DenoisingTrainer('config.yaml')

    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs']
    )

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation PSNR: {trainer.best_psnr:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
