"""
test.py — Evaluate the trained DenoisingAutoEncoder on the validation set.

Usage (from project root, inside venv):
    python src/test.py                          # uses config.yaml defaults
    python src/test.py --checkpoint logs/autoencoder_best_psnr_XX.XX.pth
    python src/test.py --image path/to/img.png  # single-image mode
"""

import argparse
import sys
from pathlib import Path

# ── Allow imports from src/ when called from project root ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import cv2
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.autoencoder import DenoisingAutoEncoder
from data.dataset import MedicalImageDataset
from data.preprocessing import ImagePreprocessor
from utils.metrics import MetricsCalculator
from utils.visualization import plot_denoising_comparison, plot_batch_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str | None, config: dict, device: torch.device):
    """Load DenoisingAutoEncoder; random weights if no checkpoint given."""
    model = DenoisingAutoEncoder(
        in_channels=1,
        initial_filters=config['model']['initial_filters']
    ).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        epoch = ckpt.get('epoch', '?')
        psnr  = ckpt.get('psnr',  '?')
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print(f"  (epoch={epoch}, saved PSNR={psnr})")
    else:
        print("⚠  No checkpoint provided — using random weights (expect poor results).")

    model.eval()
    return model


def evaluate_dataset(model, val_loader, device, results_dir: Path):
    """Run model over the full validation set and collect metrics."""
    all_metrics = {'psnr': [], 'ssim': [], 'mse': [], 'mae': []}
    calculator  = MetricsCalculator()

    saved_samples = []   # (noisy, denoised, clean) triples for visualisation

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            output = model(noisy)

            # ── Metrics (numpy, scaled to [0, 255]) ──────────────────────────
            out_np   = (output.cpu().numpy() * 255).clip(0, 255)
            clean_np = (clean.cpu().numpy()  * 255).clip(0, 255)

            m = calculator.compute_all_metrics(out_np, clean_np)
            for k in all_metrics:
                all_metrics[k].append(m[k])

            # ── Collect first batch for visualisation ─────────────────────────
            if batch_idx == 0:
                noisy_np = (noisy.cpu().numpy() * 255).clip(0, 255)
                saved_samples = (noisy_np, out_np, clean_np)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return summary, saved_samples


def save_comparison_images(samples, results_dir: Path, n: int = 4):
    """Save side-by-side comparison PNGs for the first n samples."""
    noisy_batch, denoised_batch, clean_batch = samples
    n = min(n, noisy_batch.shape[0])

    for i in range(n):
        noisy_img    = noisy_batch[i, 0].astype(np.uint8)
        denoised_img = denoised_batch[i, 0].astype(np.uint8)
        clean_img    = clean_batch[i, 0].astype(np.uint8)

        sep = np.ones((256, 4), dtype=np.uint8) * 128
        row = np.hstack([noisy_img, sep, denoised_img, sep, clean_img])
        out_path = results_dir / f"comparison_{i:02d}.png"
        cv2.imwrite(str(out_path), row)

    print(f"✓ Saved {n} comparison images to {results_dir}/")

    # Also save a matplotlib grid
    plot_batch_results(
        noisy_batch / 255.0,
        denoised_batch / 255.0,
        n_samples=n,
        save_path=str(results_dir / "batch_grid.png")
    )


def evaluate_single_image(image_path: str, model, device, results_dir: Path):
    """Denoise a single image and save result."""
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    clean, noisy = preprocessor.preprocess(image_path)

    noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(noisy_t)

    noisy_uint8    = (noisy    * 255).clip(0, 255).astype(np.uint8)
    denoised_uint8 = (output.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    clean_uint8    = (clean    * 255).clip(0, 255).astype(np.uint8)

    # Metrics
    calc = MetricsCalculator()
    m = calc.compute_all_metrics(
        denoised_uint8[np.newaxis, np.newaxis],
        clean_uint8[np.newaxis, np.newaxis]
    )
    print(f"\n── Single-image metrics ────────────────────────")
    for k, v in m.items():
        print(f"  {k.upper():6s}: {v:.4f}")
    print(f"────────────────────────────────────────────────\n")

    out_path = results_dir / f"single_{Path(image_path).stem}_result.png"
    plot_denoising_comparison(
        noisy_uint8, denoised_uint8, clean_uint8,
        title=f"Denoising: {Path(image_path).name}",
        save_path=str(out_path)
    )
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical Image Denoiser")
    parser.add_argument('--config',     default='config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--image',      default=None,
                        help='Single image path (skips dataset evaluation)')
    parser.add_argument('--n-samples',  type=int, default=4,
                        help='Number of comparison images to save')
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Medical Image Denoising — Evaluation")
    print(f"  Device : {device}")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    # Auto-discover best checkpoint if none provided
    checkpoint = args.checkpoint
    if checkpoint is None:
        logs_dir = Path(config['paths'].get('logs_dir', 'logs'))
        checkpoints = sorted(logs_dir.glob('autoencoder_best_psnr_*.pth'))
        if checkpoints:
            checkpoint = str(checkpoints[-1])
            print(f"Auto-selected checkpoint: {checkpoint}")

    model = load_model(checkpoint, config, device)

    # ── Single-image mode ─────────────────────────────────────────────────────
    if args.image:
        evaluate_single_image(args.image, model, device, results_dir)
        return

    # ── Dataset evaluation mode ───────────────────────────────────────────────
    print("Loading validation dataset...")
    val_dataset = MedicalImageDataset(
        config['paths']['val_data_dir'],
        noise_std=config.get('data', {}).get('noise_std', 25)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0          # safe for Windows
    )
    print(f"Validation samples: {len(val_dataset)}\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    summary, saved_samples = evaluate_dataset(model, val_loader, device, results_dir)

    print(f"\n{'='*60}")
    print(f"  Evaluation Summary")
    print(f"{'='*60}")
    print(f"  PSNR (dB) : {summary['psnr']:.2f}")
    print(f"  SSIM      : {summary['ssim']:.4f}")
    print(f"  MSE       : {summary['mse']:.4f}")
    print(f"  MAE       : {summary['mae']:.4f}")
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    save_comparison_images(saved_samples, results_dir, n=args.n_samples)

    # Persist summary as YAML
    summary_path = results_dir / 'evaluation_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump({'metrics': summary, 'checkpoint': checkpoint}, f)
    print(f"✓ Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
