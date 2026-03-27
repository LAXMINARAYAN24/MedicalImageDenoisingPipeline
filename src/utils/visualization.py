import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — safe for scripts & servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def plot_denoising_comparison(noisy: np.ndarray,
                               denoised: np.ndarray,
                               clean: np.ndarray | None = None,
                               title: str = 'Denoising Result',
                               save_path: str | None = None) -> None:
    """
    Plot noisy / denoised / clean images side-by-side.

    Args:
        noisy:     uint8 [H, W] noisy input
        denoised:  uint8 [H, W] model output
        clean:     uint8 [H, W] ground truth (optional)
        title:     figure title
        save_path: if given, saves the figure to this path
    """
    n_cols = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Noisy Input', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(denoised, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Denoised Output', fontsize=12)
    axes[1].axis('off')

    if clean is not None:
        axes[2].imshow(clean, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title('Ground Truth', fontsize=12)
        axes[2].axis('off')

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_history(history: dict,
                           save_path: str | None = None) -> None:
    """
    Plot training & validation loss / PSNR / SSIM curves.

    Args:
        history:   dict with keys train_loss, val_loss,
                   train_psnr, val_psnr, val_ssim (lists)
        save_path: if given, saves the figure to this path
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # ── Loss ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   label='Val',   linewidth=2, linestyle='--')
    ax1.set_title('Loss (MSE)', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    # ── PSNR ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, history['train_psnr'], label='Train', linewidth=2)
    ax2.plot(epochs, history['val_psnr'],   label='Val',   linewidth=2, linestyle='--')
    ax2.set_title('PSNR (dB) ↑', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('PSNR (dB)')
    ax2.legend(); ax2.grid(alpha=0.3)

    # ── SSIM ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(epochs, history['val_ssim'], label='Val SSIM', linewidth=2,
             color='green')
    ax3.set_title('SSIM ↑', fontweight='bold')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('SSIM')
    ax3.legend(); ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 1)

    fig.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_batch_results(noisy_batch: np.ndarray,
                        denoised_batch: np.ndarray,
                        n_samples: int = 4,
                        save_path: str | None = None) -> None:
    """
    Plot a grid of noisy/denoised pairs from a batch.

    Args:
        noisy_batch:    [B, 1, H, W] or [B, H, W] float array [0,1]
        denoised_batch: same shape as noisy_batch
        n_samples:      number of samples to show (≤ batch size)
        save_path:      if given, saves the figure
    """
    # Squeeze channel dim if present
    if noisy_batch.ndim == 4:
        noisy_batch    = noisy_batch[:, 0]
        denoised_batch = denoised_batch[:, 0]

    n = min(n_samples, noisy_batch.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle('Batch Results — Top: Noisy | Bottom: Denoised',
                 fontsize=13, fontweight='bold')

    for i in range(n):
        noisy_img    = (noisy_batch[i]    * 255).clip(0, 255).astype(np.uint8)
        denoised_img = (denoised_batch[i] * 255).clip(0, 255).astype(np.uint8)

        axes[0, i].imshow(noisy_img,    cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'Noisy #{i+1}', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(denoised_img, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'Denoised #{i+1}', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Internal helper ───────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
