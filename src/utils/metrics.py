import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(output: np.ndarray, target: np.ndarray,
                   max_pixel: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        output: Predicted image array (any shape), values in [0, 255]
        target: Ground-truth image array (same shape), values in [0, 255]
        max_pixel: Maximum possible pixel value (default 255)

    Returns:
        PSNR value in dB. Returns 100.0 for identical images.
    """
    mse = np.mean((output.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def calculate_ssim(output: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate mean Structural Similarity Index (SSIM) over a batch.

    Args:
        output: Predicted batch array of shape [B, C, H, W], values in [0, 255]
        target: Ground-truth batch array of same shape, values in [0, 255]

    Returns:
        Mean SSIM value across the batch in [0, 1].
    """
    output = output.astype(np.float64)
    target = target.astype(np.float64)

    # Handle both [B, C, H, W] and [H, W] inputs
    if output.ndim == 4:
        ssim_values = []
        for i in range(output.shape[0]):
            # Take first channel for grayscale; squeeze to [H, W]
            pred = output[i, 0]
            gt = target[i, 0]
            val = ssim(pred, gt, data_range=255.0)
            ssim_values.append(val)
        return float(np.mean(ssim_values))
    else:
        return float(ssim(output, target, data_range=255.0))


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Squared Error between two arrays."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Absolute Error between two arrays."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


class MetricsCalculator:
    """Convenience wrapper — compute all metrics in one call."""

    @staticmethod
    def compute_all_metrics(predicted: np.ndarray,
                            ground_truth: np.ndarray) -> dict:
        """
        Compute PSNR, SSIM, MSE, and MAE.

        Args:
            predicted:    Model output array, values in [0, 255]
            ground_truth: Ground-truth array, same shape, values in [0, 255]

        Returns:
            dict with keys: psnr, ssim, mse, mae
        """
        return {
            'psnr': calculate_psnr(predicted, ground_truth),
            'ssim': calculate_ssim(predicted, ground_truth),
            'mse':  calculate_mse(predicted, ground_truth),
            'mae':  calculate_mae(predicted, ground_truth),
        }
