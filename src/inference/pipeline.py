import torch
import numpy as np
import cv2
from pathlib import Path

from models.autoencoder import DenoisingAutoEncoder
from models.classifier import MedicalImageClassifier
from data.preprocessing import ImagePreprocessor


class DenoisingPipeline:
    """
    End-to-end inference pipeline:
      1. Load & preprocess a medical image
      2. Denoise it with the AutoEncoder
      3. Classify it (Normal / Abnormal) with the CNN
    """

    CLASS_LABELS = {0: 'Normal', 1: 'Abnormal'}

    def __init__(self,
                 denoiser_checkpoint: str | None = None,
                 classifier_checkpoint: str | None = None,
                 initial_filters: int = 32,
                 num_classes: int = 2,
                 device: str | None = None):

        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f"Pipeline running on: {self.device}")

        # ── Denoiser ──────────────────────────────────────────────────────────
        self.denoiser = DenoisingAutoEncoder(
            in_channels=1, initial_filters=initial_filters
        ).to(self.device)
        if denoiser_checkpoint:
            self._load_weights(self.denoiser, denoiser_checkpoint)
        self.denoiser.eval()

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = MedicalImageClassifier(
            in_channels=1, num_classes=num_classes
        ).to(self.device)
        if classifier_checkpoint:
            self._load_weights(self.classifier, classifier_checkpoint)
        self.classifier.eval()

        self.preprocessor = ImagePreprocessor(target_size=(256, 256))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_weights(self, model: torch.nn.Module, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state)
        print(f"Loaded weights from {checkpoint_path}")

    def _read_grayscale(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img

    def _to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        """Normalize [0,255] uint8 → [0,1] float tensor [1, 1, H, W]"""
        tensor = torch.from_numpy(
            image_np.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0)          # Add batch + channel dims
        return tensor.to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output tensor [1,1,H,W] → uint8 [H,W]"""
        arr = tensor.squeeze().cpu().detach().numpy()
        return np.clip(arr * 255, 0, 255).astype(np.uint8)

    # ── Public API ────────────────────────────────────────────────────────────

    def denoise(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Denoise an image from disk.

        Returns:
            (noisy_image, denoised_image) as uint8 numpy arrays [H, W]
        """
        raw = self._read_grayscale(image_path)
        resized = self.preprocessor.resize_image(raw)
        noisy = self.preprocessor.add_gaussian_noise(resized, noise_std=25)

        tensor = self._to_tensor(noisy)
        with torch.no_grad():
            output = self.denoiser(tensor)
        denoised = self._to_numpy(output)

        return noisy, denoised

    def classify(self, image_np: np.ndarray) -> dict:
        """
        Classify a grayscale image (numpy uint8 [H,W]).

        Returns:
            dict with keys: label (str), class_id (int), confidence (float),
                            probabilities (list[float])
        """
        tensor = self._to_tensor(image_np)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.squeeze().cpu().numpy().tolist()
        class_id = int(np.argmax(probs_np))

        return {
            'label':         self.CLASS_LABELS[class_id],
            'class_id':      class_id,
            'confidence':    probs_np[class_id],
            'probabilities': probs_np
        }

    def run(self, image_path: str) -> dict:
        """
        Full pipeline: denoise → classify.

        Returns:
            dict with keys: noisy, denoised (np.ndarray),
                            classification (dict)
        """
        noisy, denoised = self.denoise(image_path)
        classification = self.classify(denoised)

        print(f"\n── Inference Results ──────────────────────────")
        print(f"  Image      : {Path(image_path).name}")
        print(f"  Label      : {classification['label']}")
        print(f"  Confidence : {classification['confidence']*100:.1f}%")
        print(f"  Probs      : Normal={classification['probabilities'][0]:.3f} "
              f"| Abnormal={classification['probabilities'][1]:.3f}")
        print(f"───────────────────────────────────────────────\n")

        return {
            'noisy':          noisy,
            'denoised':       denoised,
            'classification': classification
        }

    def save_result(self, result: dict, output_dir: str = 'results/'):
        """Save noisy and denoised images side-by-side to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        noisy = result['noisy']
        denoised = result['denoised']

        # Side-by-side comparison
        separator = np.ones((noisy.shape[0], 4), dtype=np.uint8) * 128
        comparison = np.hstack([noisy, separator, denoised])

        label = result['classification']['label']
        conf = result['classification']['confidence']
        tag = f"{label}_{conf*100:.0f}pct"

        out_path = output_dir / f"result_{tag}.png"
        cv2.imwrite(str(out_path), comparison)
        print(f"Result saved to {out_path}")

        return str(out_path)
