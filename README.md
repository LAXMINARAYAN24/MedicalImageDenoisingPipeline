# Medical Image Denoising Pipeline 🏥

> An end-to-end deep learning pipeline for **denoising medical images** and classifying them as Normal / Abnormal — built with PyTorch, U-Net style AutoEncoder, and a CNN classifier.

---

## 📁 Project Structure

```
MedicalImageDenoisingPipeline/
├── config.yaml                   # Central configuration (all hyperparams & paths)
├── requirements.txt              # Python dependencies
├── datasets/
│   ├── generate_synthetic.py     # Generate synthetic medical-like training data
│   └── raw/
│       └── synthetic_medical/    # Generated PNG images live here
├── src/
│   ├── train.py                  # Phase 1: Train the DenoisingAutoEncoder
│   ├── train_classifier.py       # Phase 2: Train the CNN classifier
│   ├── test.py                   # Evaluate denoiser (PSNR / SSIM / MAE / MSE)
│   ├── models/
│   │   ├── autoencoder.py        # U-Net style DenoisingAutoEncoder
│   │   └── classifier.py        # CNN binary classifier (Normal vs Abnormal)
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset (clean-noisy pairs)
│   │   └── preprocessing.py     # Resize, noise injection, normalisation
│   ├── inference/
│   │   └── pipeline.py           # End-to-end inference: denoise → classify
│   └── utils/
│       ├── metrics.py            # PSNR, SSIM, MSE, MAE + MetricsCalculator
│       └── visualization.py     # Matplotlib plots (comparison, training curves)
├── logs/                         # Saved model checkpoints (.pth)
├── results/                      # Output images & evaluation YAML summaries
└── notebooks/
    └── exploration.ipynb         # Jupyter exploration notebook
```

---

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python datasets/generate_synthetic.py
```

This creates **500 synthetic medical-like PNG images** in `datasets/raw/synthetic_medical/`.

### 3. Train the Denoiser (AutoEncoder)

```bash
cd src
python train.py
```

- Trains a **U-Net style AutoEncoder** for Gaussian noise removal
- Saves best checkpoint to `logs/autoencoder_best_psnr_XX.XX.pth`
- Metrics: **PSNR** and **SSIM** tracked per epoch

### 4. Train the Classifier (CNN)

```bash
python src/train_classifier.py
# Or with a specific denoiser checkpoint:
python src/train_classifier.py --denoiser-checkpoint logs/autoencoder_best_psnr_XX.XX.pth
```

- Runs denoised images through the CNN to classify **Normal vs. Abnormal**
- Saves best checkpoint to `logs/classifier_best_acc_XX.X.pth`

### 5. Evaluate the Denoiser

```bash
python src/test.py
# Single image mode:
python src/test.py --image datasets/raw/synthetic_medical/image_0001.png
```

Outputs:
- Per-metric summary (PSNR, SSIM, MSE, MAE)
- Side-by-side comparison PNGs in `results/`
- `results/evaluation_summary.yaml`

### 6. Run Full Inference Pipeline

```python
from src.inference.pipeline import DenoisingPipeline

pipeline = DenoisingPipeline(
    denoiser_checkpoint='logs/autoencoder_best_psnr_XX.XX.pth',
    classifier_checkpoint='logs/classifier_best_acc_XX.X.pth'
)
result = pipeline.run('path/to/image.png')
pipeline.save_result(result, output_dir='results/')
```

---

## 🧠 Model Architecture

### DenoisingAutoEncoder (U-Net style)

```
Input (1×256×256)
  ↓
Encoder:   [32] → [64] → [128]   (with MaxPool2d)
  ↓
Bottleneck (128-channel feature map)
  ↓
Decoder:   [128] → [64] → [32]   (ConvTranspose2d + skip connections)
  ↓
Output (1×256×256) — tanh activation
```

Skip connections concatenate encoder feature maps at each scale — exactly like U-Net.

### CNN Classifier

```
Input (1×256×256)
  ↓ Conv Block ×2  → 32ch  → MaxPool
  ↓ Conv Block ×2  → 64ch  → MaxPool
  ↓ Conv Block ×2  → 128ch → MaxPool
  ↓ GlobalAvgPool
  ↓ FC(128→256) → Dropout(0.5) → FC(256→2)
  ↓
Softmax → {Normal, Abnormal}
```

---

## 📊 Metrics

| Metric | Description | Ideal |
|--------|-------------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | Higher ↑ |
| **SSIM** | Structural Similarity Index | Closer to 1 ↑ |
| **MSE** | Mean Squared Error | Lower ↓ |
| **MAE** | Mean Absolute Error | Lower ↓ |

---

## ⚙️ Configuration (`config.yaml`)

All hyperparameters live in `config.yaml`. Key sections:

| Section | Key Settings |
|---------|-------------|
| `paths` | Dataset directories, results & checkpoint dirs |
| `model` | Filter sizes, architecture type, class count |
| `training` | Epochs, batch size, LR, weight decay, scheduler |
| `data` | Image size, noise std, DataLoader workers |
| `inference` | Device, ONNX export, output format |
| `noise` | Gaussian std, Poisson, salt-and-pepper flags |
| `logging` | Log frequency, save interval, verbose |

---

## 🔧 Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | Deep learning framework |
| `opencv-python` | Image I/O and preprocessing |
| `scikit-image` | SSIM computation |
| `matplotlib` | Visualisation |
| `numpy` | Numerical operations |
| `tqdm` | Progress bars |
| `PyYAML` | Configuration loading |
| `onnx` + `onnxruntime` | Model export / inference |

---

## 📝 Reference

Inspired by techniques from:
- [Computer Vision Projects](https://github.com/avs-abhishek123/Computer-Vision-Projects) — avs-abhishek123  
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
