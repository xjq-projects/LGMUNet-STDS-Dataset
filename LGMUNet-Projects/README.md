# LGMUNet:  Global Mamba UNet for 2D Medical Image Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LGMUNet is a 2D medical image segmentation framework that combines **Mamba** (State Space Models) with a **U-Net** architecture. It leverages the SS2D (Selective Scan 2D) module as a replacement for traditional self-attention, achieving a strong balance between performance and efficiency.

## Architecture

```
Input → PatchEmbed → Encoder (LGBlocks + SS2D) → Bottleneck → Decoder (LGBlocks + Skip Connections) → Output
```

### Key Components

- **SS2D (Selective Scan 2D)**: Four-directional scanning mechanism that captures global context efficiently
- **LGMamba Block**: Dual-branch design combining SS2D scanning with depthwise convolution
- **LGBlock**: Integrated block with convolution branch, MLP, and LGMamba attention
- **Hierarchical Encoder-Decoder**: Multi-scale feature extraction with skip connections

## Project Structure

```
LGMUNet-Projects/
├── configs/
│   └── Config_LGMUNet.py       # Model, data, training, and loss configurations
├── models/
│   └── LGFMUNet/
│       ├── LGMUNet.py           # LGMUNet factory and wrapper
│       └── UNet_VMamba.py       # Core model implementation (Encoder, Decoder, SS2D, etc.)
├── dataset.py                   # Dataset loading and augmentation
├── engine_lvmunet.py            # Training, validation, and testing loops
├── engine.py                    # Additional engine utilities
├── utils.py                     # Loss functions, metrics, optimizers, schedulers, visualization
├── train_LGMUNet.py             # Main training script with multi-seed support
└── data/                        # Dataset directory
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (recommended)

### Install Dependencies

```bash
# Install PyTorch (see https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Mamba SSM
pip install mamba_ssm

# Install other dependencies
pip install numpy scipy scikit-learn matplotlib tqdm tensorboard
pip install timm einops thop torchsummary
pip install SimpleITK medpy pillow opencv-python
```

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| ISIC 2017 | Skin lesion segmentation |
| ISIC 2018 | Skin lesion segmentation |
| STDS | Skin tumor dataset |
| MICCAI 2023 | Medical image segmentation challenge |

### Data Preparation

Organize your dataset as follows:

```
data/
└── MICCAI2023/
    └── train/
        ├── images/          # Training images (.png, .jpg, etc.)
        └── masks/           # Binary masks (.png, grayscale)
```

## Usage

### Quick Start

```bash
python train_LGMUNet.py
```

### Multi-Seed Training

The script supports training with multiple random seeds for robust evaluation:

```python
# In train_LGMUNet.py
SEEDS = [42, 3407, 1234]
```

Results are saved to `results/{model_name}_{dataset}_{timestamp}/` including:
- `seed_{N}/best_model.pth` — Best model weights per seed
- `seed_{N}/latest.pth` — Latest checkpoint
- `seed_{N}/tensorboard/` — TensorBoard logs
- `seed_{N}/visualizations/` — Prediction visualizations
- `multi_seed_results.json` — Aggregated results across all seeds

### Configuration

All settings are managed through dataclass configurations in [configs/Config_LGMUNet.py](configs/Config_LGMUNet.py):

```python
# Model Config
config.model.encoder_depths = [3, 3, 3, 3]   # Encoder depth per stage
config.model.decoder_depths = [3, 3, 3]      # Decoder depth per stage
config.model.embed_dim = 96                   # Embedding dimension
config.model.d_state = 16                     # SSM state dimension

# Data Config
config.data.batch_size = 8
config.data.input_size = (448, 448)

# Training Config
config.train.epochs = 300
config.train.opt = 'AdamW'                    # Optimizer
config.train.sch = 'CosineAnnealingLR'        # Scheduler
```

### Loss Functions

The following loss functions are available in [utils.py](utils.py):

| Loss | Description |
|------|-------------|
| `BCELoss` | Binary Cross-Entropy Loss |
| `DiceLoss` | Dice Similarity Coefficient Loss |
| `BceDiceLoss` | Combined BCE + Dice Loss (default) |
| `CeDiceLoss` | Cross-Entropy + Dice for multi-class |
| `GT_BceDiceLoss` | Deep supervision with multi-scale BCE+Dice |

### Data Augmentation

Custom augmentation pipeline (applied to both image and mask):

- `myResize` — Resize to target dimensions
- `myRandomHorizontalFlip` / `myRandomVerticalFlip` — Spatial flipping
- `myRandomRotation` — Random rotation (0°–360°)
- `myColorJitter` — Brightness, contrast, saturation, hue
- `myGaussianBlur` — Gaussian blur
- `myNormalize` — Dataset-specific normalization

## Evaluation Metrics

The following metrics are computed during validation and testing:

| Metric | Description |
|--------|-------------|
| DSC / F1 | Dice Similarity Coefficient |
| mIoU | Mean Intersection over Union |
| Accuracy | Overall pixel accuracy |
| Sensitivity | True positive rate (recall) |
| Specificity | True negative rate |

## TensorBoard Visualization

```bash
tensorboard --logdir=results/{model_name}_{dataset}_{timestamp}/seed_42/tensorboard
```

## Key Hyperparameters

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `embed_dim` | 96 | Embedding dimension |
| `d_state` | 16 | SS2D state dimension |
| `encoder_depths` | [3, 3, 3, 3] | Encoder blocks per stage |
| `decoder_depths` | [3, 3, 3] | Decoder blocks per stage |
| `drop_path_rate` | 0.1–0.2 | Stochastic depth rate |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `epochs` | 300 | Total training epochs |

## License

This project is for research purposes. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite the relevant works:

- VMamba: Visual State Space Model
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- U-Net: Convolutional Networks for Biomedical Image Segmentation
