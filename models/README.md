# Models Directory

This directory contains the trained models used by the composite attractiveness analyzer.

## Required Files

### 1. `beta_vae_utkface.pth`
- **Purpose**: Beta-VAE model trained on face data for aesthetic scoring
- **Usage**: Used by the face aesthetic scoring component
- **Format**: PyTorch state dict (.pth file)
- **Source**: Your trained Beta-VAE model

### 2. `gold_standard_female.pth`
- **Purpose**: Gold standard embedding vector for face comparison
- **Usage**: Reference vector for computing face aesthetic distances
- **Format**: PyTorch tensor (.pth file)
- **Source**: Extracted from your training data

## Model Architecture

The Beta-VAE model expects:
- **Input**: 64x64 RGB face images
- **Architecture**: 
  - Encoder: 4 convolutional layers (3→32→64→128→256 channels)
  - Latent space: 128 dimensions
  - Decoder: 4 transposed convolutional layers
- **Output**: Reconstruction + latent embeddings (mu, logvar)

## Usage

The models are automatically loaded by the `CompositeAttractivenessAnalyzer` class:

```python
from features.attractiveness_composite import composite_attractiveness_analyzer

# Models are loaded during initialization
result = composite_attractiveness_analyzer.compute_attractiveness_for_reel(video_path)
```

## Fallback Behavior

If model files are missing:
- The system will log warnings but continue to run
- Background analysis will still work (uses DeepLabV3 from torchvision)
- Face aesthetic scoring will return None/default values
- The composite score will be computed using available metrics only

## File Placement

Simply copy your trained model files to this directory:
```
models/
├── beta_vae_utkface.pth      # Your trained Beta-VAE model
├── gold_standard_female.pth  # Your gold standard vector
└── README.md                 # This file
```

The system will automatically detect and load them on startup.