# Stage 1 — SimpleCNN From Scratch

## Goal

Before using any pre-trained model, this stage establishes a working 
understanding of the common CNN pipeline — from raw data ingestion to 
training loop design — by building everything from scratch

## What This Stage Covers

- data read and custom `Dataset` class with Oxford `.mat` label file parsing
- Per-channel mean/std computed from the training set (not ImageNet defaults)
- Data augmentation with `torchvision.transforms`
- `SimpleCNN` built from scratch: Conv → BN → ReLU → MaxPool blocks
- SGD optimizer
- Training loop with per-epoch validation tracking
- Training/validation loss curves and accuracy visualization

## File Structure
```
📁 01_custom_cnn/
├── main.py # Full pipeline: data loading → training → visualization
├── 📁 model/ # SimpleCNN basic block, structure, inspection tools and training loop setup
├── 📁 preprocess/ # data manipulate tools
└── README.md # detailed procedures for custom CNN model training and key findings
```

## Key Design Decisions

**1. Data augmentation strategy**  
Except basic resize and center crop transformation, applied random horizontal flip and random rotation 
during training to improve generalization on a relatively small dataset (8,189 images). 
Validation and test sets use only resize and center crop — no augmentation.

**2. Manual architecture before transfer learning**  
Building SimpleCNN from scratch before using EfficientNet can help me obtain a 
clear understanding of some basic CNN knowledge and pipeline, e.g., how data augmentation, gradient flow, receptive fields, and 
loss function definition and validation in practice.

## Results

| Metric | Value |
|--------|-------|
| Dataset | Oxford 102 Flowers |
| Top-1 Accuracy | 32% |
| Epochs | 40 |
| Optimizer | SGD, lr=1e-3, weight_decay=1e-4 |

## Key Finding

Using customed SimpleCNN,the accuracy plateaus quickly, but only at poor accuracy. Compared to the SOTA model, e.g, 99.847% for Efficient Adaptive Ensembling, 99.74% for Vision Transformer ViT-L/16
and 97.3% for EfficientNet-B0, better and advanced model should be employed. Considering limited computer resource and current learning stage, this directly motivated the switch to pre-trained EfficientNet in Stage 2.
