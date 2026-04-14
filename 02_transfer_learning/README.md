
# Stage 2 — Transfer Learning

## Goal

Stage 1 established a working CNN pipeline on the Oxford 102 Flowers dataset,
but the custom SimpleCNN reached only ~42% top-1 accuracy — well below the
SOTA benchmark of 99.85% (Efficient Adaptive Ensembling) and even the
EfficientNet-B0 paper result of 97.3%.

The accuracy gap points to a fundamental limitation: training a shallow CNN
from scratch on a small dataset (8,189 images, 102 classes) cannot match
the rich feature representations learned from large-scale pre-training.

This stage applies **transfer learning** — leveraging an ImageNet pre-trained
EfficientNet-B4 backbone — to close this gap while keeping the model
lightweight enough for futher CNN learning.

## What This Stage Covers

- head modified
- last layer fine tuning
- last three layer fine tuning

## File Structure
```
📁 01_custom_cnn/
├── main.py # Full pipeline: data loading → training → visualization
├── 📁 postprocess/ # 
├── 📁 preprocess/ # data manipulate tools
├── README.md # detailed procedures for custom CNN model training and key findings
└── result.png # loss and accuracy plots
```

## Key Design Decisions

**1. **  

## Results

| Metric | Value |
|--------|-------|
| Dataset | Oxford 102 Flowers |
| Top-1 Accuracy | ?% |
| Epochs | 40 |
| Optimizer | SGD, lr=1e-3, weight_decay=1e-4 |

## Key Finding
