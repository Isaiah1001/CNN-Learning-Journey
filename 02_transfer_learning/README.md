
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
EfficientNet-B0 backbone — to close this gap while keeping the model
lightweight enough for futher CNN learning.

## What This Stage Covers

- Classifier header fine-tuning
- Last layer fine-tuning
- Last three layer fine-tuning

## File Structure
```
📁 01_custom_cnn/
├── main.py # Full pipeline: data loading → pretrained model load and modified → training → visualization
├── 📁 model/ # training loop definition
├── 📁 postprocess/ # plot figures tools
├── 📁 preprocess/ # data manipulate tools
├── README.md # detailed procedures for pretrained model, unfreeze proccess, training and key findings
└── result.png # loss and accuracy plots
```

## Key Design Decisions

**1. Why EfficientNet-B0 **  
EfficientNet-B0 (5.3M parameters) is the baseline of the EfficientNet family. For a 102-class fine-grained
classification task on a small dataset (8,189 images), B0 offers a strong accuracy and efficiency trade-off.
Given limited compute resources, a small backbone like EfficientNet‑B0 makes fast experimentation and iteration possible.

**2. Why classifier head -> last layer -> last three layers fine-tuning**
To better understanding the magic of fine-tuning art and the performance of CNN backbone, gradual unfreezing benefits. Besides, comparison among classifier head, last layer and last three layers unfreezing will show inference accuracy improving gradually. Last, with gradual unfreezing strategy, the model training process will be under control.

## Results

| Metric | Value |
|--------|-------|
|Model | Classifier Head Fine-tuning|
| Top-1 Accuracy | 92.02% |
| Epochs | 40 |
| Optimizer | SGD, lr=0.1, weight_decay=1e-4 |

![Loss and Accuracy](result_classifier_header.png)

## Key Finding
**1. Classifier head fine-tuning**  
With only 20 epoches, inference accuracy reach around 90%, which is a hugh improvement, compared with model trained at **Stage 1**. This shows that the shallow layers and backbone, which have already been trained on large‑scale data, provide generic features that capture common visual characteristics of objects. Large learning rate should be used for this stage, since the weights for head are randomly initialized and large lr will help them converge quickly.

