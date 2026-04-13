# CNN-Learning-Journey — From Scratch to EfficientNet Deployment
> A physics-trained engineer's structured path through deep learning:  
> building CNN intuition from first principles, then scaling to production-ready deployment.

## Background: PhD in Wind Science & Engineering
With expertise in computational fluid dynamics, large eddy simulation, 
and physical laboratory experimentation (tornado simulator). 
This repository documents the transition from physical simulation 
and experimental fluid dynamics to computer vision engineering.

## Motivation

This project is part of a self-directed, 700+ hour deep learning curriculum 
spanning Feb 2025 – present, built entirely outside of formal CS training:

| Period | Focus | Hours |
|--------|-------|-------|
| Feb – May 2025 | Statistical learning foundations (*Elements of Statistical Learning*) | ~100h |
| May – Sep 2025 | Coursera: ML Specialization → Deep Learning Specialization → AI Agent Developer | ~347h |
| Sep 2025 – Feb 2026 | Kaggle projects (tabular + image classification), YOLO, CS231N | ~247h |
| Feb 2026 – present | CNN architecture deep dive, fine-tuning, MLflow, pruning, quantization, ONNX | ongoing |

**Why this matters:** Most CV portfolios show the end result. This repository 
documents the reasoning behind each technical decision — the questions asked, 
the experiments that failed, and what was learned from them.

## Learning Roadmap
```
Stage 1: SimpleCNN from scratch (PyTorch implementation)
↓
Stage 2: Transfer Learning with EfficientNet (fine-tuning, training tricks)
↓
Stage 3: Experiment Management (MLflow tracking, hyperparameter search)
↓
Stage 4: Model Compression & Deployment (pruning → INT8 quantization → ONNX)
```
## Repository Structure
```
📁 01_custom_cnn/
├── main.py # Full pipeline: data loading → training → visualization
├── model.py # SimpleCNN built from scratch + debug utilities
├── preprocess.py # Custom Dataset, mean/std computation, dataloaders
└── README.md # Design decisions and key findings

📁 02_transfer_learning/
├── efficientnet_finetune.py
└── README.md

📁 03_experiment_tracking/
├── mlflow_tracking.py
└── README.md

📁 04_compression_deployment/
├── pruning.py
├── quantization.py
├── export_onnx.py
├── inference.py
└── README.md

📁 99_flower_data/ # The data for model training

📁 assets/ # Training curves, Grad-CAM, confusion matrix
README.md
requirements.txt


```
