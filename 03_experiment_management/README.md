# Stage 3 — Experiment Management

## Goal

Stage 1&2 established a strong CNN and transfer learning pipeline on the Oxford 102 Flowers dataset, with the best EfficientNet-B0 fine-tuning result reaching 97.31% top-1 accuracy.

But the training process itself was messy: no unified way to log hyperparameters and metrics, no visibility into GPU/CPU utilization, which parts delaying training, and comparing runs required manually checking saved
files one by one, etc.. These problems slow down iteration. As the Chinese saying goes: '工欲善其事，必先利其器' (if a craftsman wants to do good work,  he must first sharpen his tools). 
Fortunately，the ML community has developed dedicated tooling to address exactly this class of workflow friction.

This stage introduces experiment management — setting up proper tooling to track, compare, and visualize training runs, then using this workflow to explore how key hyperparameters
affect model accuracy.

## What This Stage Covers
- Lighting module: 
- MLFlow: save and track
- hyperparameters: lr selections, optimizers, different training setups (batch, num_workers), etc..
- Visualization: saliency and CAM showing inference accuracy for interpretability

## File Structure
```
📁 03_experiment_management/
├── 📁 01_lighting_module/ #FlowersDataModule, FlowerLightModule, Trainer
├── 📁 02_mlflow/  # 
├── 📁 03_hyperparameters/
├── 📁 04_visualization/
└── README.md 
```
## Key Design Decisions

**1. Why EfficientNet-B0 **  

## Results
**1. Classifier head fine-tuning**  
**Code:** classifier_head.py  
**Artifact:** ./checkpoints/efficientnet_b0_flower.pth  
| Metric | Value |
|--------|-------|
| Dataset | Oxford 102 Flowers |
| Top-1 Accuracy | 93.49% (best:93.73%) |
| Epochs | 40 |
| Optimizer | SGD, lr=0.1, weight_decay=1e-4 |

Notes: Two types of accuracy are provided, one is accuracy after final epoch and another is best during training. Same for other tables
![Loss and Accuracy](./plot_results/classifier_head.png)

## Key Finding
**1. Classifier head fine-tuning**  


## Questions  

## Reference
- Wizwand, [Oxford Flowers-102 Classification Leaderboard](https://www.wizwand.com/sota/image-classification-on-oxford-flowers-102-test), accessed April 2026.
