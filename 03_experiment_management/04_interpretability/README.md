
# Stage 3.4 — Interpretability

This stage focuses on understanding and visualizing what the CNN model learns through various interpretability techniques.

## File Structure
```
📁 04_interpretability/
├── 📁 preprocess/          # Dataset utilities
├── 📁 outputs/             # Generated visualizations and analysis results
├── base.yaml               # Base YAML configuration
├── checkpoint_base_epoch=34_val_acc=0.9568.ckpt  # Trained model checkpoint
├── hyperparameters_flower.py  # Model and data module definitions
├── run_code.py             # Main entry point for running interpretability analysis
│
├── # Interpretability Scripts
├── gradcam_flower.py           # Grad-CAM visualization for correct predictions
├── gradcam_flower_true.py      # Grad-CAM for specific true class predictions
├── saliency_flower.py          # Saliency map visualization
├── saliency_flower_true.py     # Saliency maps for specific true class
├── error_analysis_lightning.py    # Analysis of misclassified samples
├── select_right_prediction.py  # Tools to select correctly classified samples
├── select_wrong_prediction.py  # Tools to select misclassified samples
```

## Results
**Code:** 'run_code.py' 
| Script | Technique | Purpose |
|--------|-----------|---------|
| `gradcam_flower*.py` | Grad-CAM | Visualize important regions in input images for CNN predictions |
| `saliency_flower*.py` | Saliency Maps | Show pixel-level importance based on gradient |
| `error_analysis_lightning.py` | Error Analysis | Analyze misclassified samples to understand model weaknesses |
| `select_right_prediction.py` | Sample Selection | Filter correctly classified samples for analysis |
| `select_wrong_prediction.py` | Sample Selection | Filter misclassified samples for analysis |
**Artifact:**  './output'

## Results
**Code:** `hyperparameters_flower.py` and corresponding yaml files  
**Artifact:** `checkpoint_base_epoch=34_val_acc=0.9568.ckpt`, `./outputs/`

### Learning Rate Comparison
| Run name | lr     | Final val acc | Final val loss | Notes |
|----------|--------|--------------:|---------------:|-------|
| 1e-4     | 1e-4   | 0.2769        | 3.96           | Very slow learning |
| 1e-3     | 1e-3   | 0.8461        | 1.04           | Slow learning, underfits after 40 epochs |
| 1e-2     | 1e-2   | 0.9568        | 0.19           | Fast, stable convergence; ~0.9 val acc by epoch 10 |
| 1e-1     | 1e-1   | 0.9739        | 0.11           | Fast learning, best validation acc |

### Optimizer Comparison
| Run name | lr   | Optimizer | Final val acc | Final val loss | Notes |
|----------|------|-----------|--------------:|---------------:|-------|
| AdamW    | 1e-2 | AdamW     | 0.9650        | 0.20           | Fast, stable, slightly best acc |
| Adam     | 1e-2 | Adam      | 0.9471        | 0.25           | Quick convergence, slightly lower final acc |
| base     | 1e-2 | SGD       | 0.9568        | 0.20           | Strong, stable baseline |
| RMSprop  | 1e-2 | RMSprop   | 0.4414        | 5.42           | Numerically unstable, poor acc |

## Key Findings
- **Learning rate** is critical: lr=1e-2 reaches ~0.9 validation accuracy by epoch 10 and stabilizes around 0.96
- With lr=1e-2, SGD, Adam, and AdamW all achieve similar final accuracy (~0.95–0.97); AdamW is slightly best
- **Default configuration**: SGD (lr=1e-2, momentum=0.9, weight_decay=1e-4) — simple, stable, competitive
- Using **LightningCLI + YAML configs**, experiments are defined declaratively with MLflow logging

