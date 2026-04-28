
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

## Code
`run_code.py`: the bash of running those following codes  
`hyperparameters_flower.py`: lightning data and model module with hyperparameters defined in `base.yaml`to produce `checkpoint_base_epoch=34_val_acc=0.9568.ckpt`

| Script | Technique | Purpose |
|--------|-----------|---------|
| `gradcam_flower*.py` | Grad-CAM | Visualize important regions in input images for CNN predictions |
| `saliency_flower*.py` | Saliency Maps | Show pixel-level importance based on gradient |
| `error_analysis_lightning.py` | Error Analysis | Evaluate the checkpoint on the 410-image test set, show the inference statistics |
| `select_right_prediction.py` | Sample Selection | Filter correctly classified samples for CAM and Saliency map |
| `select_wrong_prediction.py` | Sample Selection | Filter misclassified samples for CAM and Saliency map |

## Artifact
The artifacts are inside folder `./outputs`
| Name |  Purpose |
|--------|---------|
| `all_predictions.csv` | the results for all 410 test cases |
| `condusion_matrix.png` | plot to show confusion matrix of all 410 test cases |
| `per_class_accuracy.csv` | inference accuracy for every class, from worst to best |
| `summary.txt` | accuracy, f1 scores, precision recall calculations and statistics are here |
| `wrong_predictions.csv` |  misclassified samples |
| `gradcam_targets.csv` | filter 4 types of misclassified samples for CAM and Saliency map  |
| `gradcam_targets_true.csv` | filter 4 types of correctly classified samples for CAM and Saliency map |



## Results
### Statistics

### CAM and Saliency

## Key Findings
- **Learning rate** is critical: lr=1e-2 reaches ~0.9 validation accuracy by epoch 10 and stabilizes around 0.96
- With lr=1e-2, SGD, Adam, and AdamW all achieve similar final accuracy (~0.95–0.97); AdamW is slightly best
- **Default configuration**: SGD (lr=1e-2, momentum=0.9, weight_decay=1e-4) — simple, stable, competitive
- Using **LightningCLI + YAML configs**, experiments are defined declaratively with MLflow logging

