
# Stage 3.3 — Hyperparameters

## File Structure
```
📁 03_hyperparameters/
├── 📁 preprocess/  # dataset access and split utilities
├── 📁 logs/  # checkpoints
├── 📁 profiler_output/  # Lightning profiler outputs and trace files
├── MLflow_flower.py  # DataModule, LightningModule, Trainer setup
├── mlflow.db
├── mlflow.png
└── README.md 
```

## Results
**Code:** 'MLflow_flower.py'  
**Artifact:** './logs', './profiler_output', 'mlflow.db', 'mlflow.png'
| Metric | Value |
|--------|-------|
| Dataset | Oxford 102 Flowers |
| Top-1 Accuracy | 95.3% (best:95.68%) |
| Epochs | 40 |
| Optimizer | SGD, lr=0.01, momentum =0.9, weight_decay=1e-4, |

![Loss, Accuracy and Lr](./mlflow.png)  



## Key Finding  
