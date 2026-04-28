# Stage 4 — Compression Deployment

## Goal


## What This Stage Covers

## File Structure
```
📁 04_compression_deployment/
├── 📁 01_lightning_module/ 
├── 📁 03_hyperparameters/
├── 📁 04_interpretability/

└── README.md 
```

## Key Design Decisions

**1. Why PyTorch Lightning**


---  
## Hyperparameter Experiment Plan

### Group 1 — Learning Rate
Fixed: `optimizer=SGD`, `batch_size=128`, `epochs = 40`

| LR | Val Acc (%) | Best Epoch | Notes |
|----|-------------|------------|-------|
| 1e-4| 0.2769     | 40    | Very slow learning    |
| 1e-3| 0.8461     | 40    | slow learning and still underfits after 40 epochs |
| 1e-2| 0.9568     | 34    | Fast, stable convergence; reaches ~0.9 validation acc by epoch 10        |
| 1e-1| 0.9739     | 35    | fast learning, better validation acc       |

---

## Results

| Group      | Best config | Val acc (%) | Key finding                                           |
|-----------|-------------|-------------|-------------------------------------------------------|
| LR sweep  | 1e-1        | 97.39       | A relatively large learning rate works best for the newly added classifier head. |
| Optimizer | AdamW       | 96.50       | The choice of optimizer has a noticeable impact on convergence and final accuracy. |

---


## Key Findings

## Questions


---

## References
