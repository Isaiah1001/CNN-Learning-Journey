# Stage 4 — Compression Deployment

## Goal
This part explores the compression techniques, like unstructured, structured pruning and quantization. 

## What This Stage Covers
- unstructured l1 pruning
- structured l1 pruning
- structured l1 pruning and remove the corresponding filters
- quantization
- benchmark
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
