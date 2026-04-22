
# Stage 3.1 — Lightning Module
This subfolder refactors the EfficientNet‑B0 fine‑tuning pipeline from a hand‑written training loop (Stage 2) into a PyTorch Lightning setup. It keeps the same model architecture and dataset（only training classifier head）, 
but moves all training orchestration (epochs, device placement, checkpointing, LR logging, profiling) into `Trainer` and callbacks.

## File Structure
```
📁 01_Lightning_module/
├── 📁 preprocess/  # dataset access and split utilities
├── 📁 logs/  # CSV logs and checkpoints
├── 📁 profiler_output/  # Lightning profiler outputs and trace files
├── lightning_flower.py  # DataModule, LightningModule, Trainer setup
└── README.md 
```

## Results
**Code:** 'lightning_flower.py'  
**Artifact:** './logs', './profiler_output'
| Metric | Value |
|--------|-------|
| Dataset | Oxford 102 Flowers |
| Top-1 Accuracy | 95.28% (best:95.68%) |
| Epochs | 40 |
| Optimizer | SGD, lr=0.01, momentum =0.9, weight_decay=1e-4, |

![Loss, Accuracy and Lr](./Lightning.png)  

The folder './profiler_output' inspects the cost of different operators and RAM inside the training model - both on the CPU and GPU. The trace files can be inspected with Perfetto for detailed performance analysis. A deeper interpretation of profiler bottlenecks and anomalies will be added in a later stage.
## Key Finding
