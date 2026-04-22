
# Stage 3.1 — Lightning Module
This subfolder refactors the EfficientNet‑B0 fine‑tuning pipeline from a hand‑written training loop (Stage 2) into a PyTorch Lightning setup. It keeps the same model architecture and dataset（only training classifier head）, 
but moves all training orchestration (epochs, device placement, checkpointing, LR logging, profiling) into `Trainer` and callbacks.
## File Structure
```

```

## Results

## Key Finding
