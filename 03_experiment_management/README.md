# Stage 3 — Experiment Management

## Goal

Stage 1&2 established a strong CNN and transfer learning pipeline on the Oxford 102 Flowers dataset, with the best EfficientNet-B0 fine-tuning result reaching 97.31% top-1 accuracy.

But the training process itself was messy: no unified way to log hyperparameters and metrics, no visibility into GPU/CPU utilization, which parts delaying training, and comparing runs required manually checking saved
files one by one, how I can interpret the prediction results, etc.. These problems slow down iteration. As the Chinese saying goes: '工欲善其事，必先利其器' (if a craftsman wants to do good work,  he must first sharpen his tools). 
Fortunately，the ML community has developed dedicated tooling to address exactly this class of workflow issues.

This stage introduces experiment management: setting up proper tooling to organize training code, track and compare runs, then using this workflow to study how key hyperparameters affect accuracy. Last, use CAM and saliency map to interpret model behavior.

## What This Stage Covers
- Lighting module: Refactor EfficientNet-B0 training into `LightningDataModule` + `LightningModule`, replacing the hand-written training loop with Trainer-managed epochs, built-in LR logging, and `ModelCheckpoint` callbacks
- MLFlow: Every training run automatically logs hyperparameters, per-epoch metrics, epoch time, and model artifacts; compare runs visually via `mlflow ui`.
- Hyperparameters: Use the 'LightningCLI' + MLflow workflow to systematically compare learning rates, optimizers; produce a clean results table.
- Interpretability: Analyze error to show more prediction details, use saliency maps and CAM/Grad-CAM to visualize which regions drive predictions.

## File Structure
```
📁 03_experiment_management/
├── 📁 01_lightning_module/ #Data Module, Model Module, Trainer and callbacks
    ├── 📁 preprocess/  
    ├── 📁 logs/ 
    ├── 📁 profiler_output/  
    ├── lightning_flower.py
    └── README.md 
├── 📁 02_mlflow/  # mlflow to show experiment
    ├── 📁 preprocess/  
    ├── 📁 logs/  
    ├── 📁 profiler_output/  
    ├── MLflow_flower.py
    ├── mlflow.db
    ├── mlflow.png
    └── README.md 
├── 📁 03_hyperparameters/ # different lr and optimizers experiment by lightningCLI
    ├── 📁 preprocess/  
    ├── 📁 logs/  
    ├── 📁 profiler_output/ 
    ├── 📁 yaml_lr/
    ├── 📁 yaml_optimizer/
    ├── hyperparameters_flower.py 
    ├── mlflow.db
    ├── lr.png
    ├── optimizer.png
    ├── run_config.py
    └── README.md 
├── 📁 04_interpretability/ # error analysis, Grad-CAM and saliency map for interpretation
    ├── 📁 preprocess/          
    ├── 📁 outputs/            
    ├── base.yaml               
    ├── checkpoint_base_epoch=34_val_acc=0.9568.ckpt 
    ├── hyperparameters_flower.py
    ├── run_code.py          
    │
    ├── # Interpretability Scripts
    ├── gradcam_flower.py         
    ├── gradcam_flower_true.py    
    ├── saliency_flower.py        
    ├── saliency_flower_true.py    
    ├── error_analysis_lightning.py
    ├── select_right_prediction.py 
    ├── select_wrong_prediction.py
└── README.md 
```

## Key Design Decisions

**1. Why PyTorch Lightning**

The Stage 2 training loop (`model/training_loop.py`) was reasonably clean, but it still required manually handling the epoch loop, metric accumulation, best‑model tracking, device placement, and learning‑rate scheduling. 
PyTorch Lightning handles all of this via `Trainer`, so the module code only needs to define *what* happens (forward pass, loss, optimizer) rather than *how* the training loop is executed.

In this stage, I also take advantage of several built‑in and custom callbacks:

- `LearningRateMonitor`: logs the learning rate each epoch automatically, with no manual tracking code.
- `ModelCheckpoint`: saves the best and/or last checkpoints based on `val_acc` without any custom save logic.
- `ProgressiveBackboneFinetuning`: encodes the gradual unfreezing strategy as a reusable callback, instead of hard‑coding it into the training loop.
- `PostFreezeModelSummary`: prints the total and trainable parameter counts whenever the finetuning state changes.
- `MLFlowLogger` (in folder '02_mlflow'): plugs directly into `Trainer` so that hyperparameters, metrics, and artifacts are logged to MLflow with minimal extra code.

Other useful tools:

- Profiler: records detailed timing and memory information during training to help identify performance bottlenecks.

**2. Why MLflow**

Stage 2 produced six nearly identical scripts to track six experimental configurations. MLflow
solves this by treating each training run as a named artifact with full parameter + metric history.
Key capabilities used:

- `mlflow ui`: browser-based dashboard to compare runs side by side
- Artifact logging: best checkpoint `.pth` stored alongside its run, always traceable
- No external server needed: local SQLite backend (`mlflow.db`) sufficient for solo projects

**3. Experiment Design: Change One Variable at a Time**

At this stage, lightningCLI is used for speedup, in case of messy  each experiment group has only one variable, like lr or optimizer. This makes direct comparison clear for variable changing.

**4. Interpretability: use tools to understand model behavior instead of relying only on accuracy**
- Saliency and (Grad-)CAM heatmaps to show which image regions drive each prediction.
- Confusion matrix, per-class accuracy, and a few failure examples to see which flower classes are hardest and how the model makes mistakes.
  
---  
## Hyperparameter Experiment Plan

### Group 1 — Learning Rate
Fixed: `optimizer=AdamW`, `batch_size=64`

| LR | Val Acc (%) | Best Epoch | Notes |
|----|-------------|------------|-------|
| 1e-4 | — | — | |
| 3e-4 | — | — | |
| 1e-3 | — | — | |
| 3e-3 | — | — | |

### Group 2 — Optimizer
Fixed: `lr=3e-4`, `batch_size=64`

| Optimizer | Val Acc (%) | Best Epoch | Notes |
|-----------|-------------|------------|-------|
| SGD + momentum=0.9 | — | — | Stage 2 baseline |
| Adam | — | — | |
| AdamW | — | — | |

### Group 3 — Batch Size & DataLoader Efficiency
Fixed: `optimizer=AdamW`, `lr=3e-4`

| Batch Size | num_workers | Val Acc (%) | Epoch Time (s) | Notes |
|------------|-------------|-------------|----------------|-------|
| 32 | 4 | — | — | |
| 64 | 4 | — | — | |
| 128 | 4 | — | — | |
| 64 | 2 | — | — | dataloader bottleneck check |
| 64 | 8 | — | — | dataloader bottleneck check |

*All tables will be filled in after experiments complete. Final results summary in `03_hyperparameters/results.md`.*

---

## Results

*(To be filled after experiments complete)*

| Group | Best Config | Val Acc (%) | Key Finding |
|-------|-------------|-------------|-------------|
| LR sweep | — | — | — |
| Optimizer | — | — | — |
| Batch size | — | — | — |

---

## How to Run

**1. Install dependencies**
```bash
pip install pytorch-lightning mlflow
```

**2. Train with default config**
```bash
cd 01_lightning_module
python train.py
```

**3. Launch MLflow dashboard**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in browser
```

**4. Run full hyperparameter sweep**
```bash
cd 03_hyperparameters
python run_experiments.py
```

---

## Connection to Stage 2

Stage 2 ended with a practical question (quoted from Stage 2 README):

> *"How to log metrics and artifacts without producing an unmanageable number of files,
> and how to inspect the training process in enough detail to decide when to stop or
> adjust hyperparameters?"*

Stage 3 directly answers this. The six separate scripts from Stage 2 collapse into a single
`train.py` entry point, and all run history is stored and queryable in MLflow.

---

## References

- Falcon et al., [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), Lightning AI, 2019.
- Zaharia et al., [MLflow: A Machine Learning Lifecycle Platform](https://mlflow.org), Databricks, 2018.
- Selvaraju et al., [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391), ICCV 2017.
- Kolesnikov et al., [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370), ECCV 2020.
