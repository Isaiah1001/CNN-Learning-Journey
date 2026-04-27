
# Stage 3.4 — interpretability

## File Structure
```
📁 04_hyperparameters/
├── 📁 preprocess/  # dataset access and split utilities
├── 📁 logs/  # checkpoints
├── 📁 profiler_output/  # Lightning profiler outputs and trace files
├── 📁 yaml_lr/ # cli yaml files for different lr running
├── 📁 yaml_optimizer/ # cli yaml files for different optimizers running
├── hyperparameters_flower.py  # DataModule, LightningModule, lightningCLI setup
├── mlflow.db # MLflow tracking database for all runs in this stage
├── base.yaml #  YAML config
└── README.md 
```

## Results
**Code:** 'hyperparameters_flower.py'  and corresponding yaml files  
**Artifact:** './logs', './profiler_output', 'mlflow.db', 'lr.png', 'optimizer.png'
### Different lr 
| Run name     | lr     | Final val acc | Final val loss | Notes                 |
|-------------|--------|--------------:|---------------:|-----------------------|
| 1e-4  | 1e-4   | 0.2769          | 3.96           | Very slow learning    |
| 1e-3  | 1e-3   | 0.8461        | 1.04           | slow learning and still underfits after 40 epochs |
| 1e-2  | 1e-2   | 0.9568          | 0.19           | Fast, stable convergence; reaches ~0.9 validation acc by epoch 10        |
| 1e-1  | 1e-1   | 0.9739        | 0.11           | fast learning, better validation acc       |

![Loss, Accuracy and Lr](./lr.png)  

### Different optimizers

| Run name | lr   | Optimizer | Final val acc | Final val loss | Notes                                      |
|---------|------|-----------|--------------:|---------------:|--------------------------------------------|
| AdamW   | 1e-2 | AdamW     | 0.9650        | 0.20           | Fast, stable convergence, slightly best acc |
| Adam    | 1e-2 | Adam      | 0.9471        | 0.25           | Converges quickly, final acc slightly lower |
| base    | 1e-2 | SGD       | 0.9568        | 0.20           | Strong, stable baseline with simple SGD     |
| RMSprop | 1e-2 | RMSprop   | 0.4414        | 5.42           | numerically unstable and poor final acc  |

![Loss, Accuracy and Lr](./optimizer.png)  

## Key Findings
- Learning rate matters more than optimizer choice in this setup: 1e‑4 and 1e‑3 clearly underfit, while 1e‑2 reaches ~0.9 val accuracy by epoch 10 and stabilizes around 0.96.  
- With lr = 1e‑2, SGD, Adam, and AdamW all reach similar final accuracy (~0.95–0.97); AdamW is slightly best, RMSprop is clearly worse and unstable on validation.  
- For later stages, I keep **SGD (lr=1e‑2, momentum=0.9, weight_decay=1e‑4)** as the default: simple, stable, and competitive with Adam/AdamW.  
- Using **LightningCLI + YAML configs + `run_config.py`**, each experiment is defined declaratively (no code changes), and full sweeps over learning rates or optimizers can be launched with a single command while MLflow logs all runs.

