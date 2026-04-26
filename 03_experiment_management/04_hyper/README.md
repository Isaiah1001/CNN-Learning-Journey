
# Stage 3.3 — Hyperparameters

## File Structure
```
📁 03_hyperparameters/
├── 📁 preprocess/  # dataset access and split utilities
├── 📁 logs/  # checkpoints
├── 📁 profiler_output/  # Lightning profiler outputs and trace files
├── 📁 yaml_lr/ # cli yaml files for different lr running
├── 📁 yaml_optimizer/ # cli yaml files for different optimizers running
├── hyperparameters_flower.py  # DataModule, LightningModule, lightningCLI setup
├── mlflow.db
├── lr.png
├── optimizer.png
├── run_config.py # helper scripts to run multiple YAML config
└── README.md 
```

## Results
**Code:** 'hyperparameters_flower.py'  and corresponding yaml files
**Artifact:** './logs', './profiler_output', 'mlflow.db', 'lr.png', 'optimizer.png'
### Different lr 
| Run name     | lr     | Final val acc | Final val loss | Notes                 |
|-------------|--------|--------------:|---------------:|-----------------------|
| sgd_lr1e-4  | 1e-4   | 0.2769          | 3.96           | Very slow learning    |
| sgd_lr1e-3  | 1e-3   | 0.8461        | 1.04           | Underfits after 40 ep |
| sgd_lr1e-2  | 1e-2   | 0.9568          | 0.19           | Best trade-off        |
| sgd_lr1e-1  | 1e-1   | 0.9739        | 0.11           | Diverges early        |
![Loss, Accuracy and Lr](./lr.png)  



## Key Finding  
