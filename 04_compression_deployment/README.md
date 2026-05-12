# Stage 4 — Compression & Deployment

## Goal
This stage explores model compression and deployment optimization for the flower classification task. The main focus is to evaluate whether pruning and quantization can reduce model size and improve inference efficiency while maintaining strong validation accuracy.

## What This Stage Covers
- Unstructured L1 pruning
- Structured L1 pruning
- Structured L1 pruning with physical channel removal
- Quantization
- Benchmark compressed models

## File Structure
```
📁 04_compression_deployment/
├── 📁 __pycache__/                
├── 📁 logs/                       # Checkpoints, benchmark logs, and compression experiment outputs
├── 📁 preprocess/                 # Dataset preprocessing and data pipeline utilities
├── 📁 profiler_output/            # Profiling outputs for runtime and performance analysis
├── README.md                      # Stage 4 overview, experimental workflow, results, and deployment findings
├── base.yaml                      # Configuration file for baseline model training
├── base_flower.py                 # Train the baseline EfficientNet-B0 model used for later compression
├── benchmark.py                   # Benchmark compressed PyTorch models in terms of accuracy, size, and latency
├── benchmark_onnx.py              # Benchmark exported ONNX models with ONNX Runtime
├── export_and_quant_ort.py        # Export the baseline model to ONNX and generate an INT8 quantized version
├── flower_efficientnet_basemodel.onnx      # Baseline ONNX model exported from PyTorch
├── flower_efficientnet_quantization.onnx   # INT8-quantized ONNX model for faster and smaller deployment
├── mlflow.db                      # Local MLflow database for tracking runs and benchmark results
├── pruning_l1structured.py        # Structured L1 pruning experiment
├── pruning_l1unstructured.py      # Unstructured L1 pruning experiment
├── pruning_physical_channel.py    # Structured pruning with physical channel removal for real compression
├── runbash.py                     # Utility script for running experiment commands more conveniently
└── save_base_model.py             # Save a trained checkpoint as a standalone PyTorch .pth model
```

## Overview
This stage starts from a baseline EfficientNet B0 model, then fine-tunes the classifier  and the last three layers, then applies multiple compression methods and benchmarks their performance.

## Key Design Decisions

### 1. Compare different compression techniques
Three pruning strategies are tested in this stage: unstructured L1 pruning, structured L1 pruning, and structured pruning with physical channel removal, quantization. This allows direct comparison between theoretical sparsity and practical deployment benefit.

## Experimental Workflow

### 1. Train the base model and export `.pth`
First, train the baseline model with the Lightning configuration file:

```bash
python3.10 base_flower.py fit -c base.yaml
```

Then export the trained checkpoint into a PyTorch weight file:

```bash
python3.10 save_base_model.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt
```

### 2. Run pruning experiments
Three pruning variants are evaluated from the same baseline checkpoint.

#### Structured L1 pruning
```bash
python3.10 pruning_l1structured.py \
  --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt \
  --sparsity 0.3 \
  --finetune_epochs 10
```

#### Unstructured L1 pruning
```bash
python3.10 pruning_l1unstructured.py \
  --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt \
  --sparsity 0.3 \
  --finetune_epochs 10
```

#### Structured pruning with physical channel removal
```bash
python3.10 pruning_physical_channel.py \
  --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt \
  --pruning_ratio 0.15 \
  --global_pruning \
  --finetune_epochs 10
```

### 3. Benchmark PyTorch models
After pruning, benchmark the baseline and pruned models:

```bash
python3.10 benchmark.py --model_path logs/base_models/base_epoch=15_val_acc=0.9788.pth --run_name base
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_l1_unst_30.pth --run_name l1_unstructured
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_l1_st_30.pth --run_name l1_structured
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_physical_15.pth --run_name l1_structured_remove
```

### 4. Export and quantize ONNX models
Export the base model to ONNX and generate the quantized ONNX model:

```bash
python3.10 export_and_quant_ort.py
```

### 5. Benchmark ONNX Runtime models
Benchmark both the original ONNX model and the quantized ONNX model:

```bash
python3.10 benchmark_onnx.py --model_path flower_efficientnet_basemodel.onnx --run_name base
python3.10 benchmark_onnx.py --model_path flower_efficientnet_quantization.onnx --run_name quantized
```

### 6. Launch MLflow
To inspect experiment logs and benchmark records:

```bash
python3.10 -m mlflow server --backend-store-uri sqlite:///mlflow.db
```

## Results

### PyTorch benchmark
| Method | Setting | Accuracy | F1 Macro | Model size (MB) | Latency (ms) |
|--------|---------|----------|----------|:------------------:|:--------------:|
| Baseline | -- | 0.9854 | 0.9852 | 16.13 | 85.49 |
| Unstructured L1 pruning | sparsity = 0.3 | 0.9976 | 0.9970 | 16.13 | 85.72 |
| Structured L1 pruning | sparsity = 0.3 | 0.9780 | 0.9740 | 16.13 | 85.06 |
| Physical channel removal | pruning ratio = 0.15 | 0.9805 | 0.9742 | 13.66 | 68.76 |

### ONNX Runtime benchmark
| Method | Setting | Accuracy | F1 Macro | Model size (MB) | Latency (ms) |
|--------|---------|----------|----------|:------------------:|:--------------:|
| Baseline | FP32 | 0.9976 | 0.9946 | 15.78 | 10.36 |
| Quantization | INT8 | 0.9488 | 0.9474 | 4.78 | 4.94 |


## Key Findings
### 1. Unstructured pruning improved accuracy but not deployment efficiency
Unstructured L1 pruning slightly improved both accuracy and macro F1 compared with the PyTorch baseline. This suggests that the original model may have been slightly overfitted, and zeroing out less important small-magnitude weights acted as a form of regularization.

However, unstructured pruning did not reduce model size or latency. This is because pruning masks only set selected weights to zero, while the model still stores the same tensor shape and performs nearly the same dense computation during inference.

### 2. Structured pruning alone still gave limited runtime benefit
Structured L1 pruning reduced accuracy slightly and did not produce meaningful gains in model size or latency. Although structured pruning is more deployment-friendly than unstructured pruning in theory, mask-based structured pruning still keeps the original computational graph in many implementations.

### 3. Physical channel removal produced real compression
Physical channel removal gave the clearest practical benefit among the pruning-based methods. Compared with the PyTorch baseline, model size dropped from 16.13 MB to 13.66 MB, which is about a 15.3% reduction, and latency dropped from 85.49 ms to 68.76 ms, which is about a 19.6% reduction.

This result shows that actual filter removal is much more effective than masked pruning when the goal is real deployment acceleration.

### 4. Quantization gave the best size and latency improvement
The ONNX INT8 quantized model reduced model size from 15.78 MB to 4.78 MB, which is about 3.30x smaller. Latency also improved from 10.36 ms to 4.94 ms, giving about 2.10x speedup.

The trade-off is a noticeable drop in accuracy and macro F1. Therefore, quantization provides the strongest deployment gain, but its performance loss must be considered depending on the application requirement.

## Lessons Learned & Questions
This stage shows that compression methods must be judged by real deployment benefit. While mask-based pruning provides no gains in model size and latency, physical channel removal and quantization lead to more meaningful improvements in efficiency, though often with some trade-off in accuracy. The key lesson is that deployment optimization is a multi-objective problem involving accuracy, model size, latency, and framework compatibility.

These results raise several practical questions:
- Are compression techniques necessary for specific task?
- If so, which method is the most suitable?
- How much accuracy are we willing to compromise for lower latency and smaller model size?
