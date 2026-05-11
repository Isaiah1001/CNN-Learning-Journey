# Stage 4 — Compression Deployment

## Goal
This stage explores model compression and deployment optimization for the flower classification task. The main focus is to evaluate whether pruning and quantization can reduce model size and improve inference efficiency while maintaining strong validation accuracy.

## What This Stage Covers
- Unstructured L1 pruning
- Structured L1 pruning
- Structured L1 pruning with physical channel removal
- Quantization
- Benchmark in compression models

## File Structure
```text
📁 04_compression_deployment/
├── 📁__pycache__/
├── 📁logs/
├── 📁preprocess/
├── 📁profiler_output/
├── README.md
├── base.yaml
├── base_flower.py
├── benchmark.py
├── benchmark_onnx.py
├── export_and_quant_ort.py
├── flower_efficientnet_basemodel.onnx
├── flower_efficientnet_quantization.onnx
├── mlflow.db
├── pruning_l1structured.py
├── pruning_l1unstructured.py
├── pruning_physical_channel.py
├── runbash.py
└── save_base_model.py
```

## Overview
This stage starts from a baseline EfficientNet B0 model, then fine-tunes classifier  and last three layers, then applies multiple compression methods and benchmarks their performance.

## Key Design Decisions

### 1. Compare different compression techniques
Three pruning strategies are tested in this stage: unstructured L1 pruning, structured L1 pruning, and structured pruning with physical channel removal, quantization. This allows direct comparison between theoretical sparsity and practical deployment benefit.

## Experimental Workflow

### 1. Train the base model and export `.pth`
First, train the baseline model with the Lightning configuration file:

```bash
python3.10 base_flower.py fit -c base.yaml
```

Then export the trained checkpoint into a deployable PyTorch weight file:

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
After pruning, benchmark the baseline and pruned `.pth` models:

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

### Compression benchmark
**on pth model**
| Method | Setting | Accracy| F1 Macro | Model size(mb) | Latency (ms) | 
|--------|---------|-------------|:-------------:|:------------:|:---------:|
| Baseline | -- | .9854 | .9852 | 16.13 | 10.36 |
| Unstructured L1 pruning | sparsity = 0.3 | .9976 | .9970 | 16.13 | 85.72 |
| Structured L1 pruning | sparsity = 0.3 | .9780 | .9740 | 16.13 | 85.06 |
| Physical channel removal | pruning ratio = 0.15 | .9805 | .9742 | 13.66 | 68.76 |

untructured pruning helps the model in accuracy, because our base model is overfitting, and set some tiny values in our model benefits. However, unstructured and structured pruning does not help in terms of model size and latency. This could be understood that the pruning technique just sets the less important parameters zeros, and those zeros are still stored in those models, and the calculation still involved. When we physically remove those filters, 15% in this case, we see the model size and lantency decreases by around 15% for model size and 20% in latency.

**on onnx model**
| Method | Setting | Accracy| F1 Macro | Model size(mb) | Latency (ms) | 
|--------|---------|-------------|:-------------:|:------------:|:---------:|
| Baseline | -- | .9976 | .9946 | 15.78 | 85.49 |
| Quantization | sparsity = 0.3 | .9488 | .9474 | 4.94 | 4.78 |

## Key Findings
- Unstructured pruning is useful for sparsity analysis, but not always for real acceleration.
- Structured pruning is more suitable for deployment because it preserves hardware-friendly dense computation patterns.
- Physical channel removal is important when the goal is to create a genuinely smaller and faster model.
- Quantization is a strong final-stage optimization for ONNX Runtime deployment.
- Benchmarking is necessary because fewer parameters do not automatically mean lower latency.

## Questions
- How much validation accuracy is lost after each compression method?
- Does structured pruning lead to lower latency than unstructured pruning?
- How much improvement comes specifically from physical channel removal?
- Does ONNX quantization provide the best accuracy-efficiency trade-off?
- Which compressed model should be selected for real deployment?

## References
- PyTorch: Saving and Loading Models
- Lightning AI: Saving and Loading Checkpoints
- ONNX Runtime Quantization Documentation
