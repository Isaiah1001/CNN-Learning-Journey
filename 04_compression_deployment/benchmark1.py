# benchmark.py
# Baseline assessment of model size and inference latency
# Run this BEFORE any pruning/quantization to establish baseline numbers.
# Usage: python benchmark.py --ckpt path/to/checkpoint.ckpt

import os
import time
import argparse
import torch
from base_flower import FlowerLightModule

# ==============================================
# 1) load pre-trained weights and get the default transforms
# ==============================================

def load_model(ckpt_path: str) -> torch.nn.Module:
    pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path)
    pl_model.eval()
    return pl_model.model  # raw torchvision EfficientNet-B0


# ==============================================
# 2) Model size
# ==============================================

def get_model_size(model: torch.nn.Module) -> dict:
    """Estimate model size from parameters + buffers (dtype-aware)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": param_size / 1024 ** 2,
        "buffer_size_mb": buffer_size / 1024 ** 2,
        "model_size_mb": (param_size + buffer_size) / 1024 ** 2,
    }


def get_checkpoint_file_size(ckpt_path: str) -> float:
    """Disk size of the checkpoint file (includes optimizer states etc.)."""
    return os.path.getsize(ckpt_path) / 1024 ** 2


# ==============================================
# 3) Latency benchmark
# ==============================================

def benchmark_cpu(model: torch.nn.Module,
                  input_size=(1, 3, 224, 224),
                  warmup=20,
                  runs=100) -> dict:
    model = model.cpu()
    x = torch.randn(*input_size)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return {
        "device": "cpu",
        "batch_size": input_size[0],
        "runs": runs,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p50_ms": sorted(times)[len(times) // 2],
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
    }


def benchmark_gpu(model, input_size=(1, 3, 224, 224), warmup=20, runs=100):
    """MPS backend for Apple Silicon — torch.cuda.Event not available on MPS."""
    if not torch.backends.mps.is_available():
        return {"device": "mps", "error": "MPS not available"}

    model = model.to("mps")
    x = torch.randn(*input_size, device="mps")

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
    torch.mps.synchronize()  # flush warmup

    times = []
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            torch.mps.synchronize()  # wait for MPS to actually finish
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return {
        "device": "mps",
        "batch_size": input_size[0],
        "runs": runs,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p50_ms": sorted(times)[len(times) // 2],
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
    }



# ==============================================
# 4) Report
# ==============================================

def print_report(ckpt_path, size_stats, cpu_stats, gpu_stats):
    sep = "=" * 55

    print(f"\n{sep}")
    print(" BASELINE BENCHMARK — EfficientNet-B0 Flower")
    print(sep)

    print(f"\n{'── Checkpoint ──':}")
    print(f"  File            : {ckpt_path}")
    print(f"  Checkpoint size : {get_checkpoint_file_size(ckpt_path):.2f} MB  "
          f"(includes optimizer states)")

    print(f"\n── Model Size (parameters + buffers) ──")
    print(f"  Total params    : {size_stats['total_params']:,}")
    print(f"  Trainable params: {size_stats['trainable_params']:,}")
    print(f"  Param size      : {size_stats['param_size_mb']:.2f} MB")
    print(f"  Buffer size     : {size_stats['buffer_size_mb']:.2f} MB")
    print(f"  Model size      : {size_stats['model_size_mb']:.2f} MB")

    print(f"\n── CPU Latency (batch_size={cpu_stats['batch_size']}) ──")
    print(f"  Mean  : {cpu_stats['mean_ms']:.2f} ms")
    print(f"  Min   : {cpu_stats['min_ms']:.2f} ms")
    print(f"  p50   : {cpu_stats['p50_ms']:.2f} ms")
    print(f"  p95   : {cpu_stats['p95_ms']:.2f} ms")
    print(f"  Max   : {cpu_stats['max_ms']:.2f} ms")
    
    print(f"\n── GPU Latency (batch_size={gpu_stats.get('batch_size', 'N/A')}) ──")
    if "error" in gpu_stats:
        print(f"  {gpu_stats['error']}")
    else:
        print(f"  Mean  : {gpu_stats['mean_ms']:.2f} ms")
        print(f"  Min   : {gpu_stats['min_ms']:.2f} ms")
        print(f"  p50   : {gpu_stats['p50_ms']:.2f} ms")
        print(f"  p95   : {gpu_stats['p95_ms']:.2f} ms")
        print(f"  Max   : {gpu_stats['max_ms']:.2f} ms")
        print(f"  Peak GPU mem: {gpu_stats['peak_gpu_memory_mb']:.2f} MB")

    print(f"\n{sep}\n")

# ==============================================
# 5) Main function to run the benchmark
# ==============================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for latency test (default=1 for edge deployment)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    input_size = (args.batch_size, 3, 224, 224)

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt)

    size_stats = get_model_size(model)
    cpu_stats = benchmark_cpu(model, input_size, args.warmup, args.runs)
    gpu_stats = benchmark_gpu(model, input_size, args.warmup, args.runs)

    print_report(args.ckpt, size_stats, cpu_stats, gpu_stats)


if __name__ == "__main__":
    main()


# # single image latency (most relevant for edge deployment)
# python3.10 benchmark.py --ckpt logs/checkpoints/checkpoint_base_xx_x.ckpt

# # batch latency (relevant for server throughput)
# python3.10 benchmark.py --ckpt logs/checkpoints/checkpoint_base_xx_x.ckpt --batch_size 32
