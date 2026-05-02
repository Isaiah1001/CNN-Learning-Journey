# benchmark.py
import os
import time
import argparse
import functools
import torch
import mlflow
# `from base_flower import FlowerLightModule` is performed lazily inside
# load_model so that benchmarking a whole-model .pth file does not require
# Lightning / torchmetrics to be installed.

TRACKING_URI    = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pruning"


def _is_whole_model_file(path: str) -> bool:
    """A whole-model file (torch.save(model, path)) typically uses .pth/.pt;
    Lightning checkpoints use .ckpt. We rely on the extension first, and fall
    back to inspecting the loaded object."""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".pth", ".pt")


def load_model(path: str):
    """Load either a Lightning checkpoint (.ckpt) or a whole nn.Module
    serialized via `torch.save(model, path)` (.pth/.pt).

    Whole-model files are required for physically pruned models because their
    Conv/BN/Linear shapes no longer match the original FlowerLightModule.
    """
    if _is_whole_model_file(path):
        # weights_only=False: we trust this file and need pickled module classes.
        model = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Expected an nn.Module in {path}, got {type(model).__name__}. "
                "If this is a Lightning checkpoint, rename it to .ckpt."
            )
        model.eval()
        return model

    from base_flower import FlowerLightModule  # needs Lightning + torchmetrics
    torch.serialization.add_safe_globals([functools.partial])
    pl_model = FlowerLightModule.load_from_checkpoint(path)
    pl_model.eval()
    return pl_model.model


def get_model_size(model):
    param_size  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params, "trainable_params": trainable_params,
        "param_size_mb": param_size / 1024**2, "buffer_size_mb": buffer_size / 1024**2,
        "model_size_mb": (param_size + buffer_size) / 1024**2,
    }


def get_checkpoint_file_size(ckpt_path):
    return os.path.getsize(ckpt_path) / 1024**2


def benchmark_cpu(model, input_size=(1,3,224,224), warmup=20, runs=100):
    model = model.cpu()
    x = torch.randn(*input_size)
    with torch.inference_mode():
        for _ in range(warmup): model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter(); model(x); t1 = time.perf_counter()
            times.append((t1-t0)*1000)
    s = sorted(times)
    return {"device":"cpu","batch_size":input_size[0],"runs":runs,
            "mean_ms":sum(times)/len(times),"min_ms":s[0],"max_ms":s[-1],
            "p50_ms":s[len(times)//2],"p95_ms":s[int(len(times)*0.95)]}


def benchmark_mps(model, input_size=(1,3,224,224), warmup=20, runs=100):
    if not torch.backends.mps.is_available():
        return {"device":"mps","error":"MPS not available"}
    model = model.to("mps")
    x = torch.randn(*input_size, device="mps")
    with torch.inference_mode():
        for _ in range(warmup): model(x)
    torch.mps.synchronize()
    times = []
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter(); model(x); torch.mps.synchronize(); t1 = time.perf_counter()
            times.append((t1-t0)*1000)
    s = sorted(times)
    return {"device":"mps","batch_size":input_size[0],"runs":runs,
            "mean_ms":sum(times)/len(times),"min_ms":s[0],"max_ms":s[-1],
            "p50_ms":s[len(times)//2],"p95_ms":s[int(len(times)*0.95)]}


def print_report(model_path, size_stats, cpu_stats, mps_stats):
    sep = "="*55
    print(f"\n{sep}\n  BENCHMARK — EfficientNet-B0 Flower\n{sep}")
    print(f"\n── Model File ──")
    print(f"  File            : {model_path}")
    print(f"  File size       : {get_checkpoint_file_size(model_path):.2f} MB")
    print(f"\n── Model Size (in-memory) ──")
    print(f"  Total params    : {size_stats['total_params']:,}")
    print(f"  Trainable params: {size_stats['trainable_params']:,}")
    print(f"  Param size      : {size_stats['param_size_mb']:.2f} MB")
    print(f"  Buffer size     : {size_stats['buffer_size_mb']:.2f} MB")
    print(f"  Model size      : {size_stats['model_size_mb']:.2f} MB")
    print(f"\n── CPU Latency (bs={cpu_stats['batch_size']}, runs={cpu_stats['runs']}) ──")
    for k in ["mean_ms","min_ms","p50_ms","p95_ms","max_ms"]:
        print(f"  {k:<9}: {cpu_stats[k]:.2f} ms")
    print(f"\n── MPS Latency (Apple Silicon) ──")
    if "error" in mps_stats: print(f"  {mps_stats['error']}")
    else:
        for k in ["mean_ms","min_ms","p50_ms","p95_ms","max_ms"]:
            print(f"  {k:<9}: {mps_stats[k]:.2f} ms")
    print(f"\n{sep}\n")


def log_to_mlflow(run_name, model_path, size_stats, cpu_stats, mps_stats, batch_size, warmup, runs):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model":"EfficientNet-B0","model_path":model_path,
                           "benchmark_batch":batch_size,"benchmark_warmup":warmup,"benchmark_runs":runs})
        mlflow.log_metrics({
            "model/total_params":     size_stats["total_params"],
            "model/trainable_params": size_stats["trainable_params"],
            "model/param_size_mb":    round(size_stats["param_size_mb"],3),
            "model/buffer_size_mb":   round(size_stats["buffer_size_mb"],3),
            "model/total_size_mb":    round(size_stats["model_size_mb"],3),
            "ckpt/file_size_mb":      round(get_checkpoint_file_size(model_path),2),
            "latency_cpu/mean_ms":    round(cpu_stats["mean_ms"],3),
            "latency_cpu/min_ms":     round(cpu_stats["min_ms"],3),
            "latency_cpu/p50_ms":     round(cpu_stats["p50_ms"],3),
            "latency_cpu/p95_ms":     round(cpu_stats["p95_ms"],3),
            "latency_cpu/max_ms":     round(cpu_stats["max_ms"],3),
        })
        if "error" not in mps_stats:
            mlflow.log_metrics({
                "latency_mps/mean_ms": round(mps_stats["mean_ms"],3),
                "latency_mps/min_ms":  round(mps_stats["min_ms"],3),
                "latency_mps/p50_ms":  round(mps_stats["p50_ms"],3),
                "latency_mps/p95_ms":  round(mps_stats["p95_ms"],3),
                "latency_mps/max_ms":  round(mps_stats["max_ms"],3),
            })
        print(f"\nLogged to MLflow — experiment: '{EXPERIMENT_NAME}', run: '{run_name}'")
        print(f"View: python3.10 -m mlflow ui --backend-store-uri {TRACKING_URI} --port 5001\n")


def main():
    parser = argparse.ArgumentParser()
    # --ckpt and --model_file are mutually exclusive but we accept either to
    # stay compatible with existing scripts.
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Lightning checkpoint (.ckpt)")
    parser.add_argument("--model_file", type=str, default=None,
                        help="Whole nn.Module saved with torch.save(model, path) (.pth/.pt). "
                             "Use this for physically pruned models.")
    parser.add_argument("--run_name",   type=str, default="benchmark_base")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup",     type=int, default=20)
    parser.add_argument("--runs",       type=int, default=100)
    args = parser.parse_args()

    if args.ckpt and args.model_file:
        parser.error("Pass either --ckpt or --model_file, not both.")
    model_path = args.model_file or args.ckpt
    if model_path is None:
        parser.error("One of --ckpt or --model_file is required.")

    input_size = (args.batch_size, 3, 224, 224)
    model      = load_model(model_path)
    size_stats = get_model_size(model)
    cpu_stats  = benchmark_cpu(model, input_size, args.warmup, args.runs)
    mps_stats  = benchmark_mps(model, input_size, args.warmup, args.runs)

    print_report(model_path, size_stats, cpu_stats, mps_stats)
    log_to_mlflow(args.run_name, model_path, size_stats, cpu_stats, mps_stats,
                  args.batch_size, args.warmup, args.runs)

if __name__ == "__main__":
    main()

# Examples:
# Lightning ckpt:
#   python3.10 benchmark.py --ckpt ./logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt --run_name benchmark_base
# Physically pruned whole-model file:
#   python3.10 benchmark.py --model_file ./logs/pruned/efficientnet_b0_pruned_30.pth --run_name benchmark_pruned_physical_30
