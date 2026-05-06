# benchmark.py
import os
import gzip
import io
import time
import argparse
import torch
import mlflow
from base_flower import FlowerLightModule
import functools

TRACKING_URI    = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pruning"

def load_model(ckpt_path):
    torch.serialization.add_safe_globals([functools.partial])
    pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path)
    pl_model.eval()
    return pl_model.model

def get_model_size(model):
    param_size  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Non-zero parameter accounting -- distinguishes mask-only ("structured")
    # pruning (zeros baked into full-shape tensors) from physical channel
    # surgery. For mask-only pruning, total_params and param_size_mb stay
    # constant; only nonzero_params drops.
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    effective_param_size = nonzero_params * 4  # float32 lower bound

    # Gzipped state_dict size -- the only on-disk savings you get "for free"
    # from a mask-only pruned model, because zero runs compress well.
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw_bytes = buf.getvalue()
    gz_bytes  = gzip.compress(raw_bytes, compresslevel=6)

    return {
        "total_params": total_params, "trainable_params": trainable_params,
        "nonzero_params": nonzero_params,
        "param_sparsity": 1 - nonzero_params / total_params if total_params else 0.0,
        "param_size_mb": param_size / 1024**2, "buffer_size_mb": buffer_size / 1024**2,
        "model_size_mb": (param_size + buffer_size) / 1024**2,
        "effective_param_size_mb": effective_param_size / 1024**2,
        "state_dict_raw_mb":  len(raw_bytes) / 1024**2,
        "state_dict_gzip_mb": len(gz_bytes)  / 1024**2,
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

def print_report(ckpt_path, size_stats, cpu_stats, mps_stats):
    sep = "="*55
    print(f"\n{sep}\n  BASELINE BENCHMARK — EfficientNet-B0 Flower\n{sep}")
    print(f"\n── Checkpoint ──")
    print(f"  File            : {ckpt_path}")
    print(f"  Checkpoint size : {get_checkpoint_file_size(ckpt_path):.2f} MB (includes optimizer states)")
    print(f"\n── Model Size ──")
    print(f"  Total params    : {size_stats['total_params']:,}")
    print(f"  Trainable params: {size_stats['trainable_params']:,}")
    print(f"  Non-zero params : {size_stats['nonzero_params']:,}  "
          f"(param sparsity {size_stats['param_sparsity']:.1%})")
    print(f"  Param size      : {size_stats['param_size_mb']:.2f} MB  (nominal, dense)")
    print(f"  Effective size  : {size_stats['effective_param_size_mb']:.2f} MB  "
          f"(non-zero params * 4B; only realized after channel surgery)")
    print(f"  Buffer size     : {size_stats['buffer_size_mb']:.2f} MB")
    print(f"  Model size      : {size_stats['model_size_mb']:.2f} MB")
    print(f"  state_dict raw  : {size_stats['state_dict_raw_mb']:.2f} MB")
    print(f"  state_dict gzip : {size_stats['state_dict_gzip_mb']:.2f} MB")
    if size_stats["param_sparsity"] > 0.01 and \
       abs(size_stats["effective_param_size_mb"] - size_stats["param_size_mb"]) > 0.01:
        print("  NOTE: this checkpoint contains zeroed channels but full-shape")
        print("        tensors. In-memory size and forward latency will not")
        print("        decrease until Conv2d/BN/Linear are physically resized")
        print("        (e.g. via torch-pruning DepGraph).")
    print(f"\n── CPU Latency (bs={cpu_stats['batch_size']}, runs={cpu_stats['runs']}) ──")
    for k in ["mean_ms","min_ms","p50_ms","p95_ms","max_ms"]:
        print(f"  {k:<9}: {cpu_stats[k]:.2f} ms")
    print(f"\n── MPS Latency (Apple Silicon) ──")
    if "error" in mps_stats: print(f"  {mps_stats['error']}")
    else:
        for k in ["mean_ms","min_ms","p50_ms","p95_ms","max_ms"]:
            print(f"  {k:<9}: {mps_stats[k]:.2f} ms")
    print(f"\n{sep}\n")

def log_to_mlflow(run_name, ckpt_path, size_stats, cpu_stats, mps_stats, batch_size, warmup, runs):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model":"EfficientNet-B0","ckpt_path":ckpt_path,
                           "benchmark_batch":batch_size,"benchmark_warmup":warmup,"benchmark_runs":runs})
        mlflow.log_metrics({
            "model/total_params":     size_stats["total_params"],
            "model/trainable_params": size_stats["trainable_params"],
            "model/nonzero_params":   size_stats["nonzero_params"],
            "model/param_sparsity":   round(size_stats["param_sparsity"],4),
            "model/param_size_mb":    round(size_stats["param_size_mb"],3),
            "model/effective_param_size_mb": round(size_stats["effective_param_size_mb"],3),
            "model/buffer_size_mb":   round(size_stats["buffer_size_mb"],3),
            "model/total_size_mb":    round(size_stats["model_size_mb"],3),
            "model/state_dict_raw_mb":  round(size_stats["state_dict_raw_mb"],3),
            "model/state_dict_gzip_mb": round(size_stats["state_dict_gzip_mb"],3),
            "ckpt/file_size_mb":      round(get_checkpoint_file_size(ckpt_path),2),
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
    parser.add_argument("--ckpt",       type=str, required=True)
    parser.add_argument("--run_name",   type=str, default="benchmark_base")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup",     type=int, default=20)
    parser.add_argument("--runs",       type=int, default=100)
    args = parser.parse_args()

    input_size = (args.batch_size, 3, 224, 224)
    model      = load_model(args.ckpt)
    size_stats = get_model_size(model)
    cpu_stats  = benchmark_cpu(model, input_size, args.warmup, args.runs)
    mps_stats  = benchmark_mps(model, input_size, args.warmup, args.runs)

    print_report(args.ckpt, size_stats, cpu_stats, mps_stats)
    log_to_mlflow(args.run_name, args.ckpt, size_stats, cpu_stats, mps_stats,
                  args.batch_size, args.warmup, args.runs)

if __name__ == "__main__":
    main()

# python3.10 benchmark.py --ckpt ./logs/checkpoints/checkpoint_pruned_l1_30%_epoch=00_val_acc=0.9951.ckpt --run_name benckmark_9951 --batch_size 1

# python3.10 benchmark.py --ckpt ./logs/checkpoints/checkpoint_pruned_l1_30%_epoch=04_val_acc=0.9992.ckpt --run_name benckmark_9992 --batch_size 1

# python3.10 benchmark.py --ckpt ./logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt --run_name benckmark_9764 --batch_size 1