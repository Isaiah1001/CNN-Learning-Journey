# pruning_l1structured_flower.py
# Post-training L1 structured pruning + fine-tune recovery
# Usage:
#   python3.10 pruning_l1structured_flower.py \
#     --ckpt logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt \
#     --sparsity 0.3 --finetune_epochs 5

import argparse
import functools
import torch
import torch.nn.utils.prune as prune
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from base_flower import FlowerLightModule, FlowerDataModule

TRACKING_URI    = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pruning"


# ==============================================
# 1) Weight-level sparsity (same as unstructured)
# ==============================================

def get_sparsity(model: torch.nn.Module) -> dict:
    """Compute the fraction of zero weights in Conv2d and Linear layers."""
    total_weights = 0
    zero_weights  = 0
    layer_stats   = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            w = module.weight.data
            n_total = w.numel()
            n_zero  = (w == 0).sum().item()
            total_weights += n_total
            zero_weights  += n_zero
            layer_stats[name] = {
                "total":    n_total,
                "zeros":    n_zero,
                "sparsity": n_zero / n_total,
            }

    global_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0
    return {"global_sparsity": global_sparsity, "layers": layer_stats}


def print_sparsity_report(stats: dict) -> None:
    print(f"\n── Weight Sparsity Report ──")
    print(f"  Global sparsity : {stats['global_sparsity']:.1%}")
    print(f"\n  Top sparse layers:")
    sorted_layers = sorted(
        [(n, s) for n, s in stats["layers"].items() if s["sparsity"] > 0],
        key=lambda x: x[1]["sparsity"], reverse=True
    )
    for name, s in sorted_layers[:10]:
        print(f"  {name:<50} {s['sparsity']:.1%}  ({s['zeros']:,}/{s['total']:,})")


# ==============================================
# 1b) Filter-level sparsity (important for structured)
# ==============================================

def get_filter_sparsity(model: torch.nn.Module) -> dict:
    """
    Count zeroed output channels (filters) per Conv2d layer.
    A filter is zeroed if its L1 norm across all weights == 0.
    """
    layer_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            w = module.weight.data           # [out_ch, in_ch, kH, kW]
            total_filters  = w.shape[0]
            filter_norms   = w.abs().sum(dim=[1, 2, 3])
            zeroed_filters = (filter_norms == 0).sum().item()
            layer_stats[name] = {
                "total_filters":   total_filters,
                "zeroed_filters":  zeroed_filters,
                "filter_sparsity": zeroed_filters / total_filters if total_filters > 0 else 0.0,
            }
    return layer_stats


def print_filter_report(filter_stats: dict) -> None:
    total  = sum(v["total_filters"]  for v in filter_stats.values())
    zeroed = sum(v["zeroed_filters"] for v in filter_stats.values())
    if total == 0:
        print("\n── Filter Sparsity Report ──")
        print("  No Conv2d layers found.")
        return

    print(f"\n── Filter Sparsity Report ──")
    print(f"  Zeroed filters  : {zeroed} / {total}  ({zeroed/total:.1%} global)")
    print(f"\n  Layers with zeroed filters:")
    for name, s in filter_stats.items():
        if s["zeroed_filters"] > 0:
            print(
                f"  {name:<50} {s['filter_sparsity']:.1%}"
                f"  ({s['zeroed_filters']}/{s['total_filters']} filters)"
            )


# ==============================================
# 2) L1 Structured Pruning (per-layer)
# ==============================================

def apply_pruning(model: torch.nn.Module, sparsity: float) -> torch.nn.Module:
    """
    Apply per-layer L1 structured pruning to all backbone Conv2d layers.

    - Per-layer: each Conv2d is pruned independently to the same sparsity ratio;
                 PyTorch provides no global_structured helper.
    - L1:        filters are ranked by L1 norm; lowest-norm filters are zeroed.
    - dim=0:     pruning axis is the output channel (filter) dimension.

    IMPORTANT: tensor shapes do NOT change. Zeros are applied via a mask.
               Physical size reduction requires model surgery (rebuilding Conv2d
               with fewer out_channels) -- not implemented here.
    """
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and "classifier" not in name:
            n_filters  = module.weight.shape[0]
            n_to_prune = max(1, int(n_filters * sparsity))
            if n_to_prune >= n_filters:
                print(
                    f"  Skipping {name}: only {n_filters} filters, "
                    f"cannot prune {sparsity:.0%}"
                )
                continue

            prune.ln_structured(
                module,
                name="weight",
                amount=sparsity,
                n=1,    # L1 norm
                dim=0,  # prune along output channel dimension
            )
            pruned_count += 1

    print(f"\n  Applied L1 structured pruning to {pruned_count} Conv2d layers")
    print(f"  Target filter sparsity per layer: {sparsity:.1%}")
    return model


def make_pruning_permanent(model: torch.nn.Module) -> torch.nn.Module:
    """
    Remove the pruning masks (weight_mask, weight_orig) and write the
    zeroed weights directly into the weight buffer.

    Must be called BEFORE fine-tuning; otherwise the optimizer will
    continue updating the pruned (zero) weights during recovery training.

    Note: tensor shapes remain unchanged after make_permanent.
    Zero filters are permanently baked in but still occupy float32 storage.
    Physical size reduction requires rebuilding Conv2d with fewer out_channels
    (model surgery) -- not implemented here.
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")

    print("  Pruning masks removed -- zeroed filters written permanently into weight buffers.")
    return model


# ==============================================
# 3) Fine-tune recovery (via Lightning Trainer)
# ==============================================

def finetune_after_pruning(ckpt_path: str,
                            sparsity: float,
                            finetune_epochs: int,
                            lr: float) -> None:
    """
    Full pipeline:
      load checkpoint -> apply pruning -> make permanent -> fine-tune -> save.
    """

    # ── Load checkpoint ───────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    torch.serialization.add_safe_globals([functools.partial])
    pl_model   = FlowerLightModule.load_from_checkpoint(ckpt_path)
    datamodule = FlowerDataModule()

    # ── Apply pruning ─────────────────────────────────────────
    print("\n[Step 1] Applying L1 structured pruning...")
    pl_model.model = apply_pruning(pl_model.model, sparsity)

    before_weight_stats  = get_sparsity(pl_model.model)
    before_filter_stats  = get_filter_sparsity(pl_model.model)
    print_sparsity_report(before_weight_stats)
    print_filter_report(before_filter_stats)

    # ── Make pruning permanent ────────────────────────────────
    print("\n[Step 2] Making pruning permanent...")
    pl_model.model = make_pruning_permanent(pl_model.model)

    after_weight_stats = get_sparsity(pl_model.model)
    print(f"\n  Weight sparsity after make_permanent: "
          f"{after_weight_stats['global_sparsity']:.1%}")

    # ── Patch optimizer with a lower lr for recovery fine-tune ─
    original_optimizer = pl_model.optimizer
    pl_model.optimizer = lambda params: original_optimizer(params, lr=lr)

    # ── MLflow logger ─────────────────────────────────────────
    run_name = f"pruned_l1structured_{sparsity:.0%}_ft{finetune_epochs}ep"
    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
        run_name=run_name,
    )

    # Aggregate filter stats for logging
    total_filters  = sum(v["total_filters"]  for v in before_filter_stats.values())
    zeroed_filters = sum(v["zeroed_filters"] for v in before_filter_stats.values())
    filter_sparsity_global = zeroed_filters / total_filters if total_filters > 0 else 0.0

    # ── Callbacks ─────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath="./logs/checkpoints",
        monitor="val_acc",
        filename=f"checkpoint_pruned_l1structured_{sparsity:.0%}" + "_{epoch:02d}_{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ───────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=finetune_epochs,
        accelerator="gpu",      # Apple Silicon MPS; change to "mps" if needed
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    # ── Log pruning hyperparameters into the MLflow run ───────
    logger.log_hyperparams({
        "pruning_method":            "l1_structured_per_layer",
        "pruning_sparsity_target":    sparsity,
        "pruning_dim":               "0 (output channels / filters)",
        "pruning_norm":              "L1",
        "pruned_layers":             "Conv2d only (backbone, classifier excluded)",
        "finetune_epochs":            finetune_epochs,
        "finetune_lr":                lr,
        "base_ckpt":                  ckpt_path,
        "weight_sparsity_before":    round(before_weight_stats["global_sparsity"], 4),
        "weight_sparsity_after":     round(after_weight_stats["global_sparsity"], 4),
        "filter_sparsity_before":    round(filter_sparsity_global, 4),
        "zeroed_filters":             zeroed_filters,
        "total_filters":              total_filters,
    })

    # ── Fine-tune ─────────────────────────────────────────────
    print(f"\n[Step 3] Fine-tuning for {finetune_epochs} epochs (lr={lr})...")
    trainer.fit(pl_model, datamodule=datamodule)

    print(f"\nDone.")
    print(f"  Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"  MLflow run      : '{run_name}' in experiment '{EXPERIMENT_NAME}'")
    print(f"\nNext step:")
    print(f"  python3.10 benchmark.py --ckpt {checkpoint_cb.best_model_path} "
          f"--run_name benchmark_pruned_l1structured_{sparsity:.0%}")


# ==============================================
# 4) Main
# ==============================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-training L1 structured pruning + fine-tune recovery for EfficientNet-B0 Flower"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to base Lightning checkpoint (.ckpt)")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.3,
        help="Per-layer filter sparsity, e.g. 0.3 = remove 30% of filters in each Conv2d",
    )
    parser.add_argument("--finetune_epochs", type=int, default=5,
                        help="Number of epochs to fine-tune after pruning")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine-tune recovery (should be << training lr)")
    args = parser.parse_args()

    finetune_after_pruning(
        ckpt_path       = args.ckpt,
        sparsity        = args.sparsity,
        finetune_epochs = args.finetune_epochs,
        lr              = args.lr,
    )


if __name__ == "__main__":
    main()


# ── Usage examples ────────────────────────────────────────────────────────────
#
# 30% filter sparsity per layer, 5-epoch fine-tune:
#   python3.10 pruning_l1structured_flower.py \
#     --ckpt logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt \
#     --sparsity 0.3 --finetune_epochs 5
#
# 50% filter sparsity per layer, 10-epoch fine-tune:
#   python3.10 pruning_l1structured_flower.py \
#     --ckpt logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt \
#     --sparsity 0.5 --finetune_epochs 10
#
# Then benchmark:
#   python3.10 benchmark.py \
#     --ckpt logs/checkpoints/checkpoint_pruned_l1structured_30%_xx.ckpt \
#     --run_name benchmark_pruned_l1structured_0.3
