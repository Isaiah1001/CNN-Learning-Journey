# pruning_l1unstructured_flower.py
# Post-training L1 unstructured pruning + fine-tune recovery
# Usage: python3.10 pruning_l1unstructured_flower.py --ckpt logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt
#                                  --sparsity 0.3 --finetune_epochs 5

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
# 1) Compute sparsity (used to verify pruning results)
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
    print(f"\n── Sparsity Report ──")
    print(f"  Global sparsity : {stats['global_sparsity']:.1%}")
    print(f"\n  Top sparse layers:")
    # Print only layers with sparsity > 0, sorted by descending sparsity
    sorted_layers = sorted(
        [(n, s) for n, s in stats["layers"].items() if s["sparsity"] > 0],
        key=lambda x: x[1]["sparsity"], reverse=True
    )
    for name, s in sorted_layers[:10]:
        print(f"  {name:<50} {s['sparsity']:.1%}  ({s['zeros']:,}/{s['total']:,})")


# ==============================================
# 2) L1 Unstructured Pruning
# ==============================================

def apply_pruning(model: torch.nn.Module, sparsity: float) -> torch.nn.Module:
    """
    Apply global L1 unstructured pruning across all Conv2d weight tensors.

    - Global:       all layers share a single magnitude threshold rather than
                    each layer being pruned independently to the same ratio.
    - L1:           weights are ranked by absolute value; the smallest are removed.
    - Unstructured: individual weight positions are zeroed (soft pruning);
                    tensor shapes are unchanged, so this does NOT reduce parameter
                    count or FLOPs on dense hardware -- combine with quantization
                    for a real size reduction.
    - The classifier head is intentionally left unpruned to protect accuracy.
    """
    # Only prune backbone Conv2d layers; leave the classifier head untouched
    parameters_to_prune = [
        (module, "weight")
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
        and "classifier" not in name
    ]

    print(f"\n  Applying global L1 unstructured pruning to {len(parameters_to_prune)} Conv2d layers")
    print(f"  Target sparsity: {sparsity:.1%}")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    return model


def make_pruning_permanent(model: torch.nn.Module) -> torch.nn.Module:
    """
    Remove the pruning masks (weight_mask, weight_orig) and write the
    zeroed weights directly into the weight buffer.

    Must be called BEFORE fine-tuning; otherwise the optimizer will
    continue updating the pruned (zero) weights during recovery training.

    Note: model size does NOT shrink after make_permanent -- zero weights
    still occupy float32 storage. A real size reduction requires either
    structured pruning or INT8 quantization.
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")

    print("  Pruning masks removed -- zero weights written permanently into weight buffers.")
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

    Uses a plain Trainer (not LightningCLI) for easier script-level control.
    """

    # ── Load checkpoint ───────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    torch.serialization.add_safe_globals([functools.partial])
    pl_model   = FlowerLightModule.load_from_checkpoint(ckpt_path)
    datamodule = FlowerDataModule()

    # ── Apply pruning ─────────────────────────────────────────
    print("\n[Step 1] Applying L1 unstructured pruning...")
    pl_model.model = apply_pruning(pl_model.model, sparsity)

    before_stats = get_sparsity(pl_model.model)
    print_sparsity_report(before_stats)

    # ── Make pruning permanent ────────────────────────────────
    print("\n[Step 2] Making pruning permanent...")
    pl_model.model = make_pruning_permanent(pl_model.model)

    after_stats = get_sparsity(pl_model.model)
    print(f"  Sparsity after make_permanent: {after_stats['global_sparsity']:.1%}")

    # ── Patch optimizer with a lower lr for recovery fine-tune ─
    # Pruning shifts weights only slightly; a small lr avoids over-correction.
    original_optimizer = pl_model.optimizer
    pl_model.optimizer = lambda params: original_optimizer(params, lr=lr)

    # ── MLflow logger ─────────────────────────────────────────
    run_name = f"pruned_l1unstructured_{sparsity:.0%}_ft{finetune_epochs}ep"
    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
        run_name=run_name,
    )

    # ── Callbacks ─────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath="./logs/checkpoints",
        monitor="val_acc",
        filename=f"checkpoint_pruned_l1unstructured_{sparsity:.0%}" + "_{epoch:02d}_{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ───────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=finetune_epochs,
        accelerator="gpu",      # Apple Silicon MPS
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0, # skip sanity check -- metric not yet updated at start
    )

    # ── Log pruning hyperparameters into the MLflow run ───────
    logger.log_hyperparams({
        "pruning_method":           "l1_unstructured_global",
        "pruning_sparsity":          sparsity,
        "pruned_layers":            "Conv2d only (backbone, classifier excluded)",
        "finetune_epochs":           finetune_epochs,
        "finetune_lr":               lr,
        "base_ckpt":                 ckpt_path,
        "sparsity_before_finetune":  round(before_stats["global_sparsity"], 4),
    })

    # ── Fine-tune ─────────────────────────────────────────────
    print(f"\n[Step 3] Fine-tuning for {finetune_epochs} epochs (lr={lr})...")
    trainer.fit(pl_model, datamodule=datamodule)

    print(f"\nDone.")
    print(f"  Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"  MLflow run      : '{run_name}' in experiment '{EXPERIMENT_NAME}'")
    print(f"\nNext step:")
    print(f"  python3.10 benchmark.py --ckpt {checkpoint_cb.best_model_path} "
          f"--run_name benchmark_pruned_l1unstructured_{sparsity:.0%}")


# ==============================================
# 4) Main
# ==============================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-training L1 pruning + fine-tune recovery for EfficientNet-B0 Flower"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to base Lightning checkpoint (.ckpt)")
    parser.add_argument("--sparsity", type=float, default=0.3,
                        help="Global pruning sparsity, e.g. 0.3 = zero out 30%% of Conv weights")
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
# 30% sparsity, 5-epoch fine-tune:
#   python3.10 pruning_L1unstructured_flower.py --ckpt logs/checkpoints/checkpoint_base_xx.ckpt \
#                            --sparsity 0.3 --finetune_epochs 5
#
# 50% sparsity, 10-epoch fine-tune:
#   python3.10 pruning_L1unstructured_flower.py --ckpt logs/checkpoints/checkpoint_base_xx.ckpt \
#                            --sparsity 0.5 --finetune_epochs 10
#
# Then benchmark the pruned model:
#   python3.10 benchmark.py --ckpt logs/checkpoints/checkpoint_pruned_l1unstructured_30%_xx.ckpt \
#                       --run_name benchmark_pruned_l1unstructured_0.3
