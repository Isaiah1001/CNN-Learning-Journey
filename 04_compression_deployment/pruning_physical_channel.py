# pruning_physical_channel.py
# Physical channel pruning for EfficientNet-B0 trained on Flower-102.
# Uses Torch-Pruning (DepGraph + MagnitudePruner) to remove low-norm
# channels in-place, then recalibrates BatchNorm stats and optionally
# runs a short recovery fine-tune.
#
# Usage:
# python3.10 pruning_physical_channel.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt --pruning_ratio 0.15 --global_pruning --finetune_epochs 10

# Pipeline:
#   STAGE 1 — load trained Lightning checkpoint, extract nn.Module
#   STAGE 2 — prune channels in-place with DepGraph (weights preserved)
#             Per-layer (default): each layer independently drops
#             `pruning_ratio` fraction of its own channels.
#             Pass --global_pruning to use a single global ranking instead.
#   STAGE 3 — recalibrate BN running stats (momentum=1.0, ~20 batches)
#             + quick validation
#   STAGE 4 — optional recovery fine-tune (continues from pruned weights)
#   STAGE 5 — save pruned nn.Module (.pth) + metadata (.json)
#

import argparse
import functools
import json
import os

import torch
import torch.nn as nn
import torch_pruning as tp
import lightning.pytorch as pl
from base_flower import FlowerLightModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from base_flower import FlowerDataModule

TRACKING_URI    = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pruning"

# ----------------------------------------------------------------------
# STAGE 2 — Prune
# ----------------------------------------------------------------------

def prune_model(model: nn.Module, dummy: torch.Tensor,
                ratio: float, round_to: int = 1,
                global_pruning: bool = False) -> dict:
    """Physically remove low-magnitude channels in-place via DepGraph.

    Returns a stats dict with params/MACs before and after.
    """
    ignored = []
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            ignored.append(m)
    for m in model.features[0].modules():
        if isinstance(m, nn.Conv2d):
            ignored.append(m)

    def _count(m):
        m.eval()
        macs, params = tp.utils.count_ops_and_params(m, dummy)
        return int(macs), int(params)

    macs_before, params_before = _count(model)

    mode_str = "global" if global_pruning else "per-layer"
    print(f"   Pruning mode    : {mode_str}")
    print(f"   Target ratio    : {ratio:.1%}  (round_to={round_to})")

    pruner = tp.pruner.MagnitudePruner(
        model, dummy,
        importance=tp.importance.MagnitudeImportance(p=1),  # L1 norm
        pruning_ratio=ratio,
        global_pruning=global_pruning,
        round_to=round_to,
        ignored_layers=ignored,
    )
    pruner.step()

    macs_after, params_after = _count(model)
    return {
        "params_before":   params_before,
        "params_after":    params_after,
        "macs_before":     macs_before,
        "macs_after":      macs_after,
        "param_reduction": 1 - params_after  / params_before,
        "macs_reduction":  1 - macs_after    / macs_before,
    }

# ----------------------------------------------------------------------
# STAGE 3 — BN recalibration + quick validation
# ----------------------------------------------------------------------

def calibrate_bn(model: nn.Module, datamodule, n_batches: int = 20) -> None:
    """Refresh BN running_mean / running_var after pruning.

    After DepGraph resizes BN tensors the running stats still reflect the
    OLD (larger) channel distributions. Fix: run forward passes in train()
    mode with momentum=1.0 so each batch fully replaces the stale stats.
    Original momentum values are restored afterwards.
    """
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    device = next(model.parameters()).device

    backup = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            backup[name] = m.momentum
            m.momentum   = 1.0

    model.train()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            model(batch[0].to(device))

    for name, m in model.named_modules():
        if name in backup:
            m.momentum = backup[name]

    model.eval()
    print(f"   BN recalibrated over {min(n_batches, i + 1)} batches (momentum=1.0).")


def quick_validate(model: nn.Module, datamodule) -> float:
    """Simple validation loop — no Lightning overhead. Returns val_acc."""
    datamodule.setup(stage="fit")
    loader  = datamodule.val_dataloader()
    model.eval()
    device  = next(model.parameters()).device
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs   = batch[0].to(device)
            labels = batch[1].to(device)
            preds  = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total if total else 0.0
    print(f"   [Quick Val] val_acc = {acc:.4f} ({correct}/{total})")
    return acc

# ----------------------------------------------------------------------
# STAGE 4 — Fine-tune recovery
# ----------------------------------------------------------------------

def finetune(pl_model, epochs: int, lr: float,
             pruning_ratio: float) -> str:
    """Recovery fine-tune: continue training the pruned weights. Returns best ckpt path."""

    original_optimizer = pl_model.optimizer
    pl_model.optimizer = lambda params: original_optimizer(params, lr=lr)
    
    datamodule = FlowerDataModule()
    run_name   = f"pruned_physical_{pruning_ratio:.0%}_ft{epochs}ep"
    logger = MLFlowLogger(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=TRACKING_URI,
            run_name=run_name,
        )
    checkpoint_cb = ModelCheckpoint(
        dirpath="./logs/checkpoints",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"pruned_physical_{pruning_ratio:.0%}_" + "{epoch:02d}_{val_acc:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    pl_model.model.train()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(pl_model, datamodule=datamodule)
    return checkpoint_cb.best_model_path

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def run(ckpt_path: str, pruning_ratio: float, finetune_epochs: int, lr: float,
        output_dir: str, round_to: int, image_size: int,
        bn_batches: int, global_pruning: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    from base_flower import FlowerDataModule

    # ── STAGE 1: Load ──────────────────────────────────────────────────
    dummy = torch.randn(1, 3, image_size, image_size)
    print(f"\n[STAGE 1] Loading checkpoint: {ckpt_path}")
    pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    params_before = sum(p.numel() for p in pl_model.model.parameters())
    print(f"   params before pruning: {params_before:,}")
    dm = FlowerDataModule()
    # print("\n[STAGE 1.1] pre-pruning validation")
    # quick_validate(pl_model.model, dm)

    # ── STAGE 2: Prune ─────────────────────────────────────────────────
    print(f"\n[STAGE 2] Applying physical channel pruning (ratio={pruning_ratio:.1%})")
    stats = prune_model(pl_model.model, dummy, ratio=pruning_ratio,
                        round_to=round_to, global_pruning=global_pruning)
    print(f"   params : {stats['params_before']:,} -> {stats['params_after']:,}"
          f"  ({stats['param_reduction'] * 100:.1f}% reduction)")
    print(f"   MACs   : {stats['macs_before']:,} -> {stats['macs_after']:,}"
          f"  ({stats['macs_reduction'] * 100:.1f}% reduction)")

    # ── STAGE 3: BN recalibration + quick validation ───────────────────
    print("\n[STAGE 3] BN recalibration")
    if bn_batches > 0:
        calibrate_bn(pl_model.model, dm, n_batches=bn_batches)
    else:
        print("   Skipped (--bn_batches=0). Accuracy will be unreliable.")
    # print("\n[STAGE 3.1] Post-pruning validation")
    # quick_validate(pl_model.model, dm)

    # ── STAGE 4: Fine-tune (optional) ─────────────────────────────────
    if finetune_epochs > 0:
        print(f"\n[STAGE 4] Fine-tuning for {finetune_epochs} epochs (lr={lr})")
        best_ckpt = finetune(pl_model, finetune_epochs, lr, pruning_ratio)
        print("\n[STAGE 4] Post-finetune validation")
        quick_validate(pl_model.model, dm)
        print(f"   Best checkpoint : {best_ckpt}")
    else:
        print("\n[STAGE 4] Skipped (--finetune_epochs=0)")

    # ── STAGE 5: Save ─────────────────────────────────────────────────
    print("\n[STAGE 5] Saving")
    pct        = int(round(pruning_ratio * 100))
    model_path = os.path.join(output_dir, f"efficientnet_b0_pruned_physical_{pct}.pth")
    meta_path  = os.path.join(output_dir, f"efficientnet_b0_pruned_physical_{pct}.json")

    torch.save(pl_model.model.to("cpu").eval(), model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "base_ckpt":        os.path.abspath(ckpt_path),
            "pruning_method":   "physical_channel_magnitude",
            "pruning_mode":     "global" if global_pruning else "per-layer",
            "pruning_ratio":    pruning_ratio,
            "round_to":         round_to,
            "bn_calib_batches": bn_batches,
            "finetune_epochs":  finetune_epochs,
            "lr":               lr,
            **stats,
        }, f, indent=2)

    print(f"   Saved  : {model_path}")
    print(f"   Reload : model = torch.load('{model_path}', weights_only=False)")
    print(f"\n   Next step:")
    print(f"   python3.10 benchmark.py --ckpt {model_path} "
          f"--run_name benchmark_pruned_physical_{pruning_ratio:.0%}")


def main():
    pl.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True)

    p = argparse.ArgumentParser(
        description="Physical channel pruning + fine-tune recovery for EfficientNet-B0 Flower"
    )
    p.add_argument("--ckpt",            type=str,   required=True,
                   help="Path to base Lightning checkpoint (.ckpt)")
    p.add_argument("--pruning_ratio",   type=float, default=0.3,
                   help="Fraction of channels to prune per layer (or globally)")
    p.add_argument("--finetune_epochs", type=int,   default=0,
                   help="Number of epochs to fine-tune after pruning (0 = skip)")
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Learning rate for fine-tune recovery")
    p.add_argument("--output_dir",      type=str,   default="./logs/pruned",
                   help="Directory to save pruned model and metadata")
    p.add_argument("--round_to",        type=int,   default=1,
                   help="Round pruned channel counts up to nearest multiple "
                        "(set 8 for hardware alignment, 1 to disable)")
    p.add_argument("--image_size",      type=int,   default=224,
                   help="Input image size for DepGraph tracing")
    p.add_argument("--bn_batches",      type=int,   default=20,
                   help="BN recalibration batches (momentum=1.0). 20 is enough.")
    p.add_argument("--global_pruning",  action="store_true",
                   help="Use global channel ranking instead of per-layer pruning")
    args = p.parse_args()

    run(
        ckpt_path       = args.ckpt,
        pruning_ratio   = args.pruning_ratio,
        finetune_epochs = args.finetune_epochs,
        lr              = args.lr,
        output_dir      = args.output_dir,
        round_to        = args.round_to,
        image_size      = args.image_size,
        bn_batches      = args.bn_batches,
        global_pruning  = args.global_pruning,
    )


if __name__ == "__main__":
    main()

