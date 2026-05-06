# pruning_physical_channel_flower-2.py
#
# Physical channel pruning for EfficientNet-B0 trained on Flower-102.
# Uses Torch-Pruning (DepGraph + MagnitudePruner) to remove low-norm
# channels in-place, then recalibrates BatchNorm stats and optionally
# runs a short recovery fine-tune.
#
# Pipeline:
#   STAGE 1 — load trained Lightning checkpoint, extract nn.Module
#   STAGE 2 — prune channels in-place with DepGraph (weights preserved)
#   STAGE 3 — recalibrate BN running stats (momentum=1.0, ~20 batches)
#   STAGE 4 — optional recovery fine-tune (continues from pruned weights)
#   STAGE 5 — save pruned nn.Module (.pth) + metadata (.json)
#
# Usage:
#   python3.10 pruning_physical_channel_flower-2.py \
#       --ckpt logs/checkpoints/best.ckpt --pruning_ratio 0.3
#
#   python3.10 pruning_physical_channel_flower-2.py \
#       --ckpt logs/checkpoints/best.ckpt --pruning_ratio 0.3 --finetune_epochs 5

import argparse
import functools
import json
import os

import torch
import torch.nn as nn
import torch_pruning as tp

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pruning"


# ----------------------------------------------------------------------
# STAGE 1 — load checkpoint
# ----------------------------------------------------------------------

def load_model(ckpt_path: str, image_size: int) -> tuple[nn.Module, torch.Tensor]:
    """Load trained weights from a Lightning checkpoint and return the
    bare nn.Module together with a dummy input tensor for tracing."""
    from base_flower import FlowerLightModule
    torch.serialization.add_safe_globals([functools.partial])
    pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    pl_model.eval()
    model = pl_model.model          # torchvision EfficientNet-B0 with 102-class head
    dummy = torch.randn(1, 3, image_size, image_size)
    return model, dummy


# ----------------------------------------------------------------------
# STAGE 2 — prune
# ----------------------------------------------------------------------

def prune(model: nn.Module, dummy: torch.Tensor,
          ratio: float, round_to: int = 8) -> dict:
    """Physically remove low-magnitude channels in-place via DepGraph.
    Surviving channel weights are unchanged. Returns a stats dict."""

    # Do not prune the final classifier or the stem conv.
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

    pruner = tp.pruner.MagnitudePruner(
        model, dummy,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio,
        global_pruning=True,
        round_to=round_to,
        ignored_layers=ignored,
    )
    pruner.step()

    macs_after, params_after = _count(model)
    return {
        "params_before": params_before, "params_after": params_after,
        "macs_before": macs_before,     "macs_after": macs_after,
        "param_reduction": 1 - params_after / params_before,
        "macs_reduction":  1 - macs_after  / macs_before,
    }


# ----------------------------------------------------------------------
# STAGE 3 — BN recalibration
# ----------------------------------------------------------------------

def calibrate_bn(model: nn.Module, datamodule, n_batches: int = 20) -> None:
    """Refresh BN running_mean / running_var after pruning.

    After DepGraph resizes BN tensors the running stats still reflect the
    OLD (larger) channel distributions, causing random-guess accuracy in
    eval() mode. Fix: run a few forward passes in train() mode with
    momentum=1.0 so each batch fully replaces the stale stats.
    20 batches is enough; original momentum values are restored afterwards.
    """
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    device = next(model.parameters()).device

    # Back up momentum and set to 1.0 for instant overwrite.
    backup = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            backup[name] = m.momentum
            m.momentum = 1.0

    model.train()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            model(batch[0].to(device))

    # Restore original momentum.
    for name, m in model.named_modules():
        if name in backup:
            m.momentum = backup[name]

    model.eval()
    print(f"  BN recalibrated over {min(n_batches, i + 1)} batches (momentum=1.0).")


def quick_validate(model: nn.Module, datamodule) -> float:
    """Run a quick CPU validation loop without Lightning overhead.
    Returns val_acc as a plain float."""
    datamodule.setup(stage="fit")
    loader = datamodule.val_dataloader()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total else 0.0
    print(f"  [Quick Val] val_acc = {acc:.4f}  ({correct}/{total})")
    return acc


# ----------------------------------------------------------------------
# STAGE 4 — fine-tune
# ----------------------------------------------------------------------

def finetune(model: nn.Module, epochs: int, lr: float,
             pruning_ratio: float) -> None:
    """Recovery fine-tune: continue training the pruned weights (not from scratch).
    Modifies `model` in-place via Lightning. Returns nothing."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
    from torchmetrics import Accuracy
    from base_flower import FlowerDataModule

    class _Module(pl.LightningModule):
        """Minimal LightningModule that wraps the already-pruned nn.Module."""
        def __init__(self):
            super().__init__()
            self.model = model          # same object — no re-init
            self.loss_fn = nn.CrossEntropyLoss()
            self.train_acc = Accuracy(task="multiclass", num_classes=102)
            self.val_acc   = Accuracy(task="multiclass", num_classes=102)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, _):
            x, y = batch[0], batch[1]
            loss = self.loss_fn(self(x), y)
            self.train_acc(self(x), y)
            self.log("train_loss", loss,           on_epoch=True, prog_bar=True)
            self.log("train_acc",  self.train_acc,  on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, _):
            x, y = batch[0], batch[1]
            loss = self.loss_fn(self(x), y)
            self.val_acc(self(x), y)
            self.log("val_loss", loss,          on_epoch=True, prog_bar=True)
            self.log("val_acc",  self.val_acc,   on_epoch=True, prog_bar=True)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

    pl_module  = _Module()
    datamodule = FlowerDataModule()

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto", devices=1,
        precision="32",
        logger=MLFlowLogger(experiment_name=EXPERIMENT_NAME,
                            tracking_uri=TRACKING_URI,
                            run_name=f"pruned_{pruning_ratio:.0%}_ft{epochs}ep"),
        callbacks=[
            ModelCheckpoint(dirpath="./logs/checkpoints",
                            monitor="val_acc", mode="max", save_top_k=1,
                            filename=f"pruned_{pruning_ratio:.0%}_" + "{epoch:02d}_{val_acc:.4f}"),
            LearningRateMonitor("epoch"),
        ],
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(pl_module, datamodule=datamodule)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def run(ckpt_path, pruning_ratio, finetune_epochs, lr,
        output_dir, round_to, image_size, bn_batches):
    os.makedirs(output_dir, exist_ok=True)
    from base_flower import FlowerDataModule

    # STAGE 1 ── load
    print("\n[STAGE 1] Loading checkpoint")
    model, dummy = load_model(ckpt_path, image_size)
    print(f"  params before pruning: {sum(p.numel() for p in model.parameters()):,}")

    # STAGE 2 ── prune
    print("\n[STAGE 2] Pruning")
    stats = prune(model, dummy, ratio=pruning_ratio, round_to=round_to)
    print(f"  params : {stats['params_before']:,} -> {stats['params_after']:,}"
          f"  ({stats['param_reduction']*100:.1f}% reduction)")
    print(f"  MACs   : {stats['macs_before']:,} -> {stats['macs_after']:,}"
          f"  ({stats['macs_reduction']*100:.1f}% reduction)")

    # STAGE 3 ── BN recalibration + quick validation
    print("\n[STAGE 3] BN recalibration")
    dm = FlowerDataModule()
    if bn_batches > 0:
        calibrate_bn(model, dm, n_batches=bn_batches)
    else:
        print("  Skipped (--bn_batches=0). Accuracy will be unreliable.")
    print("\n[STAGE 3] Post-calibration validation")
    quick_validate(model, dm)

    # STAGE 4 ── fine-tune (optional)
    if finetune_epochs > 0:
        print(f"\n[STAGE 4] Fine-tuning for {finetune_epochs} epochs")
        finetune(model, finetune_epochs, lr, pruning_ratio)
        print("\n[STAGE 4] Post-finetune validation")
        quick_validate(model, dm)
    else:
        print("\n[STAGE 4] Skipped (--finetune_epochs=0)")

    # STAGE 5 ── save
    print("\n[STAGE 5] Saving")
    pct = int(round(pruning_ratio * 100))
    model_path = os.path.join(output_dir, f"efficientnet_b0_pruned_{pct}.pth")
    meta_path  = os.path.join(output_dir, f"efficientnet_b0_pruned_{pct}.json")

    torch.save(model.to("cpu").eval(), model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "base_ckpt": os.path.abspath(ckpt_path),
            "pruning_ratio": pruning_ratio,
            "round_to": round_to,
            "bn_calibration_batches": bn_batches,
            "finetune_epochs": finetune_epochs,
            "lr": lr,
            **stats,
        }, f, indent=2)

    print(f"  Saved: {model_path}")
    print(f"  Reload: model = torch.load('{model_path}', weights_only=False)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",            required=True)
    p.add_argument("--pruning_ratio",   type=float, default=0.3)
    p.add_argument("--finetune_epochs", type=int,   default=0)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--output_dir",      default="./logs/pruned")
    p.add_argument("--round_to",        type=int,   default=8)
    p.add_argument("--image_size",      type=int,   default=224)
    p.add_argument("--bn_batches",      type=int,   default=20,
                   help="BN recalibration batches (momentum=1.0). 20 is enough.")
    args = p.parse_args()
    run(args.ckpt, args.pruning_ratio, args.finetune_epochs, args.lr,
        args.output_dir, args.round_to, args.image_size, args.bn_batches)


if __name__ == "__main__":
    main()
