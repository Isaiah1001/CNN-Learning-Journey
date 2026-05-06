# pruning_physical_channel_flower-2.py
# Post-training PHYSICAL channel pruning using Torch-Pruning (DepGraph),
# followed by a fine-tune that *continues training the pruned weights*.
#
# Pipeline (no step here ever re-initializes the backbone):
#   (1) Load the original Lightning checkpoint with its trained weights.
#   (2) Extract the underlying nn.Module (a torchvision EfficientNet-B0
#       with a 102-class head) — this object carries the trained weights.
#   (3) Run Torch-Pruning's DepGraph + MagnitudePruner on that *same* object.
#       Pruning rewrites Conv2d / BN / Linear shapes IN-PLACE; surviving
#       channels keep their trained values. The Python object identity
#       (id(model)) is preserved — there is no `model = build_new_model()`.
#   (4) [NEW] Recalibrate BatchNorm running statistics on a few training
#       batches. After channel pruning the BN running_mean / running_var
#       tensors are resized to match the new channel count, but their values
#       were computed over the OLD (larger) feature distributions. Running a
#       short calibration pass in train() mode lets BN update those stats
#       with the actual post-pruning activations, restoring accurate inference.
#   (5) Wrap the *same* pruned nn.Module into a thin LightningModule
#       (PrunedFlowerLightModule). The wrapper assigns `self.model = pruned_model`;
#       it does NOT reconstruct a fresh torchvision model. We assert
#       `pl_module.model is pruned_model` and compare a weight fingerprint
#       before/after wrapping, so any accidental re-init shows up immediately.
#   (6) Fine-tune the wrapped module. trainer.fit moves it to the accelerator
#       and continues SGD on the existing pruned weights (i.e. recovery
#       fine-tuning, not training from scratch).
#   (7) Save the whole pruned + fine-tuned nn.Module via torch.save(model, path)
#       plus a sidecar JSON with provenance.
#
# Why a different save format?
#   After DepGraph rewrites Conv/BN/Linear shapes, the resulting nn.Module no
#   longer matches FlowerLightModule.model (which builds a stock
#   torchvision.efficientnet_b0). Loading a state_dict back into the original
#   class would fail (shape mismatch) and a Lightning checkpoint produced from
#   the original class would be useless for redeployment. So we save the entire
#   pruned model object via `torch.save(model, path)` (a .pth file), and reload
#   it with `torch.load(..., weights_only=False)`.
#
# Dependency:
#   pip install torch-pruning   # imports as `torch_pruning`
#
# Usage:
#   python3.10 pruning_physical_channel_flower-2.py \
#       --ckpt logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt \
#       --pruning_ratio 0.3 \
#       --finetune_epochs 0
#
#   # With short fine-tune recovery on the pruned weights (NOT from scratch):
#   python3.10 pruning_physical_channel_flower-2.py \
#       --ckpt logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt \
#       --pruning_ratio 0.3 --finetune_epochs 5
#
# Then benchmark with:
#   python3.10 benchmark.py \
#       --model_file logs/pruned/efficientnet_b0_pruned_30.pth \
#       --run_name benchmark_pruned_physical_30

import argparse
import functools
import json
import os

import torch
import torch_pruning as tp

# Lightning / project imports happen lazily inside the functions that need them
# so this module can be imported (and the pure pruning logic exercised) without
# Lightning installed — useful for smoke tests in lean environments.

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pruning"


# ==============================================
# 1) Helpers
# ==============================================

def _macs_params(model: torch.nn.Module, example_inputs: torch.Tensor) -> tuple[int, int]:
    """Count MACs and params via torch_pruning's helper. Runs on the model's device."""
    model.eval()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    return int(macs), int(nparams)


def _format_si(x: float) -> str:
    for unit in ["", "K", "M", "G", "T"]:
        if abs(x) < 1000:
            return f"{x:.2f}{unit}"
        x /= 1000.0
    return f"{x:.2f}P"


def _weight_fingerprint(model: torch.nn.Module) -> dict:
    """Return a cheap fingerprint of a model's parameters.

    Used as a safety check that the fine-tune wrapper is holding the *same*
    trained tensor objects rather than silently re-initializing a fresh backbone.
    The fingerprint captures total parameter count, tensor count, identity of
    the first parameter tensor, and the L1 sum of the first 8 parameter tensors.
    Any of these will change instantly if a layer is swapped for a fresh init.
    """
    params = list(model.parameters())
    total_numel = sum(p.numel() for p in params)
    sample_sum = float(sum(float(p.detach().abs().sum()) for p in params[:8]))
    return {
        "n_params": total_numel,
        "n_tensors": len(params),
        "first_param_id": id(params[0]) if params else None,
        "abs_sum_first_8": round(sample_sum, 6),
    }


# ==============================================
# 2) Physical channel pruning with DepGraph
# ==============================================

def physical_prune(
    model: torch.nn.Module,
    example_inputs: torch.Tensor,
    pruning_ratio: float = 0.3,
    round_to: int = 8,
    iterative_steps: int = 1,
    global_pruning: bool = True,
) -> tuple[torch.nn.Module, dict]:
    """Physically prune output channels of a torchvision EfficientNet-B0 model.

    DepGraph automatically discovers groups of layers that must be pruned
    together (e.g. a Conv2d + its BatchNorm2d + the next Conv2d's in_channels +
    any residual / SE branches). MagnitudePruner then drops the lowest-norm
    channels in each group, and the modules are rebuilt with smaller shapes.

    NOTE — this rewrites `model` IN-PLACE. The trained weights of the surviving
    channels are kept; only low-magnitude channels are removed. The returned
    object is the *same* Python object as `model` (id is preserved).

    Args:
        model: a torchvision EfficientNet-B0 (already loaded with trained weights).
        example_inputs: a 4D tensor (e.g. 1x3x224x224) used by DepGraph to trace
            the network.
        pruning_ratio: fraction of channels to drop globally.
        round_to: round resulting channel counts up to a multiple of this value
            (8 is hardware-friendly for most accelerators). Set to None to disable.
        iterative_steps: spread pruning across this many steps. We call .step()
            once for each step; for one-shot pruning leave this at 1.
        global_pruning: True = single global ranking; False = per-layer ratio.

    Returns:
        (pruned_model, info_dict) — pruned_model is `model` itself, mutated in-place.
    """
    # Pin the final classifier (output dim = num_classes) and the EfficientNet
    # stem conv (features[0]) — both are fragile and intentionally left intact.
    ignored_layers: list[torch.nn.Module] = []
    if hasattr(model, "classifier"):
        for m in model.classifier.modules():
            if isinstance(m, torch.nn.Linear):
                ignored_layers.append(m)
    if hasattr(model, "features") and len(model.features) > 0:
        for m in model.features[0].modules():
            if isinstance(m, torch.nn.Conv2d):
                ignored_layers.append(m)

    importance = tp.importance.MagnitudeImportance(p=2)  # L2 channel magnitude

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        iterative_steps=iterative_steps,
        global_pruning=global_pruning,
        round_to=round_to,
        ignored_layers=ignored_layers,
    )

    macs_before, params_before = _macs_params(model, example_inputs)

    for _ in range(iterative_steps):
        pruner.step()  # rewrites Conv/BN/Linear shapes IN-PLACE on `model`

    macs_after, params_after = _macs_params(model, example_inputs)

    info = {
        "pruning_ratio": pruning_ratio,
        "round_to": round_to,
        "iterative_steps": iterative_steps,
        "global_pruning": global_pruning,
        "macs_before": macs_before,
        "macs_after": macs_after,
        "params_before": params_before,
        "params_after": params_after,
        "macs_reduction": 1.0 - macs_after / macs_before if macs_before else 0.0,
        "params_reduction": 1.0 - params_after / params_before if params_before else 0.0,
        "ignored_layer_count": len(ignored_layers),
    }
    return model, info


# ==============================================
# 3) BatchNorm recalibration after pruning
# ==============================================

def calibrate_bn(
    model: torch.nn.Module,
    datamodule,
    n_batches: int = 100,
) -> None:
    """Recalibrate BatchNorm running statistics after physical channel pruning.

    WHY THIS IS NECESSARY
    ---------------------
    torch_pruning rewrites Conv2d / BN / Linear shapes in-place by removing
    the lowest-magnitude channels. The BN running_mean and running_var tensors
    are resized to match the new (smaller) channel count, but their *values*
    were accumulated under the old (larger) feature distributions. Using those
    stale statistics during eval() inference causes wildly incorrect
    normalisation — observed as val_loss > 12 and val_acc ≈ 1/num_classes
    (i.e. random-guess level), even though the surviving channel weights are
    perfectly intact.

    The fix is straightforward: run a short forward-pass loop in train() mode
    so that BN accumulates fresh running_mean / running_var from the actual
    post-pruning activations, then switch back to eval() for inference.

    Args:
        model: the pruned nn.Module (in-place update of BN running stats).
        datamodule: a FlowerDataModule instance (must already be set up, or
            setup() is called here).
        n_batches: number of training mini-batches to process. 100 batches
            (≈12,800 images at batch_size=128) is enough for stable stats.
    """
    print(f"\n[BN Calibration] Recalibrating BatchNorm stats over {n_batches} batches ...")

    # setup() is idempotent in FlowerDataModule, so calling it again is safe.
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    # train() mode activates the BN momentum update for running_mean/running_var.
    # We do NOT want gradients — this is purely a statistics pass.
    model.train()
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            imgs = batch[0].to(device)  # batch = (images, labels, extras)
            model(imgs)

    model.eval()
    print(f"  Done — BN running stats recalibrated over {min(n_batches, i + 1)} batches.")


# ==============================================
# 4) Fine-tune wrapper that PRESERVES the pruned weights
# ==============================================

def _build_pruned_lightmodule(
    pruned_model: torch.nn.Module,
    num_classes: int = 102,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
):
    """Wrap an already-pruned nn.Module into a Lightning module.

    Critical contract:
      * We do NOT call FlowerLightModule.__init__ — that would build a fresh
        stock torchvision.efficientnet_b0 and overwrite self.model, discarding
        the pruned (and previously trained) weights.
      * We assign self.model = pruned_model so the Lightning module wraps the
        *same* Python object returned from physical_prune.
      * The caller verifies pl_module.model is pruned_model after construction.

    Defined inside a helper function so this file imports cleanly when
    Lightning is not installed (e.g. pure pruning smoke tests).
    """
    import lightning.pytorch as pl
    from torchmetrics import Accuracy
    from base_flower import FlowerLightModule

    class PrunedFlowerLightModule(FlowerLightModule):
        """LightningModule that fine-tunes an already-pruned nn.Module.

        The pruned model is supplied from outside; this class never builds a
        replacement backbone. Hence pruned + trained weights survive into
        trainer.fit and are updated by the optimiser (recovery fine-tuning,
        NOT training from scratch).
        """

        def __init__(
            self,
            pruned_model: torch.nn.Module,
            num_classes: int = 102,
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
        ):
            # Skip FlowerLightModule.__init__ to avoid rebuilding a stock
            # efficientnet_b0 and clobbering the pruned weights.
            pl.LightningModule.__init__(self)
            # Exclude the live nn.Module from hyperparameter pickling.
            self.save_hyperparameters(ignore=["pruned_model"])

            # Bind the SAME object. If anything replaces self.model with a
            # freshly constructed backbone, training will resume from random
            # init instead of the pruned weights — the assert below catches it.
            self.model = pruned_model

            # Store optimiser / scheduler as callables so configure_optimizers
            # can call them over self.parameters() at training start.
            self.optimizer = lambda params: torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay
            )
            self.lr_scheduler = lambda opt: torch.optim.lr_scheduler.ConstantLR(
                opt, factor=1.0, total_iters=1
            )

            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # ------------------------------------------------------------------
        # Override epoch-end hooks so Lightning (not us) owns the reset cycle.
        # torchmetrics Accuracy objects passed to self.log() are automatically
        # computed at epoch end and reset by Lightning. Adding a manual reset
        # here would cause a double-reset and corrupt the logged metric value.
        # ------------------------------------------------------------------
        def on_train_epoch_end(self):
            pass  # Lightning handles metric reset automatically

        def on_validation_epoch_end(self):
            pass  # Lightning handles metric reset automatically

    pl_module = PrunedFlowerLightModule(
        pruned_model=pruned_model,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Hard guard: the wrapper must hold the exact same pruned object we passed
    # in. If this assertion fails, something replaced the model and fine-tuning
    # would silently start from a fresh initialisation.
    assert pl_module.model is pruned_model, (
        "PrunedFlowerLightModule.model is not the same object as the pruned model. "
        "Fine-tune would discard the pruned weights — refusing to continue."
    )
    return pl_module


def finetune_pruned(
    pruned_model: torch.nn.Module,
    finetune_epochs: int,
    lr: float,
    pruning_ratio: float,
) -> torch.nn.Module:
    """Recovery fine-tune that CONTINUES training the pruned weights.

    Returns the same nn.Module that was passed in (now with updated weights).
    BN recalibration must have been run before calling this function so that
    the first validation epoch reports a meaningful accuracy (not ~1/num_classes).
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
    from base_flower import FlowerDataModule

    # Capture weight fingerprint before wrapping to verify no silent re-init.
    fp_before = _weight_fingerprint(pruned_model)
    print(f"  [finetune] weight fingerprint pre-wrap : {fp_before}")

    pl_module = _build_pruned_lightmodule(
        pruned_model, num_classes=102, lr=lr, weight_decay=1e-4,
    )

    fp_after = _weight_fingerprint(pl_module.model)
    print(f"  [finetune] weight fingerprint post-wrap: {fp_after}")
    if fp_before != fp_after:
        raise RuntimeError(
            "Pruned weights changed during Lightning wrapping — the wrapper "
            "must reuse the same nn.Module, not rebuild one."
        )
    assert pl_module.model is pruned_model, (
        "Lightning wrapper is not holding the pruned model object — aborting "
        "to avoid fine-tuning a fresh backbone from scratch."
    )

    datamodule = FlowerDataModule()

    run_name = f"pruned_physical_{pruning_ratio:.0%}_ft{finetune_epochs}ep"
    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
        run_name=run_name,
    )

    # Lightning checkpoints from the pruned model cannot be reloaded into
    # FlowerLightModule (shape mismatch after DepGraph rewrites). We save the
    # whole nn.Module separately via torch.save; this .ckpt is only for
    # training resume if the job is interrupted.
    checkpoint_cb = ModelCheckpoint(
        dirpath="./logs/checkpoints",
        monitor="val_acc",
        filename=(
            f"checkpoint_pruned_physical_{pruning_ratio:.0%}"
            + "_{epoch:02d}_{val_acc:.4f}"
        ),
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=finetune_epochs,
        accelerator="auto",
        devices=1,
        precision="32",           # float32 — safe on all accelerators including MPS.
        logger=logger,            # bf16-mixed is NOT used: MPS bf16 support is incomplete
        callbacks=[checkpoint_cb, lr_monitor],
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    logger.log_hyperparams({
        "pruning_method": "physical_channel_depgraph",
        "pruning_ratio": pruning_ratio,
        "finetune_epochs": finetune_epochs,
        "finetune_lr": lr,
        "finetune_starts_from": "pruned_pretrained_weights",  # NOT random init
    })

    print(f"\n[Fine-tune] {finetune_epochs} epochs, lr={lr} — continuing from pruned weights")
    trainer.fit(pl_module, datamodule=datamodule)

    # Final identity guard: verify Lightning did not swap pl_module.model.
    assert pl_module.model is pruned_model, (
        "Lightning swapped pl_module.model during training — pruned weights lost."
    )
    return pl_module.model


# ==============================================
# 5) Main pipeline
# ==============================================

def run(
    ckpt_path: str,
    pruning_ratio: float,
    finetune_epochs: int,
    lr: float,
    output_dir: str,
    round_to: int,
    image_size: int,
    bn_calibration_batches: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── [STAGE 1] Load original Lightning checkpoint with trained weights ────
    from base_flower import FlowerLightModule, FlowerDataModule
    print("\n" + "=" * 60)
    print("[STAGE 1] Loading original trained Lightning checkpoint")
    print("=" * 60)
    print(f"  ckpt: {ckpt_path}")
    torch.serialization.add_safe_globals([functools.partial])
    pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    pl_model.eval()
    # Extract the underlying torchvision EfficientNet-B0 (with 102-class head).
    # All downstream stages operate on this same Python object; the original
    # backbone is NEVER reconstructed.
    model = pl_model.model
    fp_loaded = _weight_fingerprint(model)
    print(f"  Trained-weight fingerprint: {fp_loaded}")

    example_inputs = torch.randn(1, 3, image_size, image_size)

    # ── [STAGE 2] Physical channel pruning (in-place, weight-preserving) ─────
    print("\n" + "=" * 60)
    print("[STAGE 2] Physical channel pruning on the trained model (in-place)")
    print("=" * 60)
    print(f"  ratio={pruning_ratio}, round_to={round_to}")
    model_id_before = id(model)
    model, info = physical_prune(
        model=model,
        example_inputs=example_inputs,
        pruning_ratio=pruning_ratio,
        round_to=round_to,
        iterative_steps=1,
        global_pruning=True,
    )
    # Pruning must mutate in-place; a changed id means a different object was
    # returned and downstream stages would operate on the wrong tensor graph.
    assert id(model) == model_id_before, (
        "physical_prune returned a different object — expected in-place rewrite."
    )

    print(
        "\n── Pruning Report ──\n"
        f"  Params : {_format_si(info['params_before'])} -> {_format_si(info['params_after'])}"
        f"   ({info['params_reduction']*100:.1f}% reduction)\n"
        f"  MACs   : {_format_si(info['macs_before'])} -> {_format_si(info['macs_after'])}"
        f"   ({info['macs_reduction']*100:.1f}% reduction)\n"
        f"  Ignored layers : {info['ignored_layer_count']}\n"
        f"  Surviving weights kept: YES (DepGraph drops low-norm channels in place)"
    )

    # Quick sanity check that the pruned model still produces the right shape.
    print("\n[Forward-pass validation on pruned model]")
    model.eval()
    with torch.inference_mode():
        y = model(example_inputs)
    assert y.shape == (1, 102), f"Unexpected output shape: {y.shape}"
    print(f"  OK — output shape {tuple(y.shape)}")

    # ── [STAGE 3] BatchNorm recalibration ────────────────────────────────────
    # After DepGraph resizes BN tensors, the running_mean / running_var values
    # are stale (computed over the old channel count). A short calibration pass
    # in train() mode refreshes those statistics so eval() inference is correct.
    # Without this step, val_loss exceeds 12 and val_acc is near random chance.
    print("\n" + "=" * 60)
    print("[STAGE 3] BatchNorm recalibration after pruning")
    print("=" * 60)
    if bn_calibration_batches > 0:
        dm_calib = FlowerDataModule()
        calibrate_bn(model, dm_calib, n_batches=bn_calibration_batches)
    else:
        print("  Skipped (bn_calibration_batches=0). Not recommended.")

    # Validate the calibrated model before any fine-tuning so we can confirm
    # that accuracy has recovered from the pruning-induced BN disruption.
    print("\n[Post-calibration validation — before fine-tuning]")
    import lightning.pytorch as pl
    calib_trainer = pl.Trainer(
        accelerator="cpu",   # CPU avoids MPS bf16 issues for this quick check
        devices=1,
        precision="32",
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    from base_flower import FlowerDataModule as _FDM
    calib_trainer.validate(
        _build_pruned_lightmodule(model),
        datamodule=_FDM(),
    )

    # ── [STAGE 4] Fine-tune the pruned weights (NOT from scratch) ────────────
    if finetune_epochs > 0:
        print("\n" + "=" * 60)
        print("[STAGE 4] Fine-tuning the pruned model — continuing from pruned weights")
        print("=" * 60)
        print(f"  epochs={finetune_epochs}, lr={lr}")
        print("  (the wrapper reuses the pruned nn.Module; it does NOT build a fresh model)")
        try:
            ft_model = finetune_pruned(model, finetune_epochs, lr, pruning_ratio)
            assert ft_model is model, (
                "finetune_pruned returned a different object — refusing to save."
            )
            model = ft_model
        except Exception as exc:  # noqa: BLE001
            print(f"  Fine-tune failed ({exc!r}); keeping pruned-but-not-finetuned weights.")
    else:
        print("\n[STAGE 4] Fine-tune skipped (finetune_epochs=0)")

    # ── [STAGE 5] Save the whole pruned (and possibly fine-tuned) nn.Module ──
    print("\n" + "=" * 60)
    print("[STAGE 5] Saving pruned model + metadata")
    print("=" * 60)
    pct = int(round(pruning_ratio * 100))
    model_path = os.path.join(output_dir, f"efficientnet_b0_pruned_{pct}.pth")
    meta_path = os.path.join(output_dir, f"efficientnet_b0_pruned_{pct}.json")

    # Move to CPU before serialising to keep the file device-agnostic.
    model = model.to("cpu").eval()
    torch.save(model, model_path)

    metadata = {
        "base_ckpt": os.path.abspath(ckpt_path),
        "pruning_method": "physical_channel_depgraph",
        "pruning_ratio": pruning_ratio,
        "round_to": round_to,
        "example_input_shape": [1, 3, image_size, image_size],
        "model_save_path": os.path.abspath(model_path),
        "finetune_epochs": finetune_epochs,
        "finetune_starts_from": "pruned_pretrained_weights",
        "bn_calibration_batches": bn_calibration_batches,
        "lr": lr,
        **{k: v for k, v in info.items() if k not in ("ignored_layer_count",)},
    }
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(
        "\nDone.\n"
        f"  Pruned model : {model_path}\n"
        f"  Metadata     : {meta_path}\n"
        "\nReload with:\n"
        f"  model = torch.load('{model_path}', weights_only=False)\n"
        "\nBenchmark with:\n"
        f"  python3.10 benchmark.py --model_file {model_path} "
        f"--run_name benchmark_pruned_physical_{pct}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Physical channel pruning for EfficientNet-B0 Flower-102 using "
            "Torch-Pruning DepGraph, followed by BN recalibration and optional "
            "recovery fine-tuning that continues from the pruned weights (NOT from scratch)."
        )
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to base Lightning checkpoint (.ckpt) with trained weights",
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.3,
        help="Global channel-pruning ratio (e.g. 0.3 = drop 30%% of channels)",
    )
    parser.add_argument(
        "--finetune_epochs", type=int, default=0,
        help=(
            "Recovery fine-tune epochs that continue from the pruned weights "
            "(0 to skip fine-tuning and keep the BN-recalibrated weights only)"
        ),
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for recovery fine-tune",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./logs/pruned",
        help="Where to save the pruned nn.Module (.pth) and metadata (.json)",
    )
    parser.add_argument(
        "--round_to", type=int, default=8,
        help="Round pruned channel counts to a multiple of this value (8 = HW-friendly)",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Input image size used for tracing (must match base.yaml)",
    )
    parser.add_argument(
        "--bn_calibration_batches", type=int, default=100,
        help=(
            "Number of training mini-batches used to recalibrate BatchNorm running "
            "statistics after pruning. 100 batches (≈12 800 images at batch_size=128) "
            "is sufficient for stable stats. Set to 0 to skip calibration (not recommended)."
        ),
    )
    args = parser.parse_args()

    run(
        ckpt_path=args.ckpt,
        pruning_ratio=args.pruning_ratio,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        round_to=args.round_to,
        image_size=args.image_size,
        bn_calibration_batches=args.bn_calibration_batches,
    )


if __name__ == "__main__":
    main()
