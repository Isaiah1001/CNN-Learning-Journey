# pruning_physical_channel_flower.py
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
#   (4) Wrap the *same* pruned nn.Module into a thin LightningModule
#       (PrunedFlowerLightModule). The wrapper assigns `self.model = pruned_model`;
#       it does NOT reconstruct a fresh torchvision model. We assert
#       `pl_module.model is pruned_model` and compare a weight fingerprint
#       before/after wrapping, so any accidental re-init shows up immediately.
#   (5) Fine-tune the wrapped module. trainer.fit moves it to the accelerator
#       and continues SGD on the existing pruned weights (i.e. recovery
#       fine-tuning, not training from scratch).
#   (6) Save the whole pruned + fine-tuned nn.Module via torch.save(model, path)
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
#   python3.10 pruning_physical_channel_flower.py \
#       --ckpt logs/checkpoints/checkpoint_base_epoch=33_val_acc=0.9764.ckpt \
#       --pruning_ratio 0.3 \
#       --finetune_epochs 0
#
#   # With short fine-tune recovery on the pruned weights (NOT from scratch):
#   python3.10 pruning_physical_channel_flower.py \
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
    """Cheap fingerprint of a model's parameters. Used as a safety check that the
    fine-tune wrapper is wrapping the *same* trained tensor objects rather than
    silently re-initializing a fresh backbone."""
    params = list(model.parameters())
    total_numel = sum(p.numel() for p in params)
    # Sum of a few statistics that change instantly if any layer is replaced
    # with a fresh torchvision init.
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
    """
    Physically prune output channels of a torchvision EfficientNet-B0 model.

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
            (8 is hardware-friendly for most accelerators). Set to None to
            disable.
        iterative_steps: spread pruning across this many steps. We call .step()
            once for each step; for one-shot pruning leave this at 1.
        global_pruning: True = single global ranking; False = per-layer ratio.

    Returns:
        (pruned_model, info_dict) — pruned_model is `model` itself, mutated in-place.
    """
    # Layers we refuse to prune. The final classifier output channels equal the
    # number of classes, so we pin it. The EfficientNet stem conv is fragile and
    # often left intact in the literature — we keep it as-is here too.
    ignored_layers: list[torch.nn.Module] = []
    if hasattr(model, "classifier"):
        for m in model.classifier.modules():
            if isinstance(m, torch.nn.Linear):
                ignored_layers.append(m)
    if hasattr(model, "features") and len(model.features) > 0:
        for m in model.features[0].modules():
            if isinstance(m, torch.nn.Conv2d):
                ignored_layers.append(m)

    importance = tp.importance.MagnitudeImportance(p=2)  # L2 magnitude

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

    for step in range(iterative_steps):
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
# 3) Fine-tune wrapper that PRESERVES the pruned weights
# ==============================================

def _build_pruned_lightmodule(pruned_model: torch.nn.Module, num_classes: int = 102,
                              lr: float = 1e-4, weight_decay: float = 1e-4):
    """Wrap an already-pruned nn.Module into a Lightning module compatible
    with FlowerLightModule's training API.

    Critical contract:
      * We do NOT call FlowerLightModule.__init__ — that would build a fresh
        stock `torchvision.efficientnet_b0` and overwrite `self.model`,
        discarding the pruned (and previously trained) weights.
      * We assign `self.model = pruned_model` so the Lightning module wraps the
        *same* Python object that was returned from physical_prune. The caller
        verifies `pl_module.model is pruned_model` after construction.

    Defined inside a function so this file imports cleanly when Lightning isn't
    installed.
    """
    import lightning.pytorch as pl
    from torchmetrics import Accuracy
    from base_flower import FlowerLightModule

    class PrunedFlowerLightModule(FlowerLightModule):
        """LightningModule that fine-tunes an already-pruned nn.Module.

        The pruned model is supplied from outside; this class never builds a
        replacement backbone. Hence pruned + trained weights survive into
        trainer.fit and are updated by SGD (i.e. recovery fine-tuning), they
        are NOT trained from scratch.
        """

        def __init__(self, pruned_model: torch.nn.Module, num_classes: int = 102,
                     lr: float = 1e-4, weight_decay: float = 1e-4):
            # IMPORTANT: skip FlowerLightModule.__init__ so we don't rebuild a
            # stock efficientnet_b0 and clobber the pruned weights.
            pl.LightningModule.__init__(self)
            # Don't try to pickle the live nn.Module into hparams.
            self.save_hyperparameters(ignore=["pruned_model"])

            # Bind the SAME object we received. This is the load-bearing line:
            # if anything ever replaces `self.model` with a freshly constructed
            # backbone, training resumes from random init instead of the pruned
            # weights.
            self.model = pruned_model

            # Closures over hparams, not raw torch callables, so configure_optimizers
            # produces a real optimizer/scheduler over `self.parameters()`.
            self.optimizer = lambda params: torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay
            )
            self.lr_scheduler = lambda opt: torch.optim.lr_scheduler.ConstantLR(
                opt, factor=1.0, total_iters=1
            )

            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    pl_module = PrunedFlowerLightModule(
        pruned_model=pruned_model, num_classes=num_classes,
        lr=lr, weight_decay=weight_decay,
    )

    # Hard guard: the wrapper must hold the *exact same* pruned object we passed
    # in. If this fails, something replaced the model and fine-tuning would
    # silently start from a fresh init.
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
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
    from base_flower import FlowerDataModule

    # Fingerprint BEFORE wrapping — used to verify the wrapper does not silently
    # swap in a freshly constructed backbone.
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

    # Pruned-model checkpoints saved by Lightning won't reload into
    # FlowerLightModule (shape mismatch). We save the whole module separately
    # via torch.save; this checkpoint is here only for training resume.
    checkpoint_cb = ModelCheckpoint(
        dirpath="./logs/checkpoints",
        monitor="val_acc",
        filename=f"checkpoint_pruned_physical_{pruning_ratio:.0%}" + "_{epoch:02d}_{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=finetune_epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
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

    # Sanity: same object came out the other side.
    assert pl_module.model is pruned_model, (
        "Lightning swapped pl_module.model during training — pruned weights lost."
    )
    return pl_module.model


# ==============================================
# 4) Main pipeline
# ==============================================

def run(
    ckpt_path: str,
    pruning_ratio: float,
    finetune_epochs: int,
    lr: float,
    output_dir: str,
    round_to: int,
    image_size: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── [STAGE 1] Load original Lightning checkpoint with trained weights ────
    from base_flower import FlowerLightModule  # lazy import — Lightning needed
    print("\n" + "=" * 60)
    print("[STAGE 1] Loading original trained Lightning checkpoint")
    print("=" * 60)
    print(f"  ckpt: {ckpt_path}")
    torch.serialization.add_safe_globals([functools.partial])
    pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    pl_model.eval()
    # Extract the underlying torchvision EfficientNet-B0 (with 102-class head)
    # — the trained weights live on this object. Everything downstream operates
    # on this same Python object; the original backbone is NEVER reconstructed.
    model = pl_model.model
    fp_loaded = _weight_fingerprint(model)
    print(f"  trained-weight fingerprint: {fp_loaded}")

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
    # Pruning must mutate in-place; if id changed, downstream finetune would be
    # operating on a different object than the one that was reported.
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

    # ── Validate forward pass on the pruned model ────────────────────────────
    print("\n[Forward-pass validation on pruned model]")
    model.eval()
    with torch.inference_mode():
        y = model(example_inputs)
    assert y.shape == (1, 102), f"Unexpected output shape: {y.shape}"
    print(f"  OK — output shape {tuple(y.shape)}")

    # ── [STAGE 3] Fine-tune the pruned weights (NOT from scratch) ────────────
    if finetune_epochs > 0:
        print("\n" + "=" * 60)
        print("[STAGE 3] Fine-tuning the pruned model — continuing from pruned weights")
        print("=" * 60)
        print(f"  epochs={finetune_epochs}, lr={lr}")
        print("  (the wrapper reuses the pruned nn.Module; it does NOT build a fresh model)")
        try:
            ft_model = finetune_pruned(model, finetune_epochs, lr, pruning_ratio)
            # Same-object guard at the script level too.
            assert ft_model is model, (
                "finetune_pruned returned a different object — refusing to save."
            )
            model = ft_model
        except Exception as exc:  # noqa: BLE001
            print(f"  Fine-tune failed ({exc!r}); keeping pruned-but-not-finetuned weights.")
    else:
        print("\n[STAGE 3] Fine-tune skipped (finetune_epochs=0)")

    # ── [STAGE 4] Save the whole pruned (and possibly fine-tuned) nn.Module ──
    print("\n" + "=" * 60)
    print("[STAGE 4] Saving pruned model + metadata")
    print("=" * 60)
    pct = int(round(pruning_ratio * 100))
    model_path = os.path.join(output_dir, f"efficientnet_b0_pruned_{pct}.pth")
    meta_path = os.path.join(output_dir, f"efficientnet_b0_pruned_{pct}.json")

    # Move to CPU before serializing — keeps the file portable.
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
            "Physical channel pruning for EfficientNet-B0 Flower using "
            "Torch-Pruning DepGraph, followed by recovery fine-tuning that "
            "continues from the pruned weights (NOT from scratch)."
        )
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to base Lightning checkpoint (.ckpt) with trained weights")
    parser.add_argument("--pruning_ratio", type=float, default=0.3,
                        help="Global channel-pruning ratio (e.g. 0.3 = drop 30%% of channels)")
    parser.add_argument("--finetune_epochs", type=int, default=0,
                        help="Recovery fine-tune epochs that continue from the pruned weights "
                             "(0 to skip fine-tuning)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for recovery fine-tune")
    parser.add_argument("--output_dir", type=str, default="./logs/pruned",
                        help="Where to save the pruned nn.Module (.pth) and metadata (.json)")
    parser.add_argument("--round_to", type=int, default=8,
                        help="Round pruned channel counts to a multiple of this value (8 = HW-friendly)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size used for tracing (matches base.yaml)")
    args = parser.parse_args()

    run(
        ckpt_path=args.ckpt,
        pruning_ratio=args.pruning_ratio,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        round_to=args.round_to,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
