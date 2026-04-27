import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import pandas as pd
import lightning.pytorch as pl

from hyperparameters_flower import FlowerLightModule, FlowerDataModule
from preprocess import data_access  # to load original images


# ------------------
# Config
# ------------------
CHECKPOINT_PATH = "./checkpoint_base_epoch=34_val_acc=0.9568.ckpt"
DATA_PATH = "../99_flower_data"
BATCH_SIZE = 128
NUM_WORKERS = 8
TARGETS_CSV = "./outputs/gradcam_targets.csv"   # reuse the same CSV
OUTPUT_DIR = "./outputs/saliency_images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------
# Helper functions
# ------------------
def build_test_cache(dm: FlowerDataModule):
    """
    Cache transformed test tensors (same order as used for predictions).
    """
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    images = []
    labels = []
    descriptions = []

    with torch.no_grad():
        for batch in test_loader:
            x, y, desc = batch
            for i in range(len(y)):
                images.append(x[i])          # CHW tensor (normalized)
                labels.append(int(y[i].item()))
                descriptions.append(desc[i])

    return images, labels, descriptions


def build_original_image_cache():
    """
    Cache original RGB images using data_access, assuming
    sample_id == original dataset index.
    """
    da = data_access(DATA_PATH)
    orig_images = []
    for idx in range(len(da)):
        img, _, _ = da[idx]
        orig_images.append(img)
    return orig_images


def tensor_to_numpy(t: torch.Tensor):
    """
    Convert tensor (C,H,W) to numpy (H,W,C), no denorm, in [-? , ?].
    """
    x = t.detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))  # HWC
    return x


def compute_saliency(
    model: FlowerLightModule,
    image_tensor: torch.Tensor,
    target_class_idx: int,
):
    """
    Vanilla saliency: gradient of target logit w.r.t. input pixels.
    Returns saliency map as numpy (H,W) in [0,1].
    """
    model.eval()

    # Prepare input with grad
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # 1xCxHxW
    input_tensor.requires_grad_(True)

    # Forward
    outputs = model(input_tensor)           # 1 x num_classes
    target_logit = outputs[0, target_class_idx]

    # Backward
    model.zero_grad()
    target_logit.backward()

    # Take gradients w.r.t. input
    grads = input_tensor.grad.detach()[0]   # CxHxW

    # Convert to absolute and max over channels
    grads_np = grads.cpu().numpy()
    saliency = np.max(np.abs(grads_np), axis=0)  # HxW

    # Normalize to [0,1]
    sal_min, sal_max = saliency.min(), saliency.max()
    if sal_max > sal_min:
        saliency = (saliency - sal_min) / (sal_max - sal_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency  # HxW in [0,1]


def overlay_saliency_on_original(
    original_image: Image.Image,
    saliency: np.ndarray,
    alpha: float = 0.6,
):
    """
    Resize saliency to match original image, apply colormap, and overlay.
    Returns RGB uint8 array.
    """
    H, W = original_image.size[1], original_image.size[0]  # (height, width)
    saliency_resized = Image.fromarray((saliency * 255).astype(np.uint8)).resize(
        (W, H), resample=Image.BICUBIC
    )
    saliency_resized = np.array(saliency_resized) / 255.0  # HxW in [0,1]

    # Color map (jet)
    cmap = plt.get_cmap("jet")
    saliency_color = cmap(saliency_resized)[:, :, :3]  # HxWx3

    base_img = np.array(original_image).astype(np.float32) / 255.0  # HxWx3

    overlay = alpha * saliency_color + (1 - alpha) * base_img
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8


# ------------------
# Main
# ------------------
def main():
    pl.seed_everything(42, workers=True)

    print("Loading model...")
    model = FlowerLightModule.load_from_checkpoint(CHECKPOINT_PATH)
    model.to(DEVICE)

    print("Building test cache (transformed tensors)...")
    dm = FlowerDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_path=DATA_PATH,
    )
    images_cache, labels_cache, descriptions_cache = build_test_cache(dm)

    print("Building original image cache...")
    original_images = build_original_image_cache()

    print(f"Reading targets from {TARGETS_CSV} ...")
    targets_df = pd.read_csv(TARGETS_CSV)
    targets_df["true_class"] = targets_df["true_class"].astype(str).str.strip().str.replace("'", "")
    targets_df["pred_class"] = targets_df["pred_class"].astype(str).str.strip().str.replace("'", "")

    print(targets_df)

    for _, row in targets_df.iterrows():
        sample_id = int(row["sample_id"])
        true_label = int(row["true_label"])
        true_class = row["true_class"]
        pred_label = int(row["pred_label"])
        pred_class = row["pred_class"]
        confidence = float(row["confidence"])

        if sample_id < 0 or sample_id >= len(images_cache):
            print(f"Skip sample_id {sample_id}: out of range")
            continue

        img_tensor = images_cache[sample_id]
        original_img = original_images[sample_id]

        # For wrong predictions, saliency w.r.t. predicted class
        target_idx = pred_label

        print(f"Computing saliency for sample_id={sample_id}, true={true_class}, pred={pred_class}")

        saliency = compute_saliency(
            model=model,
            image_tensor=img_tensor,
            target_class_idx=target_idx,
        )

        overlay = overlay_saliency_on_original(
            original_image=original_img,
            saliency=saliency,
            alpha=0.6,
        )

        cls_dir = Path(OUTPUT_DIR) / true_class.replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)
        filename = f"id{sample_id}_true_{true_label}_pred_{pred_label}_{pred_class.replace(' ', '_')}_conf_{confidence:.3f}.png"
        out_path = cls_dir / filename

        Image.fromarray(overlay).save(out_path)

    print(f"Done. Saliency images are saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()