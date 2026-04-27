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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from hyperparameters_flower import FlowerLightModule, FlowerDataModule
from preprocess import data_access  # <-- use to load original images


# ------------------
# Config
# ------------------
CHECKPOINT_PATH = "./checkpoint_base_epoch=34_val_acc=0.9568.ckpt"
DATA_PATH = "../99_flower_data"
BATCH_SIZE = 128
NUM_WORKERS = 8
TARGETS_CSV = "./outputs/gradcam_targets_true.csv"
OUTPUT_DIR = "./outputs/gradcam_images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------
# Helper functions
# ------------------
def build_test_cache(dm: FlowerDataModule):
    """
    Run the test dataloader once and cache:
    - images (tensors, transformed)
    - labels (ints)
    - descriptions (strings)
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
    Build a list of original RGB PIL images for the *test subset*,
    in the same order as test_dataset indices.
    This assumes test_dataset is a subset of the full data_access dataset.

    We will simply load the full dataset via data_access and then
    use the same indices that get_dataset used for the test split.
    Simplest assumption: your sample_id matches original dataset index.
    """
    # data_access in your project is a class, use it to load original images
    da = data_access(DATA_PATH)  # relative to this script

    orig_images = []
    for idx in range(len(da)):
        img, _, _ = da[idx]  # (PIL.Image, label, description)
        orig_images.append(img)

    return orig_images


def create_cam(model: FlowerLightModule):
    """
    Create a GradCAM object on the last feature block of EfficientNet-B0.
    """
    target_layers = [model.model.features[-1]]
    cam = GradCAM(
        model=model.model,
        target_layers=target_layers,
        # use_cuda=(DEVICE == "cuda"),
    )
    return cam


def pil_to_numpy_resized(img: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    """
    Convert a PIL image to float32 numpy (H,W,C) in [0,1] with given size.
    """
    img_resized = img.resize((target_w, target_h), resample=Image.BICUBIC)
    x = np.array(img_resized).astype(np.float32) / 255.0  # H,W,C in [0,1]
    return x


def generate_cam_for_sample(
    cam: GradCAM,
    model: FlowerLightModule,
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class_idx: int,
    out_path: Path,
):
    """
    Generate Grad-CAM for one image and one target class index.
    CAM is computed on the normalized tensor but overlaid on the original RGB image.
    """
    model.eval()
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # 1xCxHxW

    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # HxW

    H, W = grayscale_cam.shape
    base_img = pil_to_numpy_resized(original_image, H, W)  # original RGB in [0,1]

    cam_image = show_cam_on_image(base_img, grayscale_cam, use_rgb=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cam_image).save(out_path)


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

    cam = create_cam(model)

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
        # Assume sample_id == original dataset index; adjust if you used a subset with different mapping
        original_img = original_images[sample_id]

        target_idx = pred_label

        cls_dir = Path(OUTPUT_DIR) / true_class.replace(" ", "_")
        filename = f"id{sample_id}_true_{true_label}_pred_{pred_label}_{pred_class.replace(' ', '_')}_conf_{confidence:.3f}.png"
        out_path = cls_dir / filename

        print(f"Generating CAM for sample_id={sample_id}, true={true_class}, pred={pred_class} -> {out_path}")
        generate_cam_for_sample(
            cam=cam,
            model=model,
            image_tensor=img_tensor,
            original_image=original_img,
            target_class_idx=target_idx,
            out_path=out_path,
        )

    print(f"Done. Grad-CAM images are saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()