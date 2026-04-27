import os
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg", force=True)


import torch
import pandas as pd
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from hyperparameters_flower import FlowerLightModule, FlowerDataModule


CHECKPOINT_PATH = "./checkpoint_base_epoch=34_val_acc=0.9568.ckpt"
DATA_PATH = "../99_flower_data"
BATCH_SIZE = 128
NUM_WORKERS = 8
OUTPUT_DIR = Path("./outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_base_dataset(subset_dataset):
    return subset_dataset.subset.dataset


def build_label_to_name(base_dataset, num_classes=102):
    return {i: base_dataset.retrieve_description(i) for i in range(num_classes)}


def run_inference():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42, workers=True)

    model = FlowerLightModule.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model.to(DEVICE)

    dm = FlowerDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_path=DATA_PATH,
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    base_dataset = get_base_dataset(dm.test_dataset)
    label_to_name = build_label_to_name(base_dataset, num_classes=model.hparams.num_classes)

    records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, labels, descriptions = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()

                records.append(
                    {
                        "sample_id": len(records),
                        "batch_idx": batch_idx,
                        "true_label": true_label,
                        "true_class": descriptions[i],
                        "pred_label": pred_label,
                        "pred_class": label_to_name[pred_label],
                        "confidence": confs[i].item(),
                        "correct": int(pred_label == true_label),
                    }
                )

    df = pd.DataFrame(records)
    return df, label_to_name


def save_csv_outputs(df):
    df.to_csv(OUTPUT_DIR / "all_predictions.csv", index=False)

    wrong_df = df[df["correct"] == 0].copy()
    wrong_df.to_csv(OUTPUT_DIR / "wrong_predictions.csv", index=False)

    per_class_df = (
        df.groupby(["true_label", "true_class"])
        .agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    per_class_df["accuracy"] = per_class_df["correct"] / per_class_df["total"]
    per_class_df = per_class_df.sort_values("accuracy", ascending=True)
    per_class_df.to_csv(OUTPUT_DIR / "per_class_accuracy.csv", index=False)

    return wrong_df, per_class_df


def save_confusion_matrix(df, label_to_name):
    y_true = df["true_label"].tolist()
    y_pred = df["pred_label"].tolist()

    labels = sorted(label_to_name.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix - Oxford 102 Flowers")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def get_top_confusion_pairs(df, top_k=10):
    wrong_df = df[df["correct"] == 0]
    pair_counts = Counter(zip(wrong_df["true_class"], wrong_df["pred_class"]))
    return pair_counts.most_common(top_k)


def save_summary(df, per_class_df, top_pairs):
    overall_acc = df["correct"].mean()
    wrong_count = int((df["correct"] == 0).sum())

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Overall test accuracy: {overall_acc:.4f}\n")
        f.write(f"Wrong predictions: {wrong_count}\n\n")

        f.write("Worst 10 classes by accuracy:\n")
        for _, row in per_class_df.head(10).iterrows():
            f.write(
                f"- {row['true_class']}: "
                f"{row['accuracy']:.4f} "
                f"({int(row['correct'])}/{int(row['total'])})\n"
            )

        f.write("\nTop 10 confusion pairs:\n")
        for (true_cls, pred_cls), count in top_pairs:
            f.write(f"- {true_cls} -> {pred_cls}: {count}\n")


def main():
    df, label_to_name = run_inference()
    wrong_df, per_class_df = save_csv_outputs(df)
    save_confusion_matrix(df, label_to_name)
    top_pairs = get_top_confusion_pairs(df, top_k=10)
    save_summary(df, per_class_df, top_pairs)

    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
    print(f"Overall accuracy: {df['correct'].mean():.4f}")
    print(f"Wrong predictions: {len(wrong_df)}")


if __name__ == "__main__":
    main()