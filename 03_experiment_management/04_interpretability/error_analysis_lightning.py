import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg", force=True)

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightning.pytorch as pl

from sklearn.metrics import confusion_matrix
from lightning.pytorch.callbacks import Callback

from hyperparameters_flower import FlowerLightModule, FlowerDataModule

# ==============================================
# 1) load trained weights and get file paths
# ==============================================
CHECKPOINT_PATH = "./checkpoint_base_epoch=34_val_acc=0.9568.ckpt"
DATA_PATH = "../99_flower_data"
BATCH_SIZE = 128
NUM_WORKERS = 8
OUTPUT_DIR = Path("./outputs")


# ==============================================
# 2) set Lightning test module
# ==============================================
class FlowerErrorAnalysisModule(FlowerLightModule):
    def test_step(self, batch, batch_idx):
        images, labels, descriptions = batch
        logits = self(images)
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        return {
            "true_labels": labels.detach().cpu(),
            "pred_labels": preds.detach().cpu(),
            "confidences": confs.detach().cpu(),
            "descriptions": list(descriptions),
        }
        
# ==============================================
# 3) callback for error analysis
# ==============================================     
class ErrorAnalysisCallback(Callback):
    def __init__(self, output_dir: str = "./outputs"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.records = []

    def on_test_start(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        datamodule = trainer.datamodule
        base_dataset = datamodule.test_dataset.subset.dataset

        for true_label, pred_label, conf, true_desc in zip(
            outputs["true_labels"],
            outputs["pred_labels"],
            outputs["confidences"],
            outputs["descriptions"],
        ):
            true_label = int(true_label.item())
            pred_label = int(pred_label.item())

            self.records.append(
                {
                    "sample_id": len(self.records),
                    "batch_idx": batch_idx,
                    "true_label": true_label,
                    "true_class": true_desc,
                    "pred_label": pred_label,
                    "pred_class": base_dataset.retrieve_description(pred_label),
                    "confidence": float(conf.item()),
                    "correct": int(true_label == pred_label),
                }
            )

    def on_test_end(self, trainer, pl_module):
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_dir / "all_predictions.csv", index=False)

        wrong_df = df[df["correct"] == 0].copy()
        wrong_df.to_csv(self.output_dir / "wrong_predictions.csv", index=False)

        per_class_df = (
            df.groupby(["true_label", "true_class"])
            .agg(total=("correct", "count"), correct=("correct", "sum"))
            .reset_index()
        )
        per_class_df["accuracy"] = per_class_df["correct"] / per_class_df["total"]
        per_class_df = per_class_df.sort_values("accuracy", ascending=True)
        per_class_df.to_csv(self.output_dir / "per_class_accuracy.csv", index=False)

        self._save_confusion_matrix(df)
        self._save_summary(df, per_class_df)

    def _save_confusion_matrix(self, df):
        y_true = df["true_label"].tolist()
        y_pred = df["pred_label"].tolist()
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(18, 15))
        sns.heatmap(cm, cmap="Blues", cbar=True)
        plt.title("Confusion Matrix - Oxford 102 Flowers")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300)
        plt.close()

    def _save_summary(self, df, per_class_df):
        overall_acc = df["correct"].mean()
        wrong_count = int((df["correct"] == 0).sum())
        wrong_df = df[df["correct"] == 0]
        pair_counts = Counter(zip(wrong_df["true_class"], wrong_df["pred_class"]))

        with open(self.output_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
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
            for (true_cls, pred_cls), count in pair_counts.most_common(10):
                f.write(f"- {true_cls} -> {pred_cls}: {count}\n")
                
# ==============================================
# 4) main function to run the test and error analysis
# ==============================================                   
def main():
    pl.seed_everything(42, workers=True)

    model = FlowerErrorAnalysisModule.load_from_checkpoint(CHECKPOINT_PATH)
    dm = FlowerDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_path=DATA_PATH,
    )

    error_callback = ErrorAnalysisCallback(output_dir=OUTPUT_DIR)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[error_callback],
    )

    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()