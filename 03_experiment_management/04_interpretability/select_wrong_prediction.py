import pandas as pd

df = pd.read_csv("./outputs/wrong_predictions.csv")

# choose 4 classes to focus on
target_classes = [
    'sweet pea',
    'hibiscus',
    'mexican petunia',
    'windflower',
]

focus_df = df[df["true_class"].str.contains('|'.join(target_classes), case=False, na=False)].copy()
# get top 2 most confident wrong predictions for each class
selected = (
    focus_df.sort_values("confidence", ascending=False)
            .groupby("true_class")
            .head(2)
)

selected.to_csv("outputs/gradcam_targets.csv", index=False)
print(selected[["sample_id", "true_class", "pred_class", "confidence"]])