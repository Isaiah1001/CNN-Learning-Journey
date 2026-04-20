import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
metrics = pd.read_csv('./logs/flower_experiment/version_6/metrics.csv')

# ---- 关键修正 1：前向填充 lr ----
lr_cols = [c for c in metrics.columns if c.startswith('lr-')]
metrics[lr_cols] = metrics[lr_cols].ffill()

# ---- 关键修正 2：删除没有 epoch 的行 ----
metrics = metrics[metrics['epoch'].notna()]

# ---- 关键修正 3：按 epoch 聚合 ----
metrics = metrics.groupby('epoch').mean()

# ========================
# 绘图
# ========================
plt.figure(figsize=(15, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(metrics.index, metrics['train_loss'], label='Train Loss')
plt.plot(metrics.index, metrics['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(metrics.index, metrics['train_acc'], label='Train Acc')
plt.plot(metrics.index, metrics['val_acc'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Learning Rate
plt.subplot(1, 3, 3)

if lr_cols:
    for col in lr_cols:
        plt.plot(metrics.index, metrics[col], label=col)
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')  # 如果想看对数尺度
else:
    plt.text(0.5, 0.5, 'No LR Data', ha='center')

plt.tight_layout()
plt.show()