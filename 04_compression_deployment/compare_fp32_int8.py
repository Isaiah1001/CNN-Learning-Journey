import os
import time
import numpy as np
import onnxruntime as ort

from base_flower import FlowerDataModule

FP32_MODEL = "flower_efficientnet_b0_int8_u8u8_base.onnx"
INT8_MODEL = "flower_efficientnet_b0_int8_u8u8_1.onnx"
DATA_PATH = "./99_flower_data"

BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 224


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def create_session(model_path):
    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )


def evaluate_accuracy(session, dataloader):
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0

    for batch in dataloader:
        images, labels, _ = batch
        images_np = images.cpu().numpy().astype(np.float32)
        outputs = session.run(None, {input_name: images_np})[0]
        preds = np.argmax(outputs, axis=1)

        labels_np = labels.cpu().numpy()
        correct += (preds == labels_np).sum()
        total += labels_np.shape[0]

    return correct / total


def benchmark_latency(session, dataloader, warmup_batches=5, test_batches=30):
    input_name = session.get_inputs()[0].name
    latencies = []

    all_batches = []
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches + test_batches:
            break
        images, labels, _ = batch
        images_np = images.cpu().numpy().astype(np.float32)
        all_batches.append(images_np)

    for images_np in all_batches[:warmup_batches]:
        _ = session.run(None, {input_name: images_np})

    for images_np in all_batches[warmup_batches:]:
        start = time.perf_counter()
        _ = session.run(None, {input_name: images_np})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    latencies = np.array(latencies)
    return {
        "avg_ms": float(latencies.mean()),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_img_s": float(
            (len(all_batches[warmup_batches:]) * all_batches[warmup_batches].shape[0]) / (latencies.sum() / 1000.0)
        ) if len(latencies) > 0 else 0.0
    }


def main():
    dm = FlowerDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_path=DATA_PATH,
        image_size=IMAGE_SIZE,
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()

    fp32_sess = create_session(FP32_MODEL)
    int8_sess = create_session(INT8_MODEL)

    print("Evaluating accuracy...")
    fp32_acc = evaluate_accuracy(fp32_sess, val_loader)

    dm = FlowerDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_path=DATA_PATH,
        image_size=IMAGE_SIZE,
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    int8_acc = evaluate_accuracy(int8_sess, val_loader)

    dm = FlowerDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_path=DATA_PATH,
        image_size=IMAGE_SIZE,
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()

    print("Benchmarking FP32...")
    fp32_perf = benchmark_latency(fp32_sess, val_loader)

    dm = FlowerDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_path=DATA_PATH,
        image_size=IMAGE_SIZE,
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()

    print("Benchmarking INT8...")
    int8_perf = benchmark_latency(int8_sess, val_loader)

    fp32_size = get_model_size_mb(FP32_MODEL)
    int8_size = get_model_size_mb(INT8_MODEL)

    print("\n===== Comparison =====")
    print(f"FP32 accuracy       : {fp32_acc:.4f}")
    print(f"INT8 accuracy       : {int8_acc:.4f}")
    print(f"Accuracy drop       : {fp32_acc - int8_acc:.4f}")
    print()
    print(f"FP32 avg latency    : {fp32_perf['avg_ms']:.3f} ms")
    print(f"INT8 avg latency    : {int8_perf['avg_ms']:.3f} ms")
    print(f"FP32 p95 latency    : {fp32_perf['p95_ms']:.3f} ms")
    print(f"INT8 p95 latency    : {int8_perf['p95_ms']:.3f} ms")
    print(f"FP32 throughput     : {fp32_perf['throughput_img_s']:.3f} img/s")
    print(f"INT8 throughput     : {int8_perf['throughput_img_s']:.3f} img/s")
    print()
    print(f"FP32 model size     : {fp32_size:.3f} MB")
    print(f"INT8 model size     : {int8_size:.3f} MB")
    print(f"Size reduction      : {(1 - int8_size / fp32_size) * 100:.2f}%")

if __name__ == "__main__":
    main()