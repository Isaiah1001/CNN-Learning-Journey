# CNN Learning Journey

This repository documents an end-to-end CNN learning journey on the Oxford 102 Flowers dataset[^2], starting from a custom CNN built from scratch with Pytorch[^1] and progressing through transfer learning, experiment management, and deployment-oriented compression. The project is designed not only to improve classification accuracy, but also to understand how deep learning workflows evolve from basic modeling to reproducible experimentation and practical deployment.

## Overview

This is a first CNN project, but it is structured as a complete learning path rather than a single training script. The work begins with a hand-built CNN pipeline, then moves to pretrained EfficientNet fine-tuning, introduces systematic experiment tooling with PyTorch Lightning[^3] and MLflow[^4], and finally evaluates pruning, ONNX export[^5], and quantization for deployment efficiency. 

The dataset used throughout the project is the Oxford 102 Flowers dataset, which contains 8,189 images across 102 flower categories. Its relatively small size makes it a good benchmark for understanding both the limitations of training from scratch and the practical value of transfer learning. 

## Journey Structure

| Stage | Focus | Core Question | Main Outcome |
|------|------|------|------|
| [01_custom_CNN_from_scratch](./01_custom_CNN_from_scratch) | CNN fundamentals | Why does a simple CNN trained from scratch struggle on limited data? | Built the full pipeline from data loading to training and found that a shallow CNN plateaued at 42.52% top-1 accuracy.  |
| [02_transfer_learning](./02_transfer_learning) | Pretrained models and fine-tuning | How can pretrained features improve performance efficiently? | EfficientNet-B0 fine-tuning raised accuracy from about 42% to 93%+ with classifier-head tuning and up to 97.31% with deeper unfreezing.  |
| [03_experiment_management](./03_experiment_management) | Reproducibility and interpretability | How can training runs be organized, compared, and understood more systematically? | Added PyTorch Lightning, MLflow, hyperparameter sweeps, and interpretability analysis to make experiments easier to reproduce and inspect. |
| [04_compression_deployment](./04_compression_deployment) | Compression and deployment | How can the model be made smaller and faster while keeping acceptable accuracy? | Benchmarked pruning, structural compression, ONNX export, and INT8 quantization to study size-speed-accuracy trade-offs.|

## Repository Structure

```text
CNN-Learning-Journey/
├── 01_custom_CNN_from_scratch/   # Build and train a SimpleCNN from scratch
├── 02_transfer_learning/         # Fine-tune pretrained EfficientNet-B0 models
├── 03_experiment_management/     # Add Lightning, MLflow, hyperparameter studies, and interpretability
├── 04_compression_deployment/    # Benchmark pruning, quantization, and ONNX deployment
└── README.md
```

Each stage has its own README with implementation details, experimental results, and reflections. The root README serves as the high-level guide that connects these stages into one coherent progression.

## Key Results

| Stage | Model / Method | Best Result | Main Takeaway |
|--------|------|------|------|
| Stage 1 | SimpleCNN from scratch | 42.52% top-1 accuracy | Building from scratch is valuable for understanding the full pipeline, but limited data and shallow capacity strongly constrain performance. |
| Stage 2 | EfficientNet-B0 transfer learning | 97.31% best top-1 accuracy | Pretraining provides strong generic visual representations, and staged fine-tuning is much more effective than training a shallow CNN from scratch. |
| Stage 3 | Structured experiment workflow | 97.39% best validation accuracy in the LR sweep | Better tooling improves reproducibility, comparison quality, and model understanding, not just convenience.|
| Stage 4 | Compression and deployment benchmarking | INT8 ONNX model reduced size from 15.78 MB to 4.78 MB and latency from 10.36 ms to 4.94 ms, while physical channel removal reduced PyTorch model size from 16.13 MB to 13.66 MB and latency from 85.49 ms to 68.76 ms. | Deployment optimization is a trade-off across accuracy, latency, model size, and framework support. |

## Main Takeaways

- Training a CNN from scratch is useful for learning data preprocessing, model design, and optimization, but it is often not the most practical path when the dataset is small.
- Transfer learning is a highly effective solution when limited data makes feature learning from scratch difficult. Fine-tuning a pretrained backbone can close most of the performance gap with far less effort.
- Once a strong model exists, workflow quality becomes critical. Experiment tracking, hyperparameter comparison, and interpretability tools make iteration more systematic and reproducible.
- Accuracy alone is not enough for real-world use. Practical deployment also depends on model size, inference latency, and compatibility with deployment runtimes such as ONNX Runtime.
- A deep learning project naturally evolves from modeling questions to engineering questions: first how to learn, then how to train efficiently, and finally how to deploy under constraints.

## Highlights

- Built a complete CNN pipeline from scratch, including preprocessing, augmentation, training, and visualization.
- Fine-tuned EfficientNet-B0 with staged unfreezing and compared multiple transfer learning strategies.
- Introduced PyTorch Lightning and MLflow for cleaner training code, run tracking, and structured hyperparameter experiments. 
- Used Grad-CAM and saliency maps to inspect model behavior beyond scalar metrics.
- Evaluated pruning, channel removal, ONNX export, and INT8 quantization for deployment-oriented optimization.

## Future Directions

Possible future work includes testing stronger backbone networks, exploring knowledge distillation, improving performance on visually similar flower classes, and deploying the model on real edge or industrial hardware.

The Oxford 102 Flowers dataset serves as a useful introductory benchmark for image classification, and it is necessary that the pipeline developed in this work should be further validated in real industrial scenarios.

## AI Transparency

Parts of the project documentation, README refinement, and wording cleanup were assisted by AI tools. The core project design, code implementation, experiments, result analysis, and final technical decisions were completed and verified manually. This disclosure is included to make the writing process transparent while keeping clear ownership of the technical work.

## Reference
[^1]: Paszke, A., Gross, S., Massa, F., et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019. [https://pytorch.org/](https://pytorch.org/)

[^2]: Nilsback, M.-E., and Zisserman, A. Automated Flower Classification over a Large Number of Classes. Indian Conference on Computer Vision, Graphics and Image Processing, 2008.

[^3]: Falcon et al., [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), Lightning AI, 2019.

[^4]: Zaharia et al., [MLflow: A Machine Learning Lifecycle Platform](https://mlflow.org), Databricks, 2018.

[^5]: Microsoft, [ONNX Runtime](https://onnxruntime.ai), Microsoft, 2018.
