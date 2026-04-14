# CNN-Learning-Journey — From Scratch to EfficientNet Deployment
> A physics-trained engineer's structured path through deep learning:  
> building CNN intuition from first principles, then scaling to production-ready deployment.

## Background: 
PhD in Wind Science & Engineering with expertise in computational fluid dynamics, large eddy simulation, 
and physical laboratory experimentation (tornado simulator). 
This repository documents the transition from physical simulation 
and computational fluid dynamics to computer vision engineering.

## Motivation

This project is part of a self-directed, 700+ hour deep learning curriculum 
spanning Feb 2025 – present, built entirely outside of formal CS training:

| Period | Focus | Hours |
|--------|-------|-------|
| Feb – May 2025 | Statistical learning foundations (*Elements of Statistical Learning*) | ~100h |
| May – Sep 2025 | Coursera: ML Specialization → Deep Learning Specialization → AI Agent Developer | ~347h |
| Sep 2025 – Feb 2026 | Kaggle projects (tabular + image classification), YOLO, CS231N | ~247h |
| Feb 2026 – present | CNN architecture deep dive, fine-tuning, MLflow, pruning, quantization, ONNX | ongoing |

**Why this matters:** Most CV portfolios show the end result. This repository 
documents the reasoning behind each technical decision — the questions asked, 
the experiments that failed, and what was learned from them.

## Learning Roadmap
This project is **not** a starting point. It is built on top of a self-learning curriculum of 700+ hours in machine learning and deep learning,  
including completing Coursera’s Machine Learning Specialization (4 courses) and Deep Learning Specialization (8 courses), plus multiple applied projects on Kaggle and YOLO.
```
Stage 1: SimpleCNN from scratch (PyTorch implementation)
↓
Stage 2: Transfer Learning with EfficientNet (fine-tuning)
↓
Stage 3: Experiment Management (pipeline tools, MLflow tracking, training tricks, hyperparameter search)
↓
Stage 4: Model Compression & Deployment (pruning → INT8 quantization → ONNX)
```
## Repository Structure
```
📁 01_custom_cnn/
├── main.py # Full pipeline: data loading → training → visualization
├── 📁 model/ # SimpleCNN basic block, structure, inspection tools and training loop setup
├── 📁 preprocess/ # data manipulate tools
└── README.md # Design decisions and key findings

📁 02_transfer_learning/
├── efficientnet_finetune.py
└── README.md

📁 03_experiment_tracking/
├── mlflow_tracking.py
└── README.md

📁 04_compression_deployment/
├── pruning.py
├── quantization.py
├── export_onnx.py
├── inference.py
└── README.md

📁 99_flower_data/ # The data for model training

📁 assets/ # Training curves, Grad-CAM, confusion matrix
README.md
requirements.txt
```
## AI Transparency — Who Did What?

This project was built with a clear separation between **personal work** and
**AI-assisted work**.

| Phase      | Personal work                                                                               | AI-assisted work                                                                 |
|-----------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Discover  | Selected the Oxford 102 Flowers dataset, defined the learning goals, and designed the 4-stage roadmap (scratch CNN → EfficientNet → MLflow → compression and deployment). | Asked for high-level learning resources and clarified trade-offs between different CV architectures. |
| Design    | Designed the SimpleCNN architecture, data pipeline, and training strategy (optimizer, schedule, splits). | Used as a writing partner to refine naming, restructure modules, and compare alternative designs in plain language. |
| Develop   | Implemented and debugged all training, preprocessing, and evaluation code; ran experiments; interpreted metrics and failure modes. | Used an LLM to draft some helper utilities (e.g., plotting functions, error-inspection snippets) and to sanity-check edge cases in the training loop. All AI-suggested code was reviewed, modified, and tested before inclusion. |
| Document  | Decided what to disclose, summarized key findings, and curated the learning timeline and benchmarks. | Used to polish English phrasing and improve the clarity of explanations in README. |

**Summary:** AI tools were used for **material search, code suggestions, and documentation support**, 
not as an automatic end-to-end code generator. All model architectures, training logic, and 
experimental decisions were owned, implemented, and validated by the author; the author assumes full responsibility for all code and results, 
including parts initially drafted with AI assistance.
