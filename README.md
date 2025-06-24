# DS-IB Net: Ultra-Lightweight Weakly-Supervised Video Anomaly Detection through Synergistic Dual Streams and Information Bottleneck

[![Paper-PDF](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/YOUR_PAPER_LINK) This is the official PyTorch implementation for the paper **"DS-IB Net: Ultra-Lightweight Weakly-Supervised Video Anomaly Detection through Synergistic Dual Streams and Information Bottleneck"**.

DS-IB Net is a lightweight, dual-stream framework based on the Information Bottleneck (IB) theory, designed for **Weakly-Supervised Video Anomaly Detection (WS-VAD)**. It aims to effectively model normal temporal dynamics and sensitively detect sparse anomalies while maintaining low computational costs for practical applications.

## Abstract

Weakly-Supervised Video Anomaly Detection (WS-VAD) primarily struggles to robustly model normal temporal dynamics and sensitively detect sparse anomalies using only video-level labels. Current methods often inadequately represent normal patterns or are too computationally costly due to complex architectures, hindering practical use. We introduce **DS-IB Net**, a lightweight, dual-stream Information Bottleneck (IB) based framework. Its core **Dual-Output Information Bottleneck Encoder (DOIBE)**, driven by a **Temporal Information Bottleneck Loss (TIBL)**, operationalizes IB. DOIBE produces a predictive representation ($z_{pred}$) for dynamic modeling and a compression control vector ($z_{comp}$) for redundancy compression, with TIBL assigning them distinct functional roles. Since IB compression in DOIBE discards fine-grained appearance details, a parallel **Instantaneous Mode Monitoring Path (PMP)** provides crucial compensation by extracting these neglected frame-level features. An **Adaptive Feature Arbitration (AFA)** module dynamically arbitrates information from both streams for precise anomaly localization. Extensive experiments on mainstream benchmarks (UCF-Crime, ShanghaiTech) demonstrate that DS-IB Net outperforms state-of-the-art methods while remaining ultra-lightweight.

## Key Features

- **Ultra-Lightweight**: A clean and efficient model design with low computational overhead, making it easy to deploy.
- **Synergistic Dual-Stream**: Combines an information bottleneck stream for temporal dynamics modeling with a monitoring path to compensate for appearance details, synergistically improving detection accuracy.
- **Information Bottleneck Theory**: Innovatively applies the IB principle via a Dual-Output Encoder (DOIBE) and a Temporal IB Loss (TIBL) to effectively distinguish and process predictive vs. redundant information.
- **Adaptive Fusion**: Intelligently fuses features from the dual streams using an Adaptive Feature Arbitration (AFA) module for more precise anomaly scoring.
- **State-of-the-Art Performance**: Achieves leading performance on the UCF-Crime and ShanghaiTech datasets.

## Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/DS-IB-Net.git](https://github.com/YOUR_USERNAME/DS-IB-Net.git)
    cd DS-IB-Net
    ```

2.  Create and activate a Conda environment (recommended):
    ```bash
    conda create -n dsib_env python=3.8
    conda activate dsib_env
    ```



## Dataset Preparation

This project supports the **UCF-Crime** and **ShanghaiTech** datasets.

1.  **Feature Extraction**:
    You need to pre-extract VIT features from the videos. Place the extracted feature files (typically in `.npy` format) into the corresponding dataset directory.

2.  **Directory Structure**:
    Please organize your dataset files according to the following structure, which the code expects for reading features and annotations:

    ```
    DS-IB/
    ├── dataset/
    │   ├── ucf-crime/
    │   │   ├── test_split_10crop.txt
    │   │   ├── train_split_10crop.txt
    │   │   ├── GT/
    │   │   │   └── video_label_10crop.txt
    │   │   └── # ... other feature and annotation files
    │   │
    │   └── shanghaitech/
    │       ├── test_split.txt
    │       ├── train_split.txt
    │       ├── GT/
    │       │   ├── frame_label.txt
    │       │   └── video_label.txt
    │       └── # ... other feature and annotation files
    │
    └── method/
        └── # ... source code
    ```

## Model Training

You can train the model using the `train_icic.py` script. All training arguments can be viewed and modified in the `options.py` file.

**Training Example:**

```bash
python method/train_icic.py \
    --dataset-name shanghaitech \
    --data-path ./dataset/shanghaitech/ \
    --model-name DS_IB \
    --max-epoch 50 \
    --batch-size 64 \
    --lr 0.0002 \
    --gpus 0
```

### Key Arguments (`options.py`)

-   `--dataset-name`: Name of the dataset to use (`ucf-crime` or `shanghaitech`).
-   `--data-path`: Root directory of the dataset.
-   `--model-name`: Name of the model to train (e.g., `DS_IB`).
-   `--max-epoch`: The maximum number of training epochs.
-   `--batch-size`: The training batch size.
-   `--lr`: The learning rate.
-   `--gpus`: The GPU ID(s) to use (e.g., `0` or `0,1`).
-   `--num-workers`: The number of workers for data loading.

## Model Testing & Evaluation

Use the `test.py` script to evaluate a trained model on the test set. You must provide the path to the trained model checkpoint (`--ckpt-path`).

**Testing Example:**

```bash
python method/test.py \
    --dataset-name shanghaitech \
    --data-path ./dataset/shanghaitech/ \
    --model-name DS_IB \
    --ckpt-path ./ckpt/YOUR_CHECKPOINT.pth \
    --gpus 0
```

The evaluation script (e.g., `eval_10crop_12_28.py`) will be called automatically by `test.py` to compute performance metrics like AUC and save the results as a `.npy` file.
