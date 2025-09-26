# Multi-Input Histology Image Classification with Attention Fusion

This repository contains code for a multi-input deep learning model designed to classify histology images using patches from **multiple magnification levels**. The model leverages **EfficientNet-B0** as a backbone and applies **cross-attention mechanisms** to fuse features from different scales before classification.

The experiments evaluate how the number of input magnifications (1–4) affects performance on a binary classification task (e.g., tumor vs. normal tissue).

---

## 🔍 Overview

- **Task**: Binary classification of histological images.
- **Input**: Multiple image patches per sample (from different resolutions: 256, 512, 1024, 2048 px).
- **Model**: Custom `MultiInputModule` with:
  - EfficientNet-B0 feature extractors.
  - Feature projection and cross-attention fusion.
  - MLP classifier with residual connections.
- **Training**: Stratified splits, class balancing via weighted sampling, early stopping.
- **Evaluation**: Macro-F1 score on held-out test set over multiple random seeds.

---

## 🧱 Model Architecture

The `MultiInputModule` processes each input independently through a shared EfficientNet-B0 backbone, then:

1. Projects features into a shared embedding space.
2. Applies **multi-head attention** where each input attends to all others.
3. Combines attended features and passes them to a final classifier.

This allows the model to learn **inter-scale relationships** (e.g., coarse structure informs fine details).

---

## 📁 Dataset Structure

Expected directory layout:
![Image alt](https://github.com/Ruiu4317/Multi_input_network-Histologia-/blob/main/dataset_structure.png)

> 💡 All patches for a given sample share the same `num_id` (extracted from filename prefix).

---

## ⚙️ Key Features

- **Multi-input support**: Flexible number of inputs (1–4 scales).
- **Augmentation per input**: Independent color jitter, blur, erasing.
- **Stratified splitting**: Ensures balanced distribution across patient-like IDs (`num_id`).
- **Class imbalance handling**: Weighted sampler + loss weighting.
- **Reproducibility**: Multiple runs with fixed random seeds.
- **Early stopping**: Based on validation F1-score.

---

