# Blood-Cancer-Classification-B-cell-leukemia

This repository contains the source code for the research project **"A Deep Learning Approach to Blood Cancer Classification Using ConvNeXt Transfer Learning."** The project leverages the ConvNeXt Tiny model with transfer learning for the classification of blood cancer subtypes.

## Overview

Blood cancer, specifically acute lymphoblastic leukemia (ALL), is a critical health issue. Accurate and timely diagnosis is essential for better treatment outcomes. This project introduces a deep learning-based automated classification system that identifies four categories of blood cell images:

- Benign
- Early Pre-B
- Pre-B
- Pro-B

---

### Key Features:
- **Transfer Learning:** Utilized the ConvNeXt Tiny model pre-trained on ImageNet.
- **Image Augmentation:** Enhanced dataset variability with techniques like random flipping, rotation, and brightness adjustment.
- **Performance Metrics:** Achieved a test accuracy of **98.36%**, with detailed evaluation using precision, recall, and F1-score.


## Dataset

The dataset, **Blood Cell Cancer ALL-4Class Dataset**, was sourced from Kaggle: [Blood Cell Cancer ALL-4Class Dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class). 

It consists of 3,242 images divided into:

- Training set: 70% (2,268 images)
- Validation set: 15% (486 images)
- Testing set: 15% (488 images)

Each image underwent preprocessing (resizing, normalization) and augmentation to improve model generalization.

---

## Methodology

1. **Preprocessing:** Images resized to 224x224 pixels, normalized, and augmented.
2. **Model Architecture:**
   - Base Model: ConvNeXt Tiny
   - Custom Layers: Added fully connected layers and a softmax classifier for multi-class classification.
3. **Training:**
   - Optimizer: Adam
   - Loss Function: Cross-Entropy
   - Epochs: 25
4. **Evaluation:** Accuracy, precision, recall, and F1-score were computed for all categories.

---

## Results

The ConvNeXt Tiny model outperformed other architectures such as ResNet152v2, VGG19, and DenseNet201 in this classification task. The results for the four categories are as follows:

| Category      | Recall | Precision | F1-Score |
|---------------|--------|-----------|----------|
| Benign        | 0.92   | 0.99      | 0.95     |
| Early Pre-B   | 1.00   | 0.97      | 0.98     |
| Pre-B         | 0.99   | 0.99      | 0.99     |
| Pro-B         | 1.00   | 0.99      | 1.00     |

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/blood-cancer-classification.git
   cd blood-cancer-classification
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure CUDA and cuDNN are properly configured for GPU support.

---


## Future Work

- **Model Ensembling:** Combining multiple models for improved accuracy.
- **Dataset Expansion:** Incorporating additional imaging modalities for greater robustness.


## Acknowledgments

- **Dataset:** [Blood Cell Cancer ALL-4Class Dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)
- **Framework:** [PyTorch](https://pytorch.org/)
