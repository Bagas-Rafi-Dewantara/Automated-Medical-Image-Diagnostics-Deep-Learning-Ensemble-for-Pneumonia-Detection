# 🩻 Pneumonia Classification: Deep Learning Ensemble with EfficientNet

![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)
![Model](https://img.shields.io/badge/Model-EfficientNet--B2-blue)
![Medical AI](https://img.shields.io/badge/Domain-Medical%20Imaging-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Overview

Diagnosing pneumonia variants from Chest X-Rays is a high-stakes task where precision is critical. This project implements a **Deep Learning Classification Pipeline** designed to detect pneumonia variants with high sensitivity and robustness.

To overcome the challenges of limited medical data and overfitting, this solution utilizes **Transfer Learning (EfficientNet-B2)** combined with a **Stratified 5-Fold Ensemble** strategy. Instead of relying on a single model, the system aggregates predictions from multiple models to achieve a highly generalized performance (**94% Macro F1-Score**).



## 🚀 Key Features

* **Ensemble Inference Mechanism:**
    * Implemented a "Committee of Machines" approach. The final prediction is the weighted average of 5 independent models trained on different data folds (Stratified K-Fold).
    * This significantly reduces prediction variance and ensures the model generalizes well to unseen patient data.
* **State-of-the-Art Architecture:**
    * Leverages **EfficientNet-B2** as the backbone, selected for its superior balance between parameter efficiency and feature extraction capability compared to ResNet or VGG.
* **Training Stability & Optimization:**
    * **Cosine Annealing Scheduler:** Dynamically adjusts learning rates to help the model converge to better global minima.
    * **Label Smoothing:** Applied to the loss function to prevent the model from becoming over-confident (calibration), a crucial step in medical imaging to reduce overfitting.


## 🛠️ Tech Stack

* **Core Framework:** `PyTorch`, `Torchvision`
* **Data Processing:** `Pandas`, `NumPy`, `Scikit-Learn` (StratifiedKFold)
* **Image Augmentation:** `Albumentations` / `Torchvision Transforms`
* **Visualization:** `Matplotlib`, `Seaborn`


## ⚙️ Methodology

### 1. Data Preprocessing
* Images are resized to **224x224** (or appropriate input size for EfficientNet).
* Normalization applied using ImageNet mean and standard deviation.
* **Stratified K-Fold Split:** The dataset is divided into 5 folds, ensuring each fold maintains the same percentage of pneumonia classes as the original dataset.

### 2. Model Architecture
We fine-tuned a pre-trained **EfficientNet-B2**. The classifier head was modified to match the number of target classes:
```python
model = models.efficientnet_b2(pretrained=True)
model.classifier[1] = nn.Linear(in_features=1408, out_features=num_classes)
````

### 3\. Ensemble Inference Strategy

During the testing phase, the input X-Ray is passed through all 5 trained models (Fold 0 to Fold 4). The final class probability is calculated as the mean:

$$P_{final} = \frac{1}{5} \sum_{i=1}^{5} P_{model\_i}$$

This technique minimizes the risk of a single model "memorizing" specific noise in the training data.


## 📊 Results

The model was evaluated using **Macro F1-Score** to ensure balanced performance across all classes.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Macro F1-Score** | **\~94%** | Indicates high reliability across minority and majority classes. |
| **Accuracy** | **High** | Consistent across 5 folds. |

*(Place your Loss Curves or Confusion Matrix screenshot here)*
