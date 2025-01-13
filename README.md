# Flower Species Classification and Car Detection using Deep Learning

This project leverages deep learning techniques to classify flower species and detect cars in images. It includes Jupyter notebooks for training (`Train.ipynb`) and testing (`Test.ipynb`) models that use convolutional neural networks (CNNs) and transfer learning.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Models](#training-the-models)
  - [Testing the Models](#testing-the-models)
- [Features](#features)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## About the Project

This project encompasses two tasks:

1. **Flower Species Classification**
   - Uses the MobileNetV2 model pre-trained on ImageNet for transfer learning.
   - Classifies images into multiple flower species based on hierarchical features.

2. **Car Detection**
   - Employs custom CNN architectures for bounding box regression.
   - Detects cars in images by predicting bounding boxes around detected objects.

Both tasks involve data preprocessing, model training, and evaluation. The `Test.ipynb` notebook includes metrics like accuracy, precision, recall, F1-score, confusion matrix (for classification), and Intersection over Union (IoU, for detection).

---

## Getting Started

### Prerequisites

Ensure the following libraries are installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- OpenCV

To install all dependencies:

```bash
pip install -r requirements.txt
```
Clone the repository:

```bash
git clone [https://github.com/your-repo/project-flower-car-detection.git](https://github.com/UF-MLforAISystems-Fall24/project-3-graduate-varunvemuri-143.git)

```

## Usage

### Training the Models

1. Open the `Train.ipynb` notebook in Jupyter:

   ```bash
   jupyter notebook Train.ipynb
   
2. Follow the steps to load the datasets, preprocess images, and train the models:
   - **Flower species classification**: Uses MobileNetV2 with fine-tuning.
   - **Car detection**: Employs two CNN models for bounding box regression.

3. Save the trained models. Pre-trained models can also be downloaded:

   **OneDrive Link for Pre-trained Models:** [Download Models](https://uflorida-my.sharepoint.com/:f:/g/personal/va_vemuri_ufl_edu/EldEulXgG8tPnKSlzHBCrLUBSr96GzRviVVG1UVm5VJKQg?e=19WMVW)

4. Hyperparameter tuning is implemented using techniques like:
   - `GridSearchCV` for classification
   - Callbacks (e.g., `EarlyStopping`) for detection.

### Testing the Models

1. Open the `Test.ipynb` notebook in Jupyter:

   ```bash
   jupyter notebook Test.ipynb
   
## Features

- **Flower Species Classification**: Multi-class classification using transfer learning with MobileNetV2.
- **Car Detection**: Bounding box regression with custom CNNs.
- **Dimensionality Reduction**: For feature extraction and efficient computation.
- **Model Evaluation**: Metrics for classification and object detection.
- **Visualization**: Bounding boxes for car detection and class predictions for flowers.
- **Efficient Training**: Callbacks and fine-tuning to optimize training time and performance.

---

## Contact

Name – [Varun Vemuri – varunvemuri@ufl.edu](mailto:varunvemuri@ufl.edu)

Project Link: [https://github.com/UF-MLforAISystems-Fall24/project-2-graduate-varunvemuri-143.git](https://github.com/UF-MLforAISystems-Fall24/project-2-graduate-varunvemuri-143.git)

---

## Acknowledgements

**Libraries**

-TensorFlow
-scikit-learn
-OpenCV
-pandas
-matplotlib







