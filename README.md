# Underwater Audio Classification: HHT Pipeline & 1D ResNet

This project classifies underwater acoustic signals (e.g., ambient noise, cargo ships, passenger ships, tankers, and tugboats) using a hybrid signal processing and deep learning approach. It combines the Hilbert-Huang Transform (HHT) pipeline for robust feature extraction with a 1D ResNet architecture for classification.

## Project Architecture

The pipeline is split into three main stages: data preprocessing, multi-class training, and binary testing/evaluation.

1. **`HHT.py` (Data Acquisition & Preprocessing)**
   * Downloads the underwater audio dataset via Kaggle.
   * Processes the raw `.wav` files using DEMON (Detection of Envelope Modulation on Noise), BTWD pre/post-filtering, Uniform Phase Empirical Mode Decomposition (UPEMD), and Variational Mode Decomposition (VMD).
   * Extracts statistical and entropy-based features.
   * Outputs the processed features and labels (5 classes) to `submarine_dataset.csv`.

2. **`sv_evaluate.py` (Model Training)**
   * Ingests `submarine_dataset.csv` and splits the data into training (80%) and testing (20%) sets.
   * Exports `train_set.csv` and `test_set.csv` for reproducibility.
   * Trains a 1D ResNet model (`ResNet1D`) to classify the audio into 5 distinct categories.
   * Outputs training visualizations, confusion matrices, and saves the trained model weights as `optimized_resnet1d_100ep.pth`.

3. **`test.py` (Binary Evaluation)**
   * Clones the necessary `resnet1d` repository if it does not exist locally.
   * Loads `test_set.csv` and the pre-trained weights from `optimized_resnet1d_100ep.pth`.
   * Fine-tunes the dense layer of the network for **Binary Classification** (Vessel vs. No Vessel).
   * Outputs a detailed performance report, including an optimized threshold, Normalized Confusion Matrix, and ROC/AUC curves.

## Installation & Setup

1. **Clone this repository** (or download the source files to a single directory).
2. **Install the required dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt

## Sequence of running
1. python HHT.py
2. python sv_evaluate.py
3. python test.py
