# Affective Computing — Speech Emotion Recognition

This project implements Speech Emotion Recognition (SER) using multiple deep learning architectures — CNN, LSTM, DNN, and CRNN with Attention — trained on audio features (MFCCs and LPCs) extracted from speech recordings.

## Overview
- The goal is to classify speech audio samples into eight emotion categories:
neutral, calm, happy, sad, angry, fearful, disgust, surprised
- The pipeline automatically handles:
    - Preprocessing — Extracts MFCC and LPC features from .wav files.
    - Merging — Combines all feature files into NumPy arrays (X_merged_new.npy, y_merged_new.npy).
    - Training — Trains one of several neural architectures (DNN, CNN, LSTM, or CRNN-Attention).
    - Evaluation — Computes accuracy, precision, recall, F1, and confusion matrices.

## Project Structure
```
Affective-Computing---Speech-Emotion-Recognition/
│
├── README.md                        # Project documentation
├── requirements.txt                 # Environment dependencies
│
├── Data_preprocessing/              # Generated preprocessed features
│   └── preprocessed_mfccs/          # Saved MFCC + LPC .npy files
│
├── results/                         # Training results
│   ├── saved_models/                # Saved model weights (.pth)
│   ├── metrics/                     # CSV files with accuracy, F1, etc.
│   └── plots/                       # Confusion matrix and other plots
│
├── Visualization.ipynb              # interactive notebook for exploration and demo 
└── src/
    ├── Models/                      # Neural network architectures
    │   ├── CNN.py                   # Convolutional Neural Network
    │   ├── LSTM.py                  # LSTM model
    │   ├── DNN.py                   # Feedforward Dense Neural Net
    │   └── CNN_Attention.py         # CRNN model with attention mechanism
    │
    ├── preprocessing/
    │   └── preprocessing.py         # MFCC + LPC extraction and merging
    │
    ├── utils.py                     # Data loading, scaling, training, early stopping
    ├── metrics.py                   # Metrics computation and confusion matrix plotting
    └── run.py                       # Main training and evaluation script
```

## Virtual environment (Requirements)

Create and activate a virtual environment with the following commands:

```bash
conda create -n AffComp python=3.10
conda activate AffComp
pip install -r requirements.txt
```
## Download the dataset
This project uses the RAVDESS dataset (or similar emotion-labeled speech corpus) --> https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data.
Place it in:
```bash
Affective-Computing---Speech-Emotion-Recognition/archive-5/
```
Each actor should have a subfolder with .wav files, e.g.:
```
archive-5/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
│   └── ...
```
## Running the project
```bash
cd src
python run.py
```

## Configurable parameters
In run.py you can find at the top a Config section with all the parameters which can be modified. Example:
```bash
model_name = "cnn" # Options: "dnn", "lstm", "cnn", "crnn"
feature_extraction = "mfcc" # Options: "mfcc" or "mfcc_lpc"
```
## Project Information

This project was developed as part of the course **Affective Computing** at the **University of Oulu** during the Fall semester of 2025.

## Contributors
- Sania Khan Tareen, Ece Merve Gelmez, Furkancan Özdemir, Iniyan Nachimuthu, Md Sayem Khandaker, Natalia Moreno Blasco.