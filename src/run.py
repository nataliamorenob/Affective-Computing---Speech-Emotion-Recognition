import torch
from Models.DNN import sDNN
from Models.LSTM import LSTMNet
from Models.CNN import CNNModel
from Models.CNN_Attention import CRNN_Attention
from utils import load_data
from utils import train_model, evaluate_model   
import os
import pandas as pd
import numpy as np
from preprocessing.preprocessing import generate_features, merge_features
from metrics import compute_metrics, classification_summary, plot_confusion_matrix


# Create results directories if not exist:
os.makedirs("../results/saved_models", exist_ok=True)
os.makedirs("../results/metrics", exist_ok=True)
os.makedirs("../results/plots", exist_ok=True)
os.makedirs("../Data_preprocessing/preprocessed_mfccs", exist_ok=True)


# Config (this is what you can change):
model_name = "crnn" # options: "dnn" or "lstm"
epochs = 50
lr = 0.001
batch_size = 32
n_mfcc = 56 # 13 or 39 depending on your preprocessing
time_steps = 216 # fixed number of frames
num_classes = 8

# Regularization parameters (increased to combat overfitting):
weight_decay = 0.05  # L2 regularization strength (0.0 = no regularization) - INCREASED
label_smoothing = 0.15  # Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing) - INCREASED
dropout_rate = 0.6  # Dropout rate for models (higher = more regularization) - INCREASED

# Device setup:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing (generation of MFCCs):
raw_data_dir = "/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/archive-5" # this is where your raw data is located (is what it downloads from Kaggle)
processed_dir = "/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/Data_preprocessing/preprocessed_mfccs"

# checking if merged files exist and match the selected feature extraction method:
X_path = os.path.join(processed_dir, f"X_merged_{feature_extraction}.npy")
y_path = os.path.join(processed_dir, f"y_merged_{feature_extraction}.npy")

if not (os.path.exists(X_path) and os.path.exists(y_path)):
    print(f"Generating {feature_extraction.upper()} features...")
    generate_features(raw_data_dir, processed_dir, feature_extraction)
    print("Merging feature files...")
    merge_features(processed_dir, processed_dir, feature_extraction)
else:
    print(f"Found existing {feature_extraction.upper()} dataset. Skipping feature generation.")

# Data loading:
train_loader, test_loader = load_data(data_path=processed_dir, batch_size=batch_size)

# Model selection:
if model_name.lower() == "dnn":
    model = sDNN(input_dim=n_mfcc * time_steps, num_classes=num_classes)
elif model_name.lower() == "lstm":
    model = LSTMNet(n_mfcc=n_mfcc, time_steps=time_steps, num_classes=num_classes)
elif model_name.lower() == "crnn":
    model = CRNN_Attention(n_mfcc=n_mfcc, time_steps=time_steps, num_classes=num_classes, dropout_rate=dropout_rate)
elif model_name.lower() == "cnn":
    model = CNNModel(
        n_features=n_mfcc,      # or 56 if you’re using MFCC+LPC
        time_steps=time_steps,  # 216 if you padded to 216 frames
        num_classes=num_classes
    )
else:
    raise ValueError(f"Unknown model: {model_name}")

model.to(device)
print(f"Training model: {model_name.upper()}\n")

# Training:
train_losses, test_accs = train_model(
    model, train_loader, test_loader,
    epochs=epochs, lr=lr, device=device,
    patience=5, min_delta=0.001,
    weight_decay=weight_decay,
    label_smoothing=label_smoothing
)

# Final evaluation and predictions:
final_acc, y_true, y_pred = evaluate_model(model, test_loader, device)
print(f"Final Test Accuracy: {final_acc:.3f}")

# Save predictions and analyze:
unique_preds, counts = np.unique(y_pred, return_counts=True)
print("Predicted labels:", unique_preds)
print("Counts:", counts)

# Compute metrics:
results = compute_metrics(y_true, y_pred)
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall:    {results['recall']:.3f}")
print(f"F1-score:  {results['f1']:.3f}")

results["model"] = model_name

print(f"Detailed Metrics:")
for k, v in results.items():
    print(f"{k:10s}: {v:.3f}" if isinstance(v, float) else f"{k:10s}: {v}")

metrics_path = "../results/metrics/model_metrics.csv"
pd.DataFrame([results]).to_csv(metrics_path, mode='a', header=not os.path.exists(metrics_path), index=False)
print(f"Metrics saved to: {metrics_path}")

# Classification report (per emotion):
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
plot_path = f"../results/plots/{model_name}_confusion_matrix.png"
classification_summary(y_true, y_pred, labels=emotion_labels)
plot_confusion_matrix(y_true, y_pred, labels=emotion_labels, save_path=plot_path)