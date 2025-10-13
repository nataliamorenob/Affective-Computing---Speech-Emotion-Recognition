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


# Create results directories if not exist:
os.makedirs("../results/saved_models", exist_ok=True)
os.makedirs("../results/metrics", exist_ok=True)
os.makedirs("../results/plots", exist_ok=True)


# Config (this is what you can change):
model_name = "crnn" # options: "dnn" or "lstm"
epochs = 200
lr = 0.001
batch_size = 32
n_mfcc = 39 # 13 or 39 depending on your preprocessing
time_steps = 128 # fixed number of frames
num_classes = 8

# Device setup:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data loading:
train_loader, test_loader = load_data(data_path="../Data", batch_size=batch_size)

# Model selection:
# Model selection
if model_name.lower() == "dnn":
    model = sDNN(input_dim=n_mfcc * time_steps, num_classes=num_classes)
elif model_name.lower() == "lstm":
    model = LSTMNet(n_mfcc=n_mfcc, time_steps=time_steps, num_classes=num_classes)
elif model_name.lower() == "crnn":
    model = CRNN_Attention(n_mfcc=n_mfcc, time_steps=time_steps, num_classes=num_classes)
else:
    raise ValueError(f"Unknown model: {model_name}")

model.to(device)
print(f"Training model: {model_name.upper()}\n")

# Training:
train_losses, test_accs = train_model(
    model, train_loader, test_loader,
    epochs=epochs, lr=lr, device=device,
    patience=10, min_delta=0.001
)

# Final evaluation (optional but recommended):
final_acc = evaluate_model(model, test_loader, device)
print(f"Final Test Accuracy: {final_acc:.3f}")

# Save trained model:
model_path = f"../results/saved_models/{model_name}_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Training complete. Model saved as {model_name}_model.pth")

from metrics import get_predictions, compute_metrics, plot_confusion_matrix, classification_summary

# After training:
y_true, y_pred = get_predictions(model, test_loader, device)


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