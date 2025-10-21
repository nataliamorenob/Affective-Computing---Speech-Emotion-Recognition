import os
import torch
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# === Local Imports ===
from Models.DNN import sDNN
from Models.LSTM import LSTMNet
from Models.CNN import CNNModel
from Models.CNN_Attention import CRNN_Attention
from utils import load_data, train_model, evaluate_model
from preprocessing.preprocessing import generate_features, merge_features
from metrics import get_predictions, compute_metrics, plot_confusion_matrix, classification_summary


# === Logging Setup ===
LOGS_DIR = os.path.join(".", "src", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_log_dir = os.path.join(LOGS_DIR, timestamp)
os.makedirs(run_log_dir, exist_ok=True)
log_filename = os.path.join(run_log_dir, "logs_results.log")

# Configure logging to file only (we will still print to terminal)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_filename)]
)

def log_and_print(message, level="info"):
    print(message)
    if level == "error":
        logging.error(message)
    else:
        logging.info(message)

log_and_print(f"Starting training pipeline...")
log_and_print(f"Logs and results will be saved in: {run_log_dir}\n")


# === Directory Setup ===
PROCESSED_DIR = os.path.join(".", "src", "mfcc_lpc_data")

dirs = [PROCESSED_DIR]
for d in dirs:
    os.makedirs(d, exist_ok=True)
log_and_print("Required directories verified or created.\n")


# === Configuration ===
model_name = "cnn"  # Options: "dnn", "lstm", "crnn", "cnn"
epochs = 2
lr = 0.001
batch_size = 32
n_mfcc = 56
time_steps = 216
num_classes = 8
weight_decay = 0.05       # L2 regularization strength
label_smoothing = 0.15    # Softens hard labels for smoother gradients
dropout_rate = 0.6        # Dropout rate for models that support it
feature_extraction = "mfcc_lpc"  # descriptor for metrics and logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_and_print("Model Configuration:")
log_and_print(f"Model: {model_name.upper()}")
log_and_print(f"Epochs: {epochs}")
log_and_print(f"Learning rate: {lr}")
log_and_print(f"Batch size: {batch_size}")
log_and_print(f"n_mfcc: {n_mfcc}, time_steps: {time_steps}")
log_and_print(f"Feature extraction: {feature_extraction}")
log_and_print(f"Device: {device}\n")


# === Feature Preprocessing ===
input_dir = "./data"
X_path = os.path.join(PROCESSED_DIR, "x_merged_new.npy")
y_path = os.path.join(PROCESSED_DIR, "y_merged_new.npy")

if not (os.path.exists(X_path) and os.path.exists(y_path)):
    log_and_print("Generating MFCC + LPC features...")
    generate_features(input_dir, PROCESSED_DIR)
    log_and_print("Merging feature files...")
    merge_features(PROCESSED_DIR, PROCESSED_DIR)
    log_and_print("Feature generation and merging completed.\n")
else:
    log_and_print("Found existing merged dataset. Skipping feature generation.\n")


# === Data Loading ===
log_and_print("Loading data...")
train_loader, test_loader = load_data(data_path=PROCESSED_DIR, batch_size=batch_size)
log_and_print("Data loaded successfully.\n")


# === Model Selection ===
model_name_lower = model_name.lower()

if model_name_lower == "dnn":
    model = sDNN(n_mfcc=n_mfcc * time_steps, num_classes=num_classes)
elif model_name_lower == "lstm":
    model = LSTMNet(n_mfcc=n_mfcc, time_steps=time_steps, 
                    num_classes=num_classes)
elif model_name.lower() == "crnn":
    model = CRNN_Attention(n_mfcc=n_mfcc, 
                           time_steps=time_steps, num_classes=num_classes, 
                           dropout_rate=dropout_rate)
elif model_name_lower == "cnn":
    model = CNNModel(n_features=n_mfcc, time_steps=time_steps, 
                     num_classes=num_classes)
else:
    raise ValueError(f"Unknown model name: {model_name}")

model.to(device)
log_and_print(f"Training model: {model_name.upper()}\n")


# === Training ===
log_and_print("Starting training...")
# Accept either form to remain compatible.
train_result = train_model(
    model, train_loader, test_loader,
    epochs=epochs, lr=lr, device=device,
    patience=5, min_delta=0.001,
    weight_decay=weight_decay,
    label_smoothing=label_smoothing
)


# Unpack safely depending on returned tuple length
if isinstance(train_result, tuple) and len(train_result) == 4:
    train_losses, test_accs, best_epoch, stopped_early = train_result
elif isinstance(train_result, tuple) and len(train_result) == 2:
    train_losses, test_accs = train_result
    best_epoch = None
    stopped_early = False
else:
    # Fallback: try to interpret as list-like
    try:
        train_losses = train_result[0]
        test_accs = train_result[1]
    except Exception:
        raise ValueError("train_model returned unexpected result; expected (train_losses, test_accs[, best_epoch, stopped_early])")

# Per-epoch logging
log_and_print("Per-epoch results:")
for idx, (t_loss, t_acc) in enumerate(zip(train_losses, test_accs), start=1):
    log_and_print(f"Epoch {idx:02d} | Train Loss: {t_loss:.4f} | Test Acc: {t_acc:.3f}")

if stopped_early and best_epoch is not None:
    log_and_print(f"Early stopping triggered at epoch {best_epoch}.")
elif stopped_early:
    log_and_print("Early stopping triggered (best epoch not provided by train_model).")
else:
    log_and_print(f"Training completed all {epochs} epochs.")

log_and_print("Training complete.\n")


# === Evaluation ===
final_acc = evaluate_model(model, test_loader, device)
log_and_print(f"Final Test Accuracy: {final_acc:.3f}\n")

# === Metrics & Analysis ===
y_true, y_pred = get_predictions(model, test_loader, device)

# === Save Model ===
model_save_path = os.path.join(run_log_dir, f"{model_name}_model.pth")
torch.save(model.state_dict(), model_save_path)
log_and_print(f"Model saved to: {model_save_path}\n")

unique_preds, counts = np.unique(y_pred, return_counts=True)

log_and_print("Prediction Summary:")
for u, c in zip(unique_preds, counts):
    log_and_print(f"  Label {u}: {c} samples")

results = compute_metrics(y_true, y_pred)
results.update({
    "model": model_name,
    "feature_extraction": feature_extraction,
    "final_accuracy": final_acc,
    "epochs": epochs,
    "learning_rate": lr,
    "batch_size": batch_size,
    "weight_decay": weight_decay,
    "label_smoothing": label_smoothing,
    "dropout_rate": dropout_rate,
    "best_epoch": best_epoch,
    "early_stopping": stopped_early,
    "timestamp": timestamp
})


log_and_print("\nModel Performance Metrics:")
for k, v in results.items():
    if isinstance(v, (int, float)):
        log_and_print(f"{k:15s}: {v:.3f}")
    else:
        log_and_print(f"{k:15s}: {v}")

# Save metrics CSV in the same run folder
metrics_csv_path = os.path.join(run_log_dir, "model_metrics.csv")
pd.DataFrame([results]).to_csv(metrics_csv_path, index=False)
log_and_print(f"\nMetrics saved to: {metrics_csv_path}\n")


# === Classification Report & Confusion Matrix ===
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 
                  'angry', 'fearful', 'disgust', 'surprised']
confusion_matrix_path = os.path.join(run_log_dir, f"{model_name}_confusion_matrix.png")

log_and_print("Generating classification report and confusion matrix...")
classification_summary(y_true, y_pred, labels=emotion_labels)
plot_confusion_matrix(y_true, y_pred, 
                      labels=emotion_labels, save_path=confusion_matrix_path)
log_and_print(f"Confusion matrix saved to: {confusion_matrix_path}\n")


log_and_print("All tasks completed successfully!")
log_and_print("=" * 80)
