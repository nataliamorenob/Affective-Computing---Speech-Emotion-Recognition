import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# COMPUTE METRICS:
def compute_metrics(y_true, y_pred, average="weighted"):
    """
    Compute key classification metrics given true and predicted labels.
    Args:
        y_true (array): True class labels
        y_pred (array): Predicted class labels
        average (str): Averaging method for multi-class (default: 'weighted')
    Returns:
        dict: metrics dictionary
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics



# CONFUSION MATRIX:
def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None):
    """
    Display a confusion matrix as a heatmap, and optionally save it to file.
    Args:
        y_true (array): True class labels
        y_pred (array): Predicted class labels
        labels (list): Label names for display
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot (e.g., '../results/plots/model_cm.png')
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()

    # Save the figure if path provided
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()




# CLASSIFICATION REPORT:
def classification_summary(y_true, y_pred, labels=None):
    """
    Print a detailed classification report (per-class precision, recall, F1).
    """
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


# PREDICTIONS FROM MODEL:
def get_predictions(model, data_loader, device):
    """
    Run inference on a data loader and return predicted + true labels.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return np.array(y_true), np.array(y_pred)