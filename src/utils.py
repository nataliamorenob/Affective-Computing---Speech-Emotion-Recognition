import torch
import torch.nn as nn
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path="../Data", batch_size=32, test_size=0.3, random_state=42):
    """Load preprocessed data from npy files and return DataLoaders.

    Expects files named `x_merged_new.npy` and `y_merged_new.npy` in `data_path`.
    """
    # filenames produced by preprocessing are lowercase
    X = np.load(f"{data_path}/x_merged_new.npy")
    y = np.load(f"{data_path}/y_merged_new.npy")

    # Split data:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler on TRAIN:
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    # Reshape back:
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # Convert to tensors:
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # DataLoaders:
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=batch_size
    )

    return train_loader, test_loader


def train_model(model, train_loader, test_loader,
                epochs=50, lr=0.001, device=None,
                patience=10, min_delta=0.001,
                weight_decay=0.0, label_smoothing=0.0):
    """Train the model with optional weight decay and label smoothing.

    Returns:
        train_losses, test_accuracies, best_epoch, stopped_early
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss: support optional label smoothing if available in this PyTorch version
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing and label_smoothing > 0 else nn.CrossEntropyLoss()
    except TypeError:
        # Older PyTorch versions may not support label_smoothing kwarg
        if label_smoothing and label_smoothing > 0:
            print("Warning: label_smoothing requested but not supported by this PyTorch version. Ignoring.")
        criterion = nn.CrossEntropyLoss()

    # Optimizer with optional weight decay (L2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, test_accuracies = [], []

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

        # Early Stopping Logic:
        if test_acc > best_acc + min_delta:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0  # reset counter
            best_model_state = model.state_dict()  # save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Test Acc: {best_acc:.3f}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)  # restore best model
            stopped_early = True
            break

    # Restore best weights if early stopping wasn’t triggered but best found
    if not stopped_early and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, test_accuracies, best_epoch, stopped_early


def evaluate_model(model, data_loader, device):
    """Evaluate the model on a dataset and return accuracy (float).

    This function returns only accuracy to match existing callers in `run.py`.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total if total > 0 else 0.0
