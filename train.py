"""
Phase 4: Training & Evaluation Pipeline with Early Stopping

This module handles:
1. EarlyStopping to halt training when validation loss stops improving.
2. train_one_epoch: Training loop over one epoch with backpropagation.
3. validate: Evaluation loop over test/validation data.
4. train_model: Master training loop managing epochs, metrics, and checkpointing.
"""

import os
import time
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early Stopping halts training when validation loss does not improve
    after a specified number of consecutive epochs (patience).
    Also saves the model weights with the lowest validation loss.
    """
    def __init__(self, patience=5, delta=0.001, save_path="best_model.pth"):
        """
        Parameters:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            save_path (str): Filepath to save the best model weights.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        # First epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        # Improvement detected
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter
        # No significant improvement
        else:
            self.counter += 1
            print(f"  --> EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model state dictionary when validation loss improves."""
        torch.save(model.state_dict(), self.save_path)
        print(f"  --> Validation loss improved to {val_loss:.4f}. Checkpoint saved to '{self.save_path}'.")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one training epoch.
    
    Parameters:
        model: PyTorch neural network model.
        dataloader: Training DataLoader.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Optimizer (e.g. Adam, SGD).
        device: 'cuda' or 'cpu'.
        
    Returns:
        epoch_loss (float): Average training loss for the epoch.
        epoch_acc (float): Training accuracy percentage.
    """
    model.train()  # Enable dropout and batch norm training mode
    
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Zero out existing gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 3. Backward pass (compute gradients)
        loss.backward()

        # 4. Update weights
        optimizer.step()

        # 5. Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation/test set.
    
    Parameters:
        model: PyTorch neural network model.
        dataloader: Validation/Test DataLoader.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.
        
    Returns:
        val_loss (float): Average validation loss.
        val_acc (float): Validation accuracy percentage.
    """
    model.eval()  # Disable dropout and freeze batch norm stats
    
    running_loss = 0.0
    correct = 0
    total = 0

    # No gradients needed for validation (saves memory and compute)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path="best_model.pth"):
    """
    Orchestrates the entire training loop over all epochs with early stopping.
    
    Parameters:
        model: CustomCNN model instance.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for test/validation set.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs (int): Maximum number of training epochs.
        device: 'cuda' or 'cpu'.
        save_path (str): Filename where best model weights are stored.
        
    Returns:
        model: Model loaded with the best checkpoint weights.
        history: Dictionary containing loss and accuracy records.
    """
    early_stopping = EarlyStopping(patience=5, delta=0.001, save_path=save_path)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"\n{'=' * 70}")
    print(f"Starting Training: {epochs} Max Epochs | Device: {device}")
    print(f"{'=' * 70}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train and validate for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print formatted epoch summary
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Check early stopping condition
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n[!] Early stopping triggered at epoch {epoch}. Stopping training.")
            break

    # Load best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"\n[✓] Loaded best model checkpoint from '{save_path}' with Val Loss: {early_stopping.best_loss:.4f}")

    return model, history
