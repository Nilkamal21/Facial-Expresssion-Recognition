"""
Phase 6: Comprehensive Model Evaluation & Metrics Reporting

This module computes and prints:
1. Per-Class Precision, Recall, F1-Score, and Support.
2. Formatted 7x7 Confusion Matrix.
3. Key Observations (best emotion, hardest emotion, top confusion).

It can be:
- Imported and called directly inside main.py after training.
- Run as a standalone script on any saved .pth checkpoint:
    python evaluate.py --model best_model_epochs30_bs64_lr0.001_dropout0.25_adam.pth
"""

import os
import argparse
import torch
import torch.nn as nn

from dataset import get_data_loaders, EMOTION_DICT
from model import CustomCNN


def evaluate_model(model, dataloader, device):
    """
    Runs model inference across the dataloader and collects all
    ground-truth labels and predicted labels.
    
    Parameters:
        model: Trained PyTorch CustomCNN model.
        dataloader: Test/Validation DataLoader.
        device: 'cuda' or 'cpu'.
        
    Returns:
        y_true (list of int): Actual emotion labels.
        y_pred (list of int): Predicted emotion labels.
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


def compute_and_print_report(y_true, y_pred, emotion_dict=EMOTION_DICT):
    """
    Calculates and prints the full classification metrics table,
    confusion matrix, and key observations in clean ASCII format.
    
    Parameters:
        y_true (list of int): Ground-truth labels (0 to 6).
        y_pred (list of int): Predicted labels (0 to 6).
        emotion_dict (dict): Mapping from label index to emotion name.
    """
    num_classes = len(emotion_dict)
    total_samples = len(y_true)

    # 1. Build the Confusion Matrix [num_classes x num_classes]
    # Rows: Actual class, Columns: Predicted class
    cm = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    # 2. Compute Per-Class Metrics: Precision, Recall, F1-Score
    metrics = []
    total_correct = 0

    for i in range(num_classes):
        tp = cm[i][i]  # True Positives: diagonal
        total_correct += tp

        # False Positives: sum of column i minus diagonal
        fp = sum(cm[row][i] for row in range(num_classes)) - tp

        # False Negatives: sum of row i minus diagonal
        fn = sum(cm[i][col] for col in range(num_classes)) - tp

        support = tp + fn  # Total actual samples for this class

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics.append({
            "name": emotion_dict.get(i, f"Class {i}"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })

    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    macro_f1 = sum(m["f1"] for m in metrics) / num_classes

    # 3. Print Section 1: Classification Metrics Table
    print("\n" + "=" * 80)
    print("                FINAL MODEL EVALUATION REPORT (RAF-DB TEST SET)")
    print("=" * 80)
    print("\n1. CLASSIFICATION METRICS PER EMOTION:")
    print("-" * 80)
    print(f"{'Emotion':<18} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Total Samples':>16}")
    print("-" * 80)
    for m in metrics:
        print(f"{m['name']:<18} {m['precision']:>11.1f}% {m['recall']:>11.1f}% {m['f1']:>11.1f}% {m['support']:>16d}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<18} {overall_accuracy:>11.2f}% ({total_correct:,} / {total_samples:,} correct)")
    print(f"{'Macro Average F1':<18} {macro_f1:>11.2f}%")
    print("=" * 80)

    # 4. Print Section 2: Confusion Matrix
    print("\n2. CONFUSION MATRIX (Rows: Actual Emotion | Columns: Predicted Emotion):")
    print("-" * 80)

    # Header row with abbreviated column names
    col_headers = [emotion_dict[i][:7] for i in range(num_classes)]
    header_str = f"{'Actual':<14} | " + " ".join(f"{h:>8}" for h in col_headers)
    print(header_str)
    print("-" * len(header_str))

    for i in range(num_classes):
        row_name = f"{emotion_dict[i][:12]}"
        row_values = " ".join(f"{cm[i][j]:>8d}" for j in range(num_classes))
        print(f"{row_name:<14} | {row_values}")
    print("-" * 80)

    # 5. Print Section 3: Key Observations
    # Best emotion by F1
    best_emotion = max(metrics, key=lambda m: m["f1"])
    # Hardest emotion by F1
    hardest_emotion = min(metrics, key=lambda m: m["f1"])

    # Find highest off-diagonal confusion (most common mistake)
    max_mistake_count = -1
    mistake_pair = None
    for r in range(num_classes):
        for c in range(num_classes):
            if r != c and cm[r][c] > max_mistake_count:
                max_mistake_count = cm[r][c]
                mistake_pair = (emotion_dict[r], emotion_dict[c])

    print("\n3. KEY OBSERVATIONS:")
    print(f"  * Best recognized emotion : {best_emotion['name']} ({best_emotion['f1']:.1f}% F1-Score)")
    print(f"  * Most challenging emotion: {hardest_emotion['name']} ({hardest_emotion['f1']:.1f}% F1-Score)")
    if mistake_pair:
        print(f"  * Top misclassification   : {mistake_pair[0]} mistaken as {mistake_pair[1]} ({max_mistake_count} instances)")
    print("=" * 80 + "\n")


def parse_arguments():
    """CLI arguments for standalone evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CustomCNN checkpoint on the RAF-DB test set"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved .pth checkpoint file (e.g. best_model_epochs30_...pth)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Checkpoint file not found: '{args.model}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")
    print(f"[+] Loading test data from RAF-DB...")

    # Load test data only
    _, test_loader, classes = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=2,
        img_size=112
    )

    print(f"[+] Loading model architecture and weights from '{args.model}'...")
    model = CustomCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    print("[+] Running evaluation on test set...")
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Print the report
    compute_and_print_report(y_true, y_pred)


if __name__ == "__main__":
    main()
