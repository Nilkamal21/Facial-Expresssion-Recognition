"""
Phase 5: Main Entry Point for Facial Expression Recognition

This script orchestrates the entire pipeline:
1. Parses command-line arguments (with default values for any omitted parameters in any order).
2. Loads the RAF-DB dataset using dataset.py.
3. Builds the custom 4-block CNN using model.py.
4. Trains and evaluates the model using train.py.
"""

import argparse
import torch
import torch.nn as nn

# Import modules from the previous phases
from dataset import get_data_loaders
from model import CustomCNN
from train import train_model


def parse_arguments():
    """
    Parses command-line arguments.
    
    Any parameter not provided by the user automatically falls back
    to its default value. Arguments can be passed in ANY order.
    Example:
        python main.py --lr 0.0005 --epochs 10
        (batch_size, dropout, and optimizer will automatically use their defaults)
    """
    parser = argparse.ArgumentParser(
        description="Train a custom 4-block CNN on RAF-DB for Facial Expression Recognition"
    )

    # 1. Number of epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total number of training epochs (default: 30)"
    )

    # 2. Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of images per batch (default: 64)"
    )

    # 3. Learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)"
    )

    # 4. Dropout rate
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout probability in convolutional blocks (default: 0.25)"
    )

    # 5. Optimizer choice
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Choice of optimizer: 'adam' or 'sgd' (default: 'adam')"
    )

    return parser.parse_args()


def main():
    # 1. Parse arguments
    args = parse_arguments()

    print("=" * 60)
    print("      FACIAL EXPRESSION RECOGNITION - TRAINING CONFIG")
    print("=" * 60)
    print(f"  * Epochs       : {args.epochs}")
    print(f"  * Batch Size   : {args.batch_size}")
    print(f"  * Learning Rate: {args.lr}")
    print(f"  * Dropout Rate : {args.dropout}")
    print(f"  * Optimizer    : {args.optimizer.upper()}")
    print("=" * 60)

    # 2. Select hardware device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    # 3. Phase 1 & 2: Load data
    print("[+] Loading RAF-DB dataset...")
    train_loader, test_loader, classes = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=2,
        img_size=112
    )
    print(f"[+] Found {len(classes)} emotion classes: {classes}")

    # 4. Phase 3: Instantiate custom 4-block CNN
    print("[+] Initializing Custom 4-Block CNN...")
    model = CustomCNN(num_classes=len(classes), dropout_rate=args.dropout)
    model = model.to(device)

    # 5. Define Loss Function
    criterion = nn.CrossEntropyLoss()

    # 6. Set up Optimizer
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # 7. Phase 4: Train model with early stopping
    best_model_path = "best_model.pth"
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_path=best_model_path
    )

    print("\n[✓] Training complete!")
    print(f"[✓] Best model saved as '{best_model_path}'")


if __name__ == "__main__":
    main()
