"""
Phase 1 & Phase 2: Data Loading & Preprocessing for RAF-DB Dataset

This module:
1. Downloads or locates the RAF-DB dataset using kagglehub.
2. Applies data transformations and augmentations (Resize to 112x112, RandomFlip, Rotation, Normalization).
3. Creates PyTorch DataLoader objects for training and testing.
"""

import os
import kagglehub
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# RAF-DB maps folder names '1' through '7' to these basic emotion categories:
# Folder '1' -> 0: Surprise
# Folder '2' -> 1: Fear
# Folder '3' -> 2: Disgust
# Folder '4' -> 3: Happiness
# Folder '5' -> 4: Sadness
# Folder '6' -> 5: Anger
# Folder '7' -> 6: Neutral
EMOTION_DICT = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}


def download_or_get_dataset_path():
    """
    Downloads the RAF-DB dataset via kagglehub if not cached,
    and returns the path to the 'DATASET' directory containing 'train' and 'test'.
    """
    print("Checking / Downloading RAF-DB dataset via kagglehub...")
    raw_path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
    dataset_dir = os.path.join(raw_path, "DATASET")
    
    # Verify train and test directories exist
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
        raise FileNotFoundError(f"Expected 'train' and 'test' folders inside {dataset_dir}")
        
    print(f"Dataset ready at: {dataset_dir}")
    return dataset_dir


def get_transforms(img_size=112):
    """
    Defines preprocessing and data augmentations.
    
    Why 112x112?
    Starting at 112x112, four pooling operations (dividing by 2 each) 
    produce the required 7x7 spatial feature map in the 4th block.
    
    - Training transforms:
        1. Resize to (112, 112)
        2. RandomHorizontalFlip (data augmentation)
        3. RandomRotation(10) (data augmentation)
        4. ToTensor (converts PIL image to float tensor in range [0, 1])
        5. Normalize (standard ImageNet mean and std for stable gradients)
        
    - Test transforms:
        1. Resize to (112, 112)
        2. ToTensor
        3. Normalize
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def get_data_loaders(batch_size=64, num_workers=2, img_size=112):
    """
    Creates and returns PyTorch DataLoader objects for train and test sets.
    
    Parameters:
        batch_size (int): Number of images processed in each training step.
        num_workers (int): Number of subprocesses for data loading.
        img_size (int): Image resolution (default: 112).
        
    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        classes (list): List of class names as found by ImageFolder (['1', '2', ..., '7']).
    """
    # 1. Get path to dataset
    dataset_dir = download_or_get_dataset_path()
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # 2. Get transforms
    train_transform, test_transform = get_transforms(img_size=img_size)

    # 3. Create datasets using ImageFolder (automatically reads subfolders as classes)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # 4. Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # No need to shuffle test/evaluation data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} test images.")
    return train_loader, test_loader, train_dataset.classes
