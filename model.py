"""
Phase 3: Custom 4-Block CNN Architecture for Facial Expression Recognition

Architecture Specifications:
- 4 Convolutional Blocks
- Each block contains:
    * 2 x Conv2d (3x3 kernel, padding=1)
    * 1 x BatchNorm2d
    * 1 x ReLU activation
    * 1 x Dropout2d
    * 1 x MaxPool2d (2x2 kernel, stride=2)
- Feature map output before classifier: exactly 7 x 7 x 512
- 1-Layer Classifier: nn.Linear(512 * 7 * 7, num_classes)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A single convolutional block containing:
    1. First Conv2d operation
    2. Second Conv2d operation
    3. Batch Normalization
    4. ReLU Activation
    5. 2D Spatial Dropout
    6. Max Pooling (downsamples spatial dimensions by a factor of 2)
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.25):
        super(ConvBlock, self).__init__()
        
        # 1. First convolution
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False  # bias is redundant when followed by BatchNorm
        )
        
        # 2. Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        # 3. Batch Normalization (normalizes across the channel dimension)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 4. Non-linear activation
        self.relu = nn.ReLU(inplace=True)
        
        # 5. Spatial Dropout (drops entire feature maps to prevent co-adaptation)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # 6. Max Pooling (reduces height and width by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 2 convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 1 batch normalization
        x = self.bn(x)
        
        # ReLU activation
        x = self.relu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Downsampling via MaxPool
        x = self.pool(x)
        return x


class CustomCNN(nn.Module):
    """
    Custom 4-Block CNN for RAF-DB Facial Expression Recognition.
    
    Given an input of shape [Batch, 3, 112, 112]:
    - Block 1: 3   -> 64   channels  =>  [Batch, 64, 56, 56]
    - Block 2: 64  -> 128  channels  =>  [Batch, 128, 28, 28]
    - Block 3: 128 -> 256  channels  =>  [Batch, 256, 14, 14]
    - Block 4: 256 -> 512  channels  =>  [Batch, 512, 7, 7]
    - Flatten: 512 * 7 * 7 = 25,088 values
    - Classifier (1 Layer): nn.Linear(25088, num_classes) => [Batch, num_classes]
    """
    def __init__(self, num_classes=7, dropout_rate=0.25):
        super(CustomCNN, self).__init__()
        
        # Block 1: 3 -> 64 channels
        self.block1 = ConvBlock(in_channels=3, out_channels=64, dropout_rate=dropout_rate)
        
        # Block 2: 64 -> 128 channels
        self.block2 = ConvBlock(in_channels=64, out_channels=128, dropout_rate=dropout_rate)
        
        # Block 3: 128 -> 256 channels
        self.block3 = ConvBlock(in_channels=128, out_channels=256, dropout_rate=dropout_rate)
        
        # Block 4: 256 -> 512 channels
        self.block4 = ConvBlock(in_channels=256, out_channels=512, dropout_rate=dropout_rate)
        
        # Flattened feature dimension: 512 channels * 7 height * 7 width = 25,088
        self.flatten_dim = 512 * 7 * 7
        
        # Single-layer linear classifier
        self.classifier = nn.Linear(in_features=self.flatten_dim, out_features=num_classes)

    def forward_features(self, x):
        """Passes input through the 4 convolutional blocks up to the 7x7x512 feature map."""
        x = self.block1(x)  # -> [Batch, 64, 56, 56]
        x = self.block2(x)  # -> [Batch, 128, 28, 28]
        x = self.block3(x)  # -> [Batch, 256, 14, 14]
        x = self.block4(x)  # -> [Batch, 512, 7, 7]
        return x

    def forward(self, x):
        """Full forward pass from input image to emotion class logits."""
        # 1. Extract 7x7x512 features
        features = self.forward_features(x)
        
        # 2. Flatten: [Batch, 512, 7, 7] -> [Batch, 25088]
        flattened = torch.flatten(features, start_dim=1)
        
        # 3. 1-Layer Classifier: [Batch, 25088] -> [Batch, num_classes]
        logits = self.classifier(flattened)
        return logits
