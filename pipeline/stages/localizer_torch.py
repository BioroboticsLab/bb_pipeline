"""
PyTorch port of the BeesBook localizer FCN (LocalizerEncoder).

This is a direct port of the Keras model defined in
bb_localizer/localizer/model.py (get_conv_model, initial_channels=16).

Architecture (all convolutions use valid/no padding):
  - 3 strided Conv2d (stride=2) for spatial downsampling
  - 4 bottleneck Conv2d (stride=1) blocks
  - 1 expansion Conv2d
  - 2-layer 1×1 head with sigmoid output

For a 128×128 input the output is 5×5 with 4 channels (one per class).
Stride=8, offset=47 in image-coordinate space.

Layer names and shapes match localizer_2019_weights.pt (state dict compatible):
  conv0            (16,  1, 3, 3)  stride=2, no BN
  conv1 + bn1      (32, 16, 3, 3)  stride=2
  conv2 + bn2      (64, 32, 3, 3)  stride=2
  conv3 + bn3      (64, 64, 3, 3)  stride=1  ┐
  conv4 + bn4      (64, 64, 3, 3)  stride=1  │ bottleneck
  conv5 + bn5      (64, 64, 3, 3)  stride=1  │
  conv6 + bn6      (64, 64, 3, 3)  stride=1  ┘
  conv_expand + bn_expand  (128, 64, 3, 3)  stride=1
  conv_reduce      (16, 128, 1, 1)  head, ReLU
  conv_out         ( 4,  16, 1, 1)  head, Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizerEncoder(nn.Module):
    """Fully-convolutional bee localizer, PyTorch port of the BeesBook Keras FCN.

    Forward pass order matches the Keras model: Conv → ReLU → BN → Dropout.
    At inference (model.eval()) Dropout is a no-op.

    Input:  (B, 1, H, W)  float32 in [0, 1]  — grayscale, NCHW
    Output: (B, 4, H', W') float32 in [0, 1]  — per-class saliency heatmap, NCHW
    """

    def __init__(self):
        super().__init__()

        # Downsampling blocks (stride=2, valid padding)
        self.conv0 = nn.Conv2d(1, 16, 3, stride=2, padding=0)   # no BN after this one

        self.conv1 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=0)
        self.bn2   = nn.BatchNorm2d(64)

        # Bottleneck blocks (stride=1, valid padding)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn3   = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn5   = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn6   = nn.BatchNorm2d(64)

        # Expansion block
        self.conv_expand = nn.Conv2d(64, 128, 3, padding=0)
        self.bn_expand   = nn.BatchNorm2d(128)

        # 1×1 head
        self.conv_reduce = nn.Conv2d(128, 16, 1)
        self.conv_out    = nn.Conv2d(16,   4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        x = F.dropout2d(F.relu(self.conv0(x)), p=0.1, training=self.training)
        x = F.dropout2d(self.bn1(F.relu(self.conv1(x))), p=0.1, training=self.training)
        x = F.dropout2d(self.bn2(F.relu(self.conv2(x))), p=0.1, training=self.training)

        # Bottleneck
        x = F.dropout2d(self.bn3(F.relu(self.conv3(x))), p=0.1, training=self.training)
        x = F.dropout2d(self.bn4(F.relu(self.conv4(x))), p=0.1, training=self.training)
        x = F.dropout2d(self.bn5(F.relu(self.conv5(x))), p=0.1, training=self.training)
        x = self.bn6(F.relu(self.conv6(x)))           # no dropout on last bottleneck

        # Expansion
        x = self.bn_expand(F.relu(self.conv_expand(x)))

        # 1×1 head
        x = torch.sigmoid(self.conv_out(F.relu(self.conv_reduce(x))))

        return x
