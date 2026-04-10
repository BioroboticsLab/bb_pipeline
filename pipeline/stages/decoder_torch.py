"""PyTorch implementation of the BeesBook tag decoder (custom ResNet-50).

Mirrors the Keras architecture defined in
``bb_pipeline_models/models/model_generation/pipelinemodels.py:get_custom_resnet``.

Module names match the Keras H5 layer names (e.g. ``res3a_branch2a``,
``bn3a_branch2a``, ``bit_0``) so weight conversion is a direct name mapping.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderResNet(nn.Module):
    """BeesBook tag decoder — custom ResNet-50 with 16 output heads.

    Input:  (B, 1, 32, 32) grayscale float32 in [0, 1]
    Output: list of 16 tensors matching Keras ``model.predict()`` order:
            [bit_0, ..., bit_11, x_rotation, y_rotation, z_rotation, center]

    Note: BatchNorm uses ``eps=1e-3`` to match the Keras default.
    """

    # Keras BatchNormalization default epsilon (PyTorch default is 1e-5)
    _bn_eps = 1e-3

    # Stage configurations: (stage, block, filters, stride, has_shortcut_projection)
    _stages = [
        # stage 2b: identity block
        (2, "b", [16, 16, 64], 1, False),
        # stage 3
        (3, "a", [32, 32, 128], 2, True),
        (3, "b", [32, 32, 128], 1, False),
        # stage 4
        (4, "a", [64, 64, 256], 2, True),
        (4, "b", [64, 64, 256], 1, False),
        # stage 5
        (5, "a", [128, 128, 512], 2, True),
        (5, "b", [128, 128, 512], 1, False),
        # stage 6
        (6, "a", [256, 256, 1024], 2, True),
        (6, "b", [256, 256, 1024], 1, False),
    ]

    def __init__(self):
        super().__init__()

        # ── Stage 2a (manual — matches inline code in get_custom_resnet) ──
        # Main branch: input(1) → 1×1 conv → 3×3 conv → 1×1 conv
        self.res2a_branch2a = nn.Conv2d(1, 16, kernel_size=1, bias=True)
        self.bn2a_branch2a = nn.BatchNorm2d(16, eps=self._bn_eps)
        self.res2a_branch2b = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn2a_branch2b = nn.BatchNorm2d(16, eps=self._bn_eps)
        self.res2a_branch2c = nn.Conv2d(16, 64, kernel_size=1, bias=True)
        self.bn2a_branch2c = nn.BatchNorm2d(64, eps=self._bn_eps)
        # Shortcut branch: input(1) → 1×1 conv
        self.res2a_branch1 = nn.Conv2d(1, 64, kernel_size=1, bias=True)
        self.bn2a_branch1 = nn.BatchNorm2d(64, eps=self._bn_eps)

        # ── Stages 2b through 6b ──
        in_channels = 64
        for stage, block, (f1, f2, f3), stride, has_proj in self._stages:
            prefix = f"res{stage}{block}_branch"
            bn_prefix = f"bn{stage}{block}_branch"

            # Main branch: 1×1 → 3×3 → 1×1
            setattr(self, f"{prefix}2a",
                    nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride, bias=True))
            setattr(self, f"{bn_prefix}2a", nn.BatchNorm2d(f1, eps=self._bn_eps))

            setattr(self, f"{prefix}2b",
                    nn.Conv2d(f1, f2, kernel_size=3, padding=1, bias=True))
            setattr(self, f"{bn_prefix}2b", nn.BatchNorm2d(f2, eps=self._bn_eps))

            setattr(self, f"{prefix}2c",
                    nn.Conv2d(f2, f3, kernel_size=1, bias=True))
            setattr(self, f"{bn_prefix}2c", nn.BatchNorm2d(f3, eps=self._bn_eps))

            # Shortcut projection (convolutional blocks only)
            if has_proj:
                setattr(self, f"{prefix}1",
                        nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride, bias=True))
                setattr(self, f"{bn_prefix}1", nn.BatchNorm2d(f3, eps=self._bn_eps))

            in_channels = f3

        # ── Global average pooling ──
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ── 16 output heads ──
        for i in range(12):
            setattr(self, f"bit_{i}", nn.Linear(1024, 1))
        self.x_rotation = nn.Linear(1024, 2)
        self.y_rotation = nn.Linear(1024, 2)
        self.z_rotation = nn.Linear(1024, 2)
        self.center = nn.Linear(1024, 2)

    def _bottleneck(self, x, stage, block, has_proj):
        """Run one bottleneck block (convolutional or identity)."""
        prefix = f"res{stage}{block}_branch"
        bn_prefix = f"bn{stage}{block}_branch"

        shortcut = x

        # Branch 2: 1×1 → BN → ReLU → 3×3 → BN → ReLU → 1×1 → BN
        out = getattr(self, f"{prefix}2a")(x)
        out = getattr(self, f"{bn_prefix}2a")(out)
        out = F.relu(out)

        out = getattr(self, f"{prefix}2b")(out)
        out = getattr(self, f"{bn_prefix}2b")(out)
        out = F.relu(out)

        out = getattr(self, f"{prefix}2c")(out)
        out = getattr(self, f"{bn_prefix}2c")(out)

        # Shortcut projection
        if has_proj:
            shortcut = getattr(self, f"{prefix}1")(shortcut)
            shortcut = getattr(self, f"{bn_prefix}1")(shortcut)

        return F.relu(out + shortcut)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, 1, 32, 32)
            Grayscale input in NCHW format.

        Returns
        -------
        list of 16 Tensors
            [bit_0(B,1), ..., bit_11(B,1),
             x_rotation(B,2), y_rotation(B,2), z_rotation(B,2),
             center(B,2)]
        """
        # Stage 2a (manual)
        shortcut = self.bn2a_branch1(self.res2a_branch1(x))

        out = F.relu(self.bn2a_branch2a(self.res2a_branch2a(x)))
        out = F.relu(self.bn2a_branch2b(self.res2a_branch2b(out)))
        out = self.bn2a_branch2c(self.res2a_branch2c(out))

        x = F.relu(out + shortcut)

        # Stages 2b through 6b
        for stage, block, _, _, has_proj in self._stages:
            x = self._bottleneck(x, stage, block, has_proj)

        # Pool → flatten
        x = self.avg_pool(x).flatten(1)  # (B, 1024)

        # Output heads
        bits = [torch.sigmoid(getattr(self, f"bit_{i}")(x)) for i in range(12)]
        rotations = [
            self.x_rotation(x),
            self.y_rotation(x),
            self.z_rotation(x),
        ]
        center = self.center(x)

        return bits + rotations + [center]
