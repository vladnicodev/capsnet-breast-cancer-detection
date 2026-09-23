"""Convolutional baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_relu(cin: int, cout: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def conv_trunk() -> nn.Sequential:
    """Shared feature extractor of `MatchedCNN` and `CapsNet`: 3x96x96 -> 128x24x24.

    The vanilla CapsNet puts its 9x9 primary-capsule convolution directly on the input. On a
    96x96 patch that yields 51,200 primary capsules (vs. 1,152 on MNIST) and makes routing the
    bottleneck. Two strided convolutions first bring this down to 2,048 capsules.
    """
    return nn.Sequential(conv_bn_relu(3, 64, 5, stride=2, padding=2), conv_bn_relu(64, 128, 5, stride=2, padding=2))


class BinaryClassifier(nn.Module):
    """Single-logit classifier trained with binary cross-entropy."""

    def loss(self, out: dict, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bce = F.binary_cross_entropy_with_logits(out["logit"].float(), y.float())
        return bce, {"bce": bce.detach()}

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict:
        logit = self.net(x).squeeze(1)
        return {"logit": logit, "score": torch.sigmoid(logit.float())}


class SimpleCNN(BinaryClassifier):
    """Lightweight baseline from the project's first version: 3x (conv3x3 -> ReLU -> maxpool4) -> FC-64 -> FC-1."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(4),  # 96 -> 24
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(4),  # 24 -> 6
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(4),  # 6 -> 1
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )  # fmt: skip


class MatchedCNN(BinaryClassifier):
    """Controlled counterpart of `CapsNet`.

    Identical trunk, and the same 9x9/stride-2 convolution that produces the primary capsules,
    here followed by BatchNorm + ReLU. The capsule head (2,048 x 8D inputs routed into 2 x 16D
    class capsules through 524,288 weights) becomes an ordinary fully connected head with exactly
    the same number of weights: 8x8x256 = 16,384 features -> 32 hidden units -> 1 logit. The
    comparison therefore isolates capsules and routing, not network size.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_trunk(),
            conv_bn_relu(128, 256, 9, stride=2),  # 24 -> 8
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )  # fmt: skip
