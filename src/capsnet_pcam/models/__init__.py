"""Model registry.

Every model maps a batch of [B, 3, 96, 96] images in [0, 1] to a dict holding at least
`score`, the predicted probability of metastatic tissue, and implements
`loss(out, x, y) -> (total_loss, {name: detached_component})`.
"""

from __future__ import annotations

from torch import nn

from .capsnet import CapsNet
from .cnn import MatchedCNN, SimpleCNN

MODEL_NAMES = ("simple_cnn", "matched_cnn", "capsnet")

DISPLAY_NAMES = {"simple_cnn": "Simple CNN", "matched_cnn": "Matched CNN", "capsnet": "CapsNet"}


def build_model(name: str, routings: int = 3, recon_weight: float = 0.392) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN()
    if name == "matched_cnn":
        return MatchedCNN()
    if name == "capsnet":
        return CapsNet(routings=routings, recon_weight=recon_weight)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module, inference_only: bool = True) -> int:
    """Trainable parameters, excluding the CapsNet decoder (training-only) when `inference_only`."""
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not (inference_only and name.startswith("decoder."))
    )
