"""Capsule building blocks: the squash non-linearity, primary capsules and routing-by-agreement.

Follows Sabour, Frosst & Hinton, "Dynamic Routing Between Capsules" (NeurIPS 2017).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Eq. 1 of the paper: keep the direction of `s`, map its length into [0, 1).

    Short vectors shrink towards zero and long vectors saturate just below unit length, so a
    capsule's length can be read as the probability that the entity it represents is present.
    """
    sq_norm = (s * s).sum(dim=dim, keepdim=True)
    return (sq_norm / (1.0 + sq_norm)) * s / torch.sqrt(sq_norm + eps)


class PrimaryCapsules(nn.Module):
    """Convolutional capsules.

    A single convolution whose output channels are grouped into `num_types` capsule types of
    `capsule_dim` dimensions, giving one capsule per type at every spatial position.
    """

    def __init__(
        self, in_channels: int, num_types: int = 32, capsule_dim: int = 8, kernel_size: int = 9, stride: int = 2
    ):
        super().__init__()
        self.num_types = num_types
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, num_types * capsule_dim, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)  # [B, T*D, H, W]
        b, _, h, w = out.shape
        out = out.view(b, self.num_types, self.capsule_dim, h, w).permute(0, 1, 3, 4, 2)
        return squash(out.reshape(b, self.num_types * h * w, self.capsule_dim).float())  # [B, T*H*W, D]


class RoutingCapsules(nn.Module):
    """Fully connected capsule layer with dynamic routing (Procedure 1 of the paper).

    Every input capsule i makes a prediction u_hat_{j|i} = W_ij u_i for every output capsule j.
    Routing then iteratively raises the coupling c_ij towards the outputs whose (squashed)
    weighted sum agrees with the prediction.

    Following common practice, gradients only flow through the final routing iteration, so the
    earlier iterations only decide the coupling coefficients. The layer always runs in float32:
    the iterative softmax/squash updates are sensitive to bfloat16 rounding.
    """

    def __init__(self, in_caps: int, in_dim: int, out_caps: int, out_dim: int, routings: int = 3):
        super().__init__()
        if routings < 1:
            raise ValueError(f"routings must be >= 1, got {routings}")
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_caps, in_caps, out_dim, in_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            u_hat = torch.einsum("jiod,bid->bjio", self.weight, u)  # [B, out_caps, in_caps, out_dim]
            u_hat_frozen = u_hat.detach()
            logits = u.new_zeros(u.shape[0], u_hat.shape[1], u_hat.shape[2])  # b_ij

            for i in range(self.routings):
                coupling = logits.softmax(dim=1)  # each input capsule splits its vote over the outputs
                preds = u_hat if i == self.routings - 1 else u_hat_frozen
                v = squash(torch.einsum("bji,bjio->bjo", coupling, preds))  # [B, out_caps, out_dim]
                if i < self.routings - 1:
                    logits = logits + torch.einsum("bjio,bjo->bji", preds, v)  # agreement u_hat . v
        return v


def margin_loss(
    lengths: torch.Tensor, target: torch.Tensor, m_pos: float = 0.9, m_neg: float = 0.1, lam: float = 0.5
) -> torch.Tensor:
    """Eq. 4: the true class capsule should be longer than `m_pos`, the others shorter than `m_neg`."""
    t = F.one_hot(target, lengths.shape[1]).to(lengths.dtype)
    loss = t * F.relu(m_pos - lengths).pow(2) + lam * (1 - t) * F.relu(lengths - m_neg).pow(2)
    return loss.sum(dim=1).mean()
