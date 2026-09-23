"""Capsule network with dynamic routing, adapted to 96x96 RGB histopathology patches."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .capsules import PrimaryCapsules, RoutingCapsules, margin_loss
from .cnn import conv_trunk


class ReconstructionDecoder(nn.Module):
    """Reconstructs the input patch from the masked class capsules.

    Only used during training, as a regulariser that forces capsules to encode the
    appearance of the patch. The fully connected decoder of the paper would need 28M
    parameters for a 96x96x3 output, so this version upsamples with transposed convolutions.
    """

    def __init__(self, in_features: int, image_size: int = 96):
        super().__init__()
        self.base = image_size // 16
        self.fc = nn.Sequential(nn.Linear(in_features, 128 * self.base**2), nn.ReLU(inplace=True))

        def up(cin: int, cout: int) -> list[nn.Module]:
            return [nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1), nn.ReLU(inplace=True)]

        self.deconv = nn.Sequential(
            *up(128, 64), *up(64, 32), *up(32, 16),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )  # fmt: skip

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.fc(z).view(-1, 128, self.base, self.base))


class CapsNet(nn.Module):
    """conv trunk -> 2,048 primary capsules (8D) -> dynamic routing -> 2 class capsules (16D).

    The length of each class capsule is the network's confidence that the class is present.
    The tumour score used for ROC analysis is (|v_tumour| - |v_normal| + 1) / 2, which lies in
    [0, 1] and is >= 0.5 exactly when the tumour capsule is the longer one.
    """

    def __init__(self, routings: int = 3, recon_weight: float = 0.392, num_classes: int = 2, image_size: int = 96):
        super().__init__()
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.trunk = conv_trunk()
        self.primary = PrimaryCapsules(128, num_types=32, capsule_dim=8, kernel_size=9, stride=2)
        grid = ((image_size // 4) - 9) // 2 + 1
        self.digits = RoutingCapsules(32 * grid * grid, 8, num_classes, 16, routings=routings)
        self.decoder = ReconstructionDecoder(num_classes * 16, image_size) if recon_weight > 0 else None

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, reconstruct: bool | None = None) -> dict:
        v = self.digits(self.primary(self.trunk(x)))  # [B, classes, 16]
        lengths = v.norm(dim=-1)
        out = {"capsules": v, "lengths": lengths, "score": 0.5 * (1 + lengths[:, 1] - lengths[:, 0])}

        if reconstruct is None:
            reconstruct = self.training
        if reconstruct and self.decoder is not None:
            # Training: keep only the true class capsule. Inference: keep the longest one.
            target = y if y is not None else lengths.argmax(dim=1)
            mask = F.one_hot(target, self.num_classes).to(v.dtype).unsqueeze(-1)
            out["recon"] = self.decoder((v * mask).flatten(1))
        return out

    def loss(self, out: dict, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        margin = margin_loss(out["lengths"], y)
        if "recon" not in out:
            return margin, {"margin": margin.detach()}
        recon = F.mse_loss(out["recon"].float(), x.float())
        return margin + self.recon_weight * recon, {"margin": margin.detach(), "recon": recon.detach()}
