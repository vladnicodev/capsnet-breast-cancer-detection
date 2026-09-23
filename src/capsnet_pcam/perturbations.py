"""Test-time distribution shifts used to probe robustness.

Two families target the "viewpoint" robustness usually claimed for capsule networks
(rotation by angles not covered by the 90-degree training augmentation, and changes of scale).
The other two mimic real acquisition differences between pathology labs (H&E stain variation,
out-of-focus scanning). All perturbations keep the centre 32x32 region, which determines the
label, inside the patch.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Stain vectors of haematoxylin, eosin and DAB (Ruifrok & Johnston, 2001), as in scikit-image.
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)


def _affine(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    theta = torch.cat([matrix, torch.zeros(2, 1)], dim=1).to(x).expand(x.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    if degrees == 0:
        return x
    a = math.radians(degrees)
    return _affine(x, torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]))


def zoom(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Magnify (factor > 1) or shrink (factor < 1) the patch about its centre."""
    if factor == 1:
        return x
    return _affine(x, torch.eye(2) / factor)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return x
    radius = math.ceil(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum()).to(x)
    c = x.shape[1]
    x = F.pad(x, (radius,) * 4, mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def stain_shift(x: torch.Tensor, strength: float, seed: int = 0) -> torch.Tensor:
    """HED colour augmentation (Tellez et al., 2019) used as a test-time stain shift.

    Each image is deconvolved into haematoxylin/eosin/DAB concentrations, and every channel is
    rescaled by U(1 - s, 1 + s) and offset by U(-s, s) before converting back to RGB.
    """
    if strength == 0:
        return x
    g = torch.Generator().manual_seed(seed)
    alpha = (1 + strength * (2 * torch.rand(x.shape[0], 1, 3, generator=g) - 1)).to(x)
    beta = (strength * (2 * torch.rand(x.shape[0], 1, 3, generator=g) - 1)).to(x)

    optical_density = -x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 3).clamp_min(1e-3).log()
    hed = optical_density @ HED_FROM_RGB.to(x)
    hed = hed * alpha + beta
    out = torch.exp(-(hed @ RGB_FROM_HED.to(x))).clamp(0, 1)
    return out.view(x.shape[0], x.shape[2], x.shape[3], 3).permute(0, 3, 1, 2)


# name -> (axis label, severities, function). The first severity is always the identity.
PERTURBATIONS = {
    "rotation": ("Rotation angle (degrees)", [0, 15, 30, 45], rotate),
    "zoom": ("Zoom factor", [1.0, 1.15, 1.3, 1.5], zoom),
    "blur": ("Gaussian blur sigma (px)", [0, 0.75, 1.5, 2.5], gaussian_blur),
    "stain": ("H&E stain jitter strength", [0, 0.05, 0.1, 0.2], stain_shift),
}
