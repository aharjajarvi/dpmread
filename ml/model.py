"""Per-module dark / light classifier (ML plan *Model output and labels*).

Fully convolutional: the input is a fit's classifier input (gray 0..1,
`PPM` px per module, `MARGIN` modules of margin), standardised per symbol;
the output is one logit per module (rows x cols), `p_dark = sigmoid`.
Two pooling steps take 6 px per module down to one cell per module; the
last layers see a 5 x 5 module neighbourhood.
"""

from __future__ import annotations

import torch
from torch import nn

PPM = 6
MARGIN = 2


class ModuleNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.pixel = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(),
            nn.Conv2d(16, w, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),                                 # 3 px per module
            nn.Conv2d(w, w, 3, padding=2, dilation=2), nn.ReLU(),
            nn.AvgPool2d(3),                                 # 1 cell per module
        )
        self.module = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(w, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, H, W) gray 0..1 -> (N, rows, cols) logits."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        f = self.module(self.pixel((x - mean) / std))[:, 0]
        return f[:, MARGIN:-MARGIN, MARGIN:-MARGIN]


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
