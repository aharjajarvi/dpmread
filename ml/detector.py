"""Symbol detector: finder and measurer (ML plan *Stage 2 training plan*).

Both networks are fully convolutional on gray input (0..1), standardised
per image inside the network, so a caller passes plain pixels.

  finder    whole frame, long side `FINDER_SIDE` px (area-averaged), sides
            padded to a multiple of 32; per `STRIDE` x `STRIDE` cell a
            symbol-centre logit and 8 corner offsets from the cell centre
            in cells / `OFFSET_SCALE`
  measurer  `CROP` x `CROP` crop of the full-resolution image around one
            detection (side `CROP_PAD` x the quad's longer bounding-box
            side, area-averaged); 4 refined corners in crop pixels / `CROP`
            * 2 - 1, polarity logits (dark, light) and one logit per
            ECC 200 size of `SIZES`

Corners are an unordered quad: losses take the best of the 4 cyclic
orders of a clockwise quad, and `clockwise` brings labels to that
winding. Nothing here places modules: the reader fits the grid on the
finder and timing pattern of the symbol.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

FINDER_SIDE = 512
STRIDE = 4
OFFSET_SCALE = 16.0
CROP = 192
CROP_PAD = 1.25
SIZES = ["10x10", "12x12", "14x14", "16x16", "18x18", "20x20", "22x22", "24x24", "26x26",
         "32x32", "36x36", "40x40", "44x44", "48x48", "52x52", "64x64", "72x72", "80x80",
         "88x88", "96x96", "104x104", "120x120", "132x132", "144x144",
         "8x18", "8x32", "12x26", "12x36", "16x36", "16x48"]
POLARITY = ["dark", "light"]


def clockwise(q: np.ndarray) -> np.ndarray:
    """(4, 2) quad in clockwise order on screen (y down)."""
    x, y = q[:, 0], q[:, 1]
    area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return q if area >= 0 else q[::-1].copy()


def quad_side(q: np.ndarray) -> float:
    """Mean side length of a quad."""
    return float(np.mean(np.linalg.norm(q - np.roll(q, -1, 0), axis=1)))


def quad_loss(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mean corner L1 / scale under the best cyclic order. pred, target:
    (N, 4, 2); scale: (N,). Returns (N,)."""
    errs = [(pred - target.roll(k, 1)).abs().sum(-1).mean(-1) for k in range(4)]
    return torch.stack(errs, 1).min(1).values / scale


def quad_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Largest corner distance under the best cyclic order (same units)."""
    return min(float(np.linalg.norm(pred - np.roll(target, k, 0), axis=1).max())
               for k in range(4))


def standardise(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return (x - mean) / std


def cbr(cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Block(nn.Module):
    """Depthwise 3x3 + pointwise; residual when the shape is kept."""

    def __init__(self, cin: int, cout: int, s: int = 1):
        super().__init__()
        self.f = nn.Sequential(cbr(cin, cin, 3, s, cin), cbr(cin, cout, 1))
        self.res = s == 1 and cin == cout

    def forward(self, x):
        y = self.f(x)
        return x + y if self.res else y


def stage(cin: int, cout: int, n: int, s: int = 2) -> nn.Sequential:
    return nn.Sequential(Block(cin, cout, s), *[Block(cout, cout) for _ in range(n - 1)])


class FinderNet(nn.Module):
    """Stride-32 backbone with a top-down path back to stride 4."""

    def __init__(self, w: int = 32):
        super().__init__()
        self.stem = cbr(1, 16, 3, 2)                    # /2
        self.s4 = stage(16, w, 3)                       # /4
        self.s8 = stage(w, 2 * w, 3)                    # /8
        self.s16 = stage(2 * w, 3 * w, 3)               # /16
        self.s32 = stage(3 * w, 4 * w, 3)               # /32
        d = 2 * w
        self.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in (w, 2 * w, 3 * w, 4 * w)])
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Sequential(Block(d, d), cbr(d, w, 1), nn.Conv2d(w, 9, 1))
        nn.init.constant_(self.head[-1].bias[:1], -4.0)   # rare centres

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, H, W), H and W multiples of 32 -> (N, 9, H/4, W/4):
        centre logit, then x, y offsets of the 4 corners."""
        f4 = self.s4(self.stem(standardise(x)))
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        p = self.lat[3](f32)
        for f, lat in ((f16, self.lat[2]), (f8, self.lat[1]), (f4, self.lat[0])):
            p = lat(f) + self.up(p)
        return self.head(p)


class MeasurerNet(nn.Module):
    def __init__(self, w: int = 24):
        super().__init__()
        self.body = nn.Sequential(
            cbr(1, 16, 3, 2),                            # 96
            stage(16, w, 2, 1),
            stage(w, 2 * w, 3),                          # 48
            stage(2 * w, 4 * w, 3),                      # 24
            stage(4 * w, 6 * w, 3),                      # 12
            stage(6 * w, 8 * w, 2),                      # 6
            cbr(8 * w, 64, 1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * (CROP // 32) ** 2, 256),
                                nn.ReLU(inplace=True))
        self.corners = nn.Linear(256, 8)
        self.polarity = nn.Linear(256, len(POLARITY))
        self.size = nn.Linear(256, len(SIZES))

    def forward(self, x: torch.Tensor):
        """x: (N, 1, CROP, CROP) -> corners (N, 4, 2), polarity (N, 2),
        size (N, len(SIZES)) logits."""
        h = self.fc(self.body(standardise(x)))
        return self.corners(h).view(-1, 4, 2), self.polarity(h), self.size(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def macs(m: nn.Module, shape: tuple) -> int:
    """Multiply-adds of one forward pass on an input of `shape`."""
    total = 0

    def conv(mod, inp, out):
        nonlocal total
        k = mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
        total += out.numel() * k

    def lin(mod, inp, out):
        nonlocal total
        total += out.numel() * mod.in_features

    hooks = [mod.register_forward_hook(conv) for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    hooks += [mod.register_forward_hook(lin) for mod in m.modules() if isinstance(mod, nn.Linear)]
    was = m.training
    m.eval()
    with torch.no_grad():
        m(torch.rand(*shape))
    m.train(was)
    for h in hooks:
        h.remove()
    return total


def peaks(heat: np.ndarray, offsets: np.ndarray, thresh: float, k: int = 3) -> list[dict]:
    """Up to k detections from one finder output (sigmoid heat (h, w),
    offsets (8, h, w)): local maxima of the 3x3 neighbourhood above
    `thresh`, highest first; quads in input pixels."""
    h, w = heat.shape
    pad = np.pad(heat, 1, constant_values=-1.0)
    nb = np.max([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=0)
    ys, xs = np.nonzero((heat >= nb) & (heat >= thresh))
    order = np.argsort(-heat[ys, xs])[:k]
    out = []
    for i in order:
        y, x = ys[i], xs[i]
        c = np.array([x + 0.5, y + 0.5])
        q = (c + offsets[:, y, x].reshape(4, 2) * OFFSET_SCALE) * STRIDE
        out.append({"score": float(heat[y, x]), "quad": q})
    return out
