#!/usr/bin/env python3
"""Train the module classifier on `dataset.py` output.

Batches hold one symbol size each (sizes differ in shape); every epoch
visits every sample once in random order. Scores on `--val` sets after
each epoch: BCE, module bit error rate of the model and of a median
threshold at the module centres (the floor to beat). The best epoch by
the first val set's BCE is kept.

Usage:
  python ml/train.py RUN --train DS [DS ...] --val DS [DS ...] [--epochs 20]
  python ml/train.py RUN --train DS --val DS --init RUN0/model.pt \
      --freeze-pixel --lr 3e-4             # fine-tune
Writes RUN/model.pt, RUN/model.onnx, RUN/log.tsv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds  # noqa: E402
from model import MARGIN, PPM, ModuleNet, n_params  # noqa: E402


def batches(sets: list[dict], bs: int, rng: np.random.Generator | None):
    """(x uint8, y uint8) batches of one size; shuffled when `rng`."""
    plan = []
    for si, data in enumerate(sets):
        for size, (x, _) in data.items():
            idx = np.arange(len(x))
            if rng is not None:
                rng.shuffle(idx)
            plan += [(si, size, idx[k:k + bs]) for k in range(0, len(idx), bs)]
    if rng is not None:
        rng.shuffle(plan)
    for si, size, idx in plan:
        x, y = sets[si][size]
        idx = np.sort(idx)                       # memmap reads in order
        yield np.asarray(x[idx]), np.asarray(y[idx])


def to_dev(x: np.ndarray, y: np.ndarray, dev: str, augment: bool, rng: np.random.Generator):
    xt = torch.from_numpy(x).to(dev, torch.float32)[:, None] / 255.0
    yt = torch.from_numpy(y).to(dev, torch.float32)
    if augment:
        # contrast, brightness and gamma jitter; the per-symbol
        # standardisation removes most of it, gamma and noise remain
        g = torch.from_numpy(rng.uniform(0.7, 1.4, (len(x), 1, 1, 1))).to(dev, torch.float32)
        xt = xt.clamp_min(1e-4) ** g
        xt = xt + torch.randn_like(xt) * float(rng.uniform(0.0, 0.03))
    return xt, yt


def baseline_bits(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Median threshold of the module-centre values (1 = dark)."""
    v = ds.module_values(x.astype(np.float32), rows, cols, PPM, MARGIN)
    return (v < np.median(v)).astype(np.uint8)


@torch.no_grad()
def evaluate(net, data: dict, dev: str, bs: int = 256) -> dict:
    net.eval()
    loss = n = err = base_err = 0.0
    for x, y in batches([data], bs, None):
        xt, yt = to_dev(x, y, dev, False, None)
        logit = net(xt)
        loss += F.binary_cross_entropy_with_logits(logit, yt, reduction="sum").item()
        err += ((logit > 0).float() != yt).sum().item()
        rows, cols = y.shape[1:]
        base_err += sum(int((baseline_bits(xi, rows, cols) != yi).sum()) for xi, yi in zip(x, y))
        n += y.size
    net.train()
    return {"bce": loss / max(n, 1), "ber": err / max(n, 1), "ber_base": base_err / max(n, 1),
            "modules": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--val", type=Path, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--init", type=Path, help="start from this model.pt")
    ap.add_argument("--freeze-pixel", action="store_true",
                    help="train only the module layers (fine-tuning)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = [ds.load(p) for p in args.train]
    val = [ds.load(p) for p in args.val]
    net = ModuleNet(args.width).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev))
    if args.freeze_pixel:
        for p in net.pixel.parameters():
            p.requires_grad = False
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * sum(-(-len(x) // args.bs) for d in train for x, _ in d.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(steps, 1))
    args.run.mkdir(parents=True, exist_ok=True)
    log = open(args.run / "log.tsv", "w", encoding="utf-8")
    cols = ["epoch", "train_bce", "seconds"] + [f"val{i}_{k}" for i in range(len(val))
                                                 for k in ("bce", "ber", "ber_base")]
    log.write("\t".join(cols) + "\n")
    n_train = sum(len(x) for d in train for x, _ in d.values())
    print(f"{n_params(net)} weights, {n_train} train samples, device {dev}")
    best = None
    for ep in range(1, args.epochs + 1):
        t = time.time()
        tot = cnt = 0.0
        for x, y in batches(train, args.bs, rng):
            xt, yt = to_dev(x, y, dev, True, rng)
            loss = F.binary_cross_entropy_with_logits(net(xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * y.size
            cnt += y.size
        scores = [evaluate(net, v, dev) for v in val]
        row = [ep, tot / cnt, time.time() - t] + [s[k] for s in scores
                                                  for k in ("bce", "ber", "ber_base")]
        log.write("\t".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row) + "\n")
        log.flush()
        print(" ".join(f"{c}={v:.4f}" if isinstance(v, float) else f"{c}={v}"
                       for c, v in zip(cols, row)), flush=True)
        if best is None or scores[0]["bce"] < best:
            best = scores[0]["bce"]
            torch.save(net.state_dict(), args.run / "model.pt")
    net.load_state_dict(torch.load(args.run / "model.pt", map_location=dev))
    export_onnx(net.cpu().eval(), args.run / "model.onnx")
    return 0


def export_onnx(net, path: Path) -> None:
    x = torch.rand(1, 1, (24 + 2 * MARGIN) * PPM, (24 + 2 * MARGIN) * PPM)
    torch.onnx.export(net, (x,), str(path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {2: "h", 3: "w"}, "logits": {1: "rows", 2: "cols"}},
                      opset_version=17, dynamo=False)


if __name__ == "__main__":
    sys.exit(main())
