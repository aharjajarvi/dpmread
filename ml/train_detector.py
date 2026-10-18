#!/usr/bin/env python3
"""Train the detector's finder or measurer (`detector.py`) on
`detector_data.py prep` output.

finder    focal loss on the centre heatmap (CenterNet), L1 on the corner
          offsets relative to the symbol side; no loss inside "don't
          care" boxes. A share of frames (`ZOOM`) is cut and enlarged, for
          symbols larger than the scenes have. Scores per val set after
          each epoch: recall (a detection among the top 3 above `--thresh`
          whose largest corner error is at most `MATCH` x the symbol side)
          and false detections per frame. The best epoch by the first val set's loss is kept.
measurer  crops cut from the full-resolution image around each mark,
          moved and scaled as a finder error would (`crop_box`), turned,
          mirrored and (polarity flipped) inverted at random; L1 on the
          corners in modules, cross-entropy on size and polarity. Scores:
          corner error <= 1 module, size top 1 / top 3, polarity. The best
          epoch by the first val set's loss is kept.

Fine-tuning (`--init RUN/model.pt --freeze`): the backbone is frozen, only
the finder's top-down path and head (the measurer's fully connected
layers and heads) train; `--mix` keeps a share of synthetic data in every
batch.

Usage:
  python ml/train_detector.py finder RUN --train PREP [PREP ...] --val PREP [PREP ...]
  python ml/train_detector.py measurer RUN --train PREP [...] --val PREP [...]
  python ml/train_detector.py finder RUN2 --train REAL --mix SYN 0.5 \
      --val REAL --init RUN/model.pt --freeze --lr 5e-4 --epochs 30
Writes RUN/model.pt, RUN/model.onnx, RUN/log.tsv. `--holdout K` uses the
last K frames of every train set as an extra val set instead.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector as det  # noqa: E402
from detector_data import FRAME_H, apply, crop, crop_box, load_gray  # noqa: E402

MATCH = 0.25          # finder match: largest corner error / symbol side
ZOOM = (0.3, 3.5)     # finder: share of frames enlarged, largest factor
H4, W4 = FRAME_H // det.STRIDE, det.FINDER_SIDE // det.STRIDE


class Prep:
    """One `prep` folder, optionally a slice of its frames."""

    def __init__(self, path: Path, sl: slice = slice(None)):
        self.path = path
        self._x = None
        idx = json.loads((path / "index.json").read_text(encoding="utf-8"))
        for r in idx:                    # arrays: far smaller than nested lists
            for m in r["marks"]:
                m["frame"] = np.array(m["frame"], np.float32)
                m["corners"] = np.array(m["corners"], np.float32)
        self.ids = list(range(len(idx)))[sl]
        self.index = idx

    @property
    def x(self) -> np.ndarray:
        """The frames, mapped on first use in each process: a pickled
        memmap (DataLoader workers start by forkserver) copies its data."""
        if self._x is None:
            self._x = np.load(self.path / "frames.npy", mmap_mode="r")
        return self._x

    def __getstate__(self):
        return {**self.__dict__, "_x": None}

    def __len__(self):
        return len(self.ids)


# ---------------------------------------------------------------- finder
def finder_targets(marks: list[dict], flip_x: bool, flip_y: bool,
                   zoom: tuple = (1.0, 0.0, 0.0)) -> tuple:
    """heat (H4, W4), offsets (8, H4, W4), offset weight (H4, W4) = 1 /
    symbol side in offset units where supervised, care (H4, W4). `zoom`
    (factor, x0, y0): the frame was cut at (x0, y0) and enlarged."""
    heat = np.zeros((H4, W4), np.float32)
    off = np.zeros((8, H4, W4), np.float32)
    wgt = np.zeros((H4, W4), np.float32)
    care = np.ones((H4, W4), np.float32)
    yy, xx = np.mgrid[0:H4, 0:W4]
    for m in marks:
        q = (np.array(m["frame"], float) - zoom[1:]) * zoom[0] / det.STRIDE
        if flip_x:
            q[:, 0] = W4 - q[:, 0]
        if flip_y:
            q[:, 1] = H4 - q[:, 1]
        q = det.clockwise(q)
        c = q.mean(0)
        side = det.quad_side(q)
        inside = 0 <= c[0] < W4 and 0 <= c[1] < H4
        if not m["care"] or not inside:
            poly = ((q - c) * 1.2 + c).astype(np.int32)
            mask = np.zeros((H4, W4), np.uint8)
            cv2.fillConvexPoly(mask, poly, 1)
            care[mask > 0] = 0.0
            continue
        sigma = max(0.7, side / 8.0)
        g = np.exp(-((xx + 0.5 - c[0]) ** 2 + (yy + 0.5 - c[1]) ** 2) / (2 * sigma ** 2))
        heat = np.maximum(heat, g.astype(np.float32))
        cx, cy = int(c[0]), int(c[1])
        heat[cy, cx] = 1.0
        r = max(1, int(round(side / 8.0)))
        for y in range(max(0, cy - r), min(H4, cy + r + 1)):
            for x in range(max(0, cx - r), min(W4, cx + r + 1)):
                off[:, y, x] = ((q - [x + 0.5, y + 0.5]) / det.OFFSET_SCALE).ravel()
                wgt[y, x] = det.OFFSET_SCALE / side
    return heat, off, wgt, care


class FinderSet(Dataset):
    def __init__(self, preps: list[Prep], augment: bool, length: int | None = None,
                 weights: list[float] | None = None):
        self.items = [(p, i) for p in preps for i in p.ids]
        self.preps, self.augment, self.length, self.weights = preps, augment, length, weights

    def __len__(self):
        return self.length or len(self.items)

    def pick(self, k: int, rng: np.random.Generator):
        if self.weights is None:
            return self.items[k % len(self.items)]
        p = self.preps[rng.choice(len(self.preps), p=self.weights)]
        return p, p.ids[int(rng.integers(len(p.ids)))]

    def __getitem__(self, k):
        rng = np.random.default_rng()
        p, i = self.pick(k, rng)
        x = np.array(p.x[i])
        fx = fy = False
        zoom = (1.0, 0.0, 0.0)
        marks = p.index[i]["marks"]
        if self.augment and rng.random() < ZOOM[0]:
            # symbols larger than the scenes have (up to the whole frame):
            # cut a window around a random mark (or anywhere) and enlarge it
            z = math.exp(rng.uniform(0.0, math.log(ZOOM[1])))
            w, h = det.FINDER_SIDE / z, FRAME_H / z
            care = [m for m in marks if m["care"]]
            if care:
                c = care[int(rng.integers(len(care)))]["frame"].mean(0)
                x0 = c[0] - rng.uniform(0.2, 0.8) * w
                y0 = c[1] - rng.uniform(0.2, 0.8) * h
            else:
                x0, y0 = rng.uniform(0, det.FINDER_SIDE - w), rng.uniform(0, FRAME_H - h)
            x0 = float(np.clip(x0, 0, det.FINDER_SIDE - w))
            y0 = float(np.clip(y0, 0, FRAME_H - h))
            a = np.array([[z, 0, -x0 * z], [0, z, -y0 * z]])
            x = cv2.warpAffine(x, a, (det.FINDER_SIDE, FRAME_H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            zoom = (z, x0, y0)
        if self.augment:
            fx, fy = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
            if fx:
                x = x[:, ::-1]
            if fy:
                x = x[::-1]
        heat, off, wgt, care = finder_targets(marks, fx, fy, zoom)
        return (np.ascontiguousarray(x), heat, off, wgt, care,
                np.int64(k if self.weights is None else -1))


def photometric(xt: torch.Tensor) -> torch.Tensor:
    """Gamma, blur and noise jitter on a (N, 1, H, W) 0..1 batch."""
    n = len(xt)
    g = torch.empty(n, 1, 1, 1, device=xt.device).uniform_(0.6, 1.6)
    xt = xt.clamp_min(1e-4) ** g
    if torch.rand(()) < 0.3:
        xt = F.avg_pool2d(F.pad(xt, (1, 1, 1, 1), mode="replicate"), 3, 1)
    sd = torch.empty(n, 1, 1, 1, device=xt.device).uniform_(0.0, 0.04)
    return xt + torch.randn_like(xt) * sd


def focal(logit: torch.Tensor, heat: torch.Tensor, care: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = (heat >= 1.0).float()
    lp = -((1 - p) ** 2) * torch.log(p) * pos
    ln = -((1 - heat) ** 4) * (p ** 2) * torch.log(1 - p) * (1 - pos) * care
    return (lp.sum() + ln.sum()) / pos.sum().clamp_min(1.0)


def finder_loss(out, heat, off, wgt, care):
    lh = focal(out[:, 0], heat, care)
    sel = wgt > 0
    if sel.any():
        pred = out[:, 1:].permute(0, 2, 3, 1)[sel].view(-1, 4, 2)
        tgt = off.permute(0, 2, 3, 1)[sel].view(-1, 4, 2)
        lo = det.quad_loss(pred, tgt, 1.0 / wgt[sel]).mean()
    else:
        lo = out.sum() * 0.0
    return lh + lo, lh.item(), lo.item()


@torch.no_grad()
def finder_eval(net, prep: Prep, dev: str, thresh: float) -> dict:
    net.eval()
    ds = FinderSet([prep], False)
    tot = n = 0.0
    found = marks = false = 0
    for x, heat, off, wgt, care, ks in DataLoader(ds, 32, num_workers=4):
        xt = x.to(dev, torch.float32)[:, None] / 255.0
        out = net(xt)
        loss, _, _ = finder_loss(out, heat.to(dev), off.to(dev), wgt.to(dev), care.to(dev))
        tot += loss.item() * len(x)
        n += len(x)
        hm = torch.sigmoid(out[:, 0]).cpu().numpy()
        of = out[:, 1:].cpu().numpy()
        for b, k in enumerate(ks.tolist()):
            dets = det.peaks(hm[b], of[b], thresh)
            ms = prep.index[prep.ids[k]]["marks"]
            used = set()
            for m in ms:
                q = np.array(m["frame"])
                if not m["care"]:
                    continue
                marks += 1
                side = det.quad_side(q)
                for j, d in enumerate(dets):
                    if j not in used and det.quad_error(d["quad"], q) <= MATCH * side:
                        used.add(j)
                        found += 1
                        break
            for j, d in enumerate(dets):
                if j in used:
                    continue
                c = d["quad"].mean(0)
                near = any(np.linalg.norm(c - np.mean(m["frame"], 0)) < 0.5 * det.quad_side(
                    np.array(m["frame"])) for m in ms)
                false += not near           # a duplicate on a mark is no false detection
    net.train()
    return {"loss": tot / max(n, 1), "recall": found / max(marks, 1),
            "fp": false / max(n, 1), "marks": marks}


# -------------------------------------------------------------- measurer
def mark_items(prep: Prep) -> list[tuple]:
    out = []
    for i in prep.ids:
        r = prep.index[i]
        for m in r["marks"]:
            if m["care"] and m["size"] in det.SIZES:
                out.append((r["file"], m))
    return out


class MeasurerSet(Dataset):
    def __init__(self, items: list[tuple], augment: bool, seed: int | None = None,
                 length: int | None = None, mix: tuple | None = None):
        self.items, self.augment, self.seed = items, augment, seed
        self.length, self.mix = length, mix         # mix: (other items, share)

    def __len__(self):
        return self.length or len(self.items)

    def __getitem__(self, k):
        rng = np.random.default_rng(None if self.seed is None else [self.seed, k])
        items = self.items
        if self.mix is not None and rng.random() < self.mix[1]:
            items = self.mix[0]
        random = self.length is not None or items is not self.items
        file, m = items[int(rng.integers(len(items)))] if random else items[k]
        g = load_gray(Path(file))
        q = np.array(m["corners"], float)
        cx, cy, side = crop_box(q, rng)
        angle = float(rng.uniform(0, 360)) if self.augment and rng.random() < 0.5 else 0.0
        flip = bool(self.augment and rng.random() < 0.5)
        x, a = crop(g, cx, cy, side, angle, flip)
        pol = det.POLARITY.index(m["polarity"])
        if self.augment and rng.random() < 0.15:
            x, pol = 255 - x, 1 - pol
        c = det.clockwise(apply(a, q))
        rows, cols = (int(v) for v in m["size"].split("x"))
        area = 0.5 * abs(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))
        pitch = math.sqrt(area / (rows * cols)) * 2 / det.CROP
        return (x, (c / det.CROP * 2 - 1).astype(np.float32), np.float32(max(pitch, 1e-3)),
                np.int64(pol), np.int64(det.SIZES.index(m["size"])))


def measurer_loss(out, c, pitch, pol, size):
    corners, lp, ls = out
    lc = det.quad_loss(corners, c, pitch).clamp(max=20.0).mean()
    lpo = F.cross_entropy(lp, pol)
    lsz = F.cross_entropy(ls, size)
    return lc + lpo + lsz, lc.item(), lpo.item(), lsz.item()


@torch.no_grad()
def measurer_eval(net, items: list[tuple], dev: str) -> dict:
    net.eval()
    tot = n = 0.0
    ok_c = top1 = top3 = ok_p = 0
    for x, c, pitch, pol, size in DataLoader(MeasurerSet(items, False, seed=7), 64,
                                             num_workers=8):
        xt = x.to(dev, torch.float32)[:, None] / 255.0
        out = net(xt)
        loss, *_ = measurer_loss(out, c.to(dev), pitch.to(dev), pol.to(dev), size.to(dev))
        tot += loss.item() * len(x)
        n += len(x)
        for b in range(len(x)):
            e = det.quad_error(out[0][b].cpu().numpy(), c[b].numpy()) / pitch[b].item()
            ok_c += e <= 1.0
        rank = out[2].argsort(1, descending=True).cpu()
        top1 += (rank[:, 0] == size).sum().item()
        top3 += (rank[:, :3] == size[:, None]).any(1).sum().item()
        ok_p += (out[1].argmax(1).cpu() == pol).sum().item()
    net.train()
    n = max(n, 1)
    return {"loss": tot / n, "corner1": ok_c / n, "top1": top1 / n, "top3": top3 / n,
            "pol": ok_p / n, "marks": int(n)}


# ------------------------------------------------------------------ main
def export_onnx(net, path: Path, shape: tuple, outputs: list[str], dynamic: bool) -> None:
    x = torch.rand(*shape)
    axes = {"input": {2: "h", 3: "w"}, outputs[0]: {2: "h4", 3: "w4"}} if dynamic else None
    torch.onnx.export(net, (x,), str(path), input_names=["input"], output_names=outputs,
                      dynamic_axes=axes, opset_version=17, dynamo=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("which", choices=["finder", "measurer"])
    ap.add_argument("run", type=Path)
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--val", type=Path, nargs="*", default=[])
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--mix", nargs=2, metavar=("PREP", "SHARE"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=0, help="batches per epoch (0: one pass)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--init", type=Path)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = args.holdout
    train = [Prep(p, slice(0, -k) if k else slice(None)) for p in args.train]
    val = ([Prep(p, slice(-k, None)) for p in args.train] if k else []) + \
        [Prep(p) for p in args.val]
    finder = args.which == "finder"
    net = (det.FinderNet() if finder else det.MeasurerNet()).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev))
    if args.freeze:
        frozen = [net.stem, net.s4, net.s8, net.s16, net.s32] if finder else [net.body]
        for mod in frozen:
            for p in mod.parameters():
                p.requires_grad = False
            mod.eval()                                   # BatchNorm statistics too
    mix = (Prep(Path(args.mix[0])), float(args.mix[1])) if args.mix else None

    if finder:
        if mix:
            n = sum(len(p) for p in train)
            data = FinderSet(train + [mix[0]], True, args.steps * args.bs or 4 * n,
                             [(1 - mix[1]) * len(p) / n for p in train] + [mix[1]])
        else:
            data = FinderSet(train, True, args.steps * args.bs or None)
    else:
        items = [it for p in train for it in mark_items(p)]
        data = MeasurerSet(items, True, length=args.steps * args.bs or None,
                           mix=(mark_items(mix[0]), mix[1]) if mix else None)
        val_items = [mark_items(p) for p in val]
    loader = DataLoader(data, args.bs, shuffle=not (finder and mix), num_workers=args.workers,
                        drop_last=True, persistent_workers=True)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr,
                                                total_steps=args.epochs * len(loader))
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    keys = ("loss", "recall", "fp") if finder else ("loss", "corner1", "top1", "top3", "pol")
    cols = ["epoch", "train_loss", "part_a", "part_b", "seconds"] + \
        [f"val{i}_{k}" for i in range(len(val)) for k in keys]
    log = open(args.run / "log.tsv", "w", encoding="utf-8")
    log.write("\t".join(cols) + "\n")
    print(f"{det.n_params(net)} weights, {len(data)} samples per epoch, device {dev}")
    best = None
    for ep in range(1, args.epochs + 1):
        t = time.time()
        tot = pa = pb = cnt = 0.0
        if args.freeze:
            for mod in frozen:
                mod.eval()
        for batch in loader:
            if finder:
                x, heat, off, wgt, care, _ = (b.to(dev) for b in batch)
                xt = photometric(x.float()[:, None] / 255.0)
                loss, a, b = finder_loss(net(xt), heat, off, wgt, care)
            else:
                x, c, pitch, pol, size = (b.to(dev) for b in batch)
                xt = photometric(x.float()[:, None] / 255.0)
                loss, a, b, _ = measurer_loss(net(xt), c, pitch, pol, size)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, pa, pb, cnt = tot + loss.item(), pa + a, pb + b, cnt + 1
        if finder:
            scores = [finder_eval(net, v, dev, args.thresh) for v in val]
        else:
            scores = [measurer_eval(net, it, dev) for it in val_items]
        row = [ep, tot / cnt, pa / cnt, pb / cnt, time.time() - t] + \
            [s[k] for s in scores for k in keys]
        log.write("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")
        log.flush()
        print(" ".join(f"{c}={v:.4f}" if isinstance(v, float) else f"{c}={v}"
                       for c, v in zip(cols, row)), flush=True)
        if not scores or best is None or scores[0]["loss"] < best:
            best = scores[0]["loss"] if scores else None
            torch.save(net.state_dict(), args.run / "model.pt")
    net.load_state_dict(torch.load(args.run / "model.pt", map_location=dev))
    net = net.cpu().eval()
    if finder:
        export_onnx(net, args.run / "model.onnx", (1, 1, FRAME_H, det.FINDER_SIDE),
                    ["output"], True)
    else:
        export_onnx(net, args.run / "model.onnx", (1, 1, det.CROP, det.CROP),
                    ["corners", "polarity", "size"], False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
