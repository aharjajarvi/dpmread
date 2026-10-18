#!/usr/bin/env python3
"""Offline score of the detector chain (finder -> measurer) on labelled
images (ML plan *Stage 2 training plan*, D5).

Per image: the finder on the `detector_data.py prep` frame, up to 3
detections above `--thresh`; per detection the measurer on its crop from
the full-resolution image; the refined quad goes back to image pixels.
A labelled mark (not "don't care") is found when a detection's worst
corner is within `--tol` modules of the label (module pitch from the
label's size and quad area); the same for the finder quad alone, as a
share of the symbol side. A detection on no mark and not inside a
"don't care" box is a false detection.

Reported, per image source (`--list` column `source`) and in total:
recall of the finder and of the chain, size in the measurer's top 1 /
top 3 and polarity on the found marks, false detections per frame and
per frame without a code. With `--timing` (TSV: file, seconds, exit code
of the default reader), also on the images it reads slowly (> 2 s) and
on those it misses (exit code not 0).

Usage:
  python ml/evaluate_detector.py PREP --finder F.pt --measurer M.pt \
      --list LIST.csv [--timing T.tsv] [--out DETECTIONS.tsv]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector as det  # noqa: E402
from detector_data import apply, crop, crop_box, load_gray  # noqa: E402
from train_detector import Prep  # noqa: E402


def frame_to_full(q: np.ndarray, rec: dict) -> np.ndarray:
    """Finder frame pixels back to full-resolution image pixels."""
    q = np.asarray(q, float) / rec["scale"]
    if rec["turned"]:                  # undo (x, y) -> (H - y, x)
        q = np.stack([q[:, 1], rec["height"] - q[:, 0]], 1)
    return det.clockwise(q)


def pitch(q: np.ndarray, size: str) -> float:
    rows, cols = (int(v) for v in size.split("x"))
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return math.sqrt(area / (rows * cols))


@torch.no_grad()
def run(prep: Prep, finder, measurer, thresh: float, dev: str) -> list[dict]:
    """One row per image: its record and its detections."""
    out = []
    for i in prep.ids:
        rec = prep.index[i]
        x = torch.from_numpy(np.array(prep.x[i])).to(dev, torch.float32)[None, None] / 255.0
        o = finder(x)[0].cpu().numpy()
        dets = det.peaks(1 / (1 + np.exp(-o[0])), o[1:], thresh)
        gray = load_gray(Path(rec["file"])) if dets else None
        for d in dets:
            d["finder"] = frame_to_full(d["quad"], rec)
            cx, cy, side = crop_box(d["finder"])
            img, a = crop(gray, cx, cy, side)
            t = torch.from_numpy(img).to(dev, torch.float32)[None, None] / 255.0
            corners, pol, size = measurer(t)
            c = (corners[0].cpu().numpy() + 1) / 2 * det.CROP
            inv = cv2.invertAffineTransform(a)
            d["quad"] = det.clockwise(apply(inv, c))
            d["polarity"] = det.POLARITY[int(pol[0].argmax())]
            d["polarity_p"] = float(torch.softmax(pol[0], 0).max())
            d["sizes"] = [det.SIZES[k] for k in size[0].argsort(descending=True)[:3].tolist()]
        out.append({"rec": rec, "dets": dets})
    return out


def score(rows: list[dict], tol: float) -> tuple[dict, list[dict]]:
    """Per-mark and per-detection outcomes."""
    marks, dets = [], []
    for r in rows:
        rec, ds = r["rec"], r["dets"]
        used_f, used_c = set(), set()
        for m in rec["marks"]:
            q = np.asarray(m["corners"], float)
            if not m["care"]:
                continue
            side = det.quad_side(q)
            p = pitch(q, m["size"]) if m["size"] in det.SIZES else side / 18.0
            ef = [det.quad_error(d["finder"], q) / side for d in ds]
            ec = [det.quad_error(d["quad"], q) / p for d in ds]
            jf = min((j for j in range(len(ds)) if j not in used_f), key=lambda j: ef[j],
                     default=None)
            jc = min((j for j in range(len(ds)) if j not in used_c), key=lambda j: ec[j],
                     default=None)
            hit_f = jf is not None and ef[jf] <= 0.25
            hit_c = jc is not None and ec[jc] <= tol
            if hit_f:
                used_f.add(jf)
            if hit_c:
                used_c.add(jc)
            d = ds[jc] if hit_c else None
            marks.append({"file": rec["file"], "source": m["source"], "size": m["size"],
                          "finder": hit_f, "chain": hit_c,
                          "err": ec[jc] if jc is not None else float("nan"),
                          "err_finder": min((det.quad_error(d["finder"], q) / p for d in ds),
                                            default=float("nan")),
                          "top1": bool(d and d["sizes"][0] == m["size"]),
                          "top3": bool(d and m["size"] in d["sizes"]),
                          "polarity": bool(d and d["polarity"] == m["polarity"])})
        for j, d in enumerate(ds):
            c = d["finder"].mean(0)
            on = any(np.linalg.norm(c - np.mean(m["corners"], 0)) < 0.5 * det.quad_side(
                np.asarray(m["corners"], float)) or (not m["care"] and cv2.pointPolygonTest(
                    np.asarray(m["corners"], np.float32), (float(c[0]), float(c[1])), False) >= 0)
                for m in rec["marks"])
            dets.append({"file": rec["file"], "score": d["score"], "false": not on,
                         "sizes": ",".join(d["sizes"]), "polarity": d["polarity"],
                         "quad": " ".join(f"{x:.1f},{y:.1f}" for x, y in d["quad"])})
    return marks, dets


def summary(name: str, marks: list[dict], dets: list[dict], frames: list[str],
            empty: list[str]) -> str:
    n = len(marks)
    found = [m for m in marks if m["chain"]]
    fd = [d for d in dets if d["false"] and d["file"] in set(frames)]
    fe = [d for d in dets if d["file"] in set(empty)]
    med = np.nanmedian([m["err"] for m in marks]) if marks else float("nan")

    def r(k, of):
        return f"{sum(m[k] for m in of) / max(len(of), 1):.3f}"
    return (f"| {name} | {len(frames)} | {n} | {r('finder', marks)} | {r('chain', marks)} | "
            f"{med:.2f} | {r('top1', found)} | {r('top3', found)} | {r('polarity', found)} | "
            f"{len(fd) / max(len(frames), 1):.2f} | "
            f"{len(fe) / len(empty):.2f} ({len(empty)}) |" if empty else
            f"| {name} | {len(frames)} | {n} | {r('finder', marks)} | {r('chain', marks)} | "
            f"{med:.2f} | {r('top1', found)} | {r('top3', found)} | {r('polarity', found)} | "
            f"{len(fd) / max(len(frames), 1):.2f} | — |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prep", type=Path)
    ap.add_argument("--finder", type=Path, required=True)
    ap.add_argument("--measurer", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1.0, help="corner tolerance in modules")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    finder, measurer = det.FinderNet(), det.MeasurerNet()
    finder.load_state_dict(torch.load(args.finder, map_location="cpu"))
    measurer.load_state_dict(torch.load(args.measurer, map_location="cpu"))
    finder, measurer = finder.to(dev).eval(), measurer.to(dev).eval()
    prep = Prep(args.prep)
    rows = run(prep, finder, measurer, args.thresh, dev)
    marks, dets = score(rows, args.tol)

    def rel(f):
        return os.path.normpath(os.path.relpath(f, args.root))
    info = {os.path.normpath(r["file"]): r for r in csv.DictReader(open(args.list, encoding="utf-8"))}
    src = {r["rec"]["file"]: info.get(rel(r["rec"]["file"]), {}).get("source", "?") for r in rows}
    empty = {r["rec"]["file"] for r in rows
             if info.get(rel(r["rec"]["file"]), {}).get("content") in ("part-no-code", "no-part")}
    groups: dict[str, list[str]] = defaultdict(list)
    for f, s in src.items():
        groups[s].append(f)
    if args.timing:
        t = {os.path.normpath(r["file"]): r for r in csv.DictReader(
            open(args.timing, encoding="utf-8"), delimiter="\t")}
        for f in src:
            tr = t.get(rel(f))
            if tr is None or f in empty:
                continue
            if float(tr["seconds"]) > 2.0:
                groups["reader > 2 s"].append(f)
            if tr["exit"] != "0":
                groups["reader misses"].append(f)
    groups["all"] = list(src)
    print(f"finder {args.finder}, measurer {args.measurer}, threshold {args.thresh}, "
          f"corner tolerance {args.tol} modules\n")
    print("| Images | Frames | Marks | Finder recall | Chain recall | Median corner error "
          "(modules) | Size top 1 | Size top 3 | Polarity | False / frame | "
          "Detections / no-code frame |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in sorted(groups, key=lambda g: (g == "all", g.startswith("reader"), g)):
        fs = set(groups[g])
        print(summary(g, [m for m in marks if m["file"] in fs], [d for d in dets if d["file"] in fs],
                      groups[g], [f for f in groups[g] if f in empty]))
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["file", "score", "false", "sizes", "polarity", "quad"],
                               delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(dets)
        with open(args.out.with_suffix(".marks.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, list(marks[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(marks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
