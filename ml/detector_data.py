#!/usr/bin/env python3
"""Detector data: finder frames and measurer crops.

Two kinds of source, one record per image (`records`):

  scenes  `synth_marks.py --scene` folders: every mark of a frame's JSON
          (corners, size, polarity)
  real    `ml/boxes.py merge` output (`--boxes`) with the image list
          (`--list`, for the images without any box) and the folder the
          file paths start from (`--root`); `review` boxes have no size
          and are "don't care": no loss inside them, no false detection

`prep` writes the finder input of every image once: OUT/frames.u8
(N x 384 x 512 uint8: long side 512, area-averaged; portrait images are
turned 90 degrees clockwise, the short side padded with its edge pixels)
and OUT/index.json (per frame: file, full-resolution width and height,
scale, turned, and its marks with corners in full-resolution and in
frame pixels). Measurer crops are cut at training time from the
full-resolution image (`crop`).

Usage:
  python ml/detector_data.py prep OUT --scenes DIR [DIR ...]
  python ml/detector_data.py prep OUT --boxes boxes.csv --list LIST.csv \
      --root DIR --splits val
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import CROP, CROP_PAD, FINDER_SIDE, clockwise  # noqa: E402

FRAME_H = FINDER_SIDE * 3 // 4                   # 384: 4:3 frames fill it
CODE = ("code", "code-cut")


def load_gray(path: Path) -> np.ndarray:
    """Gray uint8 with EXIF rotation applied (as the reader and a browser
    show the image)."""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("L"))


def mark(corners, size: str, polarity: str, care: bool, source: str) -> dict:
    q = clockwise(np.asarray(corners, float))
    return {"corners": q.round(2).tolist(), "size": size, "polarity": polarity,
            "care": care, "source": source}


def scene_records(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            out.append({"file": str(d / r["file"]), "marks": [
                mark(m["corners"], m["size"], m["polarity"], True, "synthetic")
                for m in r["marks"]]})
    return out


def real_records(boxes: Path, listing: Path, root: Path, splits: list[str]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(listing, encoding="utf-8")):
        if r["split"] in splits and r["content"] in (*CODE, "part-no-code", "no-part"):
            by.setdefault(os.path.normpath(r["file"]), [])
    for r in csv.DictReader(open(boxes, encoding="utf-8")):
        f = os.path.normpath(r["file"])
        if r["split"] not in splits:
            continue
        q = [[float(v) for v in p.split(",")] for p in r["corners"].split()]
        care = r["source"] != "review"
        by.setdefault(f, []).append(mark(q, r["size"], r["polarity"] or "dark", care,
                                         r["source"]))
    return [{"file": str(root / f), "marks": m} for f, m in sorted(by.items())]


def finder_frame(gray: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """(FRAME_H x FINDER_SIDE frame, scale full -> frame, turned)."""
    turned = gray.shape[0] > gray.shape[1]
    if turned:
        gray = np.ascontiguousarray(np.rot90(gray, -1))   # clockwise
    h, w = gray.shape
    s = min(FINDER_SIDE / w, FRAME_H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    frame = cv2.copyMakeBorder(small, 0, FRAME_H - nh, 0, FINDER_SIDE - nw,
                               cv2.BORDER_REPLICATE)
    return frame, s, turned


def to_frame(q: np.ndarray, full_h: int, s: float, turned: bool) -> np.ndarray:
    """Full-resolution points to frame pixels (pixel-centre convention)."""
    q = np.asarray(q, float)
    if turned:                         # clockwise: (x, y) -> (H - y, x)
        q = np.stack([full_h - q[:, 1], q[:, 0]], 1)
    return q * s


def _prep_one(rec: dict):
    g = load_gray(Path(rec["file"]))
    frame, s, turned = finder_frame(g)
    h, w = g.shape
    marks = [{**m, "frame": clockwise(to_frame(np.array(m["corners"]), h, s, turned))
              .round(2).tolist()} for m in rec["marks"]]
    return frame, {"file": rec["file"], "width": w, "height": h, "scale": s,
                   "turned": turned, "marks": marks}


def prep(args) -> int:
    if args.scenes:
        recs = scene_records(args.scenes)
    else:
        recs = real_records(args.boxes, args.list, args.root, args.splits.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(args.out / "frames.npy", "w+", np.uint8,
                                  (len(recs), FRAME_H, FINDER_SIDE))
    index = []
    with Pool(args.jobs) as pool:
        for i, (frame, info) in enumerate(pool.imap(_prep_one, recs, chunksize=16)):
            x[i] = frame
            index.append(info)
            if (i + 1) % 5000 == 0:
                print(f"{i + 1} / {len(recs)}", flush=True)
    x.flush()
    (args.out / "index.json").write_text(json.dumps(index), encoding="utf-8")
    n = sum(len(r["marks"]) for r in index)
    print(f"{len(index)} frames, {n} marks in {args.out}")
    return 0


def crop_box(q: np.ndarray, jitter: np.random.Generator | None = None) -> tuple:
    """(cx, cy, side) of the measurer crop around quad q (full-resolution
    pixels); `jitter` moves and scales it as a finder error would."""
    c = q.mean(0)
    side = CROP_PAD * float(max(np.ptp(q[:, 0]), np.ptp(q[:, 1])))
    if jitter is not None:
        c = c + jitter.normal(0.0, 0.07, 2) * side
        side *= math.exp(jitter.normal(0.0, 0.12))
    return float(c[0]), float(c[1]), max(side, 8.0)


def crop(gray: np.ndarray, cx: float, cy: float, side: float, angle: float = 0.0,
         flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """CROP x CROP crop (uint8) of the square (cx, cy, side), turned by
    `angle` degrees, and the 2 x 3 affine from image to crop pixels.
    Area-averaged: the square is cut at full resolution and resized."""
    big = side * (1.5 if angle else 1.0)          # room for the turn
    n = max(CROP, int(math.ceil(big)))
    a = n / big
    # image -> square of n px around (cx, cy), turned by angle
    m = cv2.getRotationMatrix2D((cx, cy), angle, a)
    m[:, 2] += [n / 2 - cx, n / 2 - cy]
    sq = cv2.warpAffine(gray, m, (n, n), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    if angle:                                     # keep the central `side`
        k = int(round(side * a))
        o = (n - k) // 2
        sq = sq[o:o + k, o:o + k]
        m[:, 2] -= o
        n = k
    out = cv2.resize(sq, (CROP, CROP), interpolation=cv2.INTER_AREA)
    m = m * (CROP / n)
    if flip:
        out = out[:, ::-1]
        m = np.array([[-1, 0, CROP], [0, 1, 0]], float) @ np.vstack([m, [0, 0, 1]])
    return np.ascontiguousarray(out), m


def apply(m: np.ndarray, q: np.ndarray) -> np.ndarray:
    return q @ m[:, :2].T + m[:, 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("out", type=Path)
    p.add_argument("--scenes", type=Path, nargs="+")
    p.add_argument("--boxes", type=Path)
    p.add_argument("--list", type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--splits", default="val")
    p.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if not args.scenes and not (args.boxes and args.list):
        ap.error("--scenes, or --boxes with --list")
    return prep(args)


if __name__ == "__main__":
    sys.exit(main())
