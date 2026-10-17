#!/usr/bin/env python3
"""Module classifier dataset from `dpmread --dump-fits` output.

Each dumped fit gives one sample: the classifier input (`NAME.fitN.npy`,
the fit's variant at 6 px per module with a 2-module margin) and the
label, the symbol's encoded module matrix (1 = dark) in the fit's frame.

Labels follow the fit, not the image: the reader's fit may be rotated or
mirrored against the true symbol, so the matrix is turned to match.
  - synthetic images (a `synth_marks.py` record next to the image): the
    symmetry that brings the true module centres closest to the fit's;
    fits whose mean centre error exceeds `--max-err` modules are dropped
    (a wrong grid has no module labels);
  - real images (`--labels` TSV: file, text): every encodation a marking
    system may have used (`tools/agreement.py`, `encode`) at the fit's
    size, in every symmetry; the one with the best agreement between the
    matrix and the input sampled at module centres is used; fits below
    `--min-agree` (share of data modules; a wrong label agrees about half)
    are dropped.

Output in OUT, per symbol size `RxC`: `RxC.x.u8` (N x H x W uint8, gray
x 255), `RxC.y.u8` (N x rows x cols), the reader's own classification of
the fit when the dump has it, `RxC.rb.u8` (bits, 255 = absent) and
`RxC.rc.f4` (confidences), and `index.tsv` (size, row in the size's
arrays, source image, fit number, read, centre error or agreement).

Usage:
  python ml/dataset.py OUT --fits DUMP [DUMP ...] [--synth]
  python ml/dataset.py OUT --fits DUMP --labels labels.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import agreement  # noqa: E402


def symmetries(a: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Grid symmetries of an (rows, cols, ...) array that keep its shape."""
    out = [("id", a), ("r180", a[::-1, ::-1]), ("fl", a[::-1]), ("fr", a[:, ::-1])]
    if a.shape[0] == a.shape[1]:
        t = a.swapaxes(0, 1)
        out += [("T", t), ("Tr180", t[::-1, ::-1]), ("r90", np.rot90(a, 1)),
                ("r270", np.rot90(a, 3))]
    return out


def module_values(x: np.ndarray, rows: int, cols: int, ppm: int, margin: int) -> np.ndarray:
    """Mean of the central 2x2 pixels of each module in a classifier input."""
    c0 = margin * ppm + ppm // 2 - 1
    ys = c0 + ppm * np.arange(rows)
    xs = c0 + ppm * np.arange(cols)
    v = sum(x[np.ix_(ys + dy, xs + dx)] for dy in (0, 1) for dx in (0, 1)) / 4.0
    return v


def label_synth(fit: dict, rec: dict, max_err: float) -> tuple[np.ndarray, float] | None:
    rows, cols = map(int, rec["size"].split("x"))
    if fit["size"] != rec["size"]:
        return None
    fc = np.array(fit["centres"], float).reshape(rows, cols, 2)
    tc = np.array(rec["centres"], float)
    m = np.array(rec["matrix"], np.uint8)
    ppm = rec["params"]["ppm"]
    best = None
    for (name, t), (_, mm) in zip(symmetries(tc), symmetries(m)):
        err = float(np.sqrt(((fc - t) ** 2).sum(-1)).mean() / ppm)
        if best is None or err < best[1]:
            best = (np.ascontiguousarray(mm), err)
    return best if best[1] <= max_err else None


def label_real(fit: dict, x: np.ndarray, text: str, ppm: int, margin: int,
               min_agree: float) -> tuple[np.ndarray, float] | None:
    mats = agreement.encode([text], fit["size"]).get(text, [])
    if not mats:
        return None
    sym, m0 = mats[0]
    rows, cols = m0.shape
    data = ~fixed_mask(sym)
    v = module_values(x, rows, cols, ppm, margin)
    dark = (v < np.median(v)).astype(np.uint8)
    best = None
    for _, m in mats:
        for _, mm in symmetries(m):
            agree = float((mm == dark)[data].mean())
            if best is None or agree > best[1]:
                best = (np.ascontiguousarray(mm), agree)
    return best if best[1] >= min_agree else None


def fixed_mask(sym: tuple) -> np.ndarray:
    """Finder and timing modules of every data region (True)."""
    rows, cols, dr, dc = sym[0], sym[1], sym[2], sym[3]
    r = np.arange(rows) % (dr + 2)
    c = np.arange(cols) % (dc + 2)
    return (np.isin(r, (0, dr + 1))[:, None]) | (np.isin(c, (0, dc + 1))[None, :])


class Writer:
    """Appends samples to per-size uint8 files."""

    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, tuple] = {}
        self.count: dict[str, int] = {}
        self.index = open(out / "index.tsv", "w", encoding="utf-8")
        self.index.write("size\trow\tsource\tfit\tread\tscore\n")
        self.shapes: dict[str, list[int]] = {}

    def add(self, size: str, x: np.ndarray, y: np.ndarray, source: str, k: int,
            read: bool, score: float, rbits=None, rconf=None) -> None:
        if size not in self.files:
            self.files[size] = tuple(open(self.out / f"{size}.{k}", "wb")
                                     for k in ("x.u8", "y.u8", "rb.u8", "rc.f4"))
            self.count[size] = 0
            self.shapes[size] = [*x.shape, *y.shape]
        fx, fy, fb, fc = self.files[size]
        fx.write(np.clip(np.rint(x * 255), 0, 255).astype(np.uint8).tobytes())
        fy.write(y.astype(np.uint8).tobytes())
        if rbits is None or len(rbits) != y.size:
            rbits, rconf = np.full(y.size, 255), np.full(y.size, np.nan)
        fb.write(np.asarray(rbits, np.uint8).tobytes())
        fc.write(np.asarray(rconf, np.float32).tobytes())
        self.index.write(f"{size}\t{self.count[size]}\t{source}\t{k}\t{int(read)}\t{score:.4f}\n")
        self.count[size] += 1

    def close(self) -> None:
        for fs in self.files.values():
            for f in fs:
                f.close()
        self.index.close()
        meta = {s: {"n": self.count[s], "x": self.shapes[s][:2], "y": self.shapes[s][2:]}
                for s in self.count}
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def load(out: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per size (x uint8 N x H x W, y uint8 N x rows x cols) as memmaps."""
    meta = json.loads((out / "meta.json").read_text())
    data = {}
    for s, m in meta.items():
        x = np.memmap(out / f"{s}.x.u8", np.uint8, "r", shape=(m["n"], *m["x"]))
        y = np.memmap(out / f"{s}.y.u8", np.uint8, "r", shape=(m["n"], *m["y"]))
        data[s] = (x, y)
    return data


def load_reader(out: Path) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Per size the reader's (bits uint8, conf float32), N x rows x cols;
    None for a set made before they were stored."""
    meta = json.loads((out / "meta.json").read_text())
    data = {}
    for s, m in meta.items():
        if not (out / f"{s}.rb.u8").exists():
            return None
        b = np.memmap(out / f"{s}.rb.u8", np.uint8, "r", shape=(m["n"], *m["y"]))
        c = np.memmap(out / f"{s}.rc.f4", np.float32, "r", shape=(m["n"], *m["y"]))
        data[s] = (b, c)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--fits", type=Path, nargs="+", required=True,
                    help="--dump-fits folders")
    ap.add_argument("--labels", type=Path, help="TSV file, text (real images)")
    ap.add_argument("--max-err", type=float, default=0.5,
                    help="synthetic: largest mean centre error in modules")
    ap.add_argument("--min-agree", type=float, default=0.65,
                    help="real: smallest data-module agreement of the label")
    args = ap.parse_args()
    texts = {}
    if args.labels:
        for ln in args.labels.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                f, t = ln.split("\t")[:2]
                texts[str(Path(f).resolve())] = t
    w = Writer(args.out)
    kept = dropped = 0
    for d in args.fits:
        for jf in sorted(d.glob("*.fits.json")):
            doc = json.loads(jf.read_text(encoding="utf-8"))
            src = Path(doc["file"]).resolve()
            ppm, margin = doc["input_ppm"], doc["input_margin"]
            rec_path = src.with_suffix(".json")
            rec = json.loads(rec_path.read_text()) if rec_path.exists() and not texts else None
            for k, fit in enumerate(doc["fits"]):
                x = np.load(d / fit["input"])
                if rec is not None:
                    lab = label_synth(fit, rec, args.max_err)
                elif str(src) in texts:
                    lab = label_real(fit, x, texts[str(src)], ppm, margin, args.min_agree)
                else:
                    lab = None
                if lab is None:
                    dropped += 1
                    continue
                w.add(fit["size"], x, lab[0], str(src), k, fit["read"], lab[1],
                      fit.get("bits"), fit.get("conf"))
                kept += 1
    w.close()
    print(f"{kept} samples kept, {dropped} fits dropped, {len(w.count)} sizes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
