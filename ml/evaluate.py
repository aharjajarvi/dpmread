#!/usr/bin/env python3
"""Offline score of the module classifier against a threshold baseline.

For every sample of a `dataset.py` set (a reader fit with its label):
module bit errors, codeword errors per Reed-Solomon block, and whether RS
would decode with the reader's erasure rule (`dm-ecc`
`rs_decode_symbol_frac`): the least confident codewords of a block are
erased 2 at a time up to 75 % of its check codewords, and a try with
erasures needs `2e + f <= n - margin` (margin 2, or 3 from 20 check
codewords); without erasures `2e <= n`. Codeword confidence = its least
confident module.

  model:    bit = p_dark > 0.5, confidence |p_dark - 0.5| * 2
  reader:   the reader's own bits and confidences for the fit (`--dump-fits`
            with `bits` / `conf`), when the set has them
  median:   bit = module-centre value below the symbol's median,
            confidence = distance to the median (a weak floor)

Per image, a symbol counts as decoded when any of its fits decodes.

Usage:
  python ml/evaluate.py DS RUN/model.pt [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import datamatrix_gen as gen  # noqa: E402
import dataset as ds  # noqa: E402
from model import MARGIN, PPM, ModuleNet  # noqa: E402

MAX_ERASURE_FRAC = 0.75


@functools.cache
def symbol(size: str) -> tuple:
    for s in gen.SQUARE_SYMBOLS + gen.RECT_SYMBOLS:
        if f"{s[0]}x{s[1]}" == size:
            return s
    raise KeyError(size)


@functools.cache
def cw_index(size: str) -> np.ndarray:
    """Codeword number (0-based, stream order) of each module, -1 = none."""
    rows, cols, dregr, dregc, regr, regc, *_ = symbol(size)
    place = gen.placement_map(dregr * regr, dregc * regc)
    idx = np.full((rows, cols), -1, np.int32)
    for rv in range(regr):
        for rh in range(regc):
            r0, c0 = rv * (dregr + 2), rh * (dregc + 2)
            for i in range(dregr):
                for j in range(dregc):
                    cell = place[rv * dregr + i][rh * dregc + j]
                    if cell is not None and cell[0] > 0:
                        idx[r0 + 1 + i, c0 + 1 + j] = cell[0] - 1
    return idx


def rs_decodes(wrong: np.ndarray, conf: np.ndarray, size: str) -> tuple[bool, list[int]]:
    """wrong / conf per codeword (stream order) -> (decodes, errors per block)."""
    s = symbol(size)
    blocks, n = s[8], s[7]
    margin = 2 if n < 20 else 3
    max_f = int(n * MAX_ERASURE_FRAC)
    ok, per_block = True, []
    for b in range(blocks):
        w, c = wrong[b::blocks], conf[b::blocks]
        per_block.append(int(w.sum()))
        order = np.argsort(c, kind="stable")
        good = False
        for f in range(0, max_f + 1, 2):
            e = int(w.sum() - w[order[:f]].sum())
            if (f == 0 and 2 * e <= n) or (f > 0 and 2 * e + f <= n - margin):
                good = True
                break
        ok &= good
    return ok, per_block


def codewords(bits_wrong: np.ndarray, conf: np.ndarray, size: str) -> tuple[np.ndarray, np.ndarray]:
    idx = cw_index(size)
    n_cw = idx.max() + 1
    wrong = np.zeros(n_cw, bool)
    cconf = np.full(n_cw, np.inf)
    sel = idx >= 0
    np.logical_or.at(wrong, idx[sel], bits_wrong[sel])
    np.minimum.at(cconf, idx[sel], conf[sel])
    return wrong, cconf


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ds", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ModuleNet(args.width).to(dev).eval()
    net.load_state_dict(torch.load(args.model, map_location=dev))
    data = ds.load(args.ds)
    reader = ds.load_reader(args.ds)
    names = ("model", "reader", "median") if reader else ("model", "median")
    index = [ln.split("\t") for ln in (args.ds / "index.tsv").read_text().splitlines()[1:]]
    rows_out = []
    for size, row, src, k, read, score in index:
        x, y = data[size]
        xi, yi = np.asarray(x[int(row)]), np.asarray(y[int(row)]).astype(bool)
        r, c = yi.shape
        p = torch.sigmoid(net(torch.from_numpy(xi).to(dev, torch.float32)[None, None] / 255.0))[0]
        p = p.cpu().numpy()
        v = ds.module_values(xi.astype(np.float32), r, c, PPM, MARGIN)
        thr = np.median(v)
        res = {"source": src, "fit": int(k), "read": read == "1", "size": size,
               "modules": int(yi.size)}
        cands = [("model", p > 0.5, np.abs(p - 0.5) * 2), ("median", v < thr, np.abs(v - thr))]
        if reader:
            rb, rc = reader[size]
            rbi, rci = np.asarray(rb[int(row)]), np.asarray(rc[int(row)])
            if (rbi == 255).any():
                rbi, rci = np.zeros_like(rbi), np.zeros(rbi.shape, np.float32)
            cands.insert(1, ("reader", rbi.astype(bool), rci))
        for name, bits, conf in cands:
            wrong_bits = bits != yi
            w, cc = codewords(wrong_bits, conf, size)
            dec, per_block = rs_decodes(w, cc, size)
            res[name] = {"bit_errors": int(wrong_bits.sum()), "cw_errors": per_block,
                         "decodes": dec}
        rows_out.append(res)

    def summary(sel):
        n = sum(r["modules"] for r in sel)
        out = {"fits": len(sel)}
        for name in names:
            out[name] = {"ber": round(sum(r[name]["bit_errors"] for r in sel) / max(n, 1), 4),
                         "fits_decoding": sum(r[name]["decodes"] for r in sel)}
        imgs = collections.defaultdict(list)
        for r in sel:
            imgs[r["source"]].append(r)
        out["images"] = len(imgs)
        for name in names:
            out[name]["images_decoding"] = sum(any(f[name]["decodes"] for f in fs)
                                               for fs in imgs.values())
        return out

    read_imgs = {r["source"] for r in rows_out if r["read"]}
    report = {
        "all": summary(rows_out),
        "fits that read": summary([r for r in rows_out if r["read"]]),
        "failed fits": summary([r for r in rows_out if not r["read"]]),
        "failed fits, images not read": summary([r for r in rows_out if not r["read"]
                                                 and r["source"] not in read_imgs]),
    }
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"summary": report, "fits": rows_out}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
