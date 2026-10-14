#!/usr/bin/env python3
"""Module agreement of known texts with a Data Matrix in an image.

For ground truth checks: does a label (a text the symbol is believed to hold)
fit the symbol in the image, and which of many candidate texts fits best?

1. Grid fits are found without any candidate text: rough quads (from
   `--corners` or the v2 reader's blob search, plus quads completed past the
   image edge for symbols cut by it) are refined on the finder and timing
   pattern only, for every symbol size the candidates encode to. Each is
   sampled on a straight, a timing-tracked and a warped grid (the v2 reader's
   tiers); the 16 strongest distinct fits per size are kept and thresholded.
2. Each candidate text is encoded (`datamatrix_gen.py`) and compared module by
   module with every fit of its size, as seen and mirrored across the
   anti-diagonal (that mirror keeps the finder and timing pattern, so only
   the data can rule it out). Only data modules inside the image count.
3. Per candidate the fit least likely to match by luck counts:
   `agreement` = share of data modules that match (a wrong text gives about
   0.5); `chance` = expected number of wrong candidates that would match as
   well by luck, over all candidates, fits and readings tried
   (tries * P(Bin(n, 0.5) >= k)). It stays comparable between one label and a
   search over thousands. Neighbouring values (n, n+1) share data codewords
   and agree above 0.5 with each other, so check a search winner against its
   neighbours' chance too. `cw_errors` counts codewords with a wrong module
   (or one outside the image) per Reed-Solomon block; `recoverable` = every
   block within its correction capacity, i.e. the symbol could be decoded
   from this fit with no further sampling help.

Usage:
  python tools/agreement.py IMAGE [IMAGE ...] --text T [--text T ...]
  python tools/agreement.py IMAGE --digits 4            # all 10 000 values
  python tools/agreement.py IMAGE --texts-from FILE     # one text per line
  options: --corners "x,y x,y x,y x,y" (one image), --size RxC,
           --polarity dark|light|any, --top N, --json

Corners are in pixels of the image as decoded with its EXIF orientation
(as dpmread and a browser show it), in any order; for a symbol cut by the
image edge, the corners of its visible part. Without corners the blob search
can miss faint marks: then a wrong fit gives every candidate a high chance.
Needs numpy, opencv and Pillow (the generator's import).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import datamatrix_gen as gen  # noqa: E402
import datamatrix_reader_v2 as v2  # noqa: E402

WORK_MODULE_PX = 12      # crops around given corners are scaled to ~this module size
CROP_MARGIN = 0.35       # crop margin around given corners, share of the symbol side
COARSE_KEEP = 24         # quads coarse-refined per size (as the v2 grid pass)
FINE_KEEP = 8            # of those, fully refined
FITS_KEEP = 16           # distinct fits per symbol size every candidate is scored on
MIN_INSIDE = 0.4         # share of modules that must lie inside the image
SIZE_SLACK = 4           # also try symbols up to this many spare data codewords
MAX_SIDE = 1600          # longest side searched without corners (px)
MIN_AREA = 0.004         # smallest blob searched without corners, share of the image


def parse_corners(spec: str) -> np.ndarray:
    pts = [p.split(",") for p in spec.replace(";", " ").split()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise argparse.ArgumentTypeError('corners: four points "x,y x,y x,y x,y"')
    return np.float32([[float(x), float(y)] for x, y in pts])


def ascii_cw(raw: bytes) -> list[int]:
    """ASCII encodation: digit pairs in one codeword, upper shift for > 127."""
    cw, i = [], 0
    while i < len(raw):
        c = raw[i]
        if gen.is_digit(c) and i + 1 < len(raw) and gen.is_digit(raw[i + 1]):
            cw.append(130 + int(raw[i:i + 2]))
            i += 2
        elif gen.is_extended(c):
            cw += [gen.UPPER_SHIFT, c - 128 + 1]
            i += 1
        else:
            cw.append(c + 1)
            i += 1
    return cw


def edifact_tails(raw: bytes) -> list[list[int]]:
    """EDIFACT from the latch to the end, with the endings encoders use:
    unlatch packed into the last group, unlatch alone after whole groups and
    the rest in ASCII, the last 1-2 characters in ASCII without unlatch, or
    no unlatch at all on whole groups (the last two need the symbol to end
    there; the decode check sorts that out)."""
    if not all(gen.is_edifact(c) for c in raw):
        return []
    vals = [c & 0x3F for c in raw]

    def pack(v: list[int]) -> list[int]:
        cw = []
        for i in range(0, len(v), 4):
            g = v[i:i + 4]
            x = 0
            for k in range(4):
                x = (x << 6) | (g[k] if k < len(g) else 0)
            cw += [(x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF][:(6 * len(g) + 7) // 8]
        return cw

    whole = len(vals) - len(vals) % 4
    out = [pack(vals + [0x1F]), pack(vals[:whole] + [0x1F]) + ascii_cw(raw[whole:])]
    for r in (0, 1, 2):
        if (len(vals) - r) % 4 == 0 and len(vals) > r:
            out.append(pack(vals[:len(vals) - r]) + ascii_cw(raw[len(vals) - r:]))
    return out


def triplet_tails(raw: bytes, conv) -> list[list[int]]:
    """C40 / Text / X12 from the latch to the end: whole triplets, then
    unlatch and ASCII, or no unlatch, or the last two values padded with a
    shift 1 (the decode check keeps what is valid for the symbol)."""
    vals, bounds = [], [0]
    for c in raw:
        v = conv(c)
        if v is None:
            return []
        vals += v
        bounds.append(len(vals))

    def pack(v: list[int]) -> list[int]:
        cw = []
        for i in range(0, len(v) - 2, 3):
            x = 1600 * v[i] + 40 * v[i + 1] + v[i + 2] + 1
            cw += [x >> 8, x & 0xFF]
        return cw

    j = max(k for k, b in enumerate(bounds) if b % 3 == 0)    # chars on a triplet boundary
    head = pack(vals[:bounds[j]])
    out = [head + [gen.UNLATCH_ASCII] + ascii_cw(raw[j:]), head + ascii_cw(raw[j:])]
    if len(vals) % 3 == 2:
        out.append(pack(vals + [0]))
    return out


def data_variants(raw: bytes) -> list[list[int]]:
    """Data codewords of the encodations a marking system may have used:
    the generator's choice, plain ASCII and, for texts that are not all
    digits, an ASCII prefix followed by EDIFACT, C40, Text or X12 to the end."""
    out = [gen.encode_data(raw)[0], ascii_cw(raw)]
    if all(gen.is_digit(c) for c in raw):
        return out
    tails = ((gen.EDF_MODE, edifact_tails),
             (gen.C40_MODE, lambda r: triplet_tails(r, gen.c40_value)),
             (gen.TEXT_MODE, lambda r: triplet_tails(r, gen.text_value)),
             (gen.X12_MODE, lambda r: triplet_tails(r, gen.x12_value)))
    for i in range(len(raw)):
        head = ascii_cw(raw[:i])
        for mode, fn in tails:
            out += [head + [gen.LATCH[mode]] + t for t in fn(raw[i:])]
    return out


def encode(texts: list[str], size: str | None) -> dict[str, list[tuple[tuple, np.ndarray]]]:
    """Text -> distinct (symbol, module matrix) of its valid encodations, each
    in every square and rectangular symbol with room for it and at most
    about twice the codewords it needs (marking systems do not always pick
    the smallest), or only in `size`. Valid = the v2 decoder reads the padded codewords
    back as the text."""
    syms = gen.SQUARE_SYMBOLS + gen.RECT_SYMBOLS
    out = {}
    for t in texts:
        try:
            raw = t.encode("latin-1")
        except UnicodeEncodeError:
            raw = t.encode("utf-8")
        seen, mats = set(), []
        for data in data_variants(raw):
            fit = [s for s in syms if f"{s[0]}x{s[1]}" == size] if size else \
                [s for s in syms if len(data) <= s[6] <= max(2 * len(data), len(data) + SIZE_SLACK)]
            for sym in fit:
                if sym[6] < len(data):
                    continue
                cw = list(data)
                gen.add_padding(cw, sym[6])
                key = (sym, tuple(cw))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    if v2.decode_codewords(cw).data != raw:
                        continue
                except (v2.DataError, IndexError, ValueError):
                    continue
                mats.append((sym, np.array(gen.build_matrix(gen.interleave_ecc(cw, sym)[0], sym), np.uint8)))
        if mats:
            out[t] = mats
        else:
            print(f"warning: cannot encode {t!r}" + (f" in {size}" if size else "") + "; skipped", file=sys.stderr)
    return out


class View:
    """The image the fit runs on: cropped and scaled around the corners when
    given. `to_image` maps its points back to image pixels."""

    def __init__(self, path: str, corners: np.ndarray | None):
        gray = cv2.cvtColor(v2.load_image(path), cv2.COLOR_BGR2GRAY)
        self.image_shape = gray.shape
        h, w = gray.shape
        self.off, self.scale = np.float32([0, 0]), 1.0
        if corners is not None:
            side = max(np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4))
            m = CROP_MARGIN * side
            x0, y0 = np.floor(corners.min(0) - m).astype(int)
            x1, y1 = np.ceil(corners.max(0) + m).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            gray = gray[y0:y1, x0:x1]
            self.off = np.float32([x0, y0])
            corners = corners - self.off
            # large photos: scale the symbol down to ~WORK_MODULE_PX * 26 px
            # (enough for sizes up to 26x26)
            limit = WORK_MODULE_PX * 26
            if side > limit:
                self.scale = limit / side
                gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
                corners = corners * self.scale
        elif max(h, w) > MAX_SIDE:       # whole-image search: large photos scaled down
            self.scale = MAX_SIDE / max(h, w)
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.gray, self.corners = gray, corners
        hh, ww = self.image_shape
        self.lo = (np.float32([0, 0]) - self.off) * self.scale        # image bounds in
        self.hi = (np.float32([ww - 1, hh - 1]) - self.off) * self.scale  # work pixels

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Work-pixel points that lie inside the image (not only the crop)."""
        return (x >= self.lo[0]) & (x <= self.hi[0]) & (y >= self.lo[1]) & (y <= self.hi[1])

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        return pts / self.scale + self.off


def variants(gray: np.ndarray, polarity: str) -> list[tuple[str, np.ndarray]]:
    """Raw and flattened + CLAHE; inverted for light marks on a dark ground."""
    m = min(gray.shape)
    flat = v2.clahe(v2.flatten_background(gray, m / 8))
    out = [("gray", gray), ("flatten > clahe", flat)]
    inv = [(f"invert > {n}", 255 - g) for n, g in out]
    return {"dark": out, "light": inv}.get(polarity, out + inv)


def completed(q: np.ndarray) -> list[np.ndarray]:
    """A symbol cut at the image edge shows as a short rectangle: grow the
    short side to the long one, away from either short-side edge."""
    q = v2.order_quad(q)
    out = []
    for k in (0, 1):                     # k = 0: grow along q0->q1, 1: along q1->q2
        a, b = q[k], q[k + 1]
        short, long_ = np.linalg.norm(b - a), np.linalg.norm(q[(k + 2) % 4] - b)
        if short >= 0.9 * long_:
            continue
        step = (b - a) / short * (long_ - short)
        grow_far = q.copy()              # keep the edge through a, move the other
        grow_far[k + 1] += step
        grow_far[(k + 2) % 4] += step
        grow_near = q.copy()             # keep the edge through b, move the other
        grow_near[k] -= step
        grow_near[(k + 3) % 4] -= step
        out += [grow_far, grow_near]
    return out


def rough_quads(gray: np.ndarray, corners: np.ndarray | None) -> list[np.ndarray]:
    rough = [corners] if corners is not None else v2.candidate_quads(gray, max_quads=20, min_area=MIN_AREA)
    quads = []
    for q in rough:
        q = np.asarray(q, np.float32)
        outer, inner = v2.snap_quad(gray, q), v2.snap_quad(gray, q, inward=True)
        snapped = [q, outer, inner] + v2.side_combos(outer, inner)
        quads += snapped + [c for s in snapped[:3] for c in completed(s)]
    return quads


def module_points(quads: np.ndarray, g: v2.Geometry, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Work-pixel positions (K, N) of module coordinates uv for K quads."""
    src = np.float64([[0, 0], [g.cols, 0], [g.cols, g.rows], [0, g.rows]])
    h = v2.homographies(src, quads)
    pts = np.concatenate([uv, np.ones((len(uv), 1))], 1)            # (N, 3)
    p = np.einsum("kij,nj->kni", h, pts)
    den = np.where(np.abs(p[..., 2]) < 1e-9, 1e-9, p[..., 2])
    return p[..., 0] / den, p[..., 1] / den


def inside_mask(view: View, quad: np.ndarray, g: v2.Geometry) -> np.ndarray:
    """Modules whose centre lies inside the image (not only the crop)."""
    cc, rr = np.meshgrid(np.arange(g.cols) + 0.5, np.arange(g.rows) + 0.5)
    x, y = module_points(quad[None], g, np.stack([cc.ravel(), rr.ravel()], 1))
    return view.inside(x[0], y[0]).reshape(g.rows, g.cols)


def score_quads(view: View, sampler: v2.Sampler, quads: np.ndarray, g: v2.Geometry) -> np.ndarray:
    """Contrast of the finder / timing modules inside the image only, so a
    symbol cut at the image edge is not pulled inside by the replicated
    border. The Fisher score of v2.score_quads times the square root of the
    share of the fixed pattern inside, so a few lucky modules of a mostly
    hidden fit do not win. -inf when under MIN_INSIDE of the fixed pattern
    is inside."""
    uv, dk = g.fp_uv_in, g.fp_dark_in
    x, y = module_points(quads, g, uv)
    ins = view.inside(x, y)
    v = sampler.sample_uv_many(quads, g.rows, g.cols, uv)
    d, li = ins & dk, ins & ~dk
    nd, nl = d.sum(1), li.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        md = (v * d).sum(1) / nd
        ml = (v * li).sum(1) / nl
        vd = (((v - md[:, None]) ** 2) * d).sum(1) / nd
        vl = (((v - ml[:, None]) ** 2) * li).sum(1) / nl
        s = (md - ml) / np.sqrt((vd + vl) / 2 + 4.0) * np.sqrt(ins.mean(1))
    ok = (ins.mean(1) >= MIN_INSIDE) & (nd >= 3) & (nl >= 3) & np.isfinite(s)
    return np.where(ok, s, -np.inf)


def refine_quad(view: View, sampler: v2.Sampler, quad: np.ndarray, g: v2.Geometry,
                steps: tuple = (1.0, 0.5, 0.25, 0.12), iters: int = 4) -> tuple[np.ndarray, float]:
    """Coordinate ascent on the corners (corner, edge, shift and scale moves,
    as v2.refine_quad) maximising score_quads."""
    q = quad.astype(np.float32).copy()
    mod = (np.linalg.norm(q[1] - q[0]) / g.cols + np.linalg.norm(q[3] - q[0]) / g.rows) / 2
    best = float(score_quads(view, sampler, q[None], g)[0])
    for step in [f * mod for f in steps]:
        for _ in range(iters):
            moves = []
            for k in range(4):
                for axis in (0, 1):
                    for sgn in (1, -1):
                        t = q.copy()
                        t[k, axis] += sgn * step
                        moves.append(t)
                k2 = (k + 1) % 4
                nrm = q[k2] - q[k]
                nrm = np.float32([-nrm[1], nrm[0]]) / (np.linalg.norm(nrm) + 1e-6)
                for sgn in (1, -1):
                    t = q.copy()
                    t[k] += sgn * step * nrm
                    t[k2] += sgn * step * nrm
                    moves.append(t)
            c = q.mean(0)
            moves += [q + [step, 0], q - [step, 0], q + [0, step], q - [0, step],
                      c + (q - c) * (1 + step / (mod * g.cols)), c + (q - c) * (1 - step / (mod * g.cols))]
            cands = np.array(moves, np.float32)
            sc = score_quads(view, sampler, cands, g)
            i = int(np.argmax(sc))
            if sc[i] <= best + 1e-4:
                break
            best, q = float(sc[i]), cands[i]
    return q, best


def fixed_score(vals: np.ndarray, g: v2.Geometry, inside: np.ndarray) -> float:
    """score_quads on sampled module values."""
    d = vals[g.fixed_mask & (g.fixed_val == 1) & inside]
    li = vals[g.fixed_mask & (g.fixed_val == 0) & inside]
    if inside.mean() < MIN_INSIDE or len(d) < 3 or len(li) < 3:
        return -np.inf
    seen = (g.fixed_mask & inside).sum() / g.fixed_mask.sum()
    return float((d.mean() - li.mean()) / np.sqrt((d.var() + li.var()) / 2 + 4.0) * np.sqrt(seen))


def find_fits(view: View, syms: list, polarity: str) -> list[dict]:
    """The FITS_KEEP strongest distinct finder / timing fits per candidate
    size over variants, quads and orientations; never looks at the data
    modules."""
    fits = []
    for vname, img in variants(view.gray, polarity):
        sampler = v2.Sampler(img)
        quads = rough_quads(img, view.corners)
        for sym in syms:
            g = v2.geometry(sym)
            cands = []
            for quad in quads:
                for oq, _rot, mirror in v2.orientations(quad):
                    w_ = np.linalg.norm(oq[1] - oq[0])
                    h_ = np.linalg.norm(oq[3] - oq[0])
                    if min(w_ / g.cols, h_ / g.rows) < 1.6 or not 0.6 < (w_ / g.cols) / (h_ / g.rows) < 1.6:
                        continue
                    cands += [(oq, mirror), (v2.expand_quad(oq, g), mirror)]
            if not cands:
                continue
            s = score_quads(view, sampler, np.array([c[0] for c in cands], np.float32), g)
            coarse = []
            for i in np.argsort(-s)[:COARSE_KEEP]:
                if not np.isfinite(s[i]):
                    break
                q, sc = refine_quad(view, sampler, cands[i][0], g, iters=2, steps=(1.0, 0.5))
                coarse.append((sc, q, cands[i][1]))
                # marks whose pitch drifts by more than half a module (laser on
                # wood): the v2 warped grid, which counts the timing dashes
                wf = v2.warp_fit(sampler, cands[i][0], g, float("inf"))
                if wf is not None:
                    gf, ucol, vrow = wf
                    inside = inside_mask(view, gf.quad, g)
                    vals = sampler.sample_grid(gf.quad, g.rows, g.cols, ucol, vrow)
                    fs = fixed_score(vals, g, inside)
                    if np.isfinite(fs):
                        fits.append(dict(fixed_score=fs, geom=g, quad=gf.quad, vals=vals, inside=inside,
                                         variant=vname, grid="warped", mirror=bool(cands[i][1])))
            coarse.sort(key=lambda t: -t[0])
            for _, q, mirror in coarse[:FINE_KEEP]:
                q, _ = refine_quad(view, sampler, q, g, steps=(0.5, 0.25, 0.12))
                inside = inside_mask(view, q, g)
                grids = [("straight", sampler.sample(q, g.rows, g.cols))]
                try:
                    axes = v2.tracked_axes(sampler, q, g)
                    grids.append(("tracked", sampler.sample_grid(q, g.rows, g.cols, *axes)))
                except (ValueError, np.linalg.LinAlgError, cv2.error):
                    pass
                for gname, vals in grids:
                    fs = fixed_score(vals, g, inside)
                    if np.isfinite(fs):
                        fits.append(dict(fixed_score=fs, geom=g, quad=q, vals=vals, inside=inside,
                                         variant=vname, grid=gname, mirror=bool(mirror)))
    fits.sort(key=lambda f: -f["fixed_score"])
    kept = []
    for f in fits:           # one per place: same size and corners within half a module
        same = [k for k in kept if k["geom"] is f["geom"]]
        if len(same) == FITS_KEEP:   # per size, so more candidate sizes do not crowd one out
            continue
        mod = np.linalg.norm(f["quad"][1] - f["quad"][0]) / f["geom"].cols
        if not any(np.abs(k["quad"] - f["quad"]).max() < mod / 2 for k in same):
            kept.append(f)
    return kept


def readings(fit: dict) -> list[tuple[bool, np.ndarray, np.ndarray]]:
    """(mirrored, bits, inside) for the fit as is and, for square symbols, for
    its mirror across the anti-diagonal: that mirror keeps the finder and
    timing pattern in place, so only the data can tell the two apart.
    `mirrored` is absolute: the mark is seen mirror-imaged."""
    g, inside = fit["geom"], fit["inside"]
    bits, _conf, _ = v2.classify(fit["vals"], g, "plane")
    out = [(fit["mirror"], bits, inside)]
    if g.rows == g.cols:
        out.append((not fit["mirror"], bits[::-1, ::-1].T, inside[::-1, ::-1].T))
    return out


@functools.lru_cache(maxsize=None)
def log_tail(n: int) -> np.ndarray:
    """log P(Bin(n, 0.5) >= k) for k = 0..n+1."""
    k = np.arange(n + 1)
    lp = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k]) - n * math.log(2)
    rev = np.logaddexp.accumulate(lp[::-1])[::-1]
    return np.append(rev, -np.inf)


def chance(k: int, n: int, tries: int) -> float:
    """Expected number of wrong texts that match at least k of n data modules
    by luck when `tries` text x fit x reading pairs are scored:
    tries * P(Bin(n, 0.5) >= k). A wrong text's modules are close to coin
    flips."""
    return float(f"{min(1.0, tries * math.exp(log_tail(n)[k])) if n else 1.0:.2g}")


def score(fits: list[dict], enc: dict[str, list[tuple[tuple, np.ndarray]]], top: int) -> list[dict]:
    """Every encodation of every candidate on every fit of its size and both
    readings; per candidate the pair least likely by luck counts. Details
    for the `top` best candidates."""
    items = [(t, sym, m) for t, ms in enc.items() for sym, m in ms]    # one per encodation
    reads = [(fi, r) for fi, f in enumerate(fits) for r in readings(f)]
    tries = len(items) * len(reads)
    best_k = np.full(len(items), -1)
    best_n = np.zeros(len(items), int)
    best_r = np.zeros(len(items), int)
    best_lt = np.full(len(items), np.inf)        # log P(luck), lower is better
    by_size: dict[tuple, list[int]] = {}
    for i, (_t, sym, _m) in enumerate(items):
        by_size.setdefault(sym[:2], []).append(i)
    for (rows, cols), idx in by_size.items():
        want = np.array([items[i][2] for i in idx], np.uint8)            # (C, R, C)
        for ri, (fi, (_m, bits, inside)) in enumerate(reads):
            g = fits[fi]["geom"]
            if (g.rows, g.cols) != (rows, cols):
                continue
            data = ~g.fixed_mask & inside
            k = (want[:, data] == bits[data]).sum(1)
            n = int(data.sum())
            lt = log_tail(n)[k]
            # the least likely by luck wins; on a tie the earlier (stronger) fit
            better = lt < best_lt[idx]
            sel = np.array(idx)[better]
            best_k[sel], best_n[sel], best_r[sel], best_lt[sel] = k[better], n, ri, lt[better]
    per_text: dict[str, int] = {}                # best encodation per text
    for i, (t, _s, _m) in enumerate(items):
        if best_k[i] >= 0 and (t not in per_text or best_lt[i] < best_lt[per_text[t]]):
            per_text[t] = i
    out = []
    for i in sorted(per_text.values(), key=lambda i: best_lt[i])[:top]:
        t, sym, want = items[i]
        fi, (mirrored, bits, inside) = reads[best_r[i]]
        g = fits[fi]["geom"]
        # codeword j belongs to block j % blocks (interleaving); a codeword
        # with any module outside the image counts as wrong
        wrong = ((bits != want) | ~inside)[g.cw_rows, g.cw_cols].any(1)
        per_block = [int(wrong[b::g.blocks].sum()) for b in range(g.blocks)]
        k, n = int(best_k[i]), int(best_n[i])
        out.append(dict(text=t, agreement=round(k / n, 4) if n else 0.0, modules=f"{k}/{n}",
                        chance=chance(k, n, tries), cw_errors=per_block,
                        recoverable=all(e <= g.n_ecc // g.blocks // 2 for e in per_block),
                        mirrored=mirrored, fit=fi, encodations=len(enc[t])))
    return out


def describe(fit: dict, view: View) -> dict:
    g, inside = fit["geom"], fit["inside"]
    bits = v2.classify(fit["vals"], g, "plane")[0]
    quad = view.to_image(fit["quad"])
    return dict(size=g.name, fixed_score=round(fit["fixed_score"], 2),
                fixed_agreement=round(float((bits == g.fixed_val)[g.fixed_mask & inside].mean()), 4),
                modules_inside=f"{int(inside.sum())}/{inside.size}",
                modules_px=round(float(np.linalg.norm(quad[1] - quad[0]) / g.cols
                                       + np.linalg.norm(quad[3] - quad[0]) / g.rows) / 2, 2),
                variant=fit["variant"], grid=fit["grid"],
                corners_px=[[round(float(x), 1), round(float(y), 1)] for x, y in quad])


def run(path: str, corners: np.ndarray | None, enc: dict[str, list[tuple[tuple, np.ndarray]]], top: int,
        polarity: str = "any") -> dict:
    view = View(path, corners)
    fits = find_fits(view, sorted({sym for ms in enc.values() for sym, _m in ms}), polarity)
    if not fits:
        return dict(file=path, fits=[])
    ranked = score(fits, enc, top)
    return dict(
        file=path,
        fits=[describe(f, view) for f in fits],
        candidates=len(enc),
        encodations=sum(len(m) for m in enc.values()),
        best=ranked[:top],
    )


def report(r: dict) -> str:
    if not r["fits"]:
        return f"{r['file']}: no grid fit"
    lines = [f"{r['file']}: {len(r['fits'])} fits, {r['candidates']} candidates, {r['encodations']} encodations"]
    for b in r["best"]:
        f = r["fits"][b["fit"]]
        lines.append(f"  {b['agreement']:.3f} ({b['modules']}, chance {b['chance']:.2g})  {b['text']!r}  "
                     f"codeword errors {b['cw_errors']}"
                     + ("  mirrored" if b["mirrored"] else "")
                     + ("  recoverable" if b["recoverable"] else "")
                     + f"\n      fit {b['fit']}: {f['size']}, fixed score {f['fixed_score']}, fixed pattern "
                     f"{f['fixed_agreement']:.0%}, {f['modules_inside']} modules inside, "
                     f"{f['modules_px']} px/module ({f['variant']}, {f['grid']} grid)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--text", action="append", default=[], help="candidate text (repeatable)")
    ap.add_argument("--texts-from", help="file with one candidate text per line")
    ap.add_argument("--digits", type=int, help="every N-digit number as a candidate")
    ap.add_argument("--corners", type=parse_corners, help='"x,y x,y x,y x,y" (single image only)')
    ap.add_argument("--size", help="force the symbol size of every candidate, e.g. 10x10")
    ap.add_argument("--polarity", choices=("dark", "light", "any"), default="any",
                    help="dark marks on a light ground, the reverse, or both (default)")
    ap.add_argument("--top", type=int, default=3, help="candidates to show per image (default 3)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    texts = list(a.text)
    if a.texts_from:
        texts += [ln.rstrip("\r\n") for ln in Path(a.texts_from).read_text("utf-8").splitlines() if ln.strip()]
    if a.digits:
        texts += [f"{i:0{a.digits}d}" for i in range(10 ** a.digits)]
    texts = list(dict.fromkeys(texts))
    if not texts:
        ap.error("no candidate text: give --text, --texts-from or --digits")
    if a.corners is not None and len(a.images) > 1:
        ap.error("--corners needs exactly one image")
    enc = encode(texts, a.size)
    if not enc:
        sys.exit(2)
    results = [run(p, a.corners, enc, a.top, a.polarity) for p in a.images]
    if a.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("\n".join(report(r) for r in results))


if __name__ == "__main__":
    main()
