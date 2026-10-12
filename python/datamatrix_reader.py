#!/usr/bin/env python3
"""Data Matrix (ECC 200) reader for poor quality marks.

Reads one or more images (files or folders) and prints the decoded content
together with a technical report: symbol size, data regions, encodation
segments, codewords, Reed-Solomon errors per block, orientation, module size
and estimated print quality figures.

Poor marks (laser burned / dot peened on wood, low contrast, uneven lighting,
grain texture, blur) are handled by a ladder of image enhancement steps and
three decoders tried in turn:

  zxing     zxing-cpp                  fast, used on every enhancement variant
  libdmtx   pylibdmtx                  slower, different detector
  grid      built-in ECC 200 decoder   locates the symbol, fits the module grid
                                       from the finder/timing pattern, samples
                                       modules and runs Reed-Solomon with soft
                                       erasures (low confidence codewords are
                                       marked as erasures, doubling the
                                       correction capacity for them)

Usage:
  python datamatrix_reader.py IMAGE_OR_FOLDER [...] [options]
  python datamatrix_reader.py                       (prompts for a path)

Options:
  --debug DIR      write enhancement variants and an annotated result image
  --json           print results as JSON instead of the text report
  --exhaustive     keep searching after the first good decode, keep the best
  --engines LIST   comma separated subset of zxing,libdmtx,grid
  --matrix         print the sampled module matrix as ASCII art
  --budget SEC     time limit per image (default 40)
  --jobs N         worker processes (default cpu count - 1, at most 8;
                   1 = sequential)
  --expect RxC     expected symbol size(s), e.g. --expect 24x24 (repeatable);
                   like the trained model of an industrial reader: prunes the
                   grid search to the hinted sizes (zxing/libdmtx unaffected)
  --polarity P     any = dark-on-light ladder plus a small inverted chain for
                   light-on-dark marks (default), dark = dark-on-light marks
                   only, light = light-on-dark marks (all variants inverted)

Large photos (longer side > 1400 px) are additionally scanned in up to 3x3
overlapping tiles near native resolution, so small symbols in wide shots are
not lost to the whole-image downscale. When the grid decoder fails with the
quiet-zone template it retries without it (marks flush against other print).

Dependencies: numpy, opencv-python(-headless), zxing-cpp; pylibdmtx optional.
Symbol tables and the ECC 200 placement come from datamatrix_gen.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamatrix_gen import (RECT_SYMBOLS, SQUARE_SYMBOLS, build_matrix,  # noqa: E402
                            placement_map)

try:
    import zxingcpp
except ImportError:                                  # pragma: no cover
    zxingcpp = None

try:
    from pylibdmtx import pylibdmtx
except Exception:                                    # missing DLL / distutils
    pylibdmtx = None

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
# DMRE - Data Matrix Rectangular Extension (ISO/IEC 21471), same layout
# rules as ECC 200: rows cols dregr dregc regr regc data eccblk blocks
DMRE_SYMBOLS = [
    (8, 48, 6, 22, 1, 2, 18, 15, 1), (8, 64, 6, 14, 1, 4, 24, 18, 1),
    (8, 80, 6, 18, 1, 4, 32, 22, 1), (8, 96, 6, 22, 1, 4, 38, 28, 1),
    (8, 120, 6, 18, 1, 6, 49, 32, 1), (8, 144, 6, 22, 1, 6, 63, 36, 1),
    (12, 64, 10, 14, 1, 4, 43, 27, 1), (12, 88, 10, 20, 1, 4, 64, 36, 1),
    (16, 64, 14, 14, 1, 4, 62, 36, 1), (20, 36, 18, 16, 1, 2, 44, 28, 1),
    (20, 44, 18, 20, 1, 2, 56, 34, 1), (20, 64, 18, 14, 1, 4, 84, 42, 1),
    (22, 48, 20, 22, 1, 2, 72, 38, 1), (24, 48, 22, 22, 1, 2, 80, 41, 1),
    (24, 64, 22, 14, 1, 4, 108, 46, 1), (26, 40, 24, 18, 1, 2, 70, 38, 1),
    (26, 48, 24, 22, 1, 2, 90, 42, 1), (26, 64, 24, 14, 1, 4, 118, 50, 1),
]
ALL_SYMBOLS = SQUARE_SYMBOLS + RECT_SYMBOLS + DMRE_SYMBOLS


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """Load as BGR; np.fromfile keeps non-ASCII Windows paths working."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    return img


def collect_inputs(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if os.path.splitext(name)[1].lower() in IMAGE_EXT:
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out


def parse_sizes(specs: list[str]) -> list | None:
    """--expect values (e.g. 24x24, 12x26) -> symbol tuples, or None (all)."""
    out = set()
    for spec in specs:
        for part in spec.replace("×", "x").replace("X", "x").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                r, c = part.split("x")
                out.add((int(r), int(c)))
            except ValueError:
                print(f"warning: ignoring malformed size hint {part!r}", file=sys.stderr)
    if not out:
        return None
    known = {(s[0], s[1]) for s in ALL_SYMBOLS}
    bad = out - known
    if bad:
        print(f"warning: unknown symbol size(s) {sorted(bad)} ignored", file=sys.stderr)
        out &= known
    return sorted(out) or None


# --------------------------------------------------------------------------
# Image enhancement
# --------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    img: np.ndarray            # 8-bit grayscale, symbol expected dark on light
    scale: float = 1.0         # original px = variant px / scale + (ox, oy)
    ox: float = 0.0            # crop offset in original pixels (tiles)
    oy: float = 0.0

    @property
    def tile(self) -> bool:
        return self.name.startswith("tile")


def map_quad(q: np.ndarray, src: Variant, dst: Variant) -> np.ndarray:
    """Quad in `src` variant pixels -> `dst` variant pixels."""
    if not (src.ox or src.oy or dst.ox or dst.oy):
        return q * (dst.scale / src.scale)
    orig = q / src.scale + np.float32([src.ox, src.oy])
    return ((orig - np.float32([dst.ox, dst.oy])) * dst.scale).astype(np.float32)


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def channels(bgr: np.ndarray, invert: bool = False) -> dict[str, np.ndarray]:
    """Grayscale candidates. Burn marks on wood separate best in different
    channels depending on wood colour and lighting. invert=True flips every
    candidate for light-on-dark marks (chalk / paint / light laser marks on
    dark material) - the polarity hint of an industrial reader."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    out = {
        "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        "sat": 255 - hsv[:, :, 1],          # burns are less saturated
        "blue": bgr[:, :, 0],
        "lab-L": lab[:, :, 0],
    }
    if invert:
        out = {f"inv-{k}": 255 - v for k, v in out.items()}
    return out


def clahe(g: np.ndarray, clip: float = 3.0, tiles: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)


def flatten_background(g: np.ndarray, sigma: float) -> np.ndarray:
    """Divide by a heavily blurred copy: removes lighting gradients/shadows."""
    bg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma) + 1.0
    flat = g.astype(np.float32) / bg
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def merge_dots(g: np.ndarray, k: int) -> np.ndarray:
    """Grey erosion grows dark dots so dot-peened modules touch each other."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(g, se)


def remove_texture(g: np.ndarray, k: int) -> np.ndarray:
    """Grey closing removes thin dark grain lines narrower than k."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)


def sharpen(g: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1 + amount, blur, -amount, 0)


def otsu(g: np.ndarray) -> np.ndarray:
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive(g: np.ndarray, block: int) -> np.ndarray:
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, _odd(block), 5)


def gamma(g: np.ndarray, gm: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** gm) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(g, lut)


def normalise_size(g: np.ndarray, target: int = 700) -> tuple[np.ndarray, float]:
    """Bring the longer side near `target` px (detectors have sweet spots)."""
    s = target / max(g.shape)
    if 0.8 < s < 1.25:
        return g, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(g, None, fx=s, fy=s, interpolation=interp), s


def enhancement_variants(bgr: np.ndarray, invert: bool = False,
                         inverted_extra: bool = True) -> list[Variant]:
    """Ordered from cheap/faithful to aggressive. Each variant records the
    processing chain in its name so the report can say what worked.
    inverted_extra: append a small inverted chain for polarity "any"
    (skipped when the polarity is known: dark or light)."""
    out: list[Variant] = []
    for cname, g0 in channels(bgr, invert).items():
        g, s = normalise_size(g0)
        m = min(g.shape)
        out.append(Variant(f"{cname}", g, s))
        cl = clahe(g)
        out.append(Variant(f"{cname} > clahe", cl, s))
        flat = clahe(flatten_background(g, m / 8))
        out.append(Variant(f"{cname} > flatten > clahe", flat, s))
        for sig in (1.0, 2.0):
            b = cv2.GaussianBlur(flat, (0, 0), sig)
            out.append(Variant(f"{cname} > flatten > clahe > blur{sig:g}", b, s))
            out.append(Variant(f"{cname} > flatten > clahe > blur{sig:g} > otsu", otsu(b), s))
        out.append(Variant(f"{cname} > median5 > sharpen", sharpen(cv2.medianBlur(g, 5)), s))
        for k in (3, 5, 7, 9):
            md = merge_dots(cv2.GaussianBlur(flat, (0, 0), 1), k)
            out.append(Variant(f"{cname} > flatten > clahe > merge-dots{k}", md, s))
            out.append(Variant(f"{cname} > flatten > clahe > merge-dots{k} > otsu", otsu(md), s))
        tx = remove_texture(flat, 3)
        out.append(Variant(f"{cname} > flatten > clahe > de-grain3 > merge-dots5 > otsu",
                           otsu(merge_dots(tx, 5)), s))
        out.append(Variant(f"{cname} > clahe > adaptive", adaptive(cv2.GaussianBlur(cl, (0, 0), 1.5), m / 6), s))
        out.append(Variant(f"{cname} > gamma2 > clahe", clahe(gamma(g, 2.0)), s))
    # extra scales of the most useful chain: module size matters for detectors
    g0 = channels(bgr)["gray"]
    for tgt in (350, 1100):
        g, s = normalise_size(g0, tgt)
        flat = clahe(flatten_background(g, min(g.shape) / 8))
        out.append(Variant(f"gray@{tgt}px > flatten > clahe", flat, s))
        for k in (3, 5, 9):
            out.append(Variant(f"gray@{tgt}px > flatten > clahe > merge-dots{k} > otsu",
                               otsu(merge_dots(cv2.GaussianBlur(flat, (0, 0), 1), k)), s))
    if not invert and inverted_extra:
        # polarity "any": a small inverted chain so light-on-dark marks are not
        # systematically missed (with --polarity light the whole ladder already
        # runs inverted; with --polarity dark it is skipped by request)
        out += inverted_chain(bgr)
    return out


def inverted_chain(bgr: np.ndarray) -> list[Variant]:
    """Light-on-dark marks: the useful gray chain, inverted."""
    out: list[Variant] = []
    g0 = channels(bgr, invert=True)["inv-gray"]
    g, s = normalise_size(g0)
    m = min(g.shape)
    out.append(Variant("inv-gray", g, s))
    flat = clahe(flatten_background(g, m / 8))
    out.append(Variant("inv-gray > flatten > clahe", flat, s))
    out.append(Variant("inv-gray > flatten > clahe > merge-dots5 > otsu",
                       otsu(merge_dots(cv2.GaussianBlur(flat, (0, 0), 1), 5)), s))
    return out


def grid_variants(bgr: np.ndarray, invert: bool = False,
                  inverted_extra: bool = True) -> list[Variant]:
    """Grayscale variants for the grid decoder (no binarisation: the grid
    decoder thresholds per module itself and needs the analogue values).
    inverted_extra: add one inverted variant for polarity "any"."""
    out = []
    ch = channels(bgr, invert)
    for cname in list(ch)[:3]:
        g, s = normalise_size(ch[cname], 600)
        out.append(Variant(f"{cname} > flatten", flatten_background(g, min(g.shape) / 6), s))
    gk = list(ch)[0]
    g, s = normalise_size(ch[gk], 600)
    out.append(Variant(gk, g, s))
    out.append(Variant(f"{gk} > flatten > clahe", clahe(flatten_background(g, min(g.shape) / 6), 2.0, 4), s))
    if not invert and inverted_extra:
        # polarity "any": light-on-dark marks, one inverted grid variant
        g, s = normalise_size(channels(bgr, invert=True)["inv-gray"], 600)
        out.append(Variant("inv-gray > flatten", flatten_background(g, min(g.shape) / 6), s))
    return out


TILE_TRIGGER = 1400      # longer side above which tiles are scanned
TILE_SIZE = 900          # tile size handed to the detectors (px)
TILE_OVERLAP = 0.3       # a symbol on a tile border is whole in a neighbour
TILE_MAX = 3             # tiles per axis


def _tile_axis(length: int) -> tuple[int, int]:
    """(tile count, tile length in original px) along one axis."""
    n = math.ceil((length - TILE_SIZE) / (TILE_SIZE * (1 - TILE_OVERLAP))) + 1
    n = min(TILE_MAX, max(1, n))
    t = max(TILE_SIZE, math.ceil(length / (n - (n - 1) * TILE_OVERLAP)))
    return n, min(length, t)


def tile_variants(bgr: np.ndarray, invert: bool = False) -> list[list[Variant]]:
    """Overlapping tiles of a large photo, one small variant set per tile.
    Up to 3x3 tiles; a tile larger than TILE_SIZE (photos over ~2200 px) is
    downscaled to it, which still keeps ~3x the resolution of the whole-image
    variants. Empty for images that are not large."""
    h, w = bgr.shape[:2]
    if max(h, w) <= TILE_TRIGGER:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = 255 - gray
    (ny, th), (nx, tw) = _tile_axis(h), _tile_axis(w)
    s = min(1.0, TILE_SIZE / max(th, tw))
    ys = [round(i * (h - th) / max(1, ny - 1)) for i in range(ny)]
    xs = [round(i * (w - tw) / max(1, nx - 1)) for i in range(nx)]
    out = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            g = gray[y0:y0 + th, x0:x0 + tw]
            if s < 1.0:
                g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            name = f"tile{r}x{c}"
            flat = clahe(flatten_background(g, min(g.shape) / 8))
            md = otsu(merge_dots(cv2.GaussianBlur(flat, (0, 0), 1), 5))
            out.append([Variant(f"{name} > gray", g, s, x0, y0),
                        Variant(f"{name} > gray > flatten > clahe", flat, s, x0, y0),
                        Variant(f"{name} > gray > flatten > clahe > merge-dots5 > otsu", md, s, x0, y0)])
    return out


# --------------------------------------------------------------------------
# Reed-Solomon decoding over GF(256), polynomial 0x12D, roots a^1..a^n
# --------------------------------------------------------------------------
GF_EXP = [0] * 512
GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    GF_EXP[_i] = _x
    GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x12D
for _i in range(255, 512):
    GF_EXP[_i] = GF_EXP[_i - 255]
GF_EXP_NP = np.array(GF_EXP, np.int32)
GF_LOG_NP = np.array(GF_LOG, np.int32)


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_div(a: int, b: int) -> int:
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] + 255 - GF_LOG[b]) % 255]


def gf_pow(a: int, p: int) -> int:
    return GF_EXP[(GF_LOG[a] * p) % 255]


def gf_inv(a: int) -> int:
    return GF_EXP[255 - GF_LOG[a]]


def poly_mul(p: list[int], q: list[int]) -> list[int]:
    r = [0] * (len(p) + len(q) - 1)
    for j, b in enumerate(q):
        if b:
            for i, a in enumerate(p):
                if a:
                    r[i + j] ^= gf_mul(a, b)
    return r


def poly_add(p: list[int], q: list[int]) -> list[int]:
    n = max(len(p), len(q))
    r = [0] * n
    for i, c in enumerate(p):
        r[i + n - len(p)] = c
    for i, c in enumerate(q):
        r[i + n - len(q)] ^= c
    return r


def poly_scale(p: list[int], x: int) -> list[int]:
    return [gf_mul(c, x) for c in p]


def _gf_eval_many(coef: list[int], expo: np.ndarray) -> np.ndarray:
    """XOR_k coef[k] * a^(expo[j, k]) for every row j, in the log domain
    (vectorised Horner-free polynomial evaluation)."""
    c = np.asarray(coef, np.int32)
    nz = c != 0
    if not nz.any():
        return np.zeros(expo.shape[0], np.int32)
    terms = GF_EXP_NP[(GF_LOG_NP[c[nz]][None, :] + expo[:, nz]) % 255]
    return np.bitwise_xor.reduce(terms, axis=1)


def rs_syndromes(msg: list[int], nsym: int) -> list[int]:
    """S_j = msg(a^j), j = 1..nsym; msg highest power first."""
    n = len(msg)
    expo = np.outer(np.arange(1, nsym + 1), np.arange(n - 1, -1, -1))
    return _gf_eval_many(msg, expo).tolist()


def _solve_magnitudes(synd: list[int], positions: list[int], n: int) -> list[int] | None:
    """Gaussian elimination in GF(256): S_j = sum e_k * X_k^j, j = 1..v."""
    v = len(positions)
    xs = [GF_EXP[(n - 1 - p) % 255] for p in positions]
    a = [[gf_pow(x, j + 1) for x in xs] + [synd[j]] for j in range(v)]
    for col in range(v):
        piv = next((r for r in range(col, v) if a[r][col]), None)
        if piv is None:
            return None
        a[col], a[piv] = a[piv], a[col]
        inv = gf_inv(a[col][col])
        a[col] = [gf_mul(c, inv) for c in a[col]]
        for r in range(v):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [c ^ gf_mul(f, d) for c, d in zip(a[r], a[col])]
    return [a[r][v] for r in range(v)]


def rs_correct(msg: list[int], nsym: int, erasures: list[int] = (),
               synd: list[int] | None = None) -> tuple[list[int], list[int]] | None:
    """Correct errors and erasures. Returns (corrected, error_positions) where
    error_positions excludes erasures that turned out to be right, or None.
    synd: precomputed syndromes of msg (they do not depend on the erasures)."""
    n = len(msg)
    erasures = list(erasures)
    if len(erasures) > nsym:
        return None
    if synd is None:
        synd = rs_syndromes(msg, nsym)
    if not any(synd):
        return list(msg), []
    # erasure locator Gamma(x) = prod (1 - X_i x), stored highest power first
    gamma_ = [1]
    for p in erasures:
        gamma_ = poly_mul(gamma_, [GF_EXP[(n - 1 - p) % 255], 1])
    # Forney (modified) syndromes T = Gamma * S, coefficients f .. nsym-1
    s_low = synd                                  # S_1..S_n as x^0..x^(n-1)
    g_low = gamma_[::-1]
    t = [0] * nsym
    for i, gc in enumerate(g_low):
        if gc:
            for j, sc in enumerate(s_low):
                if i + j < nsym and sc:
                    t[i + j] ^= gf_mul(gc, sc)
    f = len(erasures)
    seq = t[f:]
    # Berlekamp-Massey on the modified syndromes (lowest power first)
    c_poly, b_poly = [1], [1]
    l_, m_, b_ = 0, 1, 1
    for k in range(len(seq)):
        d = seq[k]
        for i in range(1, l_ + 1):
            if i < len(c_poly) and k - i >= 0:
                d ^= gf_mul(c_poly[i], seq[k - i])
        if d == 0:
            m_ += 1
            continue
        coef = gf_div(d, b_)
        tmp = list(c_poly)
        shifted = [0] * m_ + [gf_mul(coef, c) for c in b_poly]
        if len(shifted) > len(c_poly):
            c_poly += [0] * (len(shifted) - len(c_poly))
        for i, c in enumerate(shifted):
            c_poly[i] ^= c
        if 2 * l_ <= k:
            l_ = k + 1 - l_
            b_poly, b_, m_ = tmp, d, 1
        else:
            m_ += 1
    while len(c_poly) > 1 and c_poly[-1] == 0:
        c_poly.pop()
    n_err = len(c_poly) - 1
    if 2 * n_err + f > nsym:
        return None
    # Chien search: root at X^-1 means error at position p with X = a^(n-1-p)
    err_pos = []
    if n_err:
        # Lambda(X^-1) = XOR_k c_k * a^(-k (n-1-p)) for all positions p at once
        e = (n - 1 - np.arange(n)) % 255
        expo = (-np.outer(e, np.arange(len(c_poly)))) % 255
        err_pos = np.nonzero(_gf_eval_many(c_poly, expo) == 0)[0].tolist()
        if len(err_pos) != n_err:
            return None
    positions = sorted(set(erasures) | set(err_pos))
    mags = _solve_magnitudes(synd, positions, n)
    if mags is None:
        return None
    out = list(msg)
    for p, e in zip(positions, mags):
        out[p] ^= e
    if any(rs_syndromes(out, nsym)):
        return None
    changed = [p for p in positions if out[p] != msg[p]]
    return out, changed


# --------------------------------------------------------------------------
# Codeword stream -> data (ISO/IEC 16022 section 5.2)
# --------------------------------------------------------------------------
C40_BASIC = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEXT_BASIC = " 0123456789abcdefghijklmnopqrstuvwxyz"
SHIFT2 = "!\"#$%&'()*+,-./:;<=>?@[\\]^_"
TEXT_SHIFT3 = "`ABCDEFGHIJKLMNOPQRSTUVWXYZ{|}~\x7f"
X12_SET = "\r*> 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class DataError(Exception):
    pass


@dataclass
class Segment:
    mode: str
    cw_start: int
    cw_end: int                    # exclusive
    data: bytearray = field(default_factory=bytearray)
    note: str = ""


@dataclass
class Decoded:
    data: bytes
    segments: list[Segment]
    payload_cw: int                # codewords up to and incl. first PAD
    pad_cw: int
    gs1: bool = False
    eci: list[int] = field(default_factory=list)
    macro: str = ""
    structured_append: str = ""
    reader_programming: bool = False
    warnings: list[str] = field(default_factory=list)


def _unrand255(v: int, pos: int) -> int:
    t = v - (((149 * pos) % 255) + 1)
    return t if t >= 0 else t + 256


def _unrand253(v: int, pos: int) -> int:
    t = v - (((149 * pos) % 253) + 1)
    return t if t >= 1 else t + 254


def decode_codewords(cw: list[int]) -> Decoded:
    n = len(cw)
    out = bytearray()
    segs: list[Segment] = []
    res = Decoded(b"", segs, n, 0)
    i = 0
    upper = False
    seg = Segment("ASCII", 0, 0)
    segs.append(seg)

    def emit(b: int) -> None:
        nonlocal upper
        if upper:
            b += 128
            upper = False
        out.append(b & 0xFF)
        seg.data.append(b & 0xFF)

    def set_upper() -> None:
        nonlocal upper
        upper = True

    def new_seg(mode: str, start: int) -> Segment:
        nonlocal seg
        if seg.cw_end == 0:
            seg.cw_end = start
        seg = Segment(mode, start, 0)
        segs.append(seg)
        return seg

    while i < n:
        c = cw[i]
        if c == 0:
            raise DataError(f"codeword 0 at {i + 1} is invalid")
        if c <= 128:
            emit(c - 1); i += 1
        elif c == 129:                                   # PAD: end of data
            res.payload_cw = i
            res.pad_cw = n - i
            for k in range(i + 1, n):
                if _unrand253(cw[k], k + 1) != 129:
                    res.warnings.append(f"bad padding codeword at {k + 1}")
                    break
            break
        elif c <= 229:
            v = c - 130
            emit(48 + v // 10); emit(48 + v % 10); i += 1
        elif c in (230, 239, 238):                       # C40 / Text / X12
            mode = {230: "C40", 239: "TEXT", 238: "X12"}[c]
            new_seg(mode, i)
            i = _decode_ctx(cw, i + 1, mode, emit, set_upper)
            new_seg("ASCII", i)
        elif c == 231:                                   # Base 256
            new_seg("BASE256", i)
            i += 1
            if i >= n:
                raise DataError("BASE256 length missing")
            d1 = _unrand255(cw[i], i + 1); i += 1
            if d1 == 0:
                length = n - i
            elif d1 < 250:
                length = d1
            else:
                if i >= n:
                    raise DataError("BASE256 length truncated")
                length = 250 * (d1 - 249) + _unrand255(cw[i], i + 1); i += 1
            if i + length > n:
                raise DataError("BASE256 field runs past the symbol")
            for _ in range(length):
                emit(_unrand255(cw[i], i + 1)); i += 1
            new_seg("ASCII", i)
        elif c == 232:                                   # FNC1
            if i == 0 or (i == 1 and res.structured_append):
                res.gs1 = True
                seg.note = (seg.note + " FNC1(GS1)").strip()
            else:
                emit(29)                                 # <GS> separator
            i += 1
        elif c == 233:                                   # Structured append
            if i + 3 > n:
                raise DataError("structured append truncated")
            seq = cw[i + 1]
            res.structured_append = (f"symbol {(seq >> 4) + 1} of {17 - (seq & 15)}, "
                                     f"file id {cw[i + 2]}/{cw[i + 3]}")
            i += 4
        elif c == 234:
            res.reader_programming = True; i += 1
        elif c == 235:
            upper = True; i += 1
        elif c in (236, 237):
            res.macro = "05" if c == 236 else "06"
            for b in b"[)>\x1e" + res.macro.encode() + b"\x1d":
                emit(b)
            i += 1
        elif c == 240:                                   # EDIFACT
            new_seg("EDIFACT", i)
            i = _decode_edifact(cw, i + 1, emit)
            new_seg("ASCII", i)
        elif c == 241:                                   # ECI
            c1 = cw[i + 1] if i + 1 < n else 0
            if c1 <= 127:
                val, i = c1 - 1, i + 2
            elif c1 <= 191:
                val, i = (c1 - 128) * 254 + (cw[i + 2] - 1) + 127, i + 3
            else:
                val, i = (c1 - 192) * 64516 + (cw[i + 2] - 1) * 254 + (cw[i + 3] - 1) + 16383, i + 4
            res.eci.append(val)
            seg.note = (seg.note + f" ECI {val:06d}").strip()
        elif c == 254:
            i += 1                                       # stray unlatch, ignore
        else:
            raise DataError(f"invalid codeword {c} at {i + 1}")
    if seg.cw_end == 0:
        seg.cw_end = min(i, res.payload_cw)
    if res.macro:
        out += b"\x1e\x04"
    res.data = bytes(out)
    res.segments = [s for s in segs if s.cw_end > s.cw_start or s.data]
    return res


def _decode_ctx(cw, i, mode, emit, set_upper) -> int:
    n = len(cw)
    shift = 0
    while i < n:
        if cw[i] == 254:                                 # unlatch
            return i + 1
        if i + 1 >= n:                                   # single cw left: ASCII
            return i
        v = cw[i] * 256 + cw[i + 1] - 1
        i += 2
        for val in (v // 1600, (v // 40) % 40, v % 40):
            if mode == "X12":
                if val >= len(X12_SET):
                    raise DataError("invalid X12 value")
                emit(ord(X12_SET[val]))
                continue
            if shift == 0:
                if val < 3:
                    shift = val + 1
                else:
                    emit(ord((C40_BASIC if mode == "C40" else TEXT_BASIC)[val - 3]))
            elif shift == 1:
                emit(val); shift = 0
            elif shift == 2:
                if val < 27:
                    emit(ord(SHIFT2[val]))
                elif val == 27:
                    emit(29)                             # FNC1 -> <GS>
                elif val == 30:
                    set_upper()
                else:
                    raise DataError(f"invalid {mode} shift-2 value {val}")
                shift = 0
            else:
                emit(96 + val if mode == "C40" else ord(TEXT_SHIFT3[val]))
                shift = 0
        if i < n and cw[i] == 254:
            return i + 1
    return i


def _decode_edifact(cw, i, emit) -> int:
    n = len(cw)
    while i < n:
        if n - i <= 2:                                   # implicit return
            return i
        v = (cw[i] << 16) | (cw[i + 1] << 8) | cw[i + 2]
        for k, sh in enumerate((18, 12, 6, 0)):
            e = (v >> sh) & 0x3F
            if e == 0x1F:                                # unlatch: rest of its
                return i + min(3, (6 * (k + 1) + 7) // 8)   # codeword is filler
            emit(e | 0x40 if e < 0x20 else e)
        i += 3
    return i


# --------------------------------------------------------------------------
# GS1 element strings (subset of the AI table, enough for common labels)
# --------------------------------------------------------------------------
GS1_AI = {   # ai: (fixed length or None, title)
    "00": (18, "SSCC"), "01": (14, "GTIN"), "02": (14, "CONTENT"),
    "10": (None, "BATCH/LOT"), "11": (6, "PROD DATE"), "12": (6, "DUE DATE"),
    "13": (6, "PACK DATE"), "15": (6, "BEST BEFORE"), "16": (6, "SELL BY"),
    "17": (6, "USE BY"), "20": (2, "VARIANT"), "21": (None, "SERIAL"),
    "22": (None, "CPV"), "240": (None, "ADDITIONAL ID"), "241": (None, "CUST PART NO"),
    "250": (None, "SECONDARY SERIAL"), "30": (None, "VAR COUNT"), "37": (None, "COUNT"),
    "400": (None, "ORDER NUMBER"), "410": (13, "SHIP TO LOC"), "414": (13, "LOC NO"),
    "420": (None, "SHIP TO POST"), "7003": (10, "EXPIRY TIME"), "8004": (None, "GIAI"),
    "90": (None, "INTERNAL"), "91": (None, "INTERNAL"), "92": (None, "INTERNAL"),
    "93": (None, "INTERNAL"), "94": (None, "INTERNAL"), "95": (None, "INTERNAL"),
    "96": (None, "INTERNAL"), "97": (None, "INTERNAL"), "98": (None, "INTERNAL"),
    "99": (None, "INTERNAL"),
}


def parse_gs1(text: str) -> list[tuple[str, str, str]]:
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\x1d":
            i += 1
            continue
        ai = None
        for ln in (2, 3, 4):
            cand = text[i:i + ln]
            if cand in GS1_AI:
                ai = cand
                break
            if ln == 4 and cand[:3] in ("310", "320", "330", "390", "392", "393"):
                ai = cand
        if ai is None:                                    # measures 31nn-36nn etc.
            ai = text[i:i + 4] if text[i:i + 2] in ("31", "32", "33", "34", "35", "36") else text[i:i + 2]
            fixed, title = (6, "MEASURE") if text[i:i + 2] in ("31", "32", "33", "34", "35", "36") else (None, "?")
        else:
            fixed, title = GS1_AI.get(ai, (None, "VALUE"))
        i += len(ai)
        if fixed:
            val = text[i:i + fixed]
            i += fixed
        else:
            j = text.find("\x1d", i)
            j = len(text) if j < 0 else j
            val = text[i:j]
            i = j
        out.append((ai, title, val))
    return out


# --------------------------------------------------------------------------
# Symbol geometry
# --------------------------------------------------------------------------
@dataclass
class Geometry:
    sym: tuple
    rows: int
    cols: int
    fixed_mask: np.ndarray       # True on finder / timing modules
    fixed_val: np.ndarray        # expected value there (1 = dark)
    cw_rows: np.ndarray          # (n_cw, 8) module row of each codeword bit
    cw_cols: np.ndarray
    n_data: int
    n_ecc: int
    blocks: int
    pad_mask: np.ndarray = None  # fixed pattern + 1 module quiet zone ring
    pad_val: np.ndarray = None
    fp_uv_qz: np.ndarray = None  # (k, 2) module centres (x, y) of pad_mask
    fp_dark_qz: np.ndarray = None  # (k,) bool expected dark
    fp_uv_in: np.ndarray = None  # same for the fixed pattern only (no ring)
    fp_dark_in: np.ndarray = None

    @property
    def name(self) -> str:
        return f"{self.rows}x{self.cols}"


_GEOM_CACHE: dict[tuple, Geometry] = {}


def geometry(sym) -> Geometry:
    if sym in _GEOM_CACHE:
        return _GEOM_CACHE[sym]
    rows, cols, dregr, dregc, regr, regc, data, eccblk, blocks = sym
    n_cw = data + eccblk * blocks
    template = np.array(build_matrix([0] * n_cw, sym), dtype=np.uint8)
    mask = np.zeros((rows, cols), bool)
    for rv in range(regr):
        r0 = rv * (dregr + 2)
        mask[r0, :] = mask[r0 + dregr + 1, :] = True
    for rh in range(regc):
        c0 = rh * (dregc + 2)
        mask[:, c0] = mask[:, c0 + dregc + 1] = True
    place = placement_map(dregr * regr, dregc * regc)
    cw_r = np.zeros((n_cw, 8), np.int32)
    cw_c = np.zeros((n_cw, 8), np.int32)
    for r, line in enumerate(place):
        for c, cell in enumerate(line):
            if cell is None or cell[0] == 0:
                continue
            chrn, bit = cell
            rv, rh = divmod(r, dregr)[0], divmod(c, dregc)[0]
            fr = rv * (dregr + 2) + 1 + r % dregr
            fc = rh * (dregc + 2) + 1 + c % dregc
            cw_r[chrn - 1, bit - 1] = fr
            cw_c[chrn - 1, bit - 1] = fc
    pmask = np.ones((rows + 2, cols + 2), bool)
    pmask[1:-1, 1:-1] = mask
    pval = np.zeros((rows + 2, cols + 2), np.uint8)
    pval[1:-1, 1:-1] = template
    pr, pc = np.nonzero(pmask)
    fp_uv = np.stack([pc - 1 + 0.5, pr - 1 + 0.5], 1).astype(np.float64)
    fp_dark = pval[pr, pc] == 1
    inner = (pr >= 1) & (pr <= rows) & (pc >= 1) & (pc <= cols)
    g = Geometry(sym, rows, cols, mask, template, cw_r, cw_c, data, eccblk, blocks, pmask, pval,
                 fp_uv, fp_dark, fp_uv[inner], fp_dark[inner])
    _GEOM_CACHE[sym] = g
    return g


def deinterleave(cw: list[int], g: Geometry) -> list[tuple[list[int], list[int]]]:
    """Returns per block (positions in the stream, codewords) data+ecc."""
    out = []
    for b in range(g.blocks):
        pos = list(range(b, g.n_data, g.blocks))
        pos += [g.n_data + k * g.blocks + b for k in range(g.n_ecc)]
        out.append((pos, [cw[p] for p in pos]))
    return out


@dataclass
class RSResult:
    data_cw: list[int]
    all_cw: list[int]
    block_errors: list[int]        # codewords actually changed per block
    block_erasures: list[int]      # erasures declared per block
    corrected_positions: list[int]
    uec: float                     # unused error correction (0..1)


def rs_decode_symbol(cw: list[int], conf: np.ndarray, g: Geometry,
                     max_erasure_frac: float = 0.75) -> RSResult | None:
    """Decode all blocks; on failure retry with the least confident codewords
    declared as erasures (2 per step). Erasure decoding can "correct" garbage
    into a valid but wrong codeword, so a reserve of unused check codewords
    (2e + f <= n - margin) is required whenever erasures are used."""
    fixed = list(cw)
    errs, eras, changed_all, uecs = [], [], [], []
    margin = 2 if g.n_ecc < 20 else 3
    for pos, block in deinterleave(cw, g):
        order = sorted(range(len(block)), key=lambda k: conf[pos[k]])
        max_f = int(g.n_ecc * max_erasure_frac)
        synd = rs_syndromes(block, g.n_ecc)               # same for every erasure set
        ok = None
        for f in range(0, max_f + 1, 2):
            r = rs_correct(block, g.n_ecc, order[:f], synd)
            if r is None:
                continue
            erased = set(order[:f])
            n_err = len([p for p in r[1] if p not in erased])
            if f and 2 * n_err + f > g.n_ecc - margin:
                continue
            ok = (r, f, n_err)
            break
        if ok is None:
            return None
        (corr, changed), f, n_err = ok
        for k, p in enumerate(pos):
            fixed[p] = corr[k]
        errs.append(len(changed))
        eras.append(f)
        changed_all += [pos[k] for k in changed]
        uecs.append(1 - (2 * n_err + (len(changed) - n_err)) / g.n_ecc)
    return RSResult(fixed[:g.n_data], fixed, errs, eras, sorted(changed_all), min(uecs))


# --------------------------------------------------------------------------
# Grid decoder: locate, fit module grid, sample, RS + data decode
# --------------------------------------------------------------------------
@dataclass
class GridFit:
    geom: Geometry
    quad: np.ndarray               # 4x2 corners tl, tr, br, bl in variant px
    contrast: float                # dark minus light on fixed pattern
    values: np.ndarray | None = None
    bits: np.ndarray | None = None
    conf: np.ndarray | None = None
    threshold: float = 0.0
    fixed_errors: int = 0


def candidate_quads(g: np.ndarray, max_quads: int = 10, min_area: float = 0.02) -> list[np.ndarray]:
    """Dark blobs that could be a symbol, as minimum area rectangles.
    Thresholds at several darkness percentiles: wood grain and texture are
    usually lighter than the marks, so a strict cut isolates the symbol.
    `min_area` is the smallest blob as a fraction of the image area."""
    h, w = g.shape
    m = min(h, w)
    flat = flatten_background(g, m / 8)
    dark = 255 - cv2.GaussianBlur(flat, (0, 0), max(1.0, m / 200))
    quads = []
    for pct in (65, 75, 85):
        bw = ((dark > np.percentile(dark, pct)) * 255).astype(np.uint8)
        for frac in (0.03, 0.05, 0.08):
            k = _odd(m * frac)
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            blob = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se)
            blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, se)
            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area * h * w or area > 0.9 * h * w:
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if min(rw, rh) < 12:
                    continue
                aspect = max(rw, rh) / min(rw, rh)
                solidity = area / (rw * rh)
                if aspect > 4.5 or solidity < 0.6:
                    continue
                quads.append((area * solidity ** 2, cv2.boxPoints(rect).astype(np.float32)))
    quads.sort(key=lambda q: -q[0])
    out: list[np.ndarray] = []
    for _, b in quads:
        c = b.mean(0)
        if all(np.linalg.norm(c - o.mean(0)) > 0.04 * m or
               abs(cv2.contourArea(b) / cv2.contourArea(o) - 1) > 0.15 for o in out):
            out.append(b)
        if len(out) >= max_quads:
            break
    # tight crops: the image itself may be (nearly) the symbol
    out.append(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    return out


def order_quad(q: np.ndarray) -> np.ndarray:
    """Clockwise in image coordinates starting top-left-most."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    start = np.argmin(q.sum(1))
    return np.roll(q, -start, axis=0)


def _line_intersect(p1, d1, p2, d2) -> np.ndarray:
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
    if abs(np.linalg.det(a)) < 1e-9:
        return (p1 + p2) / 2
    t = np.linalg.solve(a, (p2 - p1).astype(np.float64))[0]
    return p1 + t * d1


def snap_quad(gray: np.ndarray, quad: np.ndarray, passes: int = 2, inward: bool = False) -> np.ndarray:
    """Fit every side of a rough quad to the symbol edge: at 9 places along
    the side find the strongest dark->light drop along the outward normal
    (symbol -> quiet zone), fit a robust line through them and intersect
    neighbouring lines. Corrects size, position and small rotations.
    inward=True looks for the strongest light->dark rise instead: with dark
    print flush against the mark there is no quiet zone edge, but the half
    light data/timing modules meet the solid neighbour at the mark boundary
    (the solid finder side lands one module inside)."""
    q = order_quad(quad).astype(np.float32)
    for _ in range(passes):
        ext = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0]))
        dark = 255 - cv2.GaussianBlur(gray, (0, 0), max(1.0, ext / 100)).astype(np.float32)
        ts = np.linspace(-0.25, 0.2, 46) * ext
        d = 3
        c = q.mean(0)
        lines = []
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            u = (b - a) / (np.linalg.norm(b - a) + 1e-6)
            nrm = np.float32([-u[1], u[0]])
            if np.dot((a + b) / 2 - c, nrm) < 0:
                nrm = -nrm
            mid = (a + b) / 2
            half = np.linalg.norm(b - a) / 2
            along = np.linspace(-0.8, 0.8, 25) * half
            best = (-1e9, mid, u)
            for th in np.radians(np.arange(-6, 6.1, 1.5)):
                ct, st = math.cos(th), math.sin(th)
                ut = np.float32([u[0] * ct - u[1] * st, u[0] * st + u[1] * ct])
                nt = np.float32([-ut[1], ut[0]])
                if np.dot(nt, nrm) < 0:
                    nt = -nt
                pts = (mid[None, None, :] + along[None, :, None] * ut[None, None, :]
                       + ts[:, None, None] * nt[None, None, :])
                prof = cv2.remap(dark, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32),
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).mean(1)
                drop = prof[2 * d:] - prof[:-2 * d] if inward else prof[:-2 * d] - prof[2 * d:]
                i = int(np.argmax(drop))
                if drop[i] > best[0]:
                    best = (drop[i], mid + ts[i + d] * nt, ut)
            lines.append((best[1], best[2]))
        q = np.array([_line_intersect(*lines[(k - 1) % 4], *lines[k]) for k in range(4)], np.float32)
    return q


def side_combos(outer: np.ndarray, inner: np.ndarray, min_diff: float = 3.0) -> list[np.ndarray]:
    """Quads taking every side from either fit (16 combinations): a mark may
    have a quiet zone on some sides and dark print flush on the others, then
    the outward fit is right on the first and the inward fit on the rest."""
    a, b = order_quad(outer), order_quad(inner)
    out = []
    for m in range(16):
        sides = []
        for k in range(4):
            q = b if m >> k & 1 else a
            sides.append((q[k].astype(np.float64), (q[(k + 1) % 4] - q[k]).astype(np.float64)))
        c = np.array([_line_intersect(*sides[(k - 1) % 4], *sides[k]) for k in range(4)], np.float32)
        if all(np.abs(c - o).max() > min_diff for o in out):
            out.append(c)
    return out


REMAP_W = 4096           # cv2.remap needs map sides < SHRT_MAX: points go in rows


def remap_points(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples of `img` at any number of points (flat result).
    A 1xN map breaks cv2.remap for N > 32767 (large symbols), so the points
    are laid out in rows of REMAP_W."""
    n = x.size
    if n <= REMAP_W:
        return cv2.remap(img, x.astype(np.float32).reshape(1, -1), y.astype(np.float32).reshape(1, -1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    rows = -(-n // REMAP_W)
    mx = np.zeros(rows * REMAP_W, np.float32)
    my = np.zeros(rows * REMAP_W, np.float32)
    mx[:n], my[:n] = x.ravel(), y.ravel()
    v = cv2.remap(img, mx.reshape(rows, REMAP_W), my.reshape(rows, REMAP_W),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return v.ravel()[:n]


def homographies(src: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Perspective transforms src (4,2) -> each of quads (K,4,2), as (K,3,3);
    getPerspectiveTransform for many quads in one batched solve."""
    q = np.asarray(quads, np.float64)
    k = len(q)
    a = np.zeros((k, 8, 8))
    b = np.zeros((k, 8))
    for i, (x, y) in enumerate(src):
        u, v = q[:, i, 0], q[:, i, 1]
        a[:, 2 * i, 0:3] = (x, y, 1)
        a[:, 2 * i, 6], a[:, 2 * i, 7] = -x * u, -y * u
        a[:, 2 * i + 1, 3:6] = (x, y, 1)
        a[:, 2 * i + 1, 6], a[:, 2 * i + 1, 7] = -x * v, -y * v
        b[:, 2 * i], b[:, 2 * i + 1] = u, v
    try:
        h = np.linalg.solve(a, b[..., None])[..., 0]
    except np.linalg.LinAlgError:                    # a degenerate quad in the batch
        h = np.stack([np.linalg.lstsq(ai, bi, rcond=None)[0] for ai, bi in zip(a, b)])
    return np.concatenate([h, np.ones((k, 1))], 1).reshape(k, 3, 3)


class Sampler:
    """Samples a darkness image at module centres for a given quad."""

    def __init__(self, gray: np.ndarray, radius: float = 0.22):
        self.dark = (255 - gray).astype(np.float32)
        r = radius
        self.OFFS = [(0, 0), (-r, 0), (r, 0), (0, -r), (0, r)]

    def sample_uv(self, quad: np.ndarray, rows: int, cols: int, uv: np.ndarray) -> np.ndarray:
        """Sample arbitrary module coordinates (x=col, y=row) in one remap."""
        src = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        h = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        offs = np.array(self.OFFS)
        pts = (uv[None, :, :] + offs[:, None, :]).reshape(-1, 2)
        den = h[2, 0] * pts[:, 0] + h[2, 1] * pts[:, 1] + h[2, 2]
        x = (h[0, 0] * pts[:, 0] + h[0, 1] * pts[:, 1] + h[0, 2]) / den
        y = (h[1, 0] * pts[:, 0] + h[1, 1] * pts[:, 1] + h[1, 2]) / den
        return remap_points(self.dark, x, y).reshape(len(offs), -1).mean(0)

    def sample_uv_many(self, quads: np.ndarray, rows: int, cols: int, uv: np.ndarray) -> np.ndarray:
        """sample_uv for K quads at once -> (K, len(uv)). One batched
        homography solve and one remap instead of K Python-level calls."""
        src = np.float64([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        h = homographies(src, quads).astype(np.float32)          # (K, 3, 3)
        offs = np.array(self.OFFS)
        pts = (uv[None, :, :] + offs[:, None, :]).reshape(-1, 2).astype(np.float32)
        px, py = pts[None, :, 0], pts[None, :, 1]
        hh = [h[:, i, j, None] for i in range(3) for j in range(3)]
        den = hh[6] * px + hh[7] * py + hh[8]
        den = np.where(np.abs(den) < 1e-9, np.float32(1e-9), den)   # degenerate quad
        x = (hh[0] * px + hh[1] * py + hh[2]) / den
        y = (hh[3] * px + hh[4] * py + hh[5]) / den
        v = remap_points(self.dark, x, y)
        return v.reshape(len(quads), len(offs), -1).mean(1)

    def sample_grid(self, quad: np.ndarray, rows: int, cols: int,
                    ucol: np.ndarray, vrow: np.ndarray) -> np.ndarray:
        """Sample module values at tracked column / row centre positions."""
        uu, vv = np.meshgrid(ucol, vrow)
        uv = np.stack([uu.ravel(), vv.ravel()], 1)
        return self.sample_uv(quad, rows, cols, uv).reshape(rows, cols)

    def sample(self, quad: np.ndarray, rows: int, cols: int, pad: int = 0) -> np.ndarray:
        src = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        hmat = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        acc = np.zeros((rows + 2 * pad, cols + 2 * pad), np.float32)
        cc, rr = np.meshgrid(np.arange(-pad, cols + pad) + 0.5, np.arange(-pad, rows + pad) + 0.5)
        for dx, dy in self.OFFS:
            pts = np.stack([cc + dx, rr + dy], -1).reshape(-1, 1, 2).astype(np.float32)
            p = cv2.perspectiveTransform(pts, hmat).reshape(acc.shape[0], acc.shape[1], 2)
            acc += cv2.remap(self.dark, p[..., 0], p[..., 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        return acc / len(self.OFFS)


def score_quads(sampler: "Sampler", quads: np.ndarray, g: Geometry, qz: bool = True) -> np.ndarray:
    """Fast Fisher score of a batch of quads (K,4,2) -> (K,): samples only
    finder/timing(/quiet-zone) modules. Scores with and without the
    quiet-zone ring are not comparable."""
    uv, dk = (g.fp_uv_qz, g.fp_dark_qz) if qz else (g.fp_uv_in, g.fp_dark_in)
    v = sampler.sample_uv_many(quads, g.rows, g.cols, uv)
    d, l_ = v[:, dk], v[:, ~dk]
    s = (d.mean(1) - l_.mean(1)) / np.sqrt((d.var(1) + l_.var(1)) / 2 + 4.0)
    return np.where(np.isfinite(s), s, -np.inf)     # NaN would win np.argmax


def orientations(quad: np.ndarray):
    """8 corner assignments (4 rotations x mirror): tl, tr, br, bl."""
    q = order_quad(quad)
    for mirror in (False, True):
        base = q[[0, 3, 2, 1]] if mirror else q
        for r in range(4):
            yield np.roll(base, -r, axis=0), r, mirror


def refine_quad(sampler: Sampler, quad: np.ndarray, g: Geometry, iters: int = 4,
                steps: tuple = (1.0, 0.5, 0.25, 0.12), qz: bool = True) -> tuple[np.ndarray, float]:
    """Coordinate ascent on the 8 corner coordinates maximising the
    finder/timing pattern contrast. Each phase scores all its moves in one
    batch: corner moves (plus the combination of every improving coordinate,
    like a full sweep), then edge / shift / scale moves."""
    q = quad.astype(np.float32).copy()
    mod = (np.linalg.norm(q[1] - q[0]) / g.cols + np.linalg.norm(q[3] - q[0]) / g.rows) / 2
    best = float(score_quads(sampler, q[None], g, qz)[0])
    for step in [f * mod for f in steps]:
        for _ in range(iters):
            improved = False
            # corner moves: index 2 * (2k + axis) + (0: +step, 1: -step)
            delta = np.zeros((16, 4, 2), np.float32)
            for k in range(4):
                for axis in (0, 1):
                    i = 2 * (2 * k + axis)
                    delta[i, k, axis], delta[i + 1, k, axis] = step, -step
            cands = q[None] + delta
            s = score_quads(sampler, cands, g, qz)
            comb = np.zeros((4, 2), np.float32)
            for j in range(8):                     # best sign per improving coordinate
                i = 2 * j + int(s[2 * j + 1] > s[2 * j])
                if s[i] > best + 1e-4:
                    comb += delta[i]
            i = int(np.argmax(s))
            if s[i] > best + 1e-4:
                best, q, improved = float(s[i]), cands[i], True
                if np.count_nonzero(comb) > 1:
                    t = (q - delta[i] + comb)[None]
                    sc = float(score_quads(sampler, t, g, qz)[0])
                    if sc > best + 1e-4:
                        best, q = sc, t[0]
            # edge moves (two corners together) and whole-quad shift / scale
            c = q.mean(0)
            moves = []
            for k in range(4):
                k2 = (k + 1) % 4
                nrm = q[k2] - q[k]
                nrm = np.float32([-nrm[1], nrm[0]]) / (np.linalg.norm(nrm) + 1e-6)
                for sgn in (1, -1):
                    t = q.copy()
                    t[k] += sgn * step * nrm
                    t[k2] += sgn * step * nrm
                    moves.append(t)
            moves += [q + [step, 0], q - [step, 0], q + [0, step], q - [0, step],
                      c + (q - c) * (1 + step / (mod * g.cols)),
                      c + (q - c) * (1 - step / (mod * g.cols))]
            cands = np.array(moves, np.float32)
            s = score_quads(sampler, cands, g, qz)
            i = int(np.argmax(s))
            if s[i] > best + 1e-4:
                best, q, improved = float(s[i]), cands[i], True
            if not improved:
                break
    return q, best


def _track_axis(sampler: Sampler, quad: np.ndarray, g: Geometry, axis: str) -> np.ndarray:
    """Measure the real centre of every module along a timing edge.
    Printers (inkjet/laser on wood) do not place modules on a perfectly even
    grid; following the timing pattern, as industrial readers do, keeps the
    data sampling on the modules. axis 'u': top row -> column centres,
    axis 'v': right column -> row centres."""
    n = g.cols if axis == "u" else g.rows
    expect = g.fixed_val[0, :] if axis == "u" else g.fixed_val[:, -1]
    step = 0.05
    t = np.arange(-0.5, n + 0.5, step) + step / 2
    prof = 0
    for off in (0.35, 0.5, 0.65):                    # 3 lines across the edge
        if axis == "u":
            uv = np.stack([t, np.full_like(t, off)], 1)
        else:
            uv = np.stack([np.full_like(t, g.cols - off), t], 1)
        prof = prof + sampler.sample_uv(quad, g.rows, g.cols, uv)
    prof = np.convolve(prof / 3, np.ones(3) / 3, mode="same")
    obs = np.full(n, np.nan)
    for c in range(n):
        lo = int(round((c + 0.5 - 0.45 + 0.5) / step))
        hi = int(round((c + 0.5 + 0.45 + 0.5) / step))
        seg = prof[lo:hi] if expect[c] else -prof[lo:hi]
        if len(seg) < 3:
            continue
        k = int(np.argmax(seg))
        if 0 < k < len(seg) - 1:                      # parabolic sub-sample
            den = seg[k - 1] - 2 * seg[k] + seg[k + 1]
            k = k + (0.5 * (seg[k - 1] - seg[k + 1]) / den if den else 0)
        else:
            continue                                  # no real extremum
        obs[c] = t[lo] + k * step
    ideal = np.arange(n) + 0.5
    off = obs - ideal
    ok = ~np.isnan(off)
    idx = np.arange(n)
    if ok.sum() < n // 2:
        # Timing extrapolation: too few timing modules were tracked, but
        # the ones that were still carry the pitch trend - extrapolate a
        # linear position error through them instead of assuming a perfect
        # grid (compensates a systematic scale/rotation misfit of the quad).
        if ok.sum() >= 6:
            slope, intercept = np.polyfit(idx[ok], off[ok], 1)
            return ideal + np.clip(intercept + slope * idx, -0.5, 0.5)
        return ideal
    # robust: drop offsets far from a running median, interpolate the gaps
    off = np.interp(idx, idx[ok], off[ok])
    med = np.array([np.median(off[max(0, i - 2):i + 3]) for i in idx])
    bad = np.abs(off - med) > 0.25
    off[bad] = med[bad]
    off = np.array([np.mean(off[max(0, i - 1):i + 2]) for i in idx])
    return ideal + np.clip(off, -0.5, 0.5)


SIZE_CHECK_PX = 512      # rectified symbol side for interior_modules


def interior_modules(sampler: Sampler, quad: np.ndarray) -> tuple[float, float]:
    """Module counts (cols, rows) inside an oriented quad, from the
    periodicity of module edges over the whole rectified symbol. The grid
    fit only looks at finder/timing modules - refine_quad makes those
    alternate at the fitted size's period, so they would always "confirm" a
    wrong size (e.g. a 10x10 fit laid over part of a 24x24 mark); the data
    area is independent evidence."""
    n = SIZE_CHECK_PX
    dst = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    w = cv2.warpPerspective(sampler.dark, cv2.getPerspectiveTransform(quad.astype(np.float32), dst),
                            (n, n), borderMode=cv2.BORDER_REPLICATE)
    out = []
    for ax in (1, 0):                                # d/dx -> columns, d/dy -> rows
        prof = np.abs(np.diff(w, axis=ax)).mean(axis=1 - ax).astype(np.float32)
        prof = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 8).ravel()
        spec = np.abs(np.fft.rfft(prof * np.hanning(len(prof)), 16 * len(prof)))
        per = np.arange(len(spec)) / 16 * n / (n - 1)    # module edges per side
        spec[(per < 6) | (per > 150)] = 0
        out.append(float(per[int(np.argmax(spec))]))
    return out[0], out[1]


def size_agree(est: tuple[float, float], g: Geometry) -> int:
    """How many axes (0..2) of the measured module count confirm the fit."""
    return sum(abs(e - n) <= max(1.0, 0.06 * n) for e, n in zip(est, (g.cols, g.rows)))


def outside_dark(sampler: Sampler, quad: np.ndarray, g: Geometry, thr: float) -> float:
    """Fraction of the 1-module ring just outside the fit that is dark. A
    fit with the right module pitch but covering only part of a symbol has
    data modules there instead of a quiet zone."""
    uv = g.fp_uv_qz
    ring = (uv[:, 0] < 0) | (uv[:, 1] < 0) | (uv[:, 0] > g.cols) | (uv[:, 1] > g.rows)
    return float((sampler.sample_uv(quad, g.rows, g.cols, uv[ring]) > thr).mean())


OUTSIDE_BUSY = 0.35      # ring darker than this: the fit is likely partial


def near_miss_key(d: dict) -> tuple:
    """Best near-miss: confirmed size first (fit scores of different sizes
    and templates are not comparable), fits with a quiet surrounding before
    partial ones, then the fit score."""
    return (d.get("size_agree", 0), d.get("outside_dark", 1.0) <= OUTSIDE_BUSY,
            d.get("score", -1e9))


def tracked_axes(sampler: Sampler, quad: np.ndarray, g: Geometry) -> tuple[np.ndarray, np.ndarray]:
    return _track_axis(sampler, quad, g, "u"), _track_axis(sampler, quad, g, "v")


def classify(vals: np.ndarray, g: Geometry, mode: str = "plane") -> tuple[np.ndarray, np.ndarray, float]:
    """Threshold module values. "plane": a smooth threshold surface is fitted
    to the known finder/timing modules so lighting gradients across the symbol
    do not flip modules. "local": threshold = mean of the 5x5 module
    neighbourhood (follows grain stripes / stains), clipped to the plane.
    Confidence = distance to threshold in class-spread units."""
    fm, fv = g.fixed_mask, g.fixed_val
    rr, cc = np.mgrid[0:g.rows, 0:g.cols]
    d_mean = vals[fm & (fv == 1)].mean()
    l_mean = vals[fm & (fv == 0)].mean()
    thr_global = (d_mean + l_mean) / 2
    # plane fit on midpoint estimates from fixed modules: value - offset
    half = (d_mean - l_mean) / 2
    target = np.where(fv == 1, vals - half, vals + half)[fm]
    a = np.stack([np.ones(fm.sum()), rr[fm] / g.rows, cc[fm] / g.cols], 1)
    try:
        coef, *_ = np.linalg.lstsq(a, target, rcond=None)
        thr = coef[0] + coef[1] * rr / g.rows + coef[2] * cc / g.cols
        # keep the plane sane
        thr = np.clip(thr, l_mean + 0.2 * (d_mean - l_mean), d_mean - 0.2 * (d_mean - l_mean))
    except np.linalg.LinAlgError:
        thr = np.full_like(vals, thr_global)
    if mode == "local":
        loc = cv2.blur(vals.astype(np.float32), (5, 5), borderType=cv2.BORDER_REFLECT)
        span = 0.3 * (d_mean - l_mean)
        thr = np.clip(loc, thr - span, thr + span)
    bits = (vals > thr).astype(np.uint8)
    conf = np.abs(vals - thr) / max(1e-3, d_mean - l_mean)
    return bits, conf, float(thr_global)


def codewords_from_bits(bits: np.ndarray, conf: np.ndarray, g: Geometry):
    b = bits[g.cw_rows, g.cw_cols]                       # (n_cw, 8)
    weights = 1 << np.arange(7, -1, -1)
    cw = (b * weights).sum(1).astype(int).tolist()
    cconf = conf[g.cw_rows, g.cw_cols].min(1)
    return cw, cconf


def try_fit(sampler: Sampler, fit: GridFit, mode: str = "plane",
            vals: np.ndarray | None = None, tracked: bool = False) -> tuple[RSResult, Decoded] | None:
    g = fit.geom
    if vals is None:
        if tracked:
            ucol, vrow = tracked_axes(sampler, fit.quad, g)
            vals = sampler.sample_grid(fit.quad, g.rows, g.cols, ucol, vrow)
        else:
            vals = sampler.sample(fit.quad, g.rows, g.cols)
    bits, conf, thr = classify(vals, g, mode)
    fit.values, fit.bits, fit.conf, fit.threshold = vals, bits, conf, thr
    fit.fixed_errors = int((bits[g.fixed_mask] != g.fixed_val[g.fixed_mask]).sum())
    cw, cconf = codewords_from_bits(bits, conf, g)
    rs = rs_decode_symbol(cw, cconf, g)
    if rs is None:
        return None
    try:
        dec = decode_codewords(rs.data_cw)
    except (DataError, IndexError):
        return None
    if not dec.data or not plausible(dec):
        return None
    return rs, dec


def plausible(dec: Decoded) -> bool:
    """Reject text with control bytes other than GS/RS/EOT/CR/LF/TAB outside
    BASE256 segments - typical of a miscorrected symbol."""
    ok = {4, 9, 10, 13, 29, 30}
    for sgm in dec.segments:
        if sgm.mode == "BASE256":
            continue
        if any((b < 32 and b not in ok) or b == 127 for b in sgm.data):
            return False
    return True


def _zscore(v: np.ndarray, g: Geometry) -> np.ndarray:
    d = v[g.fixed_mask & (g.fixed_val == 1)].mean()
    l_ = v[g.fixed_mask & (g.fixed_val == 0)].mean()
    return (v - (d + l_) / 2) / max(1e-3, d - l_) * 100


def rescue_fit(fit: GridFit, images: list[tuple[str, np.ndarray]],
               deadline: float) -> tuple[RSResult, Decoded, str] | None:
    """The grid fits but the data does not decode: re-sample the same grid
    with other channels, module-sized smoothing, other sampling footprints and
    thresholds, and finally a vote (average) over all of them."""
    g = fit.geom
    mod = (np.linalg.norm(fit.quad[1] - fit.quad[0]) / g.cols +
           np.linalg.norm(fit.quad[3] - fit.quad[0]) / g.rows) / 2
    # multi-exposure bracket (HDR-style, like CAMERA.HDR-MODE of industrial
    # readers): the same view at two extra gammas joins the fusion stack
    imgs = list(images)
    for gm in (0.6, 1.5):
        imgs.append((f"{imgs[0][0]}, gamma {gm:g}", gamma(imgs[0][1], gm)))
    stack = []
    for name, img in imgs:
        for sig in (0.0, 0.25):
            src = cv2.GaussianBlur(img, (0, 0), sig * mod) if sig else img
            axes = tracked_axes(Sampler(src, 0.0), fit.quad, g)
            for rad in (0.22, 0.1, 0.32):
                for tr in (False, True):
                    if _expired(deadline):
                        return None
                    smp = Sampler(src, rad)
                    vals = (smp.sample_grid(fit.quad, g.rows, g.cols, *axes) if tr
                            else smp.sample(fit.quad, g.rows, g.cols))
                    stack.append(_zscore(vals, g))
                    for mode in ("plane", "local"):
                        r = try_fit(None, fit, mode, vals)
                        if r:
                            how = (f"{name}, blur {sig:g} module, footprint {rad:g}, {mode} threshold"
                                   + (", timing-tracked grid" if tr else ""))
                            return r[0], r[1], how
    if stack:
        avg = np.mean(stack, 0)
        for mode in ("plane", "local"):
            r = try_fit(None, fit, mode, avg)
            if r:
                return r[0], r[1], f"average of {len(stack)} samplings, {mode} threshold"
    return None


def expand_quad(quad: np.ndarray, g: Geometry, modules: float = 1.0) -> np.ndarray:
    """Grow an oriented quad (tl, tr, br, bl) by `modules` on every side."""
    src = np.float32([[0, 0], [g.cols, 0], [g.cols, g.rows], [0, g.rows]])
    h = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    m = modules
    pts = np.float32([[-m, -m], [g.cols + m, -m], [g.cols + m, g.rows + m], [-m, g.rows + m]])
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(4, 2)


def grid_decode(gray: np.ndarray, deadline: float, extra_quads=(),
                alt_images: list[tuple[str, np.ndarray]] = (), min_area: float = 0.02,
                qz_modes: tuple = (True, False), symbols: list | None = None,
                diag: dict | None = None) -> tuple | None:
    """Returns (fit, rs, decoded, mirrored, sampling_note) or None.
    qz_modes: True = template with the 1-module quiet-zone ring (more
    specific, fewer false fits, always try first), False = ring-less retry
    for marks printed flush against other dark print.
    symbols: expected symbol table(s) only (the "trained model" hint of an
    industrial reader); None = try every known size.
    diag: if given, filled with the best failed fit's data (near-miss)."""
    sampler = Sampler(gray)
    rough = [np.asarray(q, np.float32) for q in extra_quads] + candidate_quads(gray, min_area=min_area)
    quads = []
    for q in rough:
        quads += [snap_quad(gray, q), q]
    if True in qz_modes:
        r = _grid_pass(gray, sampler, quads, deadline, alt_images, qz=True,
                       symbols=symbols, diag=diag)
        if r is not None or _expired(deadline) or False not in qz_modes:
            return r
    quads2 = []
    for q, sq in zip(rough, quads[::2]):
        quads2 += side_combos(sq, snap_quad(gray, sq, inward=True)) + [snap_quad(gray, q, inward=True), q]
    return _grid_pass(gray, sampler, quads2, deadline, alt_images, qz=False,
                      symbols=symbols, diag=diag)


def _grid_pass(gray: np.ndarray, sampler: Sampler, quads: list[np.ndarray], deadline: float,
               alt_images, qz: bool, symbols: list | None = None,
               diag: dict | None = None) -> tuple | None:
    note = "" if qz else ", no-quiet-zone"
    # ring-less scores run higher on noise: demand more before the slow rescue
    strong = STRONG_FIT if qz else STRONG_FIT * NO_QZ_STRONG_FACTOR
    sym_list = ALL_SYMBOLS if not symbols else \
        [s for s in ALL_SYMBOLS if (s[0], s[1]) in {(t[0], t[1]) for t in symbols}]
    for quad in quads:
        if _expired(deadline):
            return None
        side_a = np.linalg.norm(quad[1] - quad[0])
        side_b = np.linalg.norm(quad[2] - quad[1])
        if min(side_a, side_b) < 10:
            continue
        by_sym: dict[tuple, list] = {}               # sym -> [(quad, mirror)]
        for oq, rot, mirror in orientations(quad):
            w_ = np.linalg.norm(oq[1] - oq[0])
            h_ = np.linalg.norm(oq[3] - oq[0])
            for sym in sym_list:
                rows, cols = sym[0], sym[1]
                if not 0.7 < (w_ / cols) / (h_ / rows) < 1.4:
                    continue
                if min(w_ / cols, h_ / rows) < 1.6:      # modules under ~2 px
                    continue
                # without a quiet zone the side fit may stop at the finder's
                # inner edge: also try the quad grown by one module
                for cq in ((oq,) if qz else (oq, expand_quad(oq, geometry(sym)))):
                    by_sym.setdefault(sym, []).append((cq, mirror))
        scored = []
        for sym, cands in by_sym.items():          # one batched score per size
            if _expired(deadline):
                return None
            gm = geometry(sym)
            s = score_quads(sampler, np.array([c[0] for c in cands], np.float32), gm, qz)
            scored += [(float(sc), gm, cq, mirror) for sc, (cq, mirror) in zip(s, cands)]
        # the right size often scores low before alignment: coarse-align the
        # 16 best candidates first, then fully refine and decode the top 4
        scored.sort(key=lambda t: -t[0])
        coarse = []
        for s, gm, oq, mirror in scored[:16]:
            if _expired(deadline):
                return None
            cq, cs = refine_quad(sampler, oq, gm, iters=2, steps=(1.0, 0.5), qz=qz)
            coarse.append((cs, gm, cq, mirror))
        coarse.sort(key=lambda t: -t[0])
        for s, gm, oq, mirror in coarse[:4]:
            if _expired(deadline):
                return None
            rq, rs_ = refine_quad(sampler, oq, gm, steps=(0.5, 0.25, 0.12), qz=qz)
            fit = GridFit(gm, rq, rs_)
            r = try_fit(sampler, fit)
            if r is not None:
                return fit, r[0], r[1], mirror, "direct" + note
            for mode, tr in (("local", False), ("plane", True), ("local", True)):
                r = try_fit(sampler, fit, mode, tracked=tr)
                if r is not None:
                    return (fit, r[0], r[1], mirror,
                            f"{mode} threshold" + (", timing-tracked grid" if tr else "") + note)
            if diag is not None:
                # near-miss bookkeeping (the usability layer of industrial
                # readers: report what the best failed attempt looked like)
                est = interior_modules(sampler, rq)
                agree = size_agree(est, gm)
                top, right = np.linalg.norm(rq[1] - rq[0]), np.linalg.norm(rq[2] - rq[1])
                # px/module from the measured count when it contradicts the size
                modpx = ((top / gm.cols + right / gm.rows) / 2 if agree else
                         (top / est[0] + right / est[1]) / 2)
                out = outside_dark(sampler, rq, gm, fit.threshold)
                if (agree, out <= OUTSIDE_BUSY, rs_) > near_miss_key(diag):
                    diag.clear()
                    diag.update(score=float(rs_), size=gm.name, qz=bool(qz),
                                modules_px=float(modpx),
                                fixed_pattern_module_errors=int(fit.fixed_errors),
                                size_agree=int(agree), outside_dark=round(out, 2),
                                measured_size=f"{round(est[1])}x{round(est[0])}")
            if rs_ >= strong:
                r = rescue_fit(fit, [("same image", gray)] + list(alt_images), deadline)
                if r is not None:
                    return fit, r[0], r[1], mirror, r[2] + note
    return None


STRONG_FIT = 3.5   # finder/timing Fisher score above which the grid is trusted
NO_QZ_STRONG_FACTOR = 1.3   # same bar for the ring-less template (not comparable)


# --------------------------------------------------------------------------
# Quality metrics (estimates in the spirit of ISO/IEC 15415 / TR 29158,
# from a single uncalibrated photo - indicative only, not a verifier grade)
# --------------------------------------------------------------------------
def grade(v: float, limits: tuple) -> str:
    for g_, lim in zip("ABCD", limits):
        if v >= lim:
            return g_
    return "F"


def quality(gray_orig: np.ndarray, fit: GridFit, scale: float, rs: RSResult,
            offset: tuple[float, float] = (0.0, 0.0)) -> dict:
    g = fit.geom
    quad = fit.quad / scale + np.float32(offset)
    vals = Sampler(gray_orig).sample(quad, g.rows, g.cols)
    refl = 255 - vals                                    # back to reflectance
    bits = fit.bits
    dark, light = refl[bits == 1], refl[bits == 0]
    rmax, rmin = float(np.percentile(light, 95)), float(np.percentile(dark, 5))
    sc = (rmax - rmin) / 255 * 100
    gt = (rmax + rmin) / 2
    # modulation: worst module margin relative to symbol contrast
    margin = np.abs(refl - gt) / max(1e-3, (rmax - rmin) / 2)
    mod = float(np.percentile(margin, 5))
    w_ = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2 / g.cols
    h_ = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2 / g.rows
    anu = abs(w_ - h_) / ((w_ + h_) / 2)
    fpd = fit.fixed_errors / max(1, int(g.fixed_mask.sum()))
    top = quad[1] - quad[0]
    angle = math.degrees(math.atan2(top[1], top[0]))
    left = quad[3] - quad[0]
    skew = math.degrees(math.acos(np.clip(np.dot(top, left) /
                                          (np.linalg.norm(top) * np.linalg.norm(left)), -1, 1))) - 90
    out = {
        "symbol_contrast_pct": round(sc, 1),
        "symbol_contrast_grade": grade(sc, (70, 55, 40, 20)),
        "modulation_p5": round(float(mod), 2),
        "modulation_grade": grade(mod, (0.5, 0.4, 0.3, 0.2)),
        "axial_nonuniformity": round(float(anu), 3),
        "axial_nonuniformity_grade": grade(-anu, (-0.06, -0.08, -0.10, -0.12)),
        "unused_error_correction": round(rs.uec, 2),
        "unused_error_correction_grade": grade(rs.uec, (0.62, 0.5, 0.37, 0.25)),
        "fixed_pattern_damage_pct": round(fpd * 100, 1),
        "fixed_pattern_damage_grade": grade(-fpd, (0.0, -0.02, -0.05, -0.10)),
        "module_size_px": round(float(w_ + h_) / 2, 2),
        "symbol_size_px": (int(round(w_ * g.cols)), int(round(h_ * g.rows))),
        "rotation_deg": round(float(angle), 1),
        "skew_deg": round(float(skew), 1),
        "dark_reflectance_mean": round(float(dark.mean()), 1),
        "light_reflectance_mean": round(float(light.mean()), 1),
        "corners_px": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
    }
    gs = [out[k] for k in out if k.endswith("_grade")]
    out["overall_grade_estimate"] = max(gs)               # worst letter wins
    return out


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------
@dataclass
class ReadResult:
    path: str
    engine: str = ""
    variant: str = ""
    data: bytes = b""
    text: str = ""
    elapsed: float = 0.0
    attempts: int = 0
    fit: GridFit | None = None
    rs: RSResult | None = None
    dec: Decoded | None = None
    mirrored: bool = False
    scale: float = 1.0             # of the variant res.fit.quad refers to
    ox: float = 0.0
    oy: float = 0.0
    zx: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    agreeing: list[str] = field(default_factory=list)
    sampling: str = ""
    near_miss: dict = field(default_factory=dict)


def _zx_quad(b) -> np.ndarray:
    p = b.position
    return np.float32([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                       [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]])


def _binary(v: Variant) -> bool:
    """Variant is already black/white: every binarizer sees the same image."""
    return v.name.endswith(("otsu", "adaptive"))


def zxing_read(v: Variant, try_invert: bool = True) -> tuple:
    """(result or None, failed positions). Binarizers in turn; FixedThreshold
    (threshold 127) reads some flattened / CLAHE variants the adaptive ones
    miss and is the only one needed on binarised variants. One call per
    binarizer with return_errors=True also yields the positions of symbols
    zxing detected but could not decode (seeds for the grid decoder; taken
    from the first binarizer only). try_invert=False when the polarity is
    known (zxing otherwise also tries every image inverted)."""
    bzs = ((zxingcpp.Binarizer.FixedThreshold,) if _binary(v) else
           (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram,
            zxingcpp.Binarizer.FixedThreshold))
    failed: list[np.ndarray] = []
    for i, bz in enumerate(bzs):
        kw = dict(formats=zxingcpp.BarcodeFormat.DataMatrix, binarizer=bz,
                  text_mode=zxingcpp.TextMode.Plain, try_invert=try_invert)
        try:
            r = zxingcpp.read_barcodes(v.img, return_errors=True, **kw)
        except TypeError:                            # zxing-cpp < 2.2: no error results
            r = zxingcpp.read_barcodes(v.img, **kw)
        ok = [b for b in r if getattr(b, "valid", True)]
        if ok:
            b = ok[0]
            info = {"binarizer": str(bz).split(".")[-1],
                    "symbology_identifier": b.symbology_identifier,
                    "content_type": str(b.content_type).split(".")[-1],
                    "orientation_deg": b.orientation, "ec_level": b.ec_level}
            return (bytes(b.bytes), _zx_quad(b), info), []
        if i == 0:
            failed = [_zx_quad(b) for b in r]
    return None, failed


def libdmtx_read(v: Variant, timeout_ms: int):
    if pylibdmtx is None:
        return None
    h, w = v.img.shape
    r = pylibdmtx.decode((v.img.tobytes(), w, h), timeout=timeout_ms, max_count=1)
    if not r:
        return None
    rect = r[0].rect                                     # bottom-left origin
    x, y, rw, rh = rect.left, h - rect.top, rect.width, rect.height
    quad = np.float32([[x, y - rh], [x + rw, y - rh], [x + rw, y], [x, y]])
    return bytes(r[0].data), quad, {}


def to_text(data: bytes, dec: Decoded | None) -> tuple[str, str]:
    if dec and 26 in dec.eci:
        return data.decode("utf-8", "replace"), "UTF-8 (ECI 000026)"
    try:
        s = data.decode("utf-8")
        if any(b > 127 for b in data):
            return s, "UTF-8 (detected)"
        return s, "ASCII"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "ISO-8859-1 (default)"


def analyse_known(res: ReadResult, variants_by_name: dict[str, Variant], quad, gray_orig,
                  expect: list | None = None) -> None:
    """A library decoded the symbol: run the grid decoder at its position to
    recover module matrix, codewords and RS statistics for the report."""
    v = variants_by_name[res.variant]
    others = [x for x in variants_by_name.values() if x is not v]
    if v.tile:                                   # same tile first
        prefix = v.name.split(" > ")[0] + " > "
        others.sort(key=lambda x: not x.name.startswith(prefix))
    for gv in [v] + others[:6]:
        q = map_quad(quad, v, gv)
        r = grid_decode(gv.img, time.time() + 8, extra_quads=[q] if q is not None else [],
                        min_area=_min_area(gv), symbols=expect)
        if r and r[2].data == res.data:
            res.fit, res.rs, res.dec, res.mirrored, res.sampling = r
            res.scale, res.ox, res.oy = gv.scale, gv.ox, gv.oy
            res.quality = quality(gray_orig, res.fit, res.scale, res.rs, (res.ox, res.oy))
            return


# --------------------------------------------------------------------------
# Job runner: one decoder invocation on one variant per job, run inline
# (--jobs 1) or in a process pool. Results are consumed in submission order,
# so the outcome does not depend on which worker finishes first.
# --------------------------------------------------------------------------
_ABORT = None             # multiprocessing Event in pool workers
_MISSING = "engine-missing"
JOB_GRACE_S = 0.5         # how long past the budget a running job is awaited
# a non-tiled image has 12 grid jobs with a 25 s cap: 4 workers could not
# finish the ladder inside the default budget, so use most of the CPU
DEFAULT_JOBS = max(1, min((os.cpu_count() or 2) - 1, 8))


def _expired(deadline: float) -> bool:
    return time.time() > deadline or (_ABORT is not None and _ABORT.is_set())


def _min_area(v: Variant) -> float:
    """Tiles are wide shots: a symbol covers a small part of them."""
    return 0.004 if v.tile else 0.02


def _worker_init(abort_event) -> None:
    global _ABORT
    _ABORT = abort_event
    cv2.setNumThreads(1)                 # the pool provides the parallelism


def _scan_job(kind: str, *args):
    if kind == "zxing":
        v, want_seeds, try_invert = args
        if zxingcpp is None:
            return _MISSING
        r, failed = zxing_read(v, try_invert)
        return r, (failed if want_seeds else [])
    if kind == "grid":
        v, extra, alts, cap_s, deadline, qz, symbols = args
        gdiag: dict = {}
        r = grid_decode(v.img, min(deadline, time.time() + cap_s), extra, alts,
                        _min_area(v), (qz,), symbols, gdiag)
        return r, gdiag
    if kind == "libdmtx":
        v, timeout_ms = args
        if pylibdmtx is None:
            return _MISSING
        return libdmtx_read(v, timeout_ms)
    raise ValueError(kind)


class Runner:
    def __init__(self, jobs: int = 1):
        self.pool = None
        self.abort = None
        if jobs > 1:
            try:
                ctx = mp.get_context("spawn")    # Windows-safe, pylibdmtx is not fork-safe
                self.abort = ctx.Event()
                self.pool = cf.ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_worker_init,
                                                   initargs=(self.abort,))
            except (OSError, ValueError, ImportError) as exc:
                print(f"warning: no worker pool ({exc}), running sequentially", file=sys.stderr)
                self.pool = None

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def results(self, jobs, deadline: float | None = None):
        """Yield job results in order. Sequential mode builds each job lazily,
        after the previous result was consumed. Stops at the deadline."""
        pool = self.pool
        if pool is not None:
            jobs = list(jobs)
            try:
                futs = [pool.submit(_scan_job, *j) for j in jobs]
            except Exception as exc:                     # broken pool
                print(f"warning: worker pool failed ({exc}), running sequentially", file=sys.stderr)
                self.pool = pool = None
        if pool is None:
            for job in jobs:
                if deadline is not None and time.time() > deadline:
                    return
                yield _scan_job(*job)
            return
        stop = None if deadline is None else deadline + JOB_GRACE_S
        try:
            for job, fut in zip(jobs, futs):
                timeout = None if stop is None else max(0.0, stop - time.time())
                try:
                    r = fut.result(timeout)
                except cf.TimeoutError:
                    return
                except Exception:                        # worker died
                    r = _MISSING
                if isinstance(r, str):                   # engine unusable in the worker
                    r = _scan_job(*job)
                yield r
        finally:
            running = [f for f in futs if not f.cancel() and not f.done()]
            if running:                                  # ask them to stop, then wait
                self.abort.set()
                cf.wait(running, timeout=10)
                self.abort.clear()


def read_image(path: str, engines: set[str], exhaustive: bool = False,
               debug_dir: str | None = None, budget_s: float = 40.0,
               runner: Runner | None = None, expect: list | None = None,
               polarity: str = "any") -> ReadResult:
    t0 = time.time()
    runner = runner or Runner(1)
    res = ReadResult(path)
    bgr = load_image(path)
    gray_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    invert = polarity == "light"
    inverted_extra = polarity == "any"
    variants = enhancement_variants(bgr, invert, inverted_extra)
    tiles = tile_variants(bgr, invert)
    by_name = {v.name: v for v in variants}
    by_name.update((v.name, v) for tv in tiles for v in tv)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        for k, v in enumerate(variants):
            cv2.imencode(".png", v.img)[1].tofile(
                os.path.join(debug_dir, f"{stem}_v{k:02d}.png"))
        for k, v in enumerate(v for tv in tiles for v in tv):
            cv2.imencode(".png", v.img)[1].tofile(
                os.path.join(debug_dir, f"{stem}_t{k:02d}.png"))
    deadline = t0 + budget_s
    found: list[tuple[str, str, bytes, object]] = []

    def done() -> bool:
        return bool(found) and not exhaustive

    seeds: list[tuple[Variant, np.ndarray]] = []
    tile_seeds: list[tuple[Variant, np.ndarray]] = []
    tile_hit: set[str] = set()
    # 1. zxing on all variants (cheap), whole image first, then the tiles
    if "zxing" in engines and zxingcpp is not None:
        ladder = variants + [v for tv in tiles for v in tv]
        # with a known polarity the ladder already has it: no inverted retries
        jobs = (("zxing", v, len(seeds) < 6 or v.tile, polarity == "any") for v in ladder)
        with contextlib.closing(runner.results(jobs)) as it:
            for v, (r, failed) in zip(ladder, it):
                res.attempts += 1
                if r:
                    found.append(("zxing", v.name, r[0], (r[1], r[2])))
                    tile_hit.add(v.name.split(" > ")[0])
                    if done():
                        break
                elif not v.tile and len(seeds) < 6:
                    seeds += [(v, q) for q in failed]
                elif v.tile and len(tile_seeds) < 12:
                    tile_seeds += [(v, q) for q in failed]
    # 2. built-in grid decoder on analogue variants, then on the tiles that
    #    zxing did not read (their gray / flatten variants, not binarised);
    #    all of that with the quiet-zone template, then the ring-less retry
    if not done() and "grid" in engines:
        gvars = grid_variants(bgr, invert, inverted_extra)
        stage = [(v, 25.0, gvars) for v in gvars]        # (variant, time cap, alt images)
        for tv in tiles:
            if tv[0].name.split(" > ")[0] not in tile_hit:
                stage += [(v, 10.0, tv[:2]) for v in tv[:2]]
        ladder = [j + (True,) for j in stage] + [j + (False,) for j in stage]
        all_seeds = seeds + tile_seeds

        def grid_job(v: Variant, cap_s: float, group: list[Variant], qz: bool) -> tuple:
            extra = [map_quad(q, sv, v) for sv, q in all_seeds]
            if v.tile:                           # only seeds that lie in this tile
                h, w = v.img.shape
                extra = [q for q in extra if 0 <= q[:, 0].mean() < w and 0 <= q[:, 1].mean() < h]
            alts = [(o.name, o.img) for o in group if o is not v and o.img.shape == v.img.shape]
            return "grid", v, extra, alts, cap_s, deadline, qz, expect

        jobs = (grid_job(*job) for job in ladder)
        with contextlib.closing(runner.results(jobs, deadline)) as it:
            for (v, *_), out in zip(ladder, it):
                res.attempts += 1
                r, gdiag = out
                if r:
                    by_name[v.name] = v
                    found.append(("grid", v.name, r[2].data, (v, r)))
                    if done():
                        break
                elif gdiag and near_miss_key(gdiag) > near_miss_key(res.near_miss):
                    gdiag["modules_px_orig"] = gdiag["modules_px"] / v.scale
                    gdiag["variant"] = v.name
                    res.near_miss = gdiag
    # 3. libdmtx on a subset (slow)
    if not done() and "libdmtx" in engines and pylibdmtx is not None:
        ladder = [v for v in variants if any(s in v.name for s in ("otsu", "clahe", "merge-dots"))]
        jobs = (("libdmtx", v, 1200) for v in ladder)
        with contextlib.closing(runner.results(jobs, deadline)) as it:
            for v, r in zip(ladder, it):
                res.attempts += 1
                if r:
                    found.append(("libdmtx", v.name, r[0], (r[1], r[2])))
                    if done():
                        break

    res.elapsed = time.time() - t0
    if not found:
        return res
    # majority content wins, earliest engine as the reported one
    counts: dict[bytes, int] = {}
    for f in found:
        counts[f[2]] = counts.get(f[2], 0) + 1
    best_data = max(counts, key=lambda d: counts[d])
    eng, vname, data, extra = next(f for f in found if f[2] == best_data)
    res.engine, res.variant, res.data = eng, vname, data
    res.agreeing = [f"{f[0]}: {f[1]}" for f in found if f[2] == best_data]
    if eng == "grid":
        v, (fit, rs, dec, mirror, how) = extra
        res.fit, res.rs, res.dec, res.mirrored = fit, rs, dec, mirror
        res.scale, res.ox, res.oy = v.scale, v.ox, v.oy
        res.sampling = how
        res.quality = quality(gray_orig, fit, v.scale, rs, (v.ox, v.oy))
    else:
        quad, info = extra
        res.zx = info
        if "grid" in engines:
            analyse_known(res, by_name, quad, gray_orig, expect)
    res.text, _ = to_text(res.data, res.dec)
    res.elapsed = time.time() - t0
    if debug_dir and res.fit is not None:
        save_annotated(bgr, res, debug_dir)
    return res


def save_annotated(bgr: np.ndarray, res: ReadResult, debug_dir: str) -> None:
    img = bgr.copy()
    g = res.fit.geom
    quad = res.fit.quad / res.scale + np.float32([res.ox, res.oy])
    src = np.float32([[0, 0], [g.cols, 0], [g.cols, g.rows], [0, g.rows]])
    hmat = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    cc, rr = np.meshgrid(np.arange(g.cols) + 0.5, np.arange(g.rows) + 0.5)
    pts = cv2.perspectiveTransform(np.stack([cc, rr], -1).reshape(-1, 1, 2).astype(np.float32), hmat)
    pts = pts.reshape(g.rows, g.cols, 2)
    rad = max(1, int(res.quality.get("module_size_px", 4) * 0.2))
    changed = set()
    if res.rs:
        for p in res.rs.corrected_positions:
            for r_, c_ in zip(g.cw_rows[p], g.cw_cols[p]):
                changed.add((int(r_), int(c_)))
    for r in range(g.rows):
        for c in range(g.cols):
            x, y = pts[r, c]
            col = (0, 0, 255) if (r, c) in changed else ((0, 200, 0) if res.fit.bits[r, c] else (255, 160, 0))
            cv2.circle(img, (int(x), int(y)), rad, col, -1)
    cv2.polylines(img, [quad.astype(np.int32)], True, (0, 255, 255), 1)
    cv2.circle(img, tuple(int(v) for v in quad[0]), rad * 3, (0, 0, 255), 2)
    stem = os.path.splitext(os.path.basename(res.path))[0]
    cv2.imencode(".png", img)[1].tofile(os.path.join(debug_dir, f"{stem}_result.png"))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def printable(data: bytes) -> str:
    names = {29: "<GS>", 30: "<RS>", 4: "<EOT>", 13: "<CR>", 10: "<LF>", 9: "<TAB>"}
    return "".join(names.get(b, f"<{b:02X}>") if b < 32 or b == 127 else chr(b) for b in data) \
        if all(b < 128 for b in data) else to_text(data, None)[0]


def matrix_art(bits: np.ndarray) -> str:
    lines = []
    for r in range(0, bits.shape[0], 2):
        top = bits[r]
        bot = bits[r + 1] if r + 1 < bits.shape[0] else np.zeros_like(top)
        lines.append("    " + "".join("█" if t and b else "▀" if t else "▄" if b else " "
                                      for t, b in zip(top, bot)))
    return "\n".join(lines)


def hexrows(cws: list[int], marks: set[int] = frozenset(), per: int = 16) -> str:
    out = []
    for i in range(0, len(cws), per):
        out.append("    %4d: " % (i + 1) + " ".join(
            (f"{c:02X}*" if i + k in marks else f"{c:02X} ") for k, c in enumerate(cws[i:i + per])))
    return "\n".join(out)


def to_dict(res: ReadResult) -> dict:
    d = {"file": res.path, "decoded": bool(res.data), "elapsed_s": round(res.elapsed, 2),
         "attempts": res.attempts}
    if not res.data:
        if res.near_miss:
            d["near_miss"] = res.near_miss
        return d
    text, charset = to_text(res.data, res.dec)
    d.update({"text": text, "hex": res.data.hex(" ").upper(), "length_bytes": len(res.data),
              "charset": charset, "engine": res.engine, "enhancement": res.variant,
              "grid_sampling": res.sampling,
              "agreeing_reads": res.agreeing, "zxing": res.zx})
    if res.fit:
        g = res.fit.geom
        rows, cols, dregr, dregc, regr, regc, data, eccblk, blocks = g.sym
        d["symbol"] = {
            "size": g.name,
            "shape": "square" if rows == cols else
                     ("rectangular DMRE (ISO/IEC 21471)" if g.sym in DMRE_SYMBOLS else "rectangular"),
            "data_regions": f"{regr}x{regc} of {dregr}x{dregc}",
            "data_codewords": data, "ecc_codewords": eccblk * blocks,
            "rs_blocks": blocks, "ecc_per_block": eccblk,
            "max_correctable_errors_per_block": eccblk // 2,
            "mirrored": res.mirrored}
    if res.rs:
        d["error_correction"] = {
            "codewords_corrected_per_block": res.rs.block_errors,
            "erasures_declared_per_block": res.rs.block_erasures,
            "corrected_positions": [p + 1 for p in res.rs.corrected_positions],
            "fixed_pattern_module_errors": res.fit.fixed_errors,
            "codewords": res.rs.all_cw}
    if res.dec:
        d["encodation"] = {
            "payload_codewords": res.dec.payload_cw,
            "pad_codewords": res.dec.pad_cw,
            "gs1": res.dec.gs1, "eci": res.dec.eci, "macro": res.dec.macro,
            "structured_append": res.dec.structured_append,
            "segments": [{"mode": s.mode, "codewords": f"{s.cw_start + 1}-{s.cw_end}",
                          "data": printable(bytes(s.data)), "note": s.note}
                         for s in res.dec.segments],
            "warnings": res.dec.warnings}
        if res.dec.gs1:
            d["gs1_elements"] = parse_gs1(text)
    if res.quality:
        d["quality_estimate"] = res.quality
    return d


def print_report(res: ReadResult, show_matrix: bool = False) -> None:
    bar = "=" * 72
    print(bar)
    print(f"File       : {res.path}")
    if not res.data:
        print(f"Result     : NOT DECODED  ({res.attempts} attempts, {res.elapsed:.1f} s)")
        if res.near_miss:
            nm = res.near_miss
            mpx = nm.get("modules_px_orig", nm.get("modules_px", 0.0))
            agree = nm.get("size_agree")
            print(f"Near-miss  : {nm['size']} grid fit, score {nm['score']:.1f}, "
                  f"{nm['fixed_pattern_module_errors']} finder/timing modules wrong, "
                  f"~{mpx:.1f} px/module" + (" (no-quiet-zone)" if nm.get("qz") is False else ""))
            if agree is not None:
                print("Size check : " + (
                    "module pitch inside the symbol confirms the size" if agree == 2 else
                    f"size only partly confirmed (module pitch inside suggests ~{nm['measured_size']})"
                    if agree == 1 else
                    f"size NOT confirmed (module pitch inside suggests ~{nm['measured_size']})"))
            if nm.get("outside_dark", 0.0) > OUTSIDE_BUSY:
                print(f"Extent     : {nm['outside_dark']:.0%} of the ring around the fit is dark: "
                      "it may cover only part of the symbol")
            hints = []
            if mpx and mpx < 2:
                hints.append(f"modules are ~{mpx:.1f} px: shoot much closer / crop tighter")
            if agree == 0:
                hints.append("symbol size uncertain: if known, pass it with --expect RxC")
            elif nm["fixed_pattern_module_errors"] >= 6:
                hints.append("finder pattern damaged: try another viewing angle or more light")
            if nm["score"] < STRONG_FIT:
                hints.append("low finder/timing contrast: reduce ambient light / add illumination")
            print("Hints      : " + ("; ".join(hints) if hints
                                     else "crop closer, avoid glare, shoot straight on, add light"))
        else:
            print("Hints      : crop closer, avoid glare, shoot straight on, add light")
        print(bar)
        return
    d = to_dict(res)
    print(f"Content    : {printable(res.data)}")
    print(f"Hex        : {d['hex']}")
    print(f"Length     : {d['length_bytes']} bytes, charset {d['charset']}")
    print(f"Decoder    : {res.engine}   ({res.attempts} attempts, {res.elapsed:.1f} s)")
    print(f"Enhancement: {res.variant}" + (f"  | sampling: {res.sampling}" if res.sampling else ""))
    if len(res.agreeing) > 1:
        print(f"Confirmed  : {len(res.agreeing)} independent reads agree")
    if res.zx:
        print("zxing info : " + ", ".join(f"{k}={v}" for k, v in res.zx.items() if v not in ("", None)))
    if "symbol" in d:
        s = d["symbol"]
        print("\nSymbol")
        print(f"  size            : {s['size']} modules ({s['shape']}), data regions {s['data_regions']}")
        print(f"  capacity        : {s['data_codewords']} data + {s['ecc_codewords']} ECC codewords, "
              f"{s['rs_blocks']} RS block(s) x {s['ecc_per_block']} ECC")
        print(f"  correctable     : {s['max_correctable_errors_per_block']} errors "
              f"or {s['ecc_per_block']} erasures per block")
        if s["mirrored"]:
            print("  mirrored        : YES (read from behind / mirrored mark)")
    if "error_correction" in d:
        e = d["error_correction"]
        print("\nError correction")
        print(f"  corrected cw    : {e['codewords_corrected_per_block']} per block"
              f"  (erasures declared {e['erasures_declared_per_block']})")
        print(f"  finder damage   : {e['fixed_pattern_module_errors']} wrong finder/timing modules")
        print("  codewords (* = corrected):")
        print(hexrows(e["codewords"], set(res.rs.corrected_positions)))
    if "encodation" in d:
        en = d["encodation"]
        print("\nEncodation")
        print(f"  payload         : {en['payload_codewords']} codewords, padding {en['pad_codewords']}")
        for s in en["segments"]:
            print(f"  {s['mode']:8s} cw {s['codewords']:>8s} : {s['data']!s:.60s} {s['note']}")
        flags = [f for f, on in (("GS1", en["gs1"]), (f"ECI {en['eci']}", en["eci"]),
                                 (f"Macro {en['macro']}", en["macro"]),
                                 (en["structured_append"], en["structured_append"])) if on]
        if flags:
            print(f"  flags           : {', '.join(flags)}")
        for w in en["warnings"]:
            print(f"  warning         : {w}")
    if "gs1_elements" in d:
        print("\nGS1 element string")
        for ai, title, val in d["gs1_elements"]:
            print(f"  ({ai}) {title:16s} {val}")
    if res.quality:
        q = res.quality
        print("\nQuality estimate (ISO/IEC 15415 style, indicative only)")
        print(f"  overall         : {q['overall_grade_estimate']}")
        print(f"  symbol contrast : {q['symbol_contrast_pct']}%  [{q['symbol_contrast_grade']}]"
              f"   dark/light reflectance {q['dark_reflectance_mean']}/{q['light_reflectance_mean']}")
        print(f"  modulation      : {q['modulation_p5']}  [{q['modulation_grade']}]")
        print(f"  unused ECC      : {q['unused_error_correction']}  [{q['unused_error_correction_grade']}]")
        print(f"  fixed pattern   : {q['fixed_pattern_damage_pct']}% damaged  [{q['fixed_pattern_damage_grade']}]")
        print(f"  axial nonunif.  : {q['axial_nonuniformity']}  [{q['axial_nonuniformity_grade']}]")
        print(f"  geometry        : module {q['module_size_px']} px, symbol {q['symbol_size_px'][0]}x"
              f"{q['symbol_size_px'][1]} px, rotation {q['rotation_deg']} deg, skew {q['skew_deg']} deg")
    if show_matrix and res.fit is not None:
        print("\nSampled module matrix")
        print(matrix_art(res.fit.bits))
    print(bar)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Robust Data Matrix (ECC 200) reader")
    ap.add_argument("inputs", nargs="*", help="image files or folders")
    ap.add_argument("--debug", metavar="DIR", help="save variants + annotated result")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--exhaustive", action="store_true", help="try every method, keep best")
    ap.add_argument("--engines", default="zxing,grid,libdmtx")
    ap.add_argument("--matrix", action="store_true", help="print sampled module matrix")
    ap.add_argument("--budget", type=float, default=40.0, help="max seconds per image")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help="worker processes (1 = sequential)")
    ap.add_argument("--expect", action="append", default=[], metavar="RxC",
                    help="expected symbol size(s), e.g. 24x24 (repeatable; like the "
                         "trained model of an industrial reader, prunes the grid search)")
    ap.add_argument("--polarity", choices=("any", "dark", "light"), default="any",
                    help="mark polarity: try both with a small inverted variant chain "
                         "(default), dark-on-light only, or light-on-dark")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    inputs = args.inputs
    if not inputs:
        p = input("Image file or folder: ").strip().strip('"')
        if not p:
            return
        inputs = [p]
    engines = {e.strip() for e in args.engines.split(",") if e.strip()}
    for name, mod, pkg in (("zxing", zxingcpp, "zxing-cpp"), ("libdmtx", pylibdmtx, "pylibdmtx")):
        if name in engines and mod is None:
            print(f"warning: {name} engine unavailable (pip install {pkg}), skipped",
                  file=sys.stderr)
    expect = parse_sizes(args.expect)
    results = []
    runner = Runner(args.jobs)
    try:
        for path in collect_inputs(inputs):
            try:
                res = read_image(path, engines, args.exhaustive, args.debug, args.budget,
                                 runner, expect, args.polarity)
            except ValueError as exc:
                print(f"{path}: {exc}", file=sys.stderr)
                continue
            results.append(res)
            if not args.json:
                print_report(res, args.matrix)
    finally:
        runner.close()
    if args.json:
        print(json.dumps([to_dict(r) for r in results], indent=2, ensure_ascii=False))
    elif len(results) > 1:
        ok = sum(1 for r in results if r.data)
        print(f"\nSummary: {ok}/{len(results)} decoded")
        for r in results:
            print(f"  {'OK  ' if r.data else 'FAIL'} {os.path.basename(r.path):40s} "
                  f"{printable(r.data) if r.data else ''}")


if __name__ == "__main__":
    main()
