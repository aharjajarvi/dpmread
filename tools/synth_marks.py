#!/usr/bin/env python3
"""Synthetic Data Matrix marks with exact ground truth.

Every image is rendered from a parameter record sampled from a seeded RNG;
the record is stored next to the image, so any image can be rendered again
exactly (`--params`) and any parameter swept on its own later.

Pipeline:
  1. content and symbol: a random text from a template, encoded with
     `datamatrix_gen.py` (the module matrix is the label),
  2. module geometry: ideal module centres plus a smooth displacement field
     (non-uniform pitch), per-module jitter, in-plane rotation, shear and
     aspect (marking head skew),
  3. marks, substrate and scene, drawn on a raster of the marking plane
     (R pixels per module): a burn map 0..1 in one style (`laser_dots`,
     `laser_fill`, `peen`, `flat`), wood (procedural end or side grain, or
     a patch of a real photo), burned digits near the symbol, a board edge,
     composited into colour,
  4. camera geometry: the plane is warped once into the image through the
     in-plane transform, a perspective tilt and a cylindrical bend, at
     SUPERSAMPLE x the final resolution, then reduced,
  5. camera photometry at the final size: lighting gradient, specular spot,
     white balance, defocus, motion blur, noise, gamma, mono, JPEG.

Output in `--out`:
  NNNNNN.png|.jpg  image
  NNNNNN.json     parameter record, file name, text, size, polarity, module
                  matrix, symbol corners and true module centres in image
                  pixels, per-module visibility (did the mark render)
  clean/NNNNNN.png  the same geometry (camera warp included) with flat
                  marks, no substrate, scene or photometric degradation
  expected.tsv    file, text (read by `score_public.py`); also in clean/
  index.tsv       file, text, size, style, ppm, seed

Coordinates are image pixels with pixel centres on integers (OpenCV
convention); `centres[r][c]` is the centre of module row r, column c, row 0
at the top of the symbol (finder L on the left and bottom edges).

Usage:
  python tools/synth_marks.py --out synth/run1 --count 300 --seed 1
  python tools/synth_marks.py --params synth/run1/000007.json --out /tmp/x
  python tools/synth_marks.py --check synth/run1     # C4 / C5 checks
  python tools/synth_marks.py --out synth/review1 --count 300 --preview \
      [--real real.tsv]                              # review.html
Needs numpy, opencv and Pillow (the generator's import).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import string
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import datamatrix_gen as gen  # noqa: E402

SUPERSAMPLE = 4          # render at this factor, reduce with INTER_AREA
SUBPIX_SHIFT = 4         # fractional bits for cv2 polygon drawing
MAX_SYMBOL_SIDE = 32     # largest square size sampled (real marks are small)
STYLES = ("laser_dots", "laser_fill", "ink", "peen", "flat")
DEFAULT_STYLES = ("laser_dots", "laser_fill", "ink", "peen")
# default share of each style; real marks so far are all laser marks
LIGHT_SHARE = 0.0        # share of light-on-dark marks
SIZE_CHOICE: list[str] = []   # --sizes: one of these, uniform, when the text fits
STYLE_WEIGHTS = {"laser_dots": 0.35, "laser_fill": 0.3, "ink": 0.25, "peen": 0.1, "flat": 0.0}
# laser on photo wood: opacity range, char colour factor, wood-detail factor
LASER_PHOTO = {"opacity": (0.8, 0.98), "char": 0.75, "detail": 0.5}
TEMPLATES = ("digits", "id", "alnum", "text")
PATCH_EXT = (".png", ".jpg", ".jpeg")
DEFAULT_READER = Path(__file__).resolve().parent.parent / "rust/target/release/dpmread"


def u(rng: np.random.Generator, lo: float, hi: float, nd: int = 3) -> float:
    return round(float(rng.uniform(lo, hi)), nd)


# --------------------------------------------------------------------------
# Parameter record
# --------------------------------------------------------------------------
def random_text(rng: np.random.Generator, template: str) -> str:
    def pick(chars: str, n: int) -> str:
        return "".join(chars[i] for i in rng.integers(0, len(chars), n))
    if template == "digits":
        return pick(string.digits, int(rng.integers(4, 17)))
    if template == "id":                     # short letter prefix + number
        return (pick(string.ascii_lowercase, int(rng.integers(1, 3)))
                + pick(string.digits, int(rng.integers(3, 7))))
    if template == "alnum":
        return pick(string.ascii_uppercase + string.digits, int(rng.integers(4, 13)))
    words = string.ascii_letters + string.digits + " -./:"
    return pick(words, int(rng.integers(3, 21))).strip() or "x"


def fitting_sizes(text: str, n_data: int, rect: bool) -> list[str]:
    syms = gen.RECT_SYMBOLS if rect else gen.SQUARE_SYMBOLS
    names = [f"{s[0]}x{s[1]}" for s in syms
             if s[6] >= n_data and s[0] <= MAX_SYMBOL_SIDE]
    # n_data comes from the smallest square symbol, where a trailing EDIFACT
    # unlatch may be dropped; a size that needs that codeword does not fit
    return [n for n in names if encodes(text, n)]


def encodes(text: str, size: str) -> bool:
    try:
        gen.generate(text, allow_rect=True, forced_size=size)
    except ValueError:
        return False
    return True


def sample_mark(rng: np.random.Generator, style: str) -> dict:
    """Style parameters; diameters and growth in modules."""
    if style == "flat":
        return {}
    m = {"strength_sd": u(rng, 0.0, 0.15), "missing": u(rng, 0.0, 0.03),
         "weak": u(rng, 0.0, 0.08)}
    if style == "laser_dots":
        # small dots and light dot centres were the hardest part of the
        # reviewed ranges (0.6-1.3, rim 0-0.4); kept to the easier part
        m.update(dot_d=u(rng, 0.8, 1.3), dot_sd=u(rng, 0.0, 0.08),
                 rim=u(rng, 0.0, 0.2), soft=u(rng, 0.03, 0.15),
                 # outline: a few low harmonics, laser dots are not round
                 wobble=u(rng, 0.0, 0.15))
    elif style == "laser_fill":
        # more growth merged modules into a printed-looking block (L23)
        m.update(grow=u(rng, -0.12, 0.15), grow_sd=u(rng, 0.0, 0.05),
                 ragged=u(rng, 0.0, 0.25))
    elif style == "ink":
        # ink-jet: each module printed as a few scan lines along the rows,
        # lines run over and break up, raster lines shifted sideways
        m.update(lines=u(rng, 2.5, 5.0), duty=u(rng, 0.6, 0.9),
                 overrun=u(rng, 0.0, 0.25), breaks=u(rng, 0.0, 0.2),
                 line_jitter=u(rng, 0.0, 0.12), wave=u(rng, 0.0, 0.15))
    else:                                    # peen
        m.update(dot_d=u(rng, 0.5, 0.95), dot_sd=u(rng, 0.0, 0.08),
                 light_deg=u(rng, 0.0, 360.0, 1), shade=u(rng, 0.15, 0.5),
                 dark=u(rng, 0.05, 0.4))
    return m


def sample_substrate(rng: np.random.Generator, patches: dict, extent: float) -> dict:
    """Wood: base colour (RGB 0..1), grain kind and scales in modules."""
    r = rng.uniform(0.82, 0.96)
    g = r * rng.uniform(0.86, 0.93)
    b = g * rng.uniform(0.70, 0.86)
    s = {"kind": "end" if rng.random() < 0.55 else "side",
         "rgb": [round(float(x), 3) for x in (r, g, b)],
         "spacing": u(rng, 1.5, 5.0), "ring_c": u(rng, 0.03, 0.22),
         "wobble": u(rng, 0.0, 0.3), "fibre_c": u(rng, 0.0, 0.15),
         "pore_c": u(rng, 0.03, 0.2), "grain_c": u(rng, 0.02, 0.10),
         "blotch_c": u(rng, 0.0, 0.1), "sat": u(rng, 0.0, 1.0), "dim": 1.0,
         "painted": False}
    if s["kind"] == "end":
        s.update(centre_dist=u(rng, 12.0, 80.0, 2), centre_deg=u(rng, 0.0, 360.0, 1))
    else:
        s.update(fibre_deg=u(rng, -90.0, 90.0, 1), phase=u(rng, 0.0, 1.0))
    if rng.random() < 0.12:                  # white primer, grain faintly through
        w = rng.uniform(0.9, 0.98)
        s.update(painted=True, rgb=[round(float(w), 3), round(float(w * 0.99), 3),
                                    round(float(w * 0.97), 3)],
                 ring_c=round(s["ring_c"] * 0.3, 3), sat=0.0)
    # only patches that cover the plane (`extent` modules) without being
    # enlarged more than 1.5x: enlarged patches looked smeared in review
    fit = sorted(n for n, v in patches.items() if v is None or v[1] * 1.5 >= extent)
    if fit and rng.random() < 0.5:
        name = str(rng.choice(fit))
        # patch pixels per module: the source PPM when known (grain keeps
        # its real size against the modules), else a guess
        src = patches[name]
        scale = src[0] * rng.uniform(0.7, 1.4) if src else rng.uniform(1.5, 5.0)
        s.update(kind="patch", patch=name, painted=False,
                 patch_scale=round(float(scale), 3), patch_deg=u(rng, 0.0, 360.0, 1),
                 patch_off=[u(rng, -1.0, 1.0), u(rng, -1.0, 1.0)])
    return s


def sample_burn(rng: np.random.Generator, wood_rgb: list[float], polarity: str,
                style: str) -> dict:
    """Burn (or ink) colour and opacity; light polarity = pale marks on a
    dark surface."""
    if style == "ink":                 # neutral black-grey, on the surface
        k = rng.uniform(0.04, 0.15)
        return {"rgb": [round(float(k * f), 3) for f in (1.0, 1.0, rng.uniform(1.0, 1.15))],
                "opacity": u(rng, 0.85, 1.0), "couple": u(rng, 0.0, 0.3),
                "texture": u(rng, 0.0, 0.25), "inherit": u(rng, 0.6, 1.2),
                "depth": 0.0, "light_deg": 0.0, "rough": u(rng, 0.0, 0.1)}
    t = rng.uniform(0.3, 1.0)      # always some char: flat orange read as ink (L23)
    brown = np.array([0.80, 0.70, 0.55]) * [0.75, 0.55, 0.35]
    black = np.array([0.10, 0.07, 0.05])
    rgb = brown * (1 - t) + black * t
    if polarity == "light":
        rgb = np.array(wood_rgb) * rng.uniform(0.9, 1.1)
    return {"rgb": [round(float(x), 3) for x in np.clip(rgb, 0, 1)],
            "opacity": u(rng, 0.6, 0.95) if polarity == "dark" else u(rng, 0.65, 1.0), "couple": u(rng, 0.0, 0.4),
            # the mark inherits the surface: wood detail through the burn
            # (inherit), darker fibres burn deeper (texture), engraved depth
            # shaded from one light direction (depth), char roughness (rough)
            # (in review, wider ranges broke the marks up on photo wood)
            "texture": u(rng, 0.0, 0.25), "inherit": u(rng, 0.6, 1.4),
            "depth": u(rng, 0.0, 0.45), "light_deg": u(rng, 0.0, 360.0, 1),
            "rough": u(rng, 0.0, 0.25)}


def laser_on_photo(bp: dict) -> None:
    """Laser burns on photo wood: the latewood rings of the patches are as
    dark as a light burn and about a module wide, so 59 % of these marks
    were not decoded against 11 % of real laser marks. Darker, more opaque
    char, less wood detail through it. A fixed remap of the sampled values,
    so the RNG stream of every other sample is unchanged."""
    lo, hi = LASER_PHOTO["opacity"]
    bp["opacity"] = round(lo + (bp["opacity"] - 0.6) / 0.35 * (hi - lo), 3)
    bp["rgb"] = [round(x * LASER_PHOTO["char"], 3) for x in bp["rgb"]]
    for k in ("texture", "inherit", "rough", "couple"):
        bp[k] = round(bp[k] * LASER_PHOTO["detail"], 3)


def sample_scene(rng: np.random.Generator) -> dict:
    sc = {"digits": None, "edge": None}
    if rng.random() < 0.4:
        txt = random_text(rng, str(rng.choice(["digits", "id"])))[:5]
        sc["digits"] = {"text": txt, "side": str(rng.choice(["top", "bottom"])),
                        "height": u(rng, 2.0, 5.0), "stroke": u(rng, 0.25, 0.45),
                        "gap": u(rng, 0.8, 2.5), "offset": u(rng, -0.3, 0.3)}
    if rng.random() < 0.5:
        sc["edge"] = {"side": int(rng.integers(0, 4)), "deg": u(rng, -8.0, 8.0, 2),
                      "gap": u(rng, 1.0, 6.0), "kind": str(rng.choice(["gap", "board"])),
                      "factor": u(rng, 0.1, 0.35) if rng.random() < 0.5
                      else u(rng, 0.7, 1.15),
                      "seam": u(rng, 0.0, 0.5)}
    return sc


def sample_camera(rng: np.random.Generator, ppm: float) -> dict:
    c = {"tilt": round(float(rng.uniform(0, 1) ** 2 * 0.08), 4),
         "tilt_dirs": [u(rng, -1.0, 1.0) for _ in range(8)],
         "bend": u(rng, 0.0, 0.05, 4) if rng.random() < 0.4 else 0.0,
         "bend_deg": u(rng, 0.0, 180.0, 1),
         "light": u(rng, 0.0, 0.35), "light_deg": u(rng, 0.0, 360.0, 1),
         "spec": None,
         "wb": [u(rng, 0.95, 1.05) for _ in range(3)],
         # pixels; scales with PPM: sharp high-PPM marks looked fake in review
         "defocus": round(float(rng.uniform(0.03, 0.15) * ppm), 3),
         "motion": [u(rng, 0.5, 2.5, 2), u(rng, 0.0, 180.0, 1)]
         if rng.random() < 0.15 else None,
         "noise": u(rng, 0.0, 0.02, 4), "gamma": u(rng, 0.8, 1.25),
         "mono": bool(rng.random() < 0.3),     # line cameras are mono
         "jpeg": int(rng.integers(60, 96)) if rng.random() < 0.6 else 0}
    if rng.random() < 0.25:
        c["spec"] = [u(rng, 0.0, 1.0), u(rng, 0.0, 1.0), u(rng, 0.1, 0.4),
                     u(rng, 0.1, 0.5)]           # x, y (fractions), sigma, strength
    return c


def sample_params(rng: np.random.Generator, seed: list[int], styles: list[str],
                  patches: dict) -> dict:
    """One parameter record. Plain JSON values only."""
    while True:
        template = str(rng.choice(TEMPLATES))
        text = random_text(rng, template)
        n_data = gen.generate(text).payload_cw
        if SIZE_CHOICE:
            fit = set(fitting_sizes(text, n_data, False) + fitting_sizes(text, n_data, True))
            sizes = [s for s in SIZE_CHOICE if s in fit]
            if sizes:
                size = str(rng.choice(sizes))
                break
            continue
        sizes = fitting_sizes(text, n_data, rect=bool(rng.random() < 0.3))
        if sizes:
            # mostly the smallest size that fits, sometimes one or two larger
            size = sizes[min(len(sizes) - 1, int(rng.choice([0, 0, 0, 1, 2])))]
            break
    w = np.array([STYLE_WEIGHTS[s] for s in styles])
    style = str(rng.choice(styles, p=w / w.sum() if w.sum() > 0 else None))
    # light on dark was flagged in every review round; only on request
    polarity = "dark" if rng.random() < 1.0 - LIGHT_SHARE else "light"
    p = {
        "seed": seed,
        "render_seed": int(rng.integers(0, 2**31)),
        "template": template,
        "text": text,
        "size": size,
        "style": style,
        "polarity": polarity,
        # log-uniform; the real images seen so far are ~10 pixels per module
        "ppm": round(float(math.exp(rng.uniform(math.log(3.0), math.log(12.0)))), 2),
        # mostly a quiet zone of a few modules, sometimes cropped tight
        "quiet": round(float(rng.uniform(1.5, 5.0) if rng.random() < 0.85
                             else rng.uniform(0.4, 1.5)), 2),
        # marks follow the board: mostly near a multiple of 90 degrees
        "rotation_deg": round(float(rng.uniform(-180.0, 180.0) if rng.random() < 0.15
                                    else rng.integers(-1, 3) * 90 + rng.normal(0.0, 6.0)), 2),
        "shear": round(float(rng.uniform(-0.04, 0.04)), 4),
        "aspect": round(float(rng.uniform(0.95, 1.05)), 4),
        "drift": round(float(rng.uniform(0.0, 0.35)), 3),
        "jitter": round(float(rng.uniform(0.0, 0.06)), 3),
        "mark": sample_mark(rng, style),
    }
    side = max(int(v) for v in size.split("x"))
    # plane seen by the image, in modules (a rotated crop needs ~1.45x)
    p["substrate"] = sample_substrate(rng, patches, (side + 2 * p["quiet"]) * 1.45)
    p["burn"] = sample_burn(rng, p["substrate"]["rgb"], polarity, style)
    if style.startswith("laser") and p["substrate"]["kind"] == "patch":
        laser_on_photo(p["burn"])
    if style == "ink":                 # printed marks skew more than laser ones
        p["shear"] = round(float(rng.uniform(-0.12, 0.12)), 4)
    if polarity == "light":
        p["substrate"]["dim"] = u(rng, 0.2, 0.4)
    p["scene"] = sample_scene(rng)
    p["camera"] = sample_camera(rng, p["ppm"])
    return p


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def drift_field(rng: np.random.Generator, rows: int, cols: int):
    """Smooth displacement (modules) of mark-plane points u, v: a few
    sinusoids of at most one cycle across the symbol, unscaled."""
    waves = [(rng.uniform(-1.0, 1.0, 2), rng.uniform(0.0, 2 * math.pi, 2),
              rng.normal(0.0, 1.0, 2)) for _ in range(3)]

    def field(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        du = np.zeros_like(u)
        dv = np.zeros_like(v)
        for (fu, fv), phase, (a, b) in waves:
            arg = 2 * math.pi * (fu * u / cols + fv * v / rows)
            du += a * np.sin(arg + phase[0])
            dv += b * np.sin(arg + phase[1])
        return du, dv
    return field


def apply_h(hom: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ hom.T
    return q[:, :2] / q[:, 2:3]


def module_geometry(p: dict, rows: int, cols: int, rng: np.random.Generator) -> dict:
    """Displaced module centres in the mark plane (modules), and the drift
    applied to any mark-plane point (for corners and the symbol outline)."""
    v, u = np.mgrid[0:rows, 0:cols].astype(float) + 0.5
    field = drift_field(rng, rows, cols)
    du, dv = field(u, v)
    # the largest displacement of a module centre is `drift`
    k = p["drift"] / (float(np.max(np.hypot(du, dv))) or 1.0)
    du, dv = du * k, dv * k
    ju = np.clip(rng.normal(0.0, p["jitter"], u.shape), -3 * p["jitter"], 3 * p["jitter"])
    jv = np.clip(rng.normal(0.0, p["jitter"], v.shape), -3 * p["jitter"], 3 * p["jitter"])

    def warp(pu: np.ndarray, pv: np.ndarray) -> np.ndarray:
        fu, fv = field(pu, pv)
        return np.stack([pu + fu * k, pv + fv * k], 1)
    return {"u": u + du + ju, "v": v + dv + jv, "warp": warp}


class Camera:
    """Mark plane (modules) -> image pixels: in-plane linear map to `flat`
    pixels centred on the symbol, perspective P, cylindrical bend, shift.
    Both directions are exact (the bend shifts along n by a function of the
    coordinate along t, which the shift leaves unchanged)."""

    def __init__(self, p: dict, rows: int, cols: int, geo: dict):
        ppm, c = p["ppm"], p["camera"]
        th = math.radians(p["rotation_deg"])
        self.lin = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) \
            @ np.array([[p["aspect"], p["shear"]], [0.0, 1.0]]) * ppm
        self.lin_inv = np.linalg.inv(self.lin)
        self.mid = np.array([cols / 2, rows / 2])
        # outline of the drifted symbol, for the half size and the image box
        t = np.linspace(0.0, 1.0, 33)
        ou = np.concatenate([t * cols, np.full_like(t, cols), (1 - t) * cols, np.zeros_like(t)])
        ov = np.concatenate([np.zeros_like(t), t * rows, np.full_like(t, rows), (1 - t) * rows])
        outline = self.to_flat(geo["warp"](ou, ov))
        self.half = float(np.max(np.abs(outline)))
        lh = self.half
        src = np.array([[-lh, -lh], [lh, -lh], [lh, lh], [-lh, lh]], np.float32)
        dst = src + np.array(c["tilt_dirs"], np.float32).reshape(4, 2) * c["tilt"] * 2 * lh
        self.P = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        self.P_inv = np.linalg.inv(self.P)
        b = math.radians(c["bend_deg"])
        self.n = np.array([math.cos(b), math.sin(b)])
        self.t = np.array([-math.sin(b), math.cos(b)])
        self.k = c["bend"]
        q = self.cam(outline)
        margin = p["quiet"] * ppm
        lo, hi = q.min(0), q.max(0)
        self.w = int(math.ceil(hi[0] - lo[0] + 2 * margin))
        self.h = int(math.ceil(hi[1] - lo[1] + 2 * margin))
        self.shift = np.array([self.w / 2 - 0.5, self.h / 2 - 0.5]) - (lo + hi) / 2

    def to_flat(self, uv: np.ndarray) -> np.ndarray:
        return (uv - self.mid) @ self.lin.T

    def bend_offset(self, q: np.ndarray) -> np.ndarray:
        s = (q @ self.t) / self.half
        return self.n * (self.k * self.half * s * s)[:, None]

    def cam(self, f: np.ndarray) -> np.ndarray:
        q = apply_h(self.P, f)
        return q + self.bend_offset(q)

    def to_image(self, uv: np.ndarray) -> np.ndarray:
        return self.cam(self.to_flat(uv)) + self.shift

    def to_plane(self, xy: np.ndarray) -> np.ndarray:
        q = xy - self.shift
        q = q - self.bend_offset(q)
        return apply_h(self.P_inv, q) @ self.lin_inv.T + self.mid


# --------------------------------------------------------------------------
# Plane raster
# --------------------------------------------------------------------------
class Plane:
    """Raster of the mark plane covering the image's preimage; pixel (i, j)
    centre at u = u0 + (j + 0.5) / R, v = v0 + (i + 0.5) / R."""

    def __init__(self, uv_min: np.ndarray, uv_max: np.ndarray, R: int):
        self.R = R
        self.u0 = math.floor(uv_min[0]) - 1
        self.v0 = math.floor(uv_min[1]) - 1
        self.W = int(math.ceil((uv_max[0] - self.u0 + 1) * R))
        self.H = int(math.ceil((uv_max[1] - self.v0 + 1) * R))
        j = (np.arange(self.W, dtype=np.float32) + 0.5) / R + self.u0
        i = (np.arange(self.H, dtype=np.float32) + 0.5) / R + self.v0
        self.U, self.V = np.meshgrid(j, i)

    def px(self, u: float, v: float) -> tuple[float, float]:
        return (u - self.u0) * self.R - 0.5, (v - self.v0) * self.R - 0.5

    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.W), np.float32)


def smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def value_noise(rng: np.random.Generator, shape: tuple[int, int], cell: float) -> np.ndarray:
    """Smooth noise, unit standard deviation, features about `cell` pixels."""
    cell = max(cell, 1.0)
    gh, gw = int(shape[0] / cell) + 4, int(shape[1] / cell) + 4
    g = rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
    big = cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
    o = int(cell)
    n = big[o:o + shape[0], o:o + shape[1]]
    return (n - n.mean()) / (n.std() + 1e-6)


def module_strengths(rng: np.random.Generator, m: dict, matrix: np.ndarray):
    """Per-module strength (0 = missing) for the dark modules."""
    s = np.clip(rng.normal(1.0, m.get("strength_sd", 0.0), matrix.shape), 0.3, 1.3)
    r = rng.random(matrix.shape)
    s[r < m.get("weak", 0.0) + m.get("missing", 0.0)] *= rng.uniform(0.25, 0.55)
    s[r < m.get("missing", 0.0)] = 0.0
    return np.where(matrix > 0, s, 0.0)


def draw_squares(pl: Plane, geo: dict, matrix: np.ndarray, side: np.ndarray,
                 value: np.ndarray) -> np.ndarray:
    out = pl.zeros()
    scale = 1 << SUBPIX_SHIFT
    for r, c in zip(*np.nonzero(matrix)):
        x, y = pl.px(geo["u"][r, c], geo["v"][r, c])
        h = side[r, c] * pl.R / 2
        pts = np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]])
        cv2.fillConvexPoly(out, np.round(pts * scale).astype(np.int32), float(value[r, c]),
                           lineType=cv2.LINE_AA, shift=SUBPIX_SHIFT)
    return out


def draw_ink(pl: Plane, m: dict, geo: dict, strength: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    """Ink-jet marks: module rectangles (run over along the rows) cut into
    scan lines along u, each line shifted sideways, broken and wavy."""
    R = pl.R
    out = pl.zeros()
    scale = 1 << SUBPIX_SHIFT
    for r, c in zip(*np.nonzero(strength)):
        x, y = pl.px(geo["u"][r, c], geo["v"][r, c])
        hx = (1 + m["overrun"] * rng.uniform(0.3, 1.0)) * R / 2
        hy = R / 2
        pts = np.array([[x - hx, y - hy], [x + hx, y - hy], [x + hx, y + hy], [x - hx, y + hy]])
        cv2.fillConvexPoly(out, np.round(pts * scale).astype(np.int32), float(strength[r, c]),
                           lineType=cv2.LINE_AA, shift=SUBPIX_SHIFT)
    # at most one scan line per ~1.5 final pixels (finer lines made moire)
    lines = min(m["lines"], max(1.0, R / SUPERSAMPLE / 1.5))
    v = pl.V[:, 0]
    line = np.floor(v * lines).astype(int)
    shift = rng.normal(0.0, m["line_jitter"], line.max() - line.min() + 1) * R
    mx = np.arange(pl.W, dtype=np.float32)[None, :] + shift[line - line.min()][:, None]
    my = np.repeat(np.arange(pl.H, dtype=np.float32)[:, None], pl.W, 1)
    out = cv2.remap(out, mx.astype(np.float32), my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    wave = m["wave"] * value_noise(rng, out.shape, 1.5 * R)
    f = (pl.V * lines + wave) % 1.0
    soft = 0.6 * lines / R                           # about one pixel
    stripe = smoothstep(0.0, soft, f) * (1 - smoothstep(m["duty"] - soft, m["duty"], f))
    gaps = smoothstep(-0.2, 0.2, value_noise(rng, out.shape, 0.35 * R) + 1.8 - 4 * m["breaks"])
    return out * stripe * gaps


def stamp_dots(pl: Plane, geo: dict, strength: np.ndarray, diam: np.ndarray, profile,
               rng: np.random.Generator | None = None, wobble: float = 0.0):
    """Max-combine a radial profile(d / radius) * strength at each centre."""
    out = pl.zeros()
    for r, c in zip(*np.nonzero(strength)):
        x, y = pl.px(geo["u"][r, c], geo["v"][r, c])
        rad = diam[r, c] * pl.R / 2
        ext = int(math.ceil(rad * 1.3)) + 2
        x0, x1 = max(0, int(x) - ext), min(pl.W, int(x) + ext + 1)
        y0, y1 = max(0, int(y) - ext), min(pl.H, int(y) + ext + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - x, yy - y) / rad
        if wobble:
            ang = np.arctan2(yy - y, xx - x)
            k, ph = rng.integers(2, 5, 2), rng.uniform(0, 2 * math.pi, 2)
            amp = wobble * rng.uniform(0.3, 1.0, 2)
            d = d / (1 + amp[0] * np.sin(k[0] * ang + ph[0]) + amp[1] * np.sin(k[1] * ang + ph[1]))
        np.maximum(out[y0:y1, x0:x1], strength[r, c] * profile(d, rad),
                   out=out[y0:y1, x0:x1])
    return out


def draw_marks(pl: Plane, p: dict, geo: dict, matrix: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Burn map, peen shading (or None) and per-module visibility."""
    style, m = p["style"], p["mark"]
    if style == "flat":
        ones = np.ones(matrix.shape)
        return draw_squares(pl, geo, matrix, ones, ones), None, matrix.copy()
    s = module_strengths(rng, m, matrix)
    visible = (s > 0).astype(np.uint8)
    if style == "laser_fill":
        side = 1 + 2 * (m["grow"] + rng.normal(0.0, m["grow_sd"], matrix.shape))
        side = np.clip(side, 0.5, 1.8)
        shape = draw_squares(pl, geo, s > 0, side, np.ones(matrix.shape))
        shape = cv2.GaussianBlur(shape, (0, 0), 0.08 * pl.R)
        shape += m["ragged"] * value_noise(rng, shape.shape, 0.2 * pl.R)
        shape = smoothstep(0.35, 0.65, shape)
        level = draw_squares(pl, geo, s > 0, np.full(matrix.shape, 1.6), s)
        level = cv2.GaussianBlur(level, (0, 0), 0.3 * pl.R)
        return shape * np.clip(level, 0, 1.3), None, visible
    if style == "ink":
        return draw_ink(pl, m, geo, s, rng), None, visible
    diam = np.clip(m["dot_d"] * (1 + rng.normal(0.0, m["dot_sd"], matrix.shape)), 0.3, 1.6)
    if style == "laser_dots":
        soft = m["soft"] * pl.R

        def dot(d, rad):
            edge = np.clip((1 - d) * rad / soft + 0.5, 0, 1)
            return edge * (1 - m["rim"] * np.clip(1 - d * d, 0, 1))
        return stamp_dots(pl, geo, s, diam, dot, rng, m.get("wobble", 0.0)), None, visible
    # peen: spherical craters, darker inside, shaded by a light direction
    height = stamp_dots(pl, geo, s, diam, lambda d, rad: np.sqrt(np.clip(1 - d * d, 0, 1)))
    gy, gx = np.gradient(height)
    a = math.radians(m["light_deg"])
    rad = m["dot_d"] * pl.R / 2
    shade = np.clip(-(gx * math.cos(a) + gy * math.sin(a)) * rad, -1, 1)
    return height * m["dark"], shade * m["shade"], visible


def wood(pl: Plane, sub: dict, rows: int, cols: int, rng: np.random.Generator,
         patch_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Wood colour (RGB 0..1) and a latewood map 0..1 on the plane raster."""
    R, U, V = pl.R, pl.U, pl.V
    shape = U.shape
    if sub["kind"] == "patch":
        if patch_dir is None:
            raise SystemExit(f"record needs --patches for {sub['patch']}")
        img = cv2.imdecode(np.fromfile(str(patch_dir / sub["patch"]), np.uint8),
                           cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        a = math.radians(sub["patch_deg"])
        pu = (U - U.mean()) * math.cos(a) - (V - V.mean()) * math.sin(a)
        pv = (U - U.mean()) * math.sin(a) + (V - V.mean()) * math.cos(a)
        # one copy of the patch covers the plane: mirrored repeats made
        # kaleidoscope patterns, so a small patch is enlarged instead
        ph, pw = img.shape[:2]
        sc = min(sub["patch_scale"], (pw - 1) / float(np.ptp(pu)), (ph - 1) / float(np.ptp(pv)))
        mx = pu * sc + pw / 2 + sub["patch_off"][0] * max(0.0, pw - 1 - float(np.ptp(pu)) * sc) / 2
        my = pv * sc + ph / 2 + sub["patch_off"][1] * max(0.0, ph - 1 - float(np.ptp(pv)) * sc) / 2
        rgb = cv2.remap(img, mx.astype(np.float32), my.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        lum = rgb.mean(2)
        late = np.clip((float(np.median(lum)) - lum) / 0.15, 0, 1)
        return rgb * sub["dim"], late
    sp = sub["spacing"]
    if sub["kind"] == "end":
        a = math.radians(sub["centre_deg"])
        cx = cols / 2 + sub["centre_dist"] * math.cos(a)
        cy = rows / 2 + sub["centre_dist"] * math.sin(a)
        dist = np.hypot(U - cx, V - cy)
        phase = (dist + sub["wobble"] * sp * value_noise(rng, shape, 20 * R)) / sp
        fine = value_noise(rng, shape, 0.12 * R)             # pores
        texture = sub["pore_c"] * np.clip(fine - 0.8, 0, None)
    else:
        a = math.radians(sub["fibre_deg"])
        across = -U * math.sin(a) + V * math.cos(a)
        across = across + sub["wobble"] * sp * value_noise(rng, shape, 25 * R)
        phase = across / sp + sub["phase"]
        # fibre streaks: 1-D noise across the fibres, broken along them
        step = 0.08
        lo = float(across.min()) - 1
        knots = rng.normal(0.0, 1.0, int((float(across.max()) - lo) / step) + 3)
        streak = np.interp(across, lo + step * np.arange(len(knots)), knots)
        streak *= 0.6 + 0.4 * value_noise(rng, shape, 2 * R)
        texture = sub["fibre_c"] * np.clip(streak, 0, None) * 0.5 \
            + sub["pore_c"] * 0.3 * np.clip(value_noise(rng, shape, 0.15 * R) - 1, 0, None)
    f = phase % 1.0
    late = (smoothstep(0.5, 0.92, f) * (1 - smoothstep(0.96, 1.0, f))).astype(np.float32)
    blotch = value_noise(rng, shape, 5 * R)
    grain = value_noise(rng, shape, 0.25 * R)
    lum = 1 - sub["ring_c"] * late - texture + sub["blotch_c"] * 0.5 * blotch \
        + sub["grain_c"] * grain
    tint = 1 - late[..., None] * sub["sat"] * np.array([0.0, 0.1, 0.25], np.float32)
    rgb = np.array(sub["rgb"], np.float32) * lum[..., None] * tint
    return (rgb * sub["dim"]).astype(np.float32), late


def draw_digits(pl: Plane, d: dict, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((pl.H, pl.W), np.uint8)       # putText draws 8-bit only
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(d["text"], font, 1.0, 1)
    scale = d["height"] * pl.R / th
    thick = max(1, round(d["stroke"] * pl.R))
    tw *= scale
    xc = cols / 2 + d["offset"] * cols
    if d["side"] == "top":
        base_v = -d["gap"]
    else:
        base_v = rows + d["gap"] + d["height"]
    x, y = pl.px(xc, base_v)
    cv2.putText(out, d["text"], (int(round(x - tw / 2)), int(round(y))), font, scale,
                255, thick, cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


def edge_mask(pl: Plane, e: dict, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Board edge beyond one symbol side: mask 0..1 beyond the edge, and a
    seam weight at the edge; the line never crosses the symbol."""
    normals = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    nx, ny = normals[e["side"]]
    a = math.radians(e["deg"])
    n = np.array([nx * math.cos(a) - ny * math.sin(a), nx * math.sin(a) + ny * math.cos(a)])
    half_across = cols / 2 if ny else rows / 2        # half length along the edge
    half_out = rows / 2 if ny else cols / 2
    gap = e["gap"] + half_across * abs(math.tan(a))
    p0 = np.array([cols / 2, rows / 2]) + np.array([nx, ny]) * (half_out + gap)
    dist = (pl.U - p0[0]) * n[0] + (pl.V - p0[1]) * n[1]
    soft = 1.0 / pl.R
    return smoothstep(-soft, soft, dist), np.exp(-(dist / 0.15) ** 2)


def compose(pl: Plane, p: dict, burn: np.ndarray, shade: np.ndarray | None,
            rows: int, cols: int, rng: np.random.Generator,
            patch_dir: Path | None) -> np.ndarray:
    rgb, late = wood(pl, p["substrate"], rows, cols, rng, patch_dir)
    sc, bp = p["scene"], p["burn"]
    if sc["digits"]:
        burn = np.maximum(burn, draw_digits(pl, sc["digits"], rows, cols))
    # fine wood detail: luminance against its smooth version (~1)
    lum = rgb.mean(2)
    detail = np.clip(lum / (cv2.GaussianBlur(lum, (0, 0), 1.5 * pl.R) + 1e-4), 0.5, 1.5)
    b = burn * bp["opacity"] * (1 + bp["couple"] * (late - 0.3))
    b *= np.clip(1 - bp["texture"] * 4 * (detail - 1), 0.6, 1.2)
    if sc["edge"]:
        e = sc["edge"]
        mask, seam = edge_mask(pl, e, rows, cols)
        b *= 1 - mask
        if e["kind"] == "gap":
            beyond = np.full_like(rgb, 0.0) + np.array(p["substrate"]["rgb"], np.float32) \
                * e["factor"] * p["substrate"]["dim"]
        else:
            beyond = rgb * e["factor"]
        rgb = rgb * (1 - mask[..., None]) + beyond * mask[..., None]
        rgb *= (1 - e["seam"] * seam)[..., None]
    b = np.clip(b, 0, 1)
    char = np.array(bp["rgb"], np.float32) \
        * (np.clip(detail, 0.75, 1.25) ** bp["inherit"])[..., None]
    char *= (1 + bp["rough"] * value_noise(rng, b.shape, 0.08 * pl.R) * b)[..., None]
    rgb = rgb * (1 - b[..., None]) + char * b[..., None]
    if bp["depth"] > 0:                  # engraved: walls lit on one side
        gy, gx = np.gradient(cv2.GaussianBlur(burn, (0, 0), 0.06 * pl.R))
        a = math.radians(bp["light_deg"])
        wall = np.clip(-(gx * math.cos(a) + gy * math.sin(a)) * 0.15 * pl.R, -1, 1)
        rgb *= (1 + bp["depth"] * wall)[..., None]
    if shade is not None:
        rgb *= (1 + shade)[..., None]
    return np.clip(rgb, 0, 1.2).astype(np.float32)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def photometry(img: np.ndarray, c: dict, rng: np.random.Generator) -> np.ndarray:
    """Final-size RGB 0..1 -> 8-bit BGR or gray."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hd = 0.5 * math.hypot(w, h)
    a = math.radians(c["light_deg"])
    img = img * (1 + c["light"] * ((xx - w / 2) * math.cos(a)
                                   + (yy - h / 2) * math.sin(a)) / hd)[..., None]
    if c["spec"]:
        sx, sy, sig, st = c["spec"]
        g = np.exp(-((xx - sx * w) ** 2 + (yy - sy * h) ** 2) / (2 * (sig * hd) ** 2))
        img = img + st * g[..., None]
    img = img * np.array(c["wb"], np.float32)
    if c["defocus"] > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), c["defocus"])
    if c["motion"]:
        length, deg = c["motion"]
        k = int(math.ceil(length)) * 2 + 1
        ker = np.zeros((k, k), np.float32)
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))]) * length / 2
        m = (k - 1) / 2
        scale = 1 << SUBPIX_SHIFT
        a0, a1 = (tuple(int(v) for v in np.round(q * scale)) for q in (m - d, m + d))
        cv2.line(ker, a0, a1, 1.0, 1, cv2.LINE_AA, SUBPIX_SHIFT)
        img = cv2.filter2D(img, -1, ker / max(float(ker.sum()), 1e-6))
    if c["noise"] > 0:
        n = c["noise"]
        img = img + rng.normal(0, n * 0.8, (h, w, 1)) + rng.normal(0, n * 0.4, (h, w, 3))
    img = np.clip(img, 0, 1) ** c["gamma"]
    img = np.round(img * 255).astype(np.uint8)
    if c["mono"]:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render(p: dict, patch_dir: Path | None = None, transfer: bool = False) -> dict:
    """Image, clean image and ground truth for one parameter record.

    With `transfer` (scenes) there is no image and no clean image but a
    `ratio`: the colour with the marks over the same surface without them
    (h x w x 3, 1 where nothing is marked), multiplied into a larger frame."""
    res = gen.generate(p["text"], allow_rect=True, forced_size=p["size"])
    matrix = np.array(res.matrix, np.uint8)
    rows, cols = matrix.shape
    rng = np.random.default_rng(p["render_seed"])
    geo = module_geometry(p, rows, cols, rng)
    cam = Camera(p, rows, cols, geo)
    # scene marks store a lower resolution for large marks (speed)
    w, h, S = cam.w, cam.h, p.get("supersample", SUPERSAMPLE)

    # mark-plane position of every supersampled pixel centre
    xs = (np.arange(w * S) + 0.5) / S - 0.5
    ys = (np.arange(h * S) + 0.5) / S - 0.5
    gx, gy = np.meshgrid(xs, ys)
    uv = cam.to_plane(np.stack([gx.ravel(), gy.ravel()], 1))
    R = int(max(12, math.ceil(p["ppm"] * S)))
    R = min(R, p.get("max_raster", R))
    pl = Plane(uv.min(0), uv.max(0), R)
    mx = ((uv[:, 0] - pl.u0) * R - 0.5).reshape(h * S, w * S).astype(np.float32)
    my = ((uv[:, 1] - pl.v0) * R - 0.5).reshape(h * S, w * S).astype(np.float32)

    def to_image(raster: np.ndarray) -> np.ndarray:
        big = cv2.remap(raster, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)

    out = {}
    if not transfer:
        ones = np.ones(matrix.shape)
        flat = to_image(draw_squares(pl, geo, matrix, ones, ones))
        out["clean"] = np.clip(np.round((1.0 - flat if p["polarity"] == "dark" else flat)
                                        * 255), 0, 255).astype(np.uint8)
    burn, shade, visible = draw_marks(pl, p, geo, matrix, rng)
    # the bare surface draws the same noise: same generator state
    rng_bare = copy.deepcopy(rng) if transfer else None
    colour = to_image(compose(pl, p, burn, shade, rows, cols, rng, patch_dir))
    if transfer:
        bare_p = dict(p, scene={"digits": None, "edge": None})
        bare = to_image(compose(pl, bare_p, pl.zeros(), None, rows, cols, rng_bare, patch_dir))
        out["ratio"] = np.clip(colour / np.maximum(bare, 1e-3), 0.0, 1.5).astype(np.float32)
    else:
        out["image"] = photometry(colour, p["camera"], rng)

    centres = cam.to_image(np.stack([geo["u"].ravel(), geo["v"].ravel()], 1))
    corners = cam.to_image(geo["warp"](np.array([0.0, cols, cols, 0.0]),
                                       np.array([0.0, 0.0, rows, rows])))
    return {**out, "truth": {
        "text": p["text"], "size": p["size"], "polarity": p["polarity"],
        "matrix": matrix.tolist(),
        # outer symbol corners after drift and camera: top-left, top-right,
        # bottom-right, bottom-left of the symbol (not of the image)
        "corners": np.round(corners, 2).tolist(),
        "centres": np.round(centres.reshape(rows, cols, 2), 2).tolist(),
        "visible": visible.tolist(),
    }}


# --------------------------------------------------------------------------
# Scenes (--scene): camera-sized frames with 0-3 marks, for the detector
# --------------------------------------------------------------------------
SCENE_MARKS = (0.05, 0.5, 0.3, 0.15)   # share of frames with 0, 1, 2, 3 marks
# symbol side / long frame side, log-uniform; real phone photos 3.1-27.5 %,
# median 10.4 %
SCENE_SIDE = (0.03, 0.28)
SCENE_CUT = 0.06         # share of marks that may cross the frame edge
SCENE_KEEP_OUT = 2.0     # modules kept free around everything marked
FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN)
DISTRACTORS = ("text", "text", "label", "barcode", "dots", "knot", "stroke", "stain")


def sample_surface(rng: np.random.Generator) -> dict:
    """One region of the frame: wood, a dark gap between boards or another
    flat surface (cardboard, concrete, plastic)."""
    k = rng.random()
    if k < 0.65:
        return scene_wood(rng)
    if k < 0.85:
        return {"kind": "gap", "level": u(rng, 0.03, 0.3), "noise": u(rng, 0.0, 0.05)}
    grey = rng.uniform(0.25, 0.95)            # low saturation: cardboard, concrete
    return {"kind": "flat", "rgb": [round(float(grey * t), 3)
                                    for t in rng.uniform(0.85, 1.1, 3)],
            "noise": u(rng, 0.01, 0.08), "cell": u(rng, 1.0, 40.0, 1)}


def scene_wood(rng: np.random.Generator) -> dict:
    """Wood across a frame: rings 2-6x wider than under one mark and the
    end-grain centre mostly out of the frame (at mark scale whole frames
    showed fine bullseye rings)."""
    s = sample_substrate(rng, {}, 0.0)
    s["dim"] = u(rng, 0.6, 1.05)
    s["spacing"] = round(s["spacing"] * float(rng.uniform(2.0, 6.0)), 3)
    s["ring_c"] = round(s["ring_c"] * 0.7, 3)
    if s["kind"] == "end":
        s["centre_dist"] = round(s["centre_dist"] * float(rng.uniform(3.0, 10.0)), 2)
    return s


def sample_distractor(rng: np.random.Generator, W: int, H: int) -> dict:
    """Shape, size (pixels) and colour; the position is set on placement."""
    kind = str(rng.choice(DISTRACTORS))
    short = min(W, H)
    ink = [round(float(v), 3) for v in rng.uniform(0.02, 0.2) * rng.uniform(0.8, 1.2, 3)]
    if rng.random() < 0.4:               # burned, brown-black
        t = rng.uniform(0.3, 1.0)
        ink = [round(float(a * (1 - t) + b * t), 3)
               for a, b in zip((0.6, 0.38, 0.19), (0.10, 0.07, 0.05))]
    d = {"kind": kind, "deg": u(rng, -180.0, 180.0, 1) if rng.random() < 0.3
         else round(float(rng.integers(-1, 3) * 90 + rng.normal(0.0, 5.0)), 1),
         "rgb": ink, "opacity": u(rng, 0.5, 1.0), "pos": None}
    if kind in ("text", "dots", "label"):
        d["text"] = random_text(rng, str(rng.choice(TEMPLATES)))[:14] or "x"
        d["font"] = int(rng.integers(0, len(FONTS)))
        d["height"] = u(rng, 0.012, 0.06) * short
        d["stroke"] = u(rng, 0.08, 0.2)
    if kind == "dots":                   # dot-matrix printing, dot peen text
        d["height"] = u(rng, 0.02, 0.08) * short
        d["dot"] = u(rng, 0.5, 0.95)
    if kind == "label":                  # paper label with text and a bar code
        d["paper"] = [u(rng, 0.85, 1.0)] * 3
        d["lines"] = int(rng.integers(1, 4))
        d["bars"] = bool(rng.random() < 0.6)
        d["seed"] = int(rng.integers(0, 2**31))
    if kind == "barcode":
        d["height"] = u(rng, 0.04, 0.15) * short
        d["bar"] = u(rng, 1.0, 4.0)       # narrowest bar, pixels
        d["n"] = int(rng.integers(20, 60))
        d["seed"] = int(rng.integers(0, 2**31))
    if kind == "knot":
        d["r"] = [u(rng, 0.01, 0.08) * short, u(rng, 0.01, 0.05) * short]
    if kind == "stroke":                 # marker pen or scratch
        d["len"] = u(rng, 0.05, 0.4) * short
        d["width"] = u(rng, 1.0, 0.01 * short + 1.5)
        d["bend"] = u(rng, -0.3, 0.3)
    if kind == "stain":
        d["r"] = u(rng, 0.03, 0.15) * short
        d["seed"] = int(rng.integers(0, 2**31))
        d["opacity"] = u(rng, 0.15, 0.5)
    return d


def sample_scene_frame(rng: np.random.Generator, seed: list[int], W: int, H: int,
                       styles: list[str], patches: dict) -> dict:
    """One scene record: marks (each a full mark record, rendered as a crop
    and multiplied into the frame), surfaces, distractors, camera. Positions
    are chosen when the frame is first rendered and stored."""
    n = int(rng.choice(len(SCENE_MARKS), p=SCENE_MARKS))
    marks = []
    for _ in range(n):
        p = sample_params(rng, seed, styles, patches)
        side = max(int(v) for v in p["size"].split("x"))
        frac = math.exp(rng.uniform(*np.log(SCENE_SIDE)))
        p["ppm"] = round(max(1.0, frac * max(W, H) / side), 2)
        # the frame has the edges; the crop needs room for the digits
        p["scene"]["edge"] = None
        p["quiet"] = 8.0 if p["scene"]["digits"] else 2.5
        # about 24 supersampled pixels per module, 48 raster pixels at most:
        # a large mark at 4x took seconds and hundreds of MB
        p["supersample"] = max(1, min(SUPERSAMPLE, math.ceil(24 / p["ppm"])))
        p["max_raster"] = 48
        marks.append({"params": p, "cut": bool(rng.random() < SCENE_CUT), "offset": None})
    ppms = [m["params"]["ppm"] for m in marks]
    unit = float(np.median(ppms)) if ppms else math.exp(rng.uniform(math.log(2), math.log(10)))
    n_edges = int(rng.choice(4, p=(0.35, 0.3, 0.25, 0.1)))
    surfaces = [sample_surface(rng) for _ in range(n_edges + 1)]
    if n and all(s["kind"] in ("gap", "flat") for s in surfaces):
        surfaces[int(rng.integers(0, len(surfaces)))] = scene_wood(rng)
    camera = sample_camera(rng, min(ppms) if ppms else unit)
    camera["jpeg"] = int(rng.integers(80, 97))         # phone frames are JPEG
    return {
        "seed": seed, "frame": [W, H],
        "render_seed": int(rng.integers(0, 2**31)),
        "place_seed": int(rng.integers(0, 2**31)),
        "unit": round(unit, 3),
        # board edges: parallel lines across the frame, near 0 or 90 degrees
        "edge_deg": round(float(rng.integers(0, 2) * 90 + rng.normal(0.0, 6.0)), 2),
        "edges": sorted(u(rng, 0.1, 0.9) for _ in range(n_edges)),
        "seam": u(rng, 0.0, 0.6),
        "surfaces": surfaces,
        "marks": marks,
        "distractors": [sample_distractor(rng, W, H) for _ in range(int(rng.poisson(3.0)))],
        "camera": camera,
    }


class Frame:
    """Region geometry of a scene: board edges as lines across the frame."""

    def __init__(self, sp: dict):
        self.W, self.H = sp["frame"]
        a = math.radians(sp["edge_deg"])
        self.n = np.array([math.cos(a), math.sin(a)])
        ext = abs(self.W * self.n[0]) + abs(self.H * self.n[1])
        mid = np.array([self.W / 2, self.H / 2]) @ self.n
        self.cuts = [mid + (f - 0.5) * ext for f in sp["edges"]]

    def region(self, xy: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.cuts, xy @ self.n)

    def maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Region index and distance to the nearest edge, per pixel."""
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        t = xx * self.n[0] + yy * self.n[1]
        reg = np.searchsorted(self.cuts, t)
        dist = np.full(t.shape, np.inf, np.float32)
        for c in self.cuts:
            dist = np.minimum(dist, np.abs(t - c))
        return reg, dist


def surface_rgb(s: dict, shape: tuple[int, int], unit: float,
                rng: np.random.Generator) -> np.ndarray:
    H, W = shape
    if s["kind"] in ("end", "side"):
        U, V = np.meshgrid((np.arange(W, dtype=np.float32) + 0.5) / unit,
                           (np.arange(H, dtype=np.float32) + 0.5) / unit)
        pl = type("P", (), {"R": unit, "U": U, "V": V})
        rgb, _ = wood(pl, s, int(H / unit), int(W / unit), rng, None)
        return rgb
    if s["kind"] == "gap":
        g = s["level"] * (1 + s["noise"] * value_noise(rng, shape, 8.0))
        return (np.repeat(g[..., None], 3, 2) * [1.0, 0.95, 0.85]).astype(np.float32)
    n = 1 + s["noise"] * value_noise(rng, shape, s["cell"]) \
        + 0.5 * s["noise"] * value_noise(rng, shape, 1.0)
    return (np.array(s["rgb"], np.float32) * n[..., None]).astype(np.float32)


def text_mask(text: str, font: int, height: float, stroke: float) -> np.ndarray:
    """Anti-aliased text, 0..1, about `height` pixels high, with a margin."""
    f = FONTS[font]
    (tw, th), base = cv2.getTextSize(text, f, 1.0, 1)
    sc = max(height, 4.0) / th
    thick = max(1, round(stroke * height))
    w, h = int(tw * sc + 2 * thick + 4), int((th + base) * sc + 2 * thick + 4)
    m = np.zeros((h, w), np.uint8)
    cv2.putText(m, text, (thick + 2, int(th * sc) + thick + 2), f, sc, 255, thick, cv2.LINE_AA)
    return m.astype(np.float32) / 255.0


def flat_patch(rgb: list[float], a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.broadcast_to(np.array(rgb, np.float32), (*a.shape, 3)), a.astype(np.float32)


def distractor_patch(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Colour (h x w x 3) and alpha (h x w) of a distractor, upright."""
    k = d["kind"]
    if k == "text":
        return flat_patch(d["rgb"], text_mask(d["text"], d["font"], d["height"], d["stroke"]))
    if k == "dots":
        # the text 7 cells high, one round dot per inked cell
        cells = text_mask(d["text"], 4, 7.0, 0.12) > 0.5
        pitch = d["height"] / 7.0
        h, w = cells.shape
        a = np.zeros((int(h * pitch) + 2, int(w * pitch) + 2), np.float32)
        r = max(0.6, d["dot"] * pitch / 2)
        for i, j in zip(*np.nonzero(cells)):
            cv2.circle(a, (int(round((j + 0.5) * pitch * 16)), int(round((i + 0.5) * pitch * 16))),
                       int(round(r * 16)), 1.0, -1, cv2.LINE_AA, 4)
        return flat_patch(d["rgb"], a)
    if k == "barcode":
        rng = np.random.default_rng(d["seed"])
        widths = rng.integers(1, 5, 2 * d["n"]) * d["bar"]
        edges = np.concatenate([[0.0], np.cumsum(widths)])
        w, h = int(edges[-1]) + 2, max(2, int(d["height"]))
        x = (np.arange(w * 4) + 0.5) / 4.0
        bar = (np.searchsorted(edges, x, side="right") % 2 == 1).astype(np.float32)
        return flat_patch(d["rgb"], np.repeat(bar.reshape(w, 4).mean(1)[None, :], h, 0))
    if k == "label":
        rng = np.random.default_rng(d["seed"])
        lines = [text_mask(d["text"] if i == 0 else random_text(rng, "alnum"), d["font"],
                           d["height"], d["stroke"]) for i in range(d["lines"])]
        if d["bars"]:
            bd = {"kind": "barcode", "seed": d["seed"], "n": int(rng.integers(15, 35)),
                  "bar": max(1.0, d["height"] / 8), "height": d["height"] * 2.5,
                  "rgb": d["rgb"]}
            lines.append(distractor_patch(bd)[1])
        pad = int(d["height"] * 0.8) + 2
        w = max(m.shape[1] for m in lines) + 2 * pad
        h = sum(m.shape[0] for m in lines) + 2 * pad
        ink = np.zeros((h, w), np.float32)
        y = pad
        for m in lines:
            ink[y:y + m.shape[0], pad:pad + m.shape[1]] = m
            y += m.shape[0]
        paper = np.array(d["paper"], np.float32)
        rgb = paper * (1 - ink[..., None]) + np.array(d["rgb"], np.float32) * ink[..., None]
        return rgb.astype(np.float32), np.ones((h, w), np.float32)
    if k == "knot":
        rx, ry = d["r"]
        w, h = int(2.6 * rx) + 3, int(2.6 * ry) + 3
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / rx, (yy - h / 2) / ry)
        return flat_patch(d["rgb"], (1 - smoothstep(0.5, 1.2, r))
                          * (0.6 + 0.4 * np.cos(r * 9) ** 2))
    if k == "stroke":
        L, wd = d["len"], d["width"]
        w, h = int(L + 2 * wd + 4), int(abs(d["bend"]) * L + 2 * wd + 4)
        a = np.zeros((h, w), np.float32)
        t = np.linspace(0, 1, 40)
        y0 = h / 2 - d["bend"] * L / 2
        pts = np.stack([wd + 2 + t * L, y0 + d["bend"] * L * 4 * t * (1 - t)], 1)
        cv2.polylines(a, [np.round(pts * 16).astype(np.int32)], False, 1.0,
                      max(1, int(round(wd))), cv2.LINE_AA, 4)
        return flat_patch(d["rgb"], a)
    # stain: a soft blob
    rng = np.random.default_rng(d["seed"])
    s = int(2.5 * d["r"]) + 4
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    r = np.hypot(xx - s / 2, yy - s / 2) / d["r"]
    return flat_patch(d["rgb"], 1 - smoothstep(0.4, 1.1, r + 0.25 * value_noise(
        rng, (s, s), d["r"] / 3)))


def rotated_box(w: int, h: int, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return w * c + h * s, w * s + h * c


def overlaps(box: tuple, boxes: list[tuple]) -> bool:
    x0, y0, x1, y1 = box
    return any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in boxes)


def paste(rgb: np.ndarray, col: np.ndarray, alpha: np.ndarray, centre, deg: float,
          opacity: float) -> None:
    """Alpha-blend a patch, turned by `deg` about its centre, at `centre`."""
    h, w = alpha.shape
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), -deg, 1.0)
    M[:, 2] += np.array(centre) - [w / 2 - 0.5, h / 2 - 0.5]
    H, W = rgb.shape[:2]
    a = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR) * opacity
    c = cv2.warpAffine(np.ascontiguousarray(col), M, (W, H), flags=cv2.INTER_LINEAR)
    rgb *= 1 - a[..., None]
    rgb += c * a[..., None]


def place_mark(m: dict, ink: tuple, sym: tuple, corners: np.ndarray, fr: Frame,
               wood_regions: set, keep_out: list, place: np.random.Generator) -> None:
    """Set `m["offset"]`: the crop lies in one wood region, clear of the
    other marks, inside the frame (a `cut` mark: up to 40 % of its symbol
    outside). Left None when nothing fits."""
    W, H = fr.W, fr.H
    sw, sh = sym[2] - sym[0], sym[3] - sym[1]
    for _ in range(200):
        if m["cut"]:
            ox = place.uniform(-sym[0] - 0.4 * sw, W - sym[2] + 0.4 * sw)
            oy = place.uniform(-sym[1] - 0.4 * sh, H - sym[3] + 0.4 * sh)
        else:
            if ink[2] - ink[0] >= W or ink[3] - ink[1] >= H:
                return
            ox = place.uniform(-ink[0], W - ink[2])
            oy = place.uniform(-ink[1], H - ink[3])
        ox, oy = int(ox), int(oy)
        box = (ink[0] + ox, ink[1] + oy, ink[2] + ox, ink[3] + oy)
        if overlaps(box, keep_out):
            continue
        # the symbol's own box decides the region (a cut box may leave the frame)
        sb = np.array([[sym[0], sym[1]], [sym[2], sym[1]], [sym[2], sym[3]],
                       [sym[0], sym[3]]]) + [ox, oy]
        regs = set(fr.region(sb).tolist())
        if len(regs) != 1 or not regs <= wood_regions:
            continue
        m["offset"] = [ox, oy]
        return


def render_scene(sp: dict, patch_dir: Path | None = None) -> dict:
    """Frame image and the corners of every mark. Mark and distractor
    positions missing from the record are chosen here (seeded) and stored
    in it; marks and distractors without room are removed from it, so the
    record renders the same frame again."""
    W, H = sp["frame"]
    rng = np.random.default_rng(sp["render_seed"])
    place = np.random.default_rng(sp["place_seed"])
    fr = Frame(sp)
    reg, dist = fr.maps()
    rgb = np.zeros((H, W, 3), np.float32)
    for i, s in enumerate(sp["surfaces"]):
        m = reg == i
        if m.any():
            rgb[m] = surface_rgb(s, (H, W), sp["unit"], rng)[m]
    if sp["edges"]:
        rgb *= (1 - sp["seam"] * np.exp(-(dist / 1.5) ** 2))[..., None]
    wood_regions = {i for i, s in enumerate(sp["surfaces"]) if s["kind"] in ("end", "side")}

    keep_out: list[tuple] = []
    marks = []
    for m in sp["marks"]:
        p = m["params"]
        r = render(p, patch_dir, transfer=True)
        ratio = r["ratio"]
        h, w = ratio.shape[:2]
        corners = np.array(r["truth"]["corners"])
        ys, xs = np.nonzero(np.abs(ratio - 1).max(2) > 0.03)
        pad = SCENE_KEEP_OUT * p["ppm"]
        sym = (*corners.min(0), *corners.max(0))
        ink = (min(xs.min(), sym[0]) - pad, min(ys.min(), sym[1]) - pad,
               max(xs.max(), sym[2]) + pad, max(ys.max(), sym[3]) + pad) if len(xs) \
            else (sym[0] - pad, sym[1] - pad, sym[2] + pad, sym[3] + pad)
        if m["offset"] is None:
            place_mark(m, ink, sym, corners, fr, wood_regions, keep_out, place)
        if m["offset"] is None:
            continue
        ox, oy = m["offset"]
        keep_out.append((ink[0] + ox, ink[1] + oy, ink[2] + ox, ink[3] + oy))
        x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
        if x0 < x1 and y0 < y1:
            rgb[y0:y1, x0:x1] *= ratio[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        quad = corners + [ox, oy]
        inside = bool(quad.min() >= 0 and quad[:, 0].max() <= W - 1
                      and quad[:, 1].max() <= H - 1)
        marks.append({"text": p["text"], "size": p["size"], "polarity": p["polarity"],
                      "style": p["style"], "ppm": p["ppm"], "cut": not inside,
                      "corners": np.round(quad, 2).tolist()})
    sp["marks"] = [m for m in sp["marks"] if m["offset"] is not None]

    for d in sp["distractors"]:
        col, alpha = distractor_patch(d)
        bw, bh = rotated_box(alpha.shape[1], alpha.shape[0], d["deg"])
        if d["pos"] is None:
            for _ in range(50):
                cx, cy = place.uniform(0, W), place.uniform(0, H)
                if not overlaps((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), keep_out):
                    d["pos"] = [round(float(cx), 1), round(float(cy), 1)]
                    break
        if d["pos"] is not None:
            paste(rgb, col, alpha, d["pos"], d["deg"], d["opacity"])
    sp["distractors"] = [d for d in sp["distractors"] if d["pos"] is not None]
    return {"image": photometry(rgb, sp["camera"], rng), "marks": marks}


def write_scene(out: Path, name: str, sp: dict, patch_dir: Path | None) -> dict:
    r = render_scene(sp, patch_dir)
    file = f"{name}.jpg"
    write_img(out / file, r["image"], sp["camera"]["jpeg"])
    rec = {"params": sp, "file": file, "frame": sp["frame"], "marks": r["marks"]}
    (out / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":")) + "\n",
                                      encoding="utf-8")
    return rec


def write_scene_tsv(out: Path, recs: list[tuple[str, dict]]) -> None:
    """index.tsv: file, marks, sizes, cut marks, seed; marks.tsv: one row
    per mark (file, text, size, corners)."""
    (out / "index.tsv").write_text("file\tmarks\tsizes\tcut\tseed\n" + "".join(
        f"{r['file']}\t{len(r['marks'])}\t{','.join(m['size'] for m in r['marks'])}\t"
        f"{sum(m['cut'] for m in r['marks'])}\t"
        f"{r['params']['seed'][0]}:{r['params']['seed'][1]}\n"
        for _, r in recs), encoding="utf-8")
    (out / "marks.tsv").write_text("file\ttext\tsize\tcorners\n" + "".join(
        f"{r['file']}\t{m['text']}\t{m['size']}\t"
        f"{' '.join(f'{x:.1f},{y:.1f}' for x, y in m['corners'])}\n"
        for _, r in recs for m in r["marks"]), encoding="utf-8")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_img(path: Path, img: np.ndarray, jpeg: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    else:
        ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"encoding failed: {path}")
    buf.tofile(str(path))


def read_gray(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)


def write_sample(out: Path, name: str, p: dict, patch_dir: Path | None) -> dict:
    r = render(p, patch_dir)
    jpeg = p["camera"]["jpeg"]
    file = f"{name}.jpg" if jpeg else f"{name}.png"
    write_img(out / file, r["image"], jpeg)
    write_img(out / "clean" / f"{name}.png", r["clean"])
    rec = {"params": p, "file": file, **r["truth"]}
    (out / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":")) + "\n",
                                      encoding="utf-8")
    return rec


def write_tsv(out: Path, recs: list[tuple[str, dict]]) -> None:
    (out / "expected.tsv").write_text(
        "".join(f"{r['file']}\t{r['text']}\n" for _, r in recs), encoding="utf-8")
    (out / "clean" / "expected.tsv").write_text(
        "".join(f"{n}.png\t{r['text']}\n" for n, r in recs), encoding="utf-8")
    idx = "file\ttext\tsize\tstyle\tppm\tseed\n" + "".join(
        f"{r['file']}\t{r['text']}\t{r['size']}\t{r['params']['style']}\t"
        f"{r['params']['ppm']}\t{r['params']['seed'][0]}:{r['params']['seed'][1]}\n"
        for _, r in recs)
    (out / "index.tsv").write_text(idx, encoding="utf-8")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def sample_centres(img: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Bilinear gray value at each centre."""
    mx = centres[..., 0].astype(np.float32)
    my = centres[..., 1].astype(np.float32)
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def check_scenes(run: Path, recs: list[Path], reader: Path, jobs: int) -> int:
    """Scenes: the reader (`--multi`, 10 s per frame) reads each frame;
    every text it reads must be one of the frame's marks."""
    files = [str(run / json.loads(f.read_text(encoding="utf-8"))["file"]) for f in recs]
    n = max(1, min(jobs, len(files)))
    procs = [subprocess.Popen([str(reader), *files[i::n], "--json", "--multi",
                               "--budget", "10", "--jobs", "1"], stdout=subprocess.PIPE)
             for i in range(n)]
    texts: dict[str, list[str]] = {}
    for proc in procs:
        out, _ = proc.communicate()
        if proc.returncode not in (0, 1):
            print(f"reader failed: exit {proc.returncode}")
            return 2
        for r in json.loads(out):
            got = [r["text"]] if r["decoded"] else []
            got += [m["text"] for m in r.get("more_symbols", []) if m.get("decoded")]
            texts[Path(r["file"]).stem] = got
    marks = read = 0
    false = []
    for f in recs:
        want = [m["text"] for m in json.loads(f.read_text(encoding="utf-8"))["marks"]]
        got = texts.get(f.stem, [])
        marks += len(want)
        read += sum(t in got for t in want)
        false += [f"{f.stem}: {t!r}" for t in got if t not in want]
    print(f"scenes: {len(recs)} frames, {read} / {marks} marks read, "
          f"{len(false)} texts not in the labels")
    for w in false[:20]:
        print("  " + w)
    return 1 if false else 0


def check(run: Path, reader: Path, jobs: int) -> int:
    """C5: clean render sampled at the true centres gives the matrix.
    C4: every clean render decodes to its text. Scene runs: `check_scenes`."""
    recs = sorted(run.glob("[0-9]*.json"))
    if recs and "marks" in json.loads(recs[0].read_text(encoding="utf-8")):
        return check_scenes(run, recs, reader, jobs)
    bad_modules = 0
    bad_files = []
    for f in recs:
        rec = json.loads(f.read_text(encoding="utf-8"))
        img = read_gray(run / "clean" / f"{f.stem}.png")
        vals = sample_centres(img, np.array(rec["centres"]))
        dark = vals < 128 if rec["polarity"] == "dark" else vals >= 128
        n = int(np.sum(dark != np.array(rec["matrix"], bool)))
        if n:
            bad_files.append(f"{f.stem}: {n}")
        bad_modules += n
    print(f"C5 module centres: {len(recs)} files, {bad_modules} mismatched modules"
          + (f" ({', '.join(bad_files[:10])})" if bad_files else ""))

    # the reader reads a file list in turn: split it over processes
    files = [str(run / "clean" / f"{f.stem}.png") for f in recs]
    n = max(1, min(jobs, len(files)))
    procs = [subprocess.Popen([str(reader), *files[i::n], "--json", "--no-analysis",
                               "--jobs", "1"], stdout=subprocess.PIPE)
             for i in range(n)]
    texts = {}
    for proc in procs:
        out, _ = proc.communicate()
        if proc.returncode not in (0, 1):    # 1: an image not decoded
            print(f"reader failed: exit {proc.returncode}")
            return 2
        texts.update({Path(r["file"]).stem: r.get("text") if r["decoded"] else None
                      for r in json.loads(out)})
    wrong = []
    for f in recs:
        want = json.loads(f.read_text(encoding="utf-8"))["text"]
        if texts.get(f.stem) != want:
            wrong.append(f"{f.stem}: {texts.get(f.stem)!r} != {want!r}")
    print(f"C4 clean decodes: {len(recs) - len(wrong)} / {len(recs)}")
    for w in wrong[:20]:
        print("  " + w)
    return 1 if bad_modules or wrong else 0


def load_patches(folder: Path) -> dict[str, tuple[float, float] | None]:
    """Patch file -> (source pixels per module, side in modules). With an
    index.tsv (columns file, ppm) only the files it lists are used."""
    idx = folder / "index.tsv"
    if idx.exists():
        lines = idx.read_text(encoding="utf-8").splitlines()
        head = lines[0].split("\t")
        f, m = head.index("file"), head.index("ppm")
        out = {}
        for ln in lines[1:]:
            if not ln.strip():
                continue
            name, ppm = ln.split("\t")[f], float(ln.split("\t")[m])
            img = cv2.imdecode(np.fromfile(str(folder / name), np.uint8), cv2.IMREAD_GRAYSCALE)
            out[name] = (ppm, min(img.shape) / ppm)
        return out
    return {f.name: None for f in folder.iterdir() if f.suffix.lower() in PATCH_EXT}


# --------------------------------------------------------------------------
# Review sheet
# --------------------------------------------------------------------------
def real_crops(run: Path, listing: Path, seed: int) -> list[str]:
    """Crop the real marks of `listing` (TSV: file, corners "x,y x,y x,y x,y"
    or empty for an image that is already a crop, pixels per module) with a
    quiet zone of 1-4 modules, scaled to the PPM range of the generator, as
    run/real/rNN.png. Names are anonymous; real/index.tsv maps them back."""
    rng = np.random.default_rng([seed, 999_999])
    names, index = [], []
    rows = [ln.split("\t") for ln in listing.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    for k, row in enumerate(rows):
        file, corners, ppm = row[0], row[1].strip(), float(row[2])
        img = cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_COLOR)
        if corners:
            pts = np.array([[float(v) for v in c.split(",")] for c in corners.split()])
            q = rng.uniform(1.0, 4.0) * ppm
            x0, y0 = np.maximum(pts.min(0) - q, 0).astype(int)
            x1, y1 = np.minimum(pts.max(0) + q, [img.shape[1], img.shape[0]]).astype(int)
            img = img[y0:y1, x0:x1]
        target = math.exp(rng.uniform(math.log(3.0), math.log(12.0)))
        if target < ppm:
            f = target / ppm
            img = cv2.resize(img, (max(1, round(img.shape[1] * f)),
                                   max(1, round(img.shape[0] * f))),
                             interpolation=cv2.INTER_AREA)
        name = f"real/r{k:02d}.png"
        write_img(run / name, img)
        names.append(name)
        index.append(f"{name}\t{file}\t{round(min(target, ppm), 2)}\n")
    (run / "real" / "index.tsv").write_text("".join(index), encoding="utf-8")
    return names


def write_review(run: Path, real: list[str], seed: int, blind_count: int = 80) -> None:
    """review.html: style review (realistic or not, per style) and a blind
    test (real or synthetic); answers download as review.csv."""
    synth = []
    for f in sorted(run.glob("[0-9]*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        p = r["params"]
        c = p["camera"]
        info = {"text": r["text"], "size": r["size"], "ppm": p["ppm"],
                "polarity": r["polarity"], "substrate": p["substrate"]["kind"]
                + (" painted" if p["substrate"].get("painted") else ""),
                "digits": bool(p["scene"]["digits"]), "edge": bool(p["scene"]["edge"]),
                "defocus": c["defocus"], "motion": c["motion"], "noise": c["noise"],
                "tilt": c["tilt"], "bend": c["bend"], "mono": c["mono"],
                "jpeg": c["jpeg"], **p["mark"]}
        synth.append({"file": r["file"], "style": p["style"], "info": info})
    rng = np.random.default_rng([seed, 888_888])
    pick = rng.permutation(len(synth))[:min(blind_count, len(synth))]
    blind = [{"file": synth[i]["file"], "t": 1} for i in pick] + \
        [{"file": f, "t": 0} for f in real]
    blind = [blind[i] for i in rng.permutation(len(blind))]
    data = json.dumps({"synth": synth, "real": real, "blind": blind,
                       "key": f"synth-review-{run.name}"}, separators=(",", ":"))
    (run / "review.html").write_text(REVIEW_HTML.replace("__DATA__", data), encoding="utf-8")


REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic marks review</title>
<style>
  :root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --line:#d8d8d2; --accent:#1f6feb;
          --ok:#2da44e; --bad:#e5534b; --tile:#ebebe6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161616; --fg:#e8e8e4; --muted:#9a9a94; --line:#333; --accent:#58a6ff;
            --ok:#3fb950; --bad:#ff7b72; --tile:#222; }
  }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line);
           padding:10px 16px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0 8px 0 0; }
  button { font:inherit; padding:4px 10px; border:1px solid var(--line); border-radius:6px;
           background:transparent; color:var(--fg); cursor:pointer; }
  button.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.yes.on { background:var(--ok); border-color:var(--ok); }
  button.no.on { background:var(--bad); border-color:var(--bad); }
  .help { color:var(--muted); max-width:900px; margin:10px 16px; }
  h2 { font-size:15px; margin:18px 16px 6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:12px; padding:0 16px; }
  .tile { border:1px solid var(--line); border-radius:8px; padding:6px; }
  .box { height:200px; background:var(--tile); display:flex; align-items:center; justify-content:center;
         border-radius:4px; overflow:hidden; cursor:zoom-in; }
  .box img { max-width:100%; max-height:100%; }
  .row { display:flex; gap:6px; margin-top:6px; align-items:center; flex-wrap:wrap; }
  .row input[type=text] { flex:1; min-width:0; font:inherit; padding:3px 6px; border:1px solid var(--line);
                          border-radius:6px; background:transparent; color:var(--fg); }
  .name { color:var(--muted); font-size:12px; }
  details { font-size:12px; color:var(--muted); margin-top:4px; word-break:break-word; }
  .refs { display:flex; gap:8px; overflow-x:auto; padding:0 16px 6px; }
  .refs img { height:110px; border-radius:4px; cursor:zoom-in; }
  .truth0 { outline:3px solid var(--ok); } .truth1 { outline:3px solid var(--bad); }
  #result { font-weight:600; }
  #zoom { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center;
          justify-content:center; z-index:5; cursor:zoom-out; }
  #zoom img { max-width:95vw; max-height:92vh; min-height:50vh; object-fit:contain; }
  .hidden { display:none; }
</style>
</head>
<body>
<header>
  <h1>Synthetic marks review</h1>
  <button id="tab-style" class="on">Style review</button>
  <button id="tab-blind">Blind test</button>
  <span id="progress" class="name"></span>
  <button id="save">Download review.csv</button>
</header>
<section id="style">
  <p class="help">For each synthetic image: <b>realistic</b> or <b>unrealistic</b> next to the real
  marks at the top of each group, and a short note on what looks wrong (e.g. "grain too regular",
  "dots too perfect"). Parameters stay hidden until you open them. Click an image to enlarge.
  Answers are kept in this browser until you download them.</p>
  <div id="groups"></div>
</section>
<section id="blind" class="hidden">
  <p class="help">Real and synthetic images, mixed. Guess <b>real</b> or <b>synthetic</b> for each one,
  then press <i>Show answers</i> (green outline = real, red = synthetic). Do this before the style review
  if you can, so the styles are not yet familiar.</p>
  <div class="row" style="padding:0 16px"><button id="reveal">Show answers</button><span id="result"></span></div>
  <div id="blind-grid" class="grid" style="margin-top:10px"></div>
</section>
<div id="zoom"><img alt=""></div>
<script>
const D = __DATA__;
let st = {};
try { st = JSON.parse(localStorage.getItem(D.key) || "{}"); } catch (e) {}
st.style = st.style || {}; st.blind = st.blind || {};
const save = () => { try { localStorage.setItem(D.key, JSON.stringify(st)); } catch (e) {} progress(); };
const zoom = document.getElementById("zoom");
zoom.onclick = () => zoom.style.display = "none";
const enlarge = src => { zoom.querySelector("img").src = src; zoom.style.display = "flex"; };
const img = src => { const i = document.createElement("img"); i.src = src; i.alt = src; i.loading = "lazy";
  i.onclick = () => enlarge(src); return i; };

function toggle(pair, val, cur) { pair.forEach(b => b.classList.toggle("on", b.dataset.v === cur)); }

function styleTile(s) {
  const a = st.style[s.file] = st.style[s.file] || {};
  const t = document.createElement("div"); t.className = "tile";
  const box = document.createElement("div"); box.className = "box"; box.appendChild(img(s.file));
  const row = document.createElement("div"); row.className = "row";
  const yes = Object.assign(document.createElement("button"), {className:"yes", textContent:"realistic"});
  const no = Object.assign(document.createElement("button"), {className:"no", textContent:"unrealistic"});
  yes.dataset.v = "yes"; no.dataset.v = "no";
  [yes, no].forEach(b => b.onclick = () => { a.ok = a.ok === b.dataset.v ? "" : b.dataset.v;
    toggle([yes, no], 0, a.ok); save(); groupStats(); });
  toggle([yes, no], 0, a.ok);
  const note = Object.assign(document.createElement("input"), {type:"text", placeholder:"note", value:a.note || ""});
  note.oninput = () => { a.note = note.value; save(); };
  row.append(yes, no);
  const row2 = document.createElement("div"); row2.className = "row"; row2.append(note);
  const det = document.createElement("details");
  det.innerHTML = `<summary>${s.file}</summary>` + Object.entries(s.info)
    .map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(" · ");
  t.append(box, row, row2, det);
  return t;
}

function groupStats() {
  document.querySelectorAll("[data-group]").forEach(h => {
    const items = D.synth.filter(s => s.style === h.dataset.group);
    const done = items.filter(s => (st.style[s.file] || {}).ok);
    const ok = done.filter(s => st.style[s.file].ok === "yes").length;
    h.textContent = `${h.dataset.group}: ${items.length} images, ${done.length} rated, ` +
      (done.length ? `${Math.round(100 * ok / done.length)} % realistic` : "");
  });
}

const groups = document.getElementById("groups");
[...new Set(D.synth.map(s => s.style))].forEach(g => {
  const h = document.createElement("h2"); h.dataset.group = g; groups.appendChild(h);
  const refs = document.createElement("div"); refs.className = "refs";
  D.real.forEach(r => refs.appendChild(img(r))); groups.appendChild(refs);
  const grid = document.createElement("div"); grid.className = "grid";
  D.synth.filter(s => s.style === g).forEach(s => grid.appendChild(styleTile(s)));
  groups.appendChild(grid);
});
groupStats();

const bgrid = document.getElementById("blind-grid");
D.blind.forEach(b => {
  const a = st.blind[b.file] = st.blind[b.file] || {};
  const t = document.createElement("div"); t.className = "tile";
  const box = document.createElement("div"); box.className = "box"; box.appendChild(img(b.file));
  box.dataset.truth = b.t;
  const row = document.createElement("div"); row.className = "row";
  const r = Object.assign(document.createElement("button"), {className:"yes", textContent:"real"});
  const s = Object.assign(document.createElement("button"), {className:"no", textContent:"synthetic"});
  r.dataset.v = "real"; s.dataset.v = "synthetic";
  [r, s].forEach(x => x.onclick = () => { a.guess = a.guess === x.dataset.v ? "" : x.dataset.v;
    toggle([r, s], 0, a.guess); save(); });
  toggle([r, s], 0, a.guess);
  row.append(r, s); t.append(box, row); bgrid.appendChild(t);
});

function result() {
  const g = D.blind.filter(b => (st.blind[b.file] || {}).guess);
  const hit = (b) => (st.blind[b.file].guess === "synthetic") === (b.t === 1);
  const sy = g.filter(b => b.t === 1), re = g.filter(b => b.t === 0);
  const pc = (a, n) => n ? Math.round(100 * a / n) + " %" : "–";
  return `${g.length} / ${D.blind.length} guessed · correct ${pc(g.filter(hit).length, g.length)} · ` +
    `synthetic found ${pc(sy.filter(hit).length, sy.length)} · real found ${pc(re.filter(hit).length, re.length)}`;
}
function reveal() {
  st.revealed = true; save();
  document.querySelectorAll("#blind-grid .box").forEach(b => b.classList.add("truth" + b.dataset.truth));
  document.getElementById("result").textContent = result();
}
document.getElementById("reveal").onclick = reveal;
if (st.revealed) reveal();

function progress() {
  const sd = D.synth.filter(s => (st.style[s.file] || {}).ok).length;
  const bd = D.blind.filter(b => (st.blind[b.file] || {}).guess).length;
  document.getElementById("progress").textContent =
    `style ${sd} / ${D.synth.length} · blind ${bd} / ${D.blind.length}`;
}
progress();

const tabs = { style: document.getElementById("tab-style"), blind: document.getElementById("tab-blind") };
Object.entries(tabs).forEach(([k, b]) => b.onclick = () => {
  Object.entries(tabs).forEach(([k2, b2]) => { b2.classList.toggle("on", k2 === k);
    document.getElementById(k2).classList.toggle("hidden", k2 !== k); });
});

document.getElementById("save").onclick = () => {
  const rows = [["mode", "file", "style", "answer", "truth", "note"]];
  D.synth.forEach(s => { const a = st.style[s.file] || {};
    rows.push(["style", s.file, s.style, a.ok || "", "synthetic", a.note || ""]); });
  D.blind.forEach(b => { const a = st.blind[b.file] || {};
    rows.push(["blind", b.file, "", a.guess || "", b.t ? "synthetic" : "real",
               st.revealed ? "revealed" : ""]); });
  const csv = rows.map(r => r.map(v => /[",\\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)
    .join(",")).join("\\n") + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "review.csv"; a.click();
};
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=0,
                    help="first sample index (split one seed over processes; "
                    "the TSV files then list only this part)")
    ap.add_argument("--style", default=",".join(DEFAULT_STYLES),
                    help=f"comma list of {', '.join(STYLES)}")
    ap.add_argument("--sizes", help="comma list of symbol sizes (e.g. 20x20,8x32); "
                    "each mark takes one of them that its text fits, uniformly")
    ap.add_argument("--patches", type=Path,
                    help="folder of wood photo patches (optional index.tsv: file, ppm); "
                    "half the samples use one")
    ap.add_argument("--params", type=Path, help="render this JSON record again")
    ap.add_argument("--scene", metavar="WxH",
                    help="frames of W x H pixels with 0-3 marks and distractors")
    ap.add_argument("--preview", action="store_true",
                    help="write review.html (style review and blind test)")
    ap.add_argument("--real", type=Path, metavar="TSV",
                    help="real marks for the review: file, corners, pixels per module")
    ap.add_argument("--check", type=Path, metavar="RUN", help="run the C4 / C5 checks")
    ap.add_argument("--reader", type=Path, default=DEFAULT_READER)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="reader processes for --check")
    args = ap.parse_args()

    if args.check:
        return check(args.check, args.reader, args.jobs)
    if not args.out:
        ap.error("--out is needed")
    if args.params:
        p = json.loads(args.params.read_text(encoding="utf-8"))["params"]
        if "frame" in p:
            rec = write_scene(args.out, args.params.stem, p, args.patches)
            write_scene_tsv(args.out, [(args.params.stem, rec)])
            return 0
        rec = write_sample(args.out, args.params.stem, p, args.patches)
        write_tsv(args.out, [(args.params.stem, rec)])
        return 0
    styles = args.style.split(",")
    for s in styles:
        if s not in STYLES:
            ap.error(f"unknown style {s}")
    if args.style != ",".join(DEFAULT_STYLES):     # named styles: equal shares
        STYLE_WEIGHTS.update({s: 1.0 for s in styles})
    if args.sizes:
        known = {f"{r[0]}x{r[1]}" for r in gen.SQUARE_SYMBOLS + gen.RECT_SYMBOLS}
        SIZE_CHOICE[:] = args.sizes.split(",")
        for s in SIZE_CHOICE:
            if s not in known:
                ap.error(f"unknown size {s}")
    patches = load_patches(args.patches) if args.patches else {}
    recs = []
    if args.scene:
        W, H = (int(v) for v in args.scene.lower().split("x"))
        for i in range(args.start, args.start + args.count):
            rng = np.random.default_rng([args.seed, i])
            sp = sample_scene_frame(rng, [args.seed, i], W, H, styles, patches)
            name = f"{i:06d}"
            recs.append((name, write_scene(args.out, name, sp, args.patches)))
        write_scene_tsv(args.out, recs)
        print(f"{len(recs)} frames, {sum(len(r['marks']) for _, r in recs)} marks "
              f"in {args.out}")
        return 0
    for i in range(args.start, args.start + args.count):
        rng = np.random.default_rng([args.seed, i])
        p = sample_params(rng, [args.seed, i], styles, patches)
        name = f"{i:06d}"
        recs.append((name, write_sample(args.out, name, p, args.patches)))
    write_tsv(args.out, recs)
    print(f"{len(recs)} samples in {args.out}")
    if args.preview:
        real = real_crops(args.out, args.real, args.seed) if args.real else []
        write_review(args.out, real, args.seed)
        print(f"review sheet: {args.out / 'review.html'} ({len(real)} real)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
