"""Write Python/OpenCV reference results for the Rust image-processing tests.

Part 2 (phase 2): load_image, colour conversions, filters, morphology,
thresholds, resize, CLAHE, contours, homographies / remap and the variant
ladders. u8 images are written as PNG, float images as raw little-endian
f32 (`.f32`, shape in the manifest). The manifest `img/manifest.json` lists
every case as {"op", "params", "in", "out"}; the Rust tests dispatch on "op".

python rust/tools/dump_fixtures_img.py [OUTDIR]     (default: fixtures/ in the private data)
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def private_dir() -> Path:
    """Private test data (fixtures, test images, baselines), not part of
    this repo: `DM_PRIVATE_DIR`."""
    d = os.environ.get("DM_PRIVATE_DIR")
    if not d:
        sys.exit("DM_PRIVATE_DIR is not set: point it at the private test data")
    return Path(d)


PRIVATE = private_dir()
sys.path.insert(0, str(ROOT / "python"))

import datamatrix_reader as rd  # noqa: E402

SEED = 20261010
TI = PRIVATE / "testimages"
# the three op inputs of the plan: an easy crop, a large photo, a tile
# images by role; the names stay in the private data
PICKS = json.loads((TI / "picks.json").read_text(encoding="utf-8"))
EASY = TI / PICKS["fixtures_img_easy"]
LARGE = TI / PICKS["fixtures_img_large"]


class Writer:
    def __init__(self, out: Path, sub: str = "img"):
        self.dir = out / sub
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.iterdir():
            f.unlink()
        self.cases: list[dict] = []
        self.n = 0

    def arr(self, a: np.ndarray, stem: str | None = None) -> dict:
        """Save an array, return its reference {"file", "shape", "dtype"}."""
        self.n += 1
        stem = stem or f"a{self.n:04d}"
        a = np.ascontiguousarray(a)
        if a.dtype == np.uint8:
            name = f"{stem}.png"
            ok, buf = cv2.imencode(".png", a, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            assert ok
            buf.tofile(str(self.dir / name))
        else:
            name = f"{stem}.f32"
            a.astype("<f4").tofile(str(self.dir / name))
        return {"file": name, "shape": list(a.shape), "dtype": str(a.dtype)}

    def case(self, op: str, params: dict, inputs, outputs) -> None:
        self.cases.append({"op": op, "params": params,
                           "in": inputs, "out": outputs})

    def finish(self) -> None:
        path = self.dir / "manifest.json"
        path.write_text(json.dumps({"cases": self.cases}, ensure_ascii=False, indent=0),
                        encoding="utf-8")
        size = sum(f.stat().st_size for f in self.dir.iterdir()) // 2**20
        print(f"{path}: {len(self.cases)} cases, {size} MiB")


# --------------------------------------------------------------------------
# P2.1 load_image
# --------------------------------------------------------------------------
def exif_jpeg(bgr: np.ndarray, orientation: int) -> bytes:
    from PIL import Image
    im = Image.fromarray(bgr[:, :, ::-1])
    exif = Image.Exif()
    exif[274] = orientation
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90, exif=exif.tobytes())
    return buf.getvalue()


def dump_load(w: Writer, rng: np.random.Generator) -> None:
    for p in sorted(TI.glob("*/*")):
        if p.suffix.lower() in rd.IMAGE_EXT:
            w.case("load", {}, {"path": str(p.relative_to(ROOT))}, w.arr(rd.load_image(str(p))))
    # synthetic files: written next to the manifest, decoded by OpenCV
    base = cv2.resize(rd.load_image(str(EASY)), (61, 47), interpolation=cv2.INTER_AREA)
    files: dict[str, bytes] = {}
    for o in range(1, 9):
        files[f"exif{o}.jpg"] = exif_jpeg(base, o)
    alpha = rng.integers(0, 256, base.shape[:2], dtype=np.uint8)
    files["rgba.png"] = cv2.imencode(".png", np.dstack([base, alpha]))[1].tobytes()
    files["rgb16.png"] = cv2.imencode(".png", base.astype(np.uint16) * 257 + rng.integers(0, 257, base.shape).astype(np.uint16))[1].tobytes()
    files["gray.png"] = cv2.imencode(".png", base[:, :, 1])[1].tobytes()
    files["gray16.png"] = cv2.imencode(".png", base[:, :, 1].astype(np.uint16) * 255)[1].tobytes()
    files["plain.bmp"] = cv2.imencode(".bmp", base)[1].tobytes()
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(base[:, :, ::-1]).convert("P", palette=Image.ADAPTIVE, colors=32).save(buf, "PNG")
    files["palette.png"] = buf.getvalue()
    for name, data in files.items():
        (w.dir / name).write_bytes(data)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        w.case("load", {}, {"path": f"fixtures/img/{name}"}, w.arr(img))


# --------------------------------------------------------------------------
# P2.2 colour conversions
# --------------------------------------------------------------------------
def all_colours() -> np.ndarray:
    """4096x4096 BGR image holding every colour once (b = i & 255 ...)."""
    i = np.arange(1 << 24, dtype=np.uint32)
    bgr = np.stack([i & 255, (i >> 8) & 255, i >> 16], -1).astype(np.uint8)
    return bgr.reshape(4096, 4096, 3)


def dump_color(w: Writer, bases: dict[str, np.ndarray]) -> None:
    allc = all_colours()
    for name, bgr in [("all_colours", allc)] + list(bases.items()):
        inp = {"synthetic": "all_colours"} if name == "all_colours" else w.arr(bgr, f"bgr_{name}")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        w.case("cvt", {"code": "gray"}, inp, w.arr(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        w.case("cvt", {"code": "hsv_s"}, inp, w.arr(hsv[:, :, 1]))
        w.case("cvt", {"code": "lab_l"}, inp, w.arr(lab[:, :, 0]))
        if name != "all_colours":
            for inv in (False, True):
                ch = rd.channels(bgr, inv)
                w.case("channels", {"invert": inv, "names": list(ch)}, inp,
                       [w.arr(v) for v in ch.values()])


def gray_inputs(bases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The three gray op inputs: easy crop, large photo at 700 px, a tile."""
    tiles = rd.tile_variants(bases["large"])
    return {"easy": cv2.cvtColor(bases["easy"], cv2.COLOR_BGR2GRAY),
            "large": rd.normalise_size(cv2.cvtColor(bases["large"], cv2.COLOR_BGR2GRAY))[0],
            "tile": tiles[4][0].img}


# --------------------------------------------------------------------------
# P2.3 filters and point ops
# --------------------------------------------------------------------------
def dump_filters(w: Writer, grays: dict[str, np.ndarray]) -> None:
    for name, g in grays.items():
        gi = w.arr(g, f"gray_{name}")
        m = min(g.shape)
        f = g.astype(np.float32)
        fi = w.arr(f, f"grayf_{name}")
        for sig in (1.0, 1.5, 2.0, 3.0, m / 200, 4.37):
            w.case("gauss_u8", {"sigma": sig}, gi, w.arr(cv2.GaussianBlur(g, (0, 0), sig)))
        for sig in (1.0, 2.3, 5.0, m / 8, m / 6):
            w.case("gauss_f32", {"sigma": sig}, fi, w.arr(cv2.GaussianBlur(f, (0, 0), sig)))
        fr = (f * 0.37 + 3.1).astype(np.float32)
        w.case("box_f32", {"k": 5, "border": "reflect"}, w.arr(fr),
               w.arr(cv2.blur(fr, (5, 5), borderType=cv2.BORDER_REFLECT)))
        w.case("median5", {}, gi, w.arr(cv2.medianBlur(g, 5)))
        b = cv2.GaussianBlur(g, (0, 0), 2.0)
        w.case("add_weighted", {"alpha": 2.5, "beta": -1.5, "gamma": 0.0}, [gi, w.arr(b)],
               w.arr(cv2.addWeighted(g, 2.5, b, -1.5, 0)))
        w.case("normalize_f32", {"a": 0, "b": 255}, w.arr(fr * fr),
               w.arr(cv2.normalize(fr * fr, None, 0, 255, cv2.NORM_MINMAX)))
        for sig in (m / 8, m / 6):
            w.case("flatten_background", {"sigma": sig}, gi, w.arr(rd.flatten_background(g, sig)))
        w.case("sharpen_median5", {}, gi, w.arr(rd.sharpen(cv2.medianBlur(g, 5))))
        w.case("gamma", {"gm": 2.0}, gi, w.arr(rd.gamma(g, 2.0)))


# --------------------------------------------------------------------------
# P2.4 morphology
# --------------------------------------------------------------------------
def dump_morph(w: Writer, grays: dict[str, np.ndarray]) -> None:
    w.case("ellipse", {}, {}, {str(k): cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).ravel().tolist()
                              for k in range(1, 100, 2)})
    for name, g in grays.items():
        gi = {"dtype": "uint8", "file": f"gray_{name}.png", "shape": list(g.shape)}
        dark = 255 - cv2.GaussianBlur(rd.flatten_background(g, min(g.shape) / 8), (0, 0), 3)
        bw = ((dark > np.percentile(dark, 75)) * 255).astype(np.uint8)
        bi = w.arr(bw)
        for k in (3, 5, 7, 9, 21, 37, 57):
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            for src, si in ((g, gi), (bw, bi)):
                w.case("morph", {"op": "erode", "k": k}, si, w.arr(cv2.erode(src, se)))
                w.case("morph", {"op": "dilate", "k": k}, si, w.arr(cv2.dilate(src, se)))
                w.case("morph", {"op": "close", "k": k}, si, w.arr(cv2.morphologyEx(src, cv2.MORPH_CLOSE, se)))
                w.case("morph", {"op": "open", "k": k}, si, w.arr(cv2.morphologyEx(src, cv2.MORPH_OPEN, se)))
        for k in (3, 5, 9):
            w.case("merge_dots", {"k": k}, gi, w.arr(rd.merge_dots(g, k)))
        w.case("remove_texture", {"k": 3}, gi, w.arr(rd.remove_texture(g, 3)))


# --------------------------------------------------------------------------
# P2.5 thresholds
# --------------------------------------------------------------------------
def dump_thresh(w: Writer, grays: dict[str, np.ndarray]) -> None:
    for name, g in grays.items():
        m = min(g.shape)
        flat = rd.clahe(rd.flatten_background(g, m / 8))
        srcs = [g, flat, cv2.GaussianBlur(flat, (0, 0), 1.0), rd.merge_dots(cv2.GaussianBlur(flat, (0, 0), 1), 5)]
        for src in srcs:
            si = w.arr(src)
            lvl, out = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            w.case("otsu", {"level": lvl}, si, w.arr(out))
        b = cv2.GaussianBlur(rd.clahe(g), (0, 0), 1.5)
        bi = w.arr(b)
        for blk in (m / 6, 31, 7):
            w.case("adaptive", {"block": blk, "k": rd._odd(blk)}, bi, w.arr(rd.adaptive(b, blk)))
        w.case("box_mean", {"k": rd._odd(m / 6)}, bi,
               w.arr(cv2.boxFilter(b, -1, (rd._odd(m / 6),) * 2, normalize=True,
                                   borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED)))
    # synthetic Otsu ties / degenerate histograms
    rng = np.random.default_rng(SEED + 5)
    for vals in ([0, 255], [10], [3, 4], [0, 100, 200], list(range(256))):
        img = rng.choice(np.array(vals, np.uint8), (37, 53))
        lvl, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        w.case("otsu", {"level": lvl}, w.arr(img), w.arr(out))
    for _ in range(40):
        img = (rng.normal(rng.uniform(40, 200), rng.uniform(1, 60), (61, 47))
               + rng.choice([0, 80], (61, 47))).clip(0, 255).astype(np.uint8)
        lvl, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        w.case("otsu", {"level": lvl}, w.arr(img), w.arr(out))


# --------------------------------------------------------------------------
# P2.6 resize
# --------------------------------------------------------------------------
def dump_resize(w: Writer, bases: dict[str, np.ndarray]) -> None:
    srcs = {"large": cv2.cvtColor(bases["large"], cv2.COLOR_BGR2GRAY),
            "easy": cv2.cvtColor(bases["easy"], cv2.COLOR_BGR2GRAY)}
    for p in sorted(TI.glob("*/*.png"))[::3]:
        srcs[p.stem[-6:]] = cv2.cvtColor(rd.load_image(str(p)), cv2.COLOR_BGR2GRAY)
    srcs["square200"] = cv2.resize(srcs["easy"][100:400, 100:400], (200, 200), interpolation=cv2.INTER_AREA)
    srcs["square256"] = np.ascontiguousarray(srcs["easy"][50:306, 200:456])
    for name, g in srcs.items():
        gi = w.arr(g, f"rs_{name}")
        for tgt in (350, 600, 700, 1100):
            out, s = rd.normalise_size(g, tgt)
            w.case("normalise_size", {"target": tgt, "scale": s}, gi, w.arr(out))
    # tiles of the large photo (crop, INTER_AREA to TILE_SIZE) and integer factors
    big = srcs["large"]
    h, wd = big.shape
    (ny, th), (nx, tw) = rd._tile_axis(h), rd._tile_axis(wd)
    s = min(1.0, rd.TILE_SIZE / max(th, tw))
    crop = np.ascontiguousarray(big[:th, :tw])
    w.case("resize", {"s": s, "interp": "area"}, w.arr(crop),
           w.arr(cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)))
    for s in (0.5, 1 / 3, 0.25, 0.3, 0.123):
        for g in (crop[:901, :777], crop[:900, :600]):
            g = np.ascontiguousarray(g)
            w.case("resize", {"s": s, "interp": "area"}, w.arr(g),
                   w.arr(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)))


# --------------------------------------------------------------------------
# P2.7 CLAHE
# --------------------------------------------------------------------------
def dump_clahe(w: Writer, grays: dict[str, np.ndarray]) -> None:
    for name, g in grays.items():
        srcs = [g, rd.flatten_background(g, min(g.shape) / 8), rd.gamma(g, 2.0),
                np.ascontiguousarray(g[:, :400]), np.ascontiguousarray(g[:203, :197])]
        for src in srcs:
            si = w.arr(src)
            for clip, tiles in ((3.0, 8), (2.0, 4)):
                w.case("clahe", {"clip": clip, "tiles": tiles}, si, w.arr(rd.clahe(src, clip, tiles)))


# --------------------------------------------------------------------------
# P2.8 contours, contourArea, minAreaRect, boxPoints
# --------------------------------------------------------------------------
def blobs(g: np.ndarray) -> list[np.ndarray]:
    """The binary images candidate_quads() runs findContours on."""
    m = min(g.shape)
    flat = rd.flatten_background(g, m / 8)
    dark = 255 - cv2.GaussianBlur(flat, (0, 0), max(1.0, m / 200))
    out = []
    for pct in (65, 75, 85):
        bw = ((dark > np.percentile(dark, pct)) * 255).astype(np.uint8)
        for frac in (0.03, 0.05, 0.08):
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rd._odd(m * frac),) * 2)
            blob = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se)
            out.append(cv2.morphologyEx(blob, cv2.MORPH_OPEN, se))
    return out


def dump_contours(w: Writer, bases: dict[str, np.ndarray], grays: dict[str, np.ndarray]) -> None:
    imgs = [rd.normalise_size(g, 600)[0] for g in grays.values()]
    for p in sorted(TI.glob("*/*"))[::4]:
        imgs.append(rd.grid_variants(rd.load_image(str(p)))[0].img)
    rng = np.random.default_rng(SEED + 8)
    bws = [b for g in imgs for b in blobs(g)]
    bws += [((rng.random((53, 71)) < 0.45) * 255).astype(np.uint8) for _ in range(6)]
    for bw in bws:
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = []
        for c in cnts:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            res.append({"pts": c.reshape(-1, 2).tolist(), "area": cv2.contourArea(c),
                        "hull": cv2.convexHull(c, clockwise=False).reshape(-1, 2).tolist(),
                        "rect": [list(rect[0]), list(rect[1]), rect[2]],
                        "box": box.tolist(), "box_area": cv2.contourArea(box)})
        w.case("contours", {}, w.arr(bw), res)


# --------------------------------------------------------------------------
# P2.9 homographies, perspectiveTransform, remap, warpPerspective
# --------------------------------------------------------------------------
def random_quads(rng: np.random.Generator, n: int, size: float) -> np.ndarray:
    c = rng.uniform(0.3, 0.7, (n, 1, 2)) * size
    r = rng.uniform(0.05, 0.4, (n, 1, 1)) * size
    ang = rng.uniform(0, 2 * np.pi, (n, 1, 1)) + np.arange(4)[None, :, None] * np.pi / 2
    jit = rng.normal(0, 0.06, (n, 4, 2)) * r
    q = c + r * np.concatenate([np.cos(ang), np.sin(ang)], 2) + jit
    return q.astype(np.float32)


def dump_geom(w: Writer, grays: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(SEED + 9)
    quads = random_quads(rng, 300, 600)
    cases = []
    for k, q in enumerate(quads):
        rows, cols = (10, 10) if k % 3 else (int(rng.integers(8, 145)), int(rng.integers(8, 145)))
        src = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        h = cv2.getPerspectiveTransform(src, q)
        h64 = rd.homographies(src.astype(np.float64), q[None].astype(np.float64))[0]
        pts = rng.uniform(-2, max(rows, cols) + 2, (20, 2)).astype(np.float32)
        pt = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)
        inv = cv2.invert(h)[1]
        cases.append({"src": src.tolist(), "quad": q.tolist(), "h": h.ravel().tolist(),
                      "h64": h64.ravel().tolist(), "pts": pts.tolist(), "pt": pt.tolist(),
                      "inv": inv.ravel().tolist()})
    w.case("homography", {}, {}, cases)
    for name, g in grays.items():
        g6 = rd.normalise_size(g, 600)[0]
        dark = (255 - g6).astype(np.float32)
        di = w.arr(dark)
        hh, ww = dark.shape
        n = 5000
        xs = rng.uniform(-30, ww + 30, n).astype(np.float32)
        ys = rng.uniform(-30, hh + 30, n).astype(np.float32)
        xs[:50] = np.round(xs[:50])                  # integer and half coordinates
        ys[50:100] = np.round(ys[50:100]) + 0.5
        xs[100:110] = [0, ww - 1, ww - 1.5, -0.5, ww, 0, ww - 1, 3.25, 7, 8]
        ys[100:110] = [0, hh - 1, 2, hh - 1, 5, hh - 0.5, hh, -1e-3, 1e30, -1e30]
        for m in (n, 4096 * 2 + 17):                 # 1xN map and the 4096-wide layout
            x, y = np.resize(xs, m), np.resize(ys, m)
            w.case("remap", {}, [di, w.arr(x), w.arr(y)], w.arr(rd.remap_points(dark, x, y)))
        for q in random_quads(rng, 4, min(hh, ww)):
            dst = np.float32([[0, 0], [512, 0], [512, 512], [0, 512]])
            m3 = cv2.getPerspectiveTransform(q, dst)
            out = cv2.warpPerspective(dark, m3, (512, 512), borderMode=cv2.BORDER_REPLICATE)
            w.case("warp", {"m": m3.ravel().tolist(), "size": [512, 512]}, di, w.arr(out))
            out = cv2.warpPerspective(dark, m3, (203, 101), borderMode=cv2.BORDER_REPLICATE)
            w.case("warp", {"m": m3.ravel().tolist(), "size": [203, 101]}, di, w.arr(out))


# --------------------------------------------------------------------------
# P2.10 variant ladders
# --------------------------------------------------------------------------
VARIANT_IMAGES = PICKS["fixtures_variant_images"]
POLARITY = {"any": (False, True), "dark": (False, False), "light": (True, False)}


def dump_variants(w: Writer) -> None:
    for k, rel in enumerate(VARIANT_IMAGES):
        bgr = rd.load_image(str(TI / rel))
        for pol, (invert, extra) in POLARITY.items():
            pixels = pol == "any" or k == 0
            sets = {"enhancement": rd.enhancement_variants(bgr, invert, extra),
                    "grid": rd.grid_variants(bgr, invert, extra),
                    "tile": [v for tv in rd.tile_variants(bgr, invert) for v in tv]}
            for kind, vs in sets.items():
                out = [{"name": v.name, "scale": v.scale, "ox": v.ox, "oy": v.oy,
                        "img": w.arr(v.img) if pixels else {"shape": list(v.img.shape)}} for v in vs]
                w.case("variants", {"kind": kind, "polarity": pol, "pixels": pixels},
                       {"path": f"testimages/{rel}"}, out)
    rng = np.random.default_rng(SEED + 10)
    mq = []
    for _ in range(200):
        q = random_quads(rng, 1, 900)[0]
        a = rd.Variant("a", np.zeros((1, 1), np.uint8), float(rng.uniform(0.1, 2)), *([0.0, 0.0] if rng.random() < 0.4 else rng.uniform(0, 3000, 2).round().tolist()))
        b = rd.Variant("b", np.zeros((1, 1), np.uint8), float(rng.uniform(0.1, 2)), *([0.0, 0.0] if rng.random() < 0.4 else rng.uniform(0, 3000, 2).round().tolist()))
        mq.append({"q": q.tolist(), "src": [a.scale, a.ox, a.oy], "dst": [b.scale, b.ox, b.oy],
                   "out": rd.map_quad(q, a, b).tolist()})
    w.case("map_quad", {}, {}, mq)
    w.case("tile_axis", {}, {}, {str(n): list(rd._tile_axis(n)) for n in range(1, 9000, 7)})


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PRIVATE / "fixtures"
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    w = Writer(out)
    dump_load(w, rng)
    bases = {"easy": rd.load_image(str(EASY)), "large": rd.load_image(str(LARGE))}
    dump_color(w, bases)
    grays = gray_inputs(bases)
    dump_filters(w, grays)
    dump_morph(w, grays)
    dump_thresh(w, grays)
    dump_resize(w, bases)
    dump_clahe(w, grays)
    dump_contours(w, bases, grays)
    dump_geom(w, grays)
    dump_variants(w)
    w.finish()


if __name__ == "__main__":
    main()
