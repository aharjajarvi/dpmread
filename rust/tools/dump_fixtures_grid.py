"""Write Python reference results for the Rust grid engine tests (phase 3).

Inputs are Python's own grid variants (`grid_variants`), saved as PNG, so
image-op differences are excluded. Output: `fixtures/grid/` (private data) with a
`manifest.json` of {"op", "params", "in", "out"} cases (same layout as
`dump_fixtures_img.py`); float arrays are raw little-endian f32/f64.

python rust/tools/dump_fixtures_grid.py [OUTDIR]     (default: fixtures/ in the private data)
python rust/tools/dump_fixtures_grid.py --v2 [OUTDIR]
    phase 4b: `datamatrix_reader_v2.py` as the reference, into `grid_v2/`:
    every `warp_fit` call of the qz grid job on the three `> flatten`
    variants of the `not_recognized` images (with `global_track` of both
    axes on the input quad), and the `grid_decode` replay of all images
    against the v2 baseline.
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib
import json
import os
import multiprocessing as mp
import sys
import time
from pathlib import Path

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
sys.path.insert(0, str(ROOT / "rust" / "tools"))

# --v2 (inherited by the spawned replay workers through the environment)
V2 = "--v2" in sys.argv or os.environ.get("DUMP_FIXTURES_V2") == "1"
if V2:
    os.environ["DUMP_FIXTURES_V2"] = "1"
rd = importlib.import_module("datamatrix_reader_v2" if V2 else "datamatrix_reader")
BASELINE = PRIVATE / "baseline" / ("python_v2_baseline.json" if V2 else "python_baseline.json")
from dump_fixtures_img import Writer, random_quads  # noqa: E402

SEED = 20261011
TI = PRIVATE / "testimages"
# grid reads of the baseline plus failures (near misses)
# (names in testimages/picks.json of the private data)
PICKS = json.loads((TI / "picks.json").read_text(encoding="utf-8"))
GRID_IMAGES = PICKS["fixtures_grid_images"]


def f32list(a) -> list:
    """f32 values as JSON numbers (exact), -inf / nan as None."""
    return [float(x) if np.isfinite(x) else None for x in np.asarray(a, np.float32).ravel()]


def qlist(q) -> list:
    return np.asarray(q, np.float32).tolist()


def grid_inputs(w: Writer) -> list[tuple[str, np.ndarray, dict]]:
    """(name, gray, arr ref) of the `gray > flatten` grid variant per image."""
    out = []
    for rel in GRID_IMAGES:
        bgr = rd.load_image(str(TI / rel))
        v = rd.grid_variants(bgr)[0]
        assert v.name == "gray > flatten", v.name
        out.append((rel, v.img, w.arr(v.img)))
    return out


def jitter(rng: np.random.Generator, q: np.ndarray, amount: float) -> np.ndarray:
    return (q + rng.normal(0, amount, q.shape)).astype(np.float32)


# --------------------------------------------------------------------------
# P3.1 Sampler, score_quads, orientations
# --------------------------------------------------------------------------
def dump_sampler(w: Writer, inputs, rng: np.random.Generator) -> None:
    for rel, g, gi in inputs:
        h, wd = g.shape
        cq = rd.candidate_quads(g)
        quads = cq[:2] + [cq[-1]] + [jitter(rng, cq[0], 4.0)] + list(random_quads(rng, 2, min(h, wd)))
        smp = rd.Sampler(g)
        for qi, q in enumerate(quads):
            ors = list(rd.orientations(q))
            w.case("orientations", {}, {"quad": qlist(q)},
                   [{"quad": qlist(o), "rot": r, "mirror": m} for o, r, m in ors])
            oq = np.array([o for o, _, _ in ors], np.float32)
            # batch with near-copies, as refine_quad scores its moves
            batch = np.concatenate([oq, jitter(rng, np.repeat(oq[:1], 8, 0), 0.7)])
            scores = []
            for sym in rd.ALL_SYMBOLS:
                gm = rd.geometry(sym)
                for qz in (True, False):
                    s = rd.score_quads(smp, batch, gm, qz)
                    scores.append({"rows": gm.rows, "cols": gm.cols, "qz": qz, "s": f32list(s)})
            w.case("score_quads", {}, {"img": gi, "quads": batch.tolist()}, scores)
            # single quads reduce in another order (refine_quad's best score)
            single = []
            for sym in rd.ALL_SYMBOLS[::3]:
                gm = rd.geometry(sym)
                for qz in (True, False):
                    s = rd.score_quads(smp, batch[:1], gm, qz)
                    single.append({"rows": gm.rows, "cols": gm.cols, "qz": qz, "s": f32list(s)})
            w.case("score_quads", {}, {"img": gi, "quads": batch[:1].tolist()}, single)
            # the per-quad samplers, a few sizes and footprints
            if qi >= 4:
                continue
            for sym in (rd.ALL_SYMBOLS[0], rd.ALL_SYMBOLS[5], rd.ALL_SYMBOLS[23], rd.ALL_SYMBOLS[30]):
                gm = rd.geometry(sym)
                o = oq[qi % 8]
                for rad in (0.22, 0.0, 0.1, 0.32):
                    sr = rd.Sampler(g, rad)
                    vals = sr.sample(o, gm.rows, gm.cols)
                    uv = np.concatenate([gm.fp_uv_qz, rng.uniform(-1, max(gm.rows, gm.cols) + 1, (50, 2))])
                    su = sr.sample_uv(o, gm.rows, gm.cols, uv)
                    ucol = np.arange(gm.cols) + 0.5 + rng.uniform(-0.4, 0.4, gm.cols)
                    vrow = np.arange(gm.rows) + 0.5 + rng.uniform(-0.4, 0.4, gm.rows)
                    sg = sr.sample_grid(o, gm.rows, gm.cols, ucol, vrow)
                    w.case("sample", {"rows": gm.rows, "cols": gm.cols, "radius": rad},
                           {"img": gi, "quad": qlist(o), "uv": uv.tolist(),
                            "ucol": ucol.tolist(), "vrow": vrow.tolist()},
                           {"sample": w.arr(vals.astype(np.float32)), "sample_uv": w.arr(su.astype(np.float32)),
                            "sample_grid": w.arr(sg.astype(np.float32))})


# --------------------------------------------------------------------------
# P3.2 candidate_quads, snap_quad, side_combos, expand_quad
# --------------------------------------------------------------------------
def dump_quads(w: Writer, rng: np.random.Generator) -> None:
    for rel in GRID_IMAGES:
        bgr = rd.load_image(str(TI / rel))
        vs = [(v, 0.02) for v in rd.grid_variants(bgr)]
        tiles = rd.tile_variants(bgr)
        vs += [(tv[0], 0.004) for tv in tiles[:2]]
        for v, min_area in vs:
            g = v.img
            cq = rd.candidate_quads(g, min_area=min_area)
            snaps = []
            for q in cq:
                sq = rd.snap_quad(g, q)
                si = rd.snap_quad(g, sq, inward=True)
                qi = rd.snap_quad(g, q, inward=True)
                combos = rd.side_combos(sq, si)
                oq = next(rd.orientations(sq))[0]
                exp = {f"{s[0]}x{s[1]}": qlist(rd.expand_quad(oq, rd.geometry(s)))
                       for s in (rd.ALL_SYMBOLS[0], rd.ALL_SYMBOLS[8], rd.ALL_SYMBOLS[30])}
                snaps.append({"snap": qlist(sq), "snap_in": qlist(si), "rough_in": qlist(qi),
                              "combos": [qlist(c) for c in combos], "oq": qlist(oq), "expand": exp})
            w.case("quads", {"variant": v.name, "min_area": min_area, "image": rel},
                   {"img": w.arr(g)}, {"candidates": [qlist(q) for q in cq], "snaps": snaps})
    # percentile and line intersections
    pcs = []
    for _ in range(300):
        n = int(rng.integers(1, 3000))
        a = rng.integers(0, 256, n).astype(np.uint8)
        if rng.random() < 0.4:
            a = (a // 51 * 51).astype(np.uint8)
        pcs.append({"a": a.tolist(), "p": [float(np.percentile(a, p)) for p in (65, 75, 85)]})
    w.case("percentile", {}, {}, pcs)
    li = []
    for _ in range(2000):
        p1, d1, p2, d2 = (rng.normal(0, 300, 2), rng.normal(0, 1, 2), rng.normal(0, 300, 2), rng.normal(0, 1, 2))
        if rng.random() < 0.05:
            d2 = d1 * rng.normal()
        li.append({"in": [p1.tolist(), d1.tolist(), p2.tolist(), d2.tolist()],
                   "out": rd._line_intersect(p1, d1, p2, d2).tolist()})
    w.case("line_intersect", {}, {}, li)


# --------------------------------------------------------------------------
# P3.3 refine_quad, _track_axis
# --------------------------------------------------------------------------
def best_candidates(smp, quads, qz: bool, top: int) -> list:
    """The best scored (score, geometry, oriented quad) of `_grid_pass`."""
    scored = []
    for quad in quads:
        by_sym = {}
        for oq, _, _ in rd.orientations(quad):
            w_, h_ = np.linalg.norm(oq[1] - oq[0]), np.linalg.norm(oq[3] - oq[0])
            for sym in rd.ALL_SYMBOLS:
                if not 0.7 < (w_ / sym[1]) / (h_ / sym[0]) < 1.4 or min(w_ / sym[1], h_ / sym[0]) < 1.6:
                    continue
                for cq in ((oq,) if qz else (oq, rd.expand_quad(oq, rd.geometry(sym)))):
                    by_sym.setdefault(sym, []).append(cq)
        for sym, cands in by_sym.items():
            gm = rd.geometry(sym)
            s = rd.score_quads(smp, np.array(cands, np.float32), gm, qz)
            scored += [(float(sc), gm, cq) for sc, cq in zip(s, cands)]
    scored.sort(key=lambda t: -t[0])
    return scored[:top]


def dump_refine(w: Writer, inputs) -> None:
    for rel, g, gi in inputs:
        smp = rd.Sampler(g)
        quads = [rd.snap_quad(g, q) for q in rd.candidate_quads(g)[:2]]
        for qz in (True, False):
            for _, gm, oq in best_candidates(smp, quads, qz, 4):
                coarse, cs = rd.refine_quad(smp, oq, gm, iters=2, steps=(1.0, 0.5), qz=qz)
                fine, fs = rd.refine_quad(smp, coarse, gm, steps=(0.5, 0.25, 0.12), qz=qz)
                full, us = rd.refine_quad(smp, oq, gm, qz=qz)
                u, v = rd.tracked_axes(smp, fine, gm)
                uo, vo = rd.tracked_axes(smp, oq, gm)
                w.case("refine", {"rows": gm.rows, "cols": gm.cols, "qz": qz, "image": rel},
                       {"img": gi, "quad": qlist(oq)},
                       {"coarse": qlist(coarse), "coarse_s": cs, "fine": qlist(fine), "fine_s": fs,
                        "full": qlist(full), "full_s": us, "u": u.tolist(), "v": v.tolist(),
                        "u0": uo.tolist(), "v0": vo.tolist()})


# --------------------------------------------------------------------------
# P3.4 classify, codewords_from_bits, try_fit, _zscore
# --------------------------------------------------------------------------
def fit_out(fit: rd.GridFit, r) -> dict:
    out = {"bits": fit.bits.ravel().tolist(), "conf": fit.conf.ravel().tolist(),
           "threshold": fit.threshold, "fixed_errors": fit.fixed_errors, "ok": r is not None}
    if r is not None:
        rs, dec = r
        out.update(data=dec.data.hex(), all_cw=rs.all_cw, block_errors=rs.block_errors,
                   block_erasures=rs.block_erasures, uec=rs.uec)
    return out


def dump_fit(w: Writer, inputs) -> None:
    for rel, g, gi in inputs:
        smp = rd.Sampler(g)
        quads = [rd.snap_quad(g, q) for q in rd.candidate_quads(g)[:2]]
        for qz in (True, False):
            for _, gm, oq in best_candidates(smp, quads, qz, 3):
                q, s = rd.refine_quad(smp, oq, gm, qz=qz)
                runs = []
                for mode, tr in (("plane", False), ("local", False), ("plane", True), ("local", True)):
                    fit = rd.GridFit(gm, q, s)
                    r = rd.try_fit(smp, fit, mode, tracked=tr)
                    vals = fit.values.astype(np.float32)
                    cw, cconf = rd.codewords_from_bits(fit.bits, fit.conf, gm)
                    runs.append({"mode": mode, "tracked": tr, "vals": w.arr(vals),
                                 "zscore": w.arr(rd._zscore(vals, gm).astype(np.float32)),
                                 "cw": cw, "cconf": cconf.tolist(), **fit_out(fit, r)})
                w.case("fit", {"rows": gm.rows, "cols": gm.cols, "qz": qz, "image": rel},
                       {"img": gi, "quad": qlist(q), "score": s}, runs)


# --------------------------------------------------------------------------
# P3.5 rescue_fit: every call made inside grid_decode, recorded
# --------------------------------------------------------------------------
# (image, grid variant): the two baseline rescue reads and one failing variant
RESCUE_CASES = [tuple(c) for c in PICKS["fixtures_rescue_cases"]]


class Recorder:
    """Wraps rd.rescue_fit and keeps (inputs, result) of each call."""

    def __init__(self):
        self.orig = rd.rescue_fit
        self.calls: list = []

    def __call__(self, fit, images, deadline):
        q, gm, c = fit.quad.copy(), fit.geom, fit.contrast
        r = self.orig(fit, images, deadline)
        self.calls.append((q, gm, c, [n for n, _ in images], r, fit))
        return r


def dump_rescue(w: Writer) -> None:
    refs: dict = {}
    for rel, vname in RESCUE_CASES:
        bgr = rd.load_image(str(TI / rel))
        gvars = rd.grid_variants(bgr)
        for v in gvars:
            if (rel, v.name) not in refs:
                refs[rel, v.name] = w.arr(v.img)
        v = next(v for v in gvars if v.name == vname)
        alts = [(o.name, o.img) for o in gvars if o is not v and o.img.shape == v.img.shape]
        rec = Recorder()
        rd.rescue_fit = rec
        try:
            rd.grid_decode(v.img, float("inf"), (), alts, 0.02, (True,))
        finally:
            rd.rescue_fit = rec.orig
        for q, gm, c, names, res, fit in rec.calls:
            out = {"ok": res is not None, "threshold": fit.threshold,
                   "fixed_errors": fit.fixed_errors, "bits": fit.bits.ravel().tolist()}
            if res is not None:
                out.update(data=res[1].data.hex(), how=res[2], all_cw=res[0].all_cw)
            w.case("rescue", {"rows": gm.rows, "cols": gm.cols, "image": rel, "variant": vname},
                   {"images": [[n, refs[rel, vname if n == "same image" else n]] for n in names],
                    "quad": qlist(q), "contrast": c}, out)


# --------------------------------------------------------------------------
# P3.6 / P3.7 grid_decode: every grid job of read_image, replayed without
# a deadline (results and near misses do not depend on timing)
# --------------------------------------------------------------------------
REPLAY_BUDGET_S = 60.0


def replay_image(rel: str, only: tuple | None = None) -> tuple[str, list[dict], dict, dict | None]:
    """Grid jobs of read_image(rel); with `only` = (variant, qz) just that
    job is run (the others return nothing at once)."""
    calls: list[dict] = []
    orig = rd._scan_job

    def scan(kind, *args):
        if kind != "grid":
            return orig(kind, *args)
        v, extra, alts, _cap, _deadline, qz, symbols = args
        if only is not None and (v.name, qz) != only:
            return None, {}
        gdiag: dict = {}
        t0 = time.time()
        r = rd.grid_decode(v.img, float("inf"), extra, alts, rd._min_area(v), (qz,), symbols, gdiag)
        rec = {"variant": v.name, "img": v.img, "extra": [np.asarray(q, np.float32) for q in extra],
               "alts": list(alts), "min_area": rd._min_area(v), "qz": qz, "diag": dict(gdiag),
               "py_s": time.time() - t0, "ok": r is not None}
        if r is not None:
            fit, rs, dec, mirror, how = r
            rec.update(data=dec.data.hex(), note=how, mirrored=bool(mirror), size=fit.geom.name,
                       quad=qlist(fit.quad), all_cw=rs.all_cw)
        calls.append(rec)
        return r, gdiag

    rd._scan_job = scan
    res = rd.read_image(str(TI / rel), {"zxing", "grid"}, budget_s=REPLAY_BUDGET_S, runner=rd.Runner(1))
    qual = None
    if res.fit is not None:
        import cv2
        fit = res.fit
        qual = {"gray": cv2.cvtColor(rd.load_image(str(TI / rel)), cv2.COLOR_BGR2GRAY),
                "size": fit.geom.name, "quad": qlist(fit.quad), "bits": fit.bits.ravel().tolist(),
                "fixed_errors": fit.fixed_errors, "contrast": fit.contrast, "scale": res.scale,
                "offset": [res.ox, res.oy], "uec": res.rs.uec, "engine": res.engine,
                "axes": [a.tolist() for a in fit.axes] if getattr(fit, "axes", None) else None,
                "quality": {k: (list(v) if isinstance(v, tuple) else v) for k, v in res.quality.items()}}
    return rel, calls, res.near_miss, qual


def all_images() -> list[str]:
    return sorted(str(p.relative_to(TI)) for p in TI.rglob("*") if p.suffix.lower() in rd.IMAGE_EXT)


def dump_grid_calls(w: Writer) -> None:
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    # the job that gave each baseline near miss (not always reached above)
    targets = [(e["file"].removeprefix("testimages/"), (e["near_miss"]["variant"], e["near_miss"]["qz"]))
               for e in base if not e["decoded"] and e.get("near_miss")]
    with cf.ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as ex:
        full = ex.map(replay_image, all_images())
        tgt = ex.map(replay_image, *zip(*targets))
        results = [(r, False) for r in full] + [(r, True) for r in tgt]
    for (rel, calls, near, qual), targeted in results:
        refs: dict = {}

        def ref(name: str, img: np.ndarray) -> dict:
            if name not in refs:
                refs[name] = w.arr(img)
            return refs[name]

        for c in calls:
            inp = {"img": ref(c["variant"], c["img"]), "extra": [qlist(q) for q in c["extra"]],
                   "alts": [[n, ref(n, a)] for n, a in c["alts"]]}
            out = {k: c[k] for k in ("ok", "diag", "py_s", "data", "note", "mirrored", "size", "quad", "all_cw")
                   if k in c}
            w.case("grid_decode", {"image": rel, "variant": c["variant"], "qz": c["qz"],
                                   "min_area": c["min_area"], "baseline_near_miss": targeted}, inp, out)
        if not targeted:
            w.case("near_miss", {"image": rel}, {}, near)
        if qual is not None and not targeted:
            gi = w.arr(qual.pop("gray"))
            w.case("quality", {"image": rel}, {"gray": gi}, qual)


# --------------------------------------------------------------------------
# P4b.3 / P4b.4 global_track, warp_fit (v2)
# --------------------------------------------------------------------------
def axis_list(a) -> list | None:
    return None if a is None else np.asarray(a, np.float64).tolist()


def dump_warp(w: Writer) -> None:
    for rel in sorted(r for r in all_images() if r.startswith("not_recognized/")):
        gvars = rd.grid_variants(rd.load_image(str(TI / rel)))
        for v in gvars[:3]:
            alts = [(o.name, o.img) for o in gvars if o is not v and o.img.shape == v.img.shape]
            calls = []
            orig = rd.warp_fit

            def rec(sampler, quad, g, deadline):
                r = orig(sampler, quad, g, deadline)
                calls.append((np.asarray(quad, np.float32).copy(), g, r))
                return r

            rd.warp_fit = rec
            try:
                rd.grid_decode(v.img, float("inf"), (), alts, rd._min_area(v), (True,))
            finally:
                rd.warp_fit = orig
            smp = rd.Sampler(v.img)
            out = []
            for q, g, r in calls:
                c = {"quad": qlist(q), "rows": g.rows, "cols": g.cols,
                     "u": axis_list(rd.global_track(smp, q, g, "u")),
                     "v": axis_list(rd.global_track(smp, q, g, "v")), "fit": None}
                if r is not None:
                    fit, ucol, vrow = r
                    c["fit"] = {"quad": qlist(fit.quad), "score": float(fit.contrast),
                                "u": axis_list(ucol), "v": axis_list(vrow)}
                out.append(c)
            print(f"{rel} [{v.name}]: {len(out)} warp_fit calls, "
                  f"{sum(c['fit'] is not None for c in out)} fits")
            w.case("warp", {"image": rel, "variant": v.name}, {"img": w.arr(v.img)}, out)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--v2"]
    out = Path(args[0]) if args else PRIVATE / "fixtures"
    if V2:
        w = Writer(out, "grid_v2")
        dump_warp(w)
        dump_grid_calls(w)
        w.finish()
        return
    rng = np.random.default_rng(SEED)
    w = Writer(out, "grid")
    inputs = grid_inputs(w)
    dump_sampler(w, inputs, rng)
    dump_quads(w, rng)
    dump_refine(w, inputs)
    dump_fit(w, inputs)
    dump_rescue(w)
    dump_grid_calls(w)
    w.finish()


if __name__ == "__main__":
    main()
