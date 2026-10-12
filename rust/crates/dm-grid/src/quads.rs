//! Rough symbol locations and side fitting: `candidate_quads`,
//! `_line_intersect`, `snap_quad`, `side_combos`, `expand_quad` (reader
//! 962–1086, 1492–1500).

use crate::np::{norm2_f32, sum_f32};
use crate::sampler::{Quad, order_quad};
use dm_imgproc::contours::to_f32;
use dm_imgproc::{
    Gray8, Image, box_points, close, contour_area, find_contours_external, flatten_background,
    gaussian_blur_u8, get_perspective_transform, min_area_rect, odd, open, perspective_transform,
    remap_points,
};
use dm_symbol::{Geometry, Symbol, all_symbols};

/// `np.percentile(img, pct)` (method "linear") of a u8 image.
pub fn percentile_u8(img: &Gray8, pct: f64) -> f64 {
    let n = img.data.len();
    let mut hist = [0usize; 256];
    for &v in &img.data {
        hist[v as usize] += 1;
    }
    // k-th smallest value
    let kth = |k: usize| -> f64 {
        let mut acc = 0;
        for (v, &c) in hist.iter().enumerate() {
            acc += c;
            if acc > k {
                return v as f64;
            }
        }
        255.0
    };
    let virt = (n - 1) as f64 * (pct / 100.0);
    let lo = virt.floor();
    let t = virt - lo;
    let lo = lo as usize;
    let (a, b) = (kth(lo), kth((lo + 1).min(n - 1)));
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

/// Mean of the 4 corners in float32 (`q.mean(0)`).
fn centre(q: &Quad) -> [f32; 2] {
    let mut c = [0f32; 2];
    for p in q {
        c[0] += p[0];
        c[1] += p[1];
    }
    [c[0] / 4.0, c[1] / 4.0]
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn quad_area(q: &Quad) -> f64 {
    contour_area(&q.map(|p| (p[0], p[1])))
}

/// Blob aspect limit of the Python readers; drops the long DMRE sizes.
pub const PY_MAX_ASPECT: f64 = 4.5;

/// Blob aspect limit for `candidate_quads`: the largest aspect of the
/// candidate sizes (`None`: all, DMRE 8x144 = 18), at least
/// `PY_MAX_ASPECT` (P4b.12).
pub fn max_aspect(symbols: Option<&[Symbol]>) -> f64 {
    let aspect = |s: &Symbol| s.rows.max(s.cols) as f64 / s.rows.min(s.cols) as f64;
    let mx = match symbols {
        Some(syms) => syms.iter().map(aspect).fold(0.0, f64::max),
        None => all_symbols().map(aspect).fold(0.0, f64::max),
    };
    mx.max(PY_MAX_ASPECT)
}

/// Dark blobs that could be a symbol, as minimum area rectangles
/// (thresholds at the 65/75/85th darkness percentile, three closing
/// sizes), best `area * solidity^2` first, near-duplicates dropped, plus the
/// whole image. `min_area` is the smallest blob as a fraction of the image,
/// `max_aspect` the longest blob (`max_aspect()`, Python: `PY_MAX_ASPECT`).
pub fn candidate_quads(g: &Gray8, max_quads: usize, min_area: f64, max_aspect: f64) -> Vec<Quad> {
    let (h, w) = (g.height, g.width);
    let m = h.min(w);
    let hw = (h * w) as f64;
    let flat = flatten_background(g, m as f64 / 8.0);
    let dark = gaussian_blur_u8(&flat, (m as f64 / 200.0).max(1.0)).map(|v| 255 - v);
    let mut quads: Vec<(f64, Quad)> = Vec::new();
    for pct in [65.0, 75.0, 85.0] {
        let p = percentile_u8(&dark, pct);
        let bw = dark.map(|v| if v as f64 > p { 255 } else { 0 });
        for frac in [0.03, 0.05, 0.08] {
            let k = odd(m as f64 * frac);
            let blob = open(&close(&bw, k), k);
            for c in find_contours_external(&blob) {
                let area = contour_area(&to_f32(&c));
                if area < min_area * hw || area > 0.9 * hw {
                    continue;
                }
                let rect = min_area_rect(&c);
                let (rw, rh) = (rect.size.0 as f64, rect.size.1 as f64);
                if rw.min(rh) < 12.0 {
                    continue;
                }
                let aspect = rw.max(rh) / rw.min(rh);
                let solidity = area / (rw * rh);
                if aspect > max_aspect || solidity < 0.6 {
                    continue;
                }
                let b = box_points(&rect).map(|(x, y)| [x, y]);
                quads.push((area * solidity.powi(2), b));
            }
        }
    }
    quads.sort_by(|a, b| {
        (-a.0)
            .partial_cmp(&-b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out: Vec<Quad> = Vec::new();
    let near = (0.04 * m as f64) as f32;
    for (_, b) in quads {
        let c = centre(&b);
        let ab = quad_area(&b);
        if out
            .iter()
            .all(|o| norm2_f32(sub(c, centre(o))) > near || (ab / quad_area(o) - 1.0).abs() > 0.15)
        {
            out.push(b);
        }
        if out.len() >= max_quads {
            break;
        }
    }
    // tight crops: the image itself may be (nearly) the symbol
    let (wf, hf) = ((w - 1) as f32, (h - 1) as f32);
    out.push([[0.0, 0.0], [wf, 0.0], [wf, hf], [0.0, hf]]);
    out
}

/// Intersection of the lines `p1 + t d1` and `p2 + s d2` (float64, the 2x2
/// LU solve of numpy / OpenBLAS: reciprocal pivot); the midpoint of `p1`
/// and `p2` if they are (nearly) parallel.
pub fn line_intersect(p1: [f64; 2], d1: [f64; 2], p2: [f64; 2], d2: [f64; 2]) -> [f64; 2] {
    let a = [[d1[0], -d2[0]], [d1[1], -d2[1]]];
    let b = [p2[0] - p1[0], p2[1] - p1[1]];
    let (p, o) = if a[0][0].abs() >= a[1][0].abs() {
        (0, 1)
    } else {
        (1, 0)
    };
    let mid = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0];
    if a[p][0] == 0.0 {
        return mid;
    }
    let l = a[o][0] * (1.0 / a[p][0]);
    let u11 = a[o][1] - l * a[p][1];
    let det = a[p][0] * u11;
    if det.abs() < 1e-9 {
        return mid;
    }
    let x1 = (b[o] - l * b[p]) / u11;
    let t = (b[p] - a[p][1] * x1) / a[p][0];
    [p1[0] + t * d1[0], p1[1] + t * d1[1]]
}

/// `np.linspace(start, stop, num)`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

const SNAP_STEPS: usize = 46; // profile samples across a side
const SNAP_ALONG: usize = 25; // profiles averaged along a side
const SNAP_D: usize = 3; // half the edge width in samples

/// Fit every side of a rough quad to the symbol edge: at 25 places along
/// the side (and 9 tilts of +-6 degrees) find the strongest dark -> light
/// drop along the outward normal, then intersect neighbouring lines.
/// `inward` looks for the strongest light -> dark rise instead.
pub fn snap_quad(gray: &Gray8, quad: &Quad, passes: usize, inward: bool) -> Quad {
    let mut q = order_quad(quad);
    let along_t = linspace(-0.8, 0.8, SNAP_ALONG);
    let ts_t = linspace(-0.25, 0.2, SNAP_STEPS);
    let tilts: Vec<(f32, f32)> = (0..9)
        .map(|i| {
            let th = (-6.0 + i as f64 * 1.5) * (std::f64::consts::PI / 180.0);
            (th.cos() as f32, th.sin() as f32)
        })
        .collect();
    let n = SNAP_STEPS * SNAP_ALONG;
    let (mut xs, mut ys) = (vec![0f32; n], vec![0f32; n]);
    for _ in 0..passes {
        let ext = norm2_f32(sub(q[1], q[0])).max(norm2_f32(sub(q[3], q[0])));
        let sigma = ((ext / 100.0) as f64).max(1.0);
        let blurred = gaussian_blur_u8(gray, sigma);
        let dark = Image::from_vec(
            blurred.width,
            blurred.height,
            blurred.data.iter().map(|&v| 255.0 - v as f32).collect(),
        );
        let ts: Vec<f64> = ts_t.iter().map(|&t| t * ext as f64).collect();
        let c = centre(&q);
        let mut lines = Vec::with_capacity(4);
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            let ba = sub(b, a);
            let len = norm2_f32(ba);
            let u = [ba[0] / (len + 1e-6), ba[1] / (len + 1e-6)];
            let mut nrm = [-u[1], u[0]];
            let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            if dot(sub(mid, c), nrm) < 0.0 {
                nrm = [-nrm[0], -nrm[1]];
            }
            let half = len / 2.0;
            let along: Vec<f64> = along_t.iter().map(|&t| t * half as f64).collect();
            let (m0, m1) = (mid[0] as f64, mid[1] as f64);
            let mut best = (-1e9f32, [m0, m1], u);
            for &(ct, st) in &tilts {
                let ut = [u[0] * ct - u[1] * st, u[0] * st + u[1] * ct];
                let mut nt = [-ut[1], ut[0]];
                if dot(nt, nrm) < 0.0 {
                    nt = [-nt[0], -nt[1]];
                }
                let (u0, u1, n0, n1) = (ut[0] as f64, ut[1] as f64, nt[0] as f64, nt[1] as f64);
                for (i, &t) in ts.iter().enumerate() {
                    for (j, &s) in along.iter().enumerate() {
                        xs[i * SNAP_ALONG + j] = (m0 + s * u0 + t * n0) as f32;
                        ys[i * SNAP_ALONG + j] = (m1 + s * u1 + t * n1) as f32;
                    }
                }
                let vals = remap_points(&dark, &xs, &ys);
                let prof: Vec<f32> = vals
                    .as_chunks::<SNAP_ALONG>()
                    .0
                    .iter()
                    .map(|r| sum_f32(r) / SNAP_ALONG as f32)
                    .collect();
                let (mut bi, mut bv) = (0, f32::NEG_INFINITY);
                for i in 0..SNAP_STEPS - 2 * SNAP_D {
                    let drop = if inward {
                        prof[i + 2 * SNAP_D] - prof[i]
                    } else {
                        prof[i] - prof[i + 2 * SNAP_D]
                    };
                    // np.argmax: first maximum, a NaN wins
                    if drop.is_nan() {
                        (bi, bv) = (i, drop);
                        break;
                    }
                    if drop > bv {
                        (bi, bv) = (i, drop);
                    }
                }
                if bv > best.0 {
                    let t = ts[bi + SNAP_D];
                    best = (bv, [m0 + t * n0, m1 + t * n1], ut);
                }
            }
            lines.push((best.1, [best.2[0] as f64, best.2[1] as f64]));
        }
        q = std::array::from_fn(|k| {
            let (p1, d1) = lines[(k + 3) % 4];
            let (p2, d2) = lines[k];
            let x = line_intersect(p1, d1, p2, d2);
            [x[0] as f32, x[1] as f32]
        });
    }
    q
}

/// Quads taking every side from either fit (16 combinations, near
/// duplicates within `min_diff` px dropped): a quiet zone on some sides,
/// dark print flush on the others.
pub fn side_combos(outer: &Quad, inner: &Quad, min_diff: f32) -> Vec<Quad> {
    let (a, b) = (order_quad(outer), order_quad(inner));
    let mut out: Vec<Quad> = Vec::new();
    for m in 0..16 {
        let sides: Vec<([f64; 2], [f64; 2])> = (0..4)
            .map(|k| {
                let q = if m >> k & 1 == 1 { &b } else { &a };
                let d = sub(q[(k + 1) % 4], q[k]);
                ([q[k][0] as f64, q[k][1] as f64], [d[0] as f64, d[1] as f64])
            })
            .collect();
        let c: Quad = std::array::from_fn(|k| {
            let (p1, d1) = sides[(k + 3) % 4];
            let (p2, d2) = sides[k];
            let x = line_intersect(p1, d1, p2, d2);
            [x[0] as f32, x[1] as f32]
        });
        let far = |o: &Quad| {
            let mut mx = f32::NEG_INFINITY;
            for k in 0..4 {
                for ax in 0..2 {
                    mx = mx.max((c[k][ax] - o[k][ax]).abs());
                }
            }
            mx > min_diff
        };
        if out.iter().all(far) {
            out.push(c);
        }
    }
    out
}

/// Grow an oriented quad (tl, tr, br, bl) by `modules` on every side. The
/// quad is returned unchanged if it is degenerate.
pub fn expand_quad(quad: &Quad, g: &Geometry, modules: f32) -> Quad {
    let (c, r) = (g.cols as f32, g.rows as f32);
    let src = [(0.0, 0.0), (c, 0.0), (c, r), (0.0, r)];
    let Some(h) = get_perspective_transform(&src, &quad.map(|p| (p[0], p[1]))) else {
        return *quad;
    };
    let m = modules;
    let pts = [(-m, -m), (c + m, -m), (c + m, r + m), (-m, r + m)];
    let p = perspective_transform(&pts, &h);
    std::array::from_fn(|k| [p[k].0, p[k].1])
}
