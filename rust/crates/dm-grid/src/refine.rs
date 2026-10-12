//! Grid alignment: `refine_quad` (coordinate ascent on the corners),
//! `_track_axis` / `tracked_axes` (module centres along the timing edges)
//! (reader 1203–1314, 1365).

use crate::np::{lstsq, norm2_f32};
use crate::sampler::{Quad, Sampler, score_quads};
use dm_symbol::Geometry;

/// `np.argmax` of float32 scores (first maximum; a NaN wins).
fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for (i, &v) in s.iter().enumerate() {
        if v.is_nan() {
            return i;
        }
        if v > s[bi] {
            bi = i;
        }
    }
    bi
}

fn mean4(q: &Quad) -> [f32; 2] {
    let mut c = [0f32; 2];
    for p in q {
        c[0] += p[0];
        c[1] += p[1];
    }
    [c[0] / 4.0, c[1] / 4.0]
}

/// Module pitch of an oriented quad in px (top and left side, float32).
pub fn module_px(q: &Quad, g: &Geometry) -> f32 {
    let top = norm2_f32([q[1][0] - q[0][0], q[1][1] - q[0][1]]);
    let left = norm2_f32([q[3][0] - q[0][0], q[3][1] - q[0][1]]);
    (top / g.cols as f32 + left / g.rows as f32) / 2.0
}

/// Default step schedule of `refine_quad` (in modules).
pub const REFINE_STEPS: [f64; 4] = [1.0, 0.5, 0.25, 0.12];

/// Coordinate ascent on the 8 corner coordinates maximising the finder /
/// timing contrast (`score_quads`). Per round: the 16 single-corner moves
/// (plus the combination of every improving coordinate), then edge, shift
/// and scale moves. Returns the quad and its score.
#[allow(clippy::needless_range_loop)] // index form of the numpy delta array
pub fn refine_quad(
    sampler: &Sampler,
    quad: &Quad,
    g: &Geometry,
    iters: usize,
    steps: &[f64],
    qz: bool,
) -> (Quad, f64) {
    let mut q = *quad;
    let md = module_px(&q, g);
    let mut best = score_quads(sampler, &[q], g, qz)[0] as f64;
    // `s > best + 1e-4` with a float32 `s`: the bound is cast to float32
    let bound = |best: f64| (best + 1e-4) as f32;
    for &f in steps {
        let step = f as f32 * md;
        for _ in 0..iters {
            let mut improved = false;
            // corner moves: index 2 * (2k + axis) + (0: +step, 1: -step)
            let mut delta = [[[0f32; 2]; 4]; 16];
            for k in 0..4 {
                for axis in 0..2 {
                    let i = 2 * (2 * k + axis);
                    delta[i][k][axis] = step;
                    delta[i + 1][k][axis] = -step;
                }
            }
            let cands: Vec<Quad> = delta
                .iter()
                .map(|d| std::array::from_fn(|k| [q[k][0] + d[k][0], q[k][1] + d[k][1]]))
                .collect();
            let s = score_quads(sampler, &cands, g, qz);
            let mut comb = [[0f32; 2]; 4];
            for j in 0..8 {
                let i = 2 * j + (s[2 * j + 1] > s[2 * j]) as usize;
                if s[i] > bound(best) {
                    for k in 0..4 {
                        comb[k][0] += delta[i][k][0];
                        comb[k][1] += delta[i][k][1];
                    }
                }
            }
            let i = argmax(&s);
            if s[i] > bound(best) {
                (best, q, improved) = (s[i] as f64, cands[i], true);
                let nonzero = comb.iter().flatten().filter(|&&v| v != 0.0).count();
                if nonzero > 1 {
                    let t: Quad = std::array::from_fn(|k| {
                        [
                            q[k][0] - delta[i][k][0] + comb[k][0],
                            q[k][1] - delta[i][k][1] + comb[k][1],
                        ]
                    });
                    let sc = score_quads(sampler, &[t], g, qz)[0] as f64;
                    if sc > best + 1e-4 {
                        (best, q) = (sc, t);
                    }
                }
            }
            // edge moves (two corners together) and whole-quad shift / scale
            let c = mean4(&q);
            let mut moves: Vec<Quad> = Vec::with_capacity(14);
            for k in 0..4 {
                let k2 = (k + 1) % 4;
                let e = [q[k2][0] - q[k][0], q[k2][1] - q[k][1]];
                let nn = norm2_f32(e) + 1e-6;
                let nrm = [-e[1] / nn, e[0] / nn];
                for sgn in [1f32, -1.0] {
                    let sv = sgn * step;
                    let mv = [sv * nrm[0], sv * nrm[1]];
                    let mut t = q;
                    for kk in [k, k2] {
                        t[kk][0] += mv[0];
                        t[kk][1] += mv[1];
                    }
                    moves.push(t);
                }
            }
            for (dx, dy) in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
                moves.push(q.map(|p| [p[0] + dx, p[1] + dy]));
            }
            let span = md * g.cols as f32;
            for f in [1.0 + step / span, 1.0 - step / span] {
                moves.push(q.map(|p| [c[0] + (p[0] - c[0]) * f, c[1] + (p[1] - c[1]) * f]));
            }
            let s = score_quads(sampler, &moves, g, qz);
            let i = argmax(&s);
            if s[i] > bound(best) {
                (best, q, improved) = (s[i] as f64, moves[i], true);
            }
            if !improved {
                break;
            }
        }
    }
    (q, best)
}

/// Timing edge followed by `track_axis`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Top row -> column centres.
    U,
    /// Right column -> row centres.
    V,
}

const TRACK_STEP: f64 = 0.05;

/// `np.arange(start, stop, step)` for floats (numpy's length and fill rule).
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil().max(0.0) as usize;
    let delta = (start + step) - start;
    (0..len)
        .map(|i| {
            if i == 0 {
                start
            } else {
                start + i as f64 * delta
            }
        })
        .collect()
}

/// Median of a short window (`np.median`).
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// `np.polyfit(x, y, 1)` -> (slope, intercept): least squares on the
/// column-scaled Vandermonde matrix, as numpy (which solves it with
/// LAPACK `gelsd`; agreement to the last few bits).
fn polyfit1(x: &[f64], y: &[f64]) -> (f64, f64) {
    let s0 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let s1 = (x.len() as f64).sqrt();
    let lhs: Vec<[f64; 2]> = x.iter().map(|v| [v / s0, 1.0 / s1]).collect();
    match lstsq(&lhs, y) {
        Some(c) => (c[0] / s0, c[1] / s1),
        None => (0.0, 0.0),
    }
}

/// `np.interp(x, xp, fp)` at integer `x` (numpy's slope formula).
fn interp(x: usize, xp: &[usize], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) as f64;
    slope * (x - xp[j]) as f64 + fp[j]
}

/// Measured centre of every module along a timing edge (in modules):
/// three lines across the edge, extrema with parabolic sub-sampling, gaps
/// interpolated, outliers replaced by a running median, smoothed. With
/// fewer than half the modules tracked, a linear trend through at least 6
/// of them, else the ideal grid.
pub fn track_axis(sampler: &Sampler, quad: &Quad, g: &Geometry, axis: Axis) -> Vec<f64> {
    let n = if axis == Axis::U { g.cols } else { g.rows };
    let expect: Vec<bool> = (0..n)
        .map(|c| match axis {
            Axis::U => g.fixed_val[c] == 1,
            Axis::V => g.fixed_val[c * g.cols + g.cols - 1] == 1,
        })
        .collect();
    let step = TRACK_STEP;
    let t: Vec<f64> = arange(-0.5, n as f64 + 0.5, step)
        .iter()
        .map(|v| v + step / 2.0)
        .collect();
    let mut prof = vec![0f32; t.len()];
    for off in [0.35, 0.5, 0.65] {
        let uv: Vec<[f64; 2]> = match axis {
            Axis::U => t.iter().map(|&v| [v, off]).collect(),
            Axis::V => t.iter().map(|&v| [g.cols as f64 - off, v]).collect(),
        };
        for (p, v) in prof
            .iter_mut()
            .zip(sampler.sample_uv(quad, g.rows, g.cols, &uv))
        {
            *p += v;
        }
    }
    // np.convolve(prof / 3, ones(3) / 3, "same") in float64
    let d: Vec<f64> = prof.iter().map(|&v| (v / 3.0) as f64).collect();
    let k = 1.0 / 3.0;
    let m = d.len();
    let prof: Vec<f64> = (0..m)
        .map(|i| {
            if m == 1 {
                d[0] * k
            } else if i == 0 {
                d[0] * k + d[1] * k
            } else if i == m - 1 {
                d[m - 2] * k + d[m - 1] * k
            } else {
                d[i - 1] * k + d[i] * k + d[i + 1] * k
            }
        })
        .collect();
    let mut obs = vec![f64::NAN; n];
    for c in 0..n {
        let lo = ((c as f64 + 0.5 - 0.45 + 0.5) / step).round_ties_even() as usize;
        let hi = ((c as f64 + 0.5 + 0.45 + 0.5) / step).round_ties_even() as usize;
        let hi = hi.min(m);
        if hi < lo + 3 {
            continue;
        }
        let seg: Vec<f64> = prof[lo..hi]
            .iter()
            .map(|&v| if expect[c] { v } else { -v })
            .collect();
        let mut kk = 0;
        for (i, &v) in seg.iter().enumerate() {
            if v.is_nan() {
                kk = i;
                break;
            }
            if v > seg[kk] {
                kk = i;
            }
        }
        if kk == 0 || kk == seg.len() - 1 {
            continue; // no real extremum
        }
        let den = seg[kk - 1] - 2.0 * seg[kk] + seg[kk + 1];
        let kf = kk as f64
            + if den != 0.0 {
                0.5 * (seg[kk - 1] - seg[kk + 1]) / den
            } else {
                0.0
            };
        obs[c] = t[lo] + kf * step;
    }
    let ideal: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
    let off: Vec<f64> = obs.iter().zip(&ideal).map(|(o, i)| o - i).collect();
    let ok: Vec<usize> = (0..n).filter(|&i| !off[i].is_nan()).collect();
    if ok.len() < n / 2 {
        if ok.len() >= 6 {
            let xs: Vec<f64> = ok.iter().map(|&i| i as f64).collect();
            let ys: Vec<f64> = ok.iter().map(|&i| off[i]).collect();
            let (slope, intercept) = polyfit1(&xs, &ys);
            return (0..n)
                .map(|i| ideal[i] + (intercept + slope * i as f64).clamp(-0.5, 0.5))
                .collect();
        }
        return ideal;
    }
    // robust: drop offsets far from a running median, interpolate the gaps
    let fp: Vec<f64> = ok.iter().map(|&i| off[i]).collect();
    let mut off: Vec<f64> = (0..n).map(|i| interp(i, &ok, &fp)).collect();
    let med: Vec<f64> = (0..n)
        .map(|i| median(&off[i.saturating_sub(2)..(i + 3).min(n)]))
        .collect();
    for i in 0..n {
        if (off[i] - med[i]).abs() > 0.25 {
            off[i] = med[i];
        }
    }
    (0..n)
        .map(|i| {
            let w = &off[i.saturating_sub(1)..(i + 2).min(n)];
            let mean = w.iter().fold(0.0, |s, v| s + v) / w.len() as f64;
            ideal[i] + mean.clamp(-0.5, 0.5)
        })
        .collect()
}

/// Column and row centres (`_track_axis` "u" and "v").
pub fn tracked_axes(sampler: &Sampler, quad: &Quad, g: &Geometry) -> (Vec<f64>, Vec<f64>) {
    (
        track_axis(sampler, quad, g, Axis::U),
        track_axis(sampler, quad, g, Axis::V),
    )
}
