//! Global timing tracking, the "warped grid" of `datamatrix_reader_v2.py`
//! (v2 1376–1521, 1767–1817). `track_axis` looks for every timing module
//! within +-0.45 module of its ideal place, so it cannot follow marks whose
//! module pitch drifts by more than that over the symbol. Here every dark
//! dash on the timing edge is found and assigned in order to the expected
//! dark modules, without any position window. The dash count must match
//! the symbol size exactly, which also makes it a cheap size / orientation
//! filter.

use crate::deadline::Deadline;
use crate::fit::GridFit;
use crate::np::lstsq;
use crate::refine::module_px;
use crate::sampler::{Quad, Sampler, fisher_scores};
use dm_imgproc::{Border, Image, box_blur_row_f32, gaussian_blur_f32_ksize};
use dm_symbol::Geometry;

/// Profile sampling step (modules).
pub const WARP_STEP: f64 = 0.05;
/// Corner refinement <-> re-tracking rounds.
pub const WARP_ROUNDS: usize = 4;
/// Max dash spacing change / median spacing (real marks <= 0.31).
pub const WARP_MAX_JERK: f64 = 0.35;
/// Tracked finder / timing score before refinement (real >= 1.3).
pub const WARP_MIN_SCORE: f64 = 1.0;
/// Warped grid only for modules of at least this size (dash detection).
pub const WARP_MIN_PX: f32 = 2.5;

/// Timing line offsets inside the edge (modules):
/// `np.arange(-0.5, 1.51, 0.125)`.
pub fn warp_shifts() -> Vec<f64> {
    let n = ((1.51 - -0.5) / 0.125f64).ceil() as usize;
    (0..n).map(|i| -0.5 + i as f64 * 0.125).collect()
}

/// Which timing edge `global_track` follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpAxis {
    /// Top row -> column centres.
    U,
    /// Right column -> row centres.
    V,
}

/// `np.interp(x, xp, fp)` for increasing `xp`.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[last] {
        return fp[last];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == last || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `np.median` of float64 values (mean of the two middle ones if even).
fn median(a: &[f64]) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.total_cmp(y));
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

/// Smoothed timing profile -> module centres (`None` if the dark dash
/// count is not `want.len()`) and the jerk: largest change of dash spacing
/// relative to the median spacing (0 = even pitch).
pub fn dash_centres(prof: &[f32], t: &[f64], want: &[usize], n: usize) -> (Option<Vec<f64>>, f64) {
    // local threshold: running mean over two modules (one dark + one light)
    let k = (2.0 / WARP_STEP) as usize | 1;
    let p = Image::from_vec(prof.len(), 1, prof.to_vec());
    let trend = box_blur_row_f32(&p, k, Border::Reflect);
    // dark runs [start, end)
    let (mut starts, mut ends) = (Vec::new(), Vec::new());
    let mut prev = false;
    for (i, (&a, &b)) in prof.iter().zip(&trend.data).enumerate() {
        let d = a > b;
        if d != prev {
            if d {
                starts.push(i)
            } else {
                ends.push(i)
            }
            prev = d;
        }
    }
    if prev {
        ends.push(prof.len());
    }
    let runs: Vec<f64> = starts
        .iter()
        .zip(&ends)
        .filter(|&(&s, &e)| (e - s) as f64 * WARP_STEP > 0.2)
        .map(|(&s, &e)| t[(s + e - 1) / 2])
        .collect();
    if runs.len() != want.len() || want.len() < 4 {
        return (None, f64::INFINITY);
    }
    let gap: Vec<f64> = runs.windows(2).map(|w| w[1] - w[0]).collect();
    let jerk_max = gap
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(
            f64::NEG_INFINITY,
            |m, x| if x > m || x.is_nan() { x } else { m },
        );
    let jerk = jerk_max / median(&gap);
    let wantf: Vec<f64> = want.iter().map(|&w| w as f64).collect();
    let mut pos: Vec<f64> = (0..n).map(|i| interp(i as f64, &wantf, &runs)).collect();
    // extrapolate the ends
    let (lo, hi) = (want[0], want[want.len() - 1]);
    if lo > 0 {
        let d = (want[1] - want[0]) as f64;
        for (i, p) in pos.iter_mut().enumerate().take(lo) {
            *p = runs[0] - (lo - i) as f64 * gap[0] / d;
        }
    }
    if hi < n - 1 {
        let d = (want[want.len() - 1] - want[want.len() - 2]) as f64;
        let (r, g) = (runs[runs.len() - 1], gap[gap.len() - 1]);
        for (i, p) in pos.iter_mut().enumerate().skip(hi + 1) {
            *p = r + (i - hi) as f64 * g / d;
        }
    }
    (Some(pos), jerk)
}

/// Module centres along a timing edge, or `None` if no line has the right
/// dash count with a smooth enough spacing. All candidate lines in one
/// `sample_uv`.
pub fn global_track(
    sampler: &Sampler,
    quad: &Quad,
    g: &Geometry,
    axis: WarpAxis,
) -> Option<Vec<f64>> {
    let _span = crate::timing::Span::new(crate::timing::Stage::Track);
    let (rows, cols) = (g.rows, g.cols);
    let (n, want): (usize, Vec<usize>) = match axis {
        WarpAxis::U => (cols, (0..cols).filter(|&c| g.fixed_val[c] == 1).collect()),
        WarpAxis::V => (
            rows,
            (0..rows)
                .filter(|&r| g.fixed_val[r * cols + cols - 1] == 1)
                .collect(),
        ),
    };
    // np.arange(0, n, WARP_STEP) + WARP_STEP / 2
    let nt = (n as f64 / WARP_STEP).ceil() as usize;
    let t: Vec<f64> = (0..nt)
        .map(|i| i as f64 * WARP_STEP + WARP_STEP / 2.0)
        .collect();
    let shifts = warp_shifts();
    let offs: Vec<f64> = shifts
        .iter()
        .flat_map(|&s| [0.35, 0.5, 0.65].map(|o| s + o))
        .collect();
    let uv: Vec<[f64; 2]> = offs
        .iter()
        .flat_map(|&o| {
            t.iter().map(move |&tt| match axis {
                WarpAxis::U => [tt, o],
                WarpAxis::V => [cols as f64 - o, tt],
            })
        })
        .collect();
    let v = sampler.sample_uv(quad, rows, cols, &uv);
    // reshape(lines, 3, len(t)).mean(1): sequential over the 3 offsets
    let mut prof = Vec::with_capacity(shifts.len() * nt);
    for l in 0..shifts.len() {
        let b = &v[l * 3 * nt..(l + 1) * 3 * nt];
        prof.extend((0..nt).map(|i| (b[i] + b[nt + i] + b[2 * nt + i]) / 3.0));
    }
    let sig = 0.08 / WARP_STEP;
    let prof = gaussian_blur_f32_ksize(
        &Image::from_vec(nt, shifts.len(), prof),
        (6.0 * sig) as usize | 1,
        1,
        sig,
    );
    // deeper lines run into data modules and give ragged spacing: keep the
    // most regular dash sequence, not the most contrasted one
    let mut best = None;
    let mut best_jerk = WARP_MAX_JERK;
    for l in 0..shifts.len() {
        let (pos, jerk) = dash_centres(prof.row(l), &t, &want, n);
        if pos.is_some() && jerk < best_jerk {
            (best, best_jerk) = (pos, jerk);
        }
    }
    best
}

/// Finder / timing Fisher score of each quad sampled on the tracked grid.
pub fn tracked_score(
    sampler: &Sampler,
    quads: &[Quad],
    g: &Geometry,
    ucol: &[f64],
    vrow: &[f64],
) -> Vec<f32> {
    let (mut uv, mut dk) = (Vec::new(), Vec::new());
    for (r, &v) in vrow.iter().enumerate() {
        for (c, &u) in ucol.iter().enumerate() {
            let i = r * g.cols + c;
            if g.fixed_mask[i] {
                uv.push([u, v]);
                dk.push(g.fixed_val[i] == 1);
            }
        }
    }
    fisher_scores(sampler, quads, g.rows, g.cols, &uv, &dk)
}

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

/// Corner coordinate ascent with the tracked axes held fixed
/// (`refine_quad` assumes an even grid and pulls a warped symbol's corners
/// off).
#[allow(clippy::needless_range_loop)] // index form of the numpy delta array
pub fn refine_tracked(
    sampler: &Sampler,
    quad: &Quad,
    g: &Geometry,
    ucol: &[f64],
    vrow: &[f64],
) -> (Quad, f64) {
    let mut q = *quad;
    let md = module_px(&q, g);
    let mut best = tracked_score(sampler, &[q], g, ucol, vrow)[0] as f64;
    for f in [0.4f32, 0.2, 0.1, 0.05] {
        let step = f * md;
        for _ in 0..8 {
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
            let s = tracked_score(sampler, &cands, g, ucol, vrow);
            let i = argmax(&s);
            // float32 `s` against a Python float: the bound is cast to float32
            if s[i] <= (best + 1e-4) as f32 {
                break;
            }
            (best, q) = (s[i] as f64, cands[i]);
        }
    }
    (q, best)
}

/// Oriented quad -> fit (with `axes` = column / row centres) on a grid that
/// follows both timing edges, or `None` when the dash counts do not match,
/// the tracked score is too low or the deadline passes.
pub fn warp_fit(
    sampler: &Sampler,
    quad: &Quad,
    g: &'static Geometry,
    deadline: &Deadline,
) -> Option<GridFit> {
    let mut ucol = global_track(sampler, quad, g, WarpAxis::U)?;
    let mut vrow = global_track(sampler, quad, g, WarpAxis::V)?;
    // dash counts also match by chance on texture: only refine grids whose
    // finder / timing pattern already shows up on the tracked axes
    let mut q = *quad;
    let mut score = tracked_score(sampler, &[q], g, &ucol, &vrow)[0] as f64;
    if score < WARP_MIN_SCORE {
        return None;
    }
    for _ in 0..WARP_ROUNDS {
        if deadline.expired() {
            return None;
        }
        (q, score) = refine_tracked(sampler, &q, g, &ucol, &vrow);
        let u2 = global_track(sampler, &q, g, WarpAxis::U);
        let v2 = global_track(sampler, &q, g, WarpAxis::V);
        let (Some(u2), Some(v2)) = (u2, v2) else {
            break;
        };
        let close = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(f64::NEG_INFINITY, f64::max)
                < 0.05
        };
        let done = close(&u2, &ucol) && close(&v2, &vrow);
        (ucol, vrow) = (u2, v2);
        if done {
            break;
        }
    }
    let mut fit = GridFit::new(g, q, score);
    fit.axes = Some((ucol, vrow));
    Some(fit)
}

/// Largest deviation (modules) of the tracked module centres from an even
/// grid fitted through them; 0 for an untracked fit.
pub fn pitch_drift(axes: Option<&(Vec<f64>, Vec<f64>)>) -> f64 {
    let Some((u, v)) = axes else {
        return 0.0;
    };
    let mut worst = 0f64;
    for a in [u, v] {
        // np.polyfit(idx, a, 1): least squares on [idx, 1]
        let rows: Vec<[f64; 2]> = (0..a.len()).map(|i| [i as f64, 1.0]).collect();
        let Some([k, d]) = lstsq(&rows, a) else {
            return f64::NAN;
        };
        for (i, &x) in a.iter().enumerate() {
            worst = worst.max((x - (k * i as f64 + d)).abs());
        }
    }
    worst
}
