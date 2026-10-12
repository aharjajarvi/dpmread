//! Near-miss bookkeeping of a failed fit: `interior_modules`, `size_agree`,
//! `outside_dark`, `near_miss_key` (reader 1315–1364).

use crate::np::sum_f32;
use crate::sampler::{Quad, Sampler};
use dm_imgproc::{GrayF32, Image, gaussian_blur_f32, get_perspective_transform, warp_perspective};
use dm_symbol::Geometry;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Rectified symbol side for `interior_modules` (px).
pub const SIZE_CHECK_PX: usize = 512;
/// Quiet-zone ring darker than this: the fit is likely partial.
pub const OUTSIDE_BUSY: f64 = 0.35;
const FFT_PAD: usize = 16;

/// `np.hanning(m)`.
fn hanning(m: usize) -> Vec<f64> {
    (0..m)
        .map(|i| {
            let n = (2 * i) as f64 + 1.0 - m as f64;
            0.5 + 0.5 * (std::f64::consts::PI * n / (m - 1) as f64).cos()
        })
        .collect()
}

/// FFT plans by length.
type Plans = HashMap<usize, Arc<dyn Fft<f64>>>;

/// `abs(np.fft.rfft(x, n))` (zero-padded to `n`) with `rustfft`; pocketfft
/// rounds differently in the last bits, only the argmax is used.
fn rfft_abs(x: &[f64], n: usize) -> Vec<f64> {
    static PLANS: OnceLock<Mutex<Plans>> = OnceLock::new();
    let fft = PLANS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(n)
        .or_insert_with(|| FftPlanner::new().plan_fft_forward(n))
        .clone();
    let mut buf: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(x.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    fft.process(&mut buf);
    buf[..n / 2 + 1].iter().map(|c| c.re.hypot(c.im)).collect()
}

/// Strongest period (module edges per side) of a module-edge profile.
fn edge_period(prof: Vec<f32>, n: usize) -> f64 {
    let m = prof.len();
    let p = Image::from_vec(m, 1, prof);
    let bg = gaussian_blur_f32(&p, 8.0);
    let win = hanning(m);
    let x: Vec<f64> = p
        .data
        .iter()
        .zip(&bg.data)
        .zip(&win)
        .map(|((&a, &b), &w)| (a - b) as f64 * w)
        .collect();
    let spec = rfft_abs(&x, FFT_PAD * m);
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, &s) in spec.iter().enumerate() {
        let per = i as f64 / FFT_PAD as f64 * n as f64 / (n - 1) as f64;
        let s = if !(6.0..=150.0).contains(&per) {
            0.0
        } else {
            s
        };
        if s > best.1 {
            best = (i, s);
        }
    }
    best.0 as f64 / FFT_PAD as f64 * n as f64 / (n - 1) as f64
}

/// Module counts (cols, rows) inside an oriented quad, from the periodicity
/// of module edges over the whole rectified symbol: independent evidence
/// for the fitted size (the finder / timing modules always "confirm" it).
pub fn interior_modules(sampler: &Sampler, quad: &Quad) -> (f64, f64) {
    let n = SIZE_CHECK_PX;
    let nf = n as f32;
    let dst = [(0.0, 0.0), (nf, 0.0), (nf, nf), (0.0, nf)];
    let Some(m) = get_perspective_transform(&quad.map(|p| (p[0], p[1])), &dst) else {
        return (0.0, 0.0);
    };
    let w: GrayF32 = warp_perspective(&sampler.dark, &m, n, n);
    // d/dx -> columns: mean over rows (outer axis, sequential)
    let mut col = vec![0f32; n - 1];
    for y in 0..n {
        let r = w.row(y);
        for (x, c) in col.iter_mut().enumerate() {
            *c += (r[x + 1] - r[x]).abs();
        }
    }
    let col: Vec<f32> = col.iter().map(|v| v / n as f32).collect();
    // d/dy -> rows: mean along each row (contiguous, pairwise)
    let mut d = vec![0f32; n];
    let row: Vec<f32> = (0..n - 1)
        .map(|y| {
            let (a, b) = (w.row(y), w.row(y + 1));
            for (x, v) in d.iter_mut().enumerate() {
                *v = (b[x] - a[x]).abs();
            }
            sum_f32(&d) / n as f32
        })
        .collect();
    (edge_period(col, n), edge_period(row, n))
}

/// How many axes (0..2) of the measured module count confirm the fit.
pub fn size_agree(est: (f64, f64), g: &Geometry) -> usize {
    [(est.0, g.cols), (est.1, g.rows)]
        .iter()
        .filter(|&&(e, n)| (e - n as f64).abs() <= (0.06 * n as f64).max(1.0))
        .count()
}

/// Fraction of the 1-module ring just outside the fit that is darker than
/// `thr` (data modules instead of a quiet zone: a partial fit).
pub fn outside_dark(sampler: &Sampler, quad: &Quad, g: &Geometry, thr: f64) -> f64 {
    let (c, r) = (g.cols as f64, g.rows as f64);
    let ring: Vec<[f64; 2]> = g
        .fp_uv_qz
        .iter()
        .filter(|p| p[0] < 0.0 || p[1] < 0.0 || p[0] > c || p[1] > r)
        .copied()
        .collect();
    let v = sampler.sample_uv(quad, g.rows, g.cols, &ring);
    let t = thr as f32;
    v.iter().filter(|&&x| x > t).count() as f64 / v.len() as f64
}

/// The best failed fit of a grid search (Python's `diag` dict).
#[derive(Clone, Debug, PartialEq)]
pub struct NearMiss {
    pub score: f64,
    /// Fitted size "RxC".
    pub size: String,
    pub qz: bool,
    pub modules_px: f64,
    pub fixed_pattern_module_errors: usize,
    pub size_agree: usize,
    /// Rounded to 2 decimals.
    pub outside_dark: f64,
    /// "RxC" from `interior_modules`.
    pub measured_size: String,
    /// Warped-grid near miss only (both timing dash counts matched),
    /// rounded to 2 decimals; `None` = key absent.
    pub pitch_drift_modules: Option<f64>,
}

/// Sort key of near misses (`near_miss_key`): confirmed size first, a quiet
/// surrounding before a partial fit, then the fit score. `None` (empty
/// dict) is (0, false, -1e9).
pub fn near_miss_key(d: Option<&NearMiss>) -> (usize, bool, f64) {
    match d {
        Some(d) => (d.size_agree, d.outside_dark <= OUTSIDE_BUSY, d.score),
        None => (0, false, -1e9),
    }
}

/// Python `round(x, 2)` (correctly rounded decimal, ties to even).
pub fn round2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap_or(x)
}
