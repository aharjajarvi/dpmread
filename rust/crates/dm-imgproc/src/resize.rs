// Derived from OpenCV 5.0 (resize.cpp (resizeArea_, computeResizeAreaTab, resizeAreaFast_)),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! `cv2.resize(g, None, fx=s, fy=s)` for 8-bit gray images.
//!
//! * `INTER_AREA` (downscale): OpenCV's `resizeArea_` (f32 weights from
//!   `computeResizeAreaTab`, same summation order, fused) or `resizeAreaFast_` for
//!   integer factors. Bit-exact.
//! * `INTER_CUBIC` (upscale): Python's OpenCV hands 8-bit cubic resizes to
//!   IPP (`ipp_resize`, warpAffine with Catmull-Rom a = -0.75 and a
//!   replicated border), not to OpenCV's own fixed-point code. The model here
//!   is that interpolation in f64, separable, rounded half to even; it
//!   matches IPP except for a few pixels per million exactly at x.5.

use crate::fma::{fma_dispatch, madd};
use crate::image::{Gray8, Image};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Area,
    Cubic,
}

/// Output size of `cv2.resize(..., fx=s, fy=s)`: `round(w * s)` (half to even).
pub fn scaled_size(w: usize, h: usize, s: f64) -> (usize, usize) {
    (
        (w as f64 * s).round_ties_even() as usize,
        (h as f64 * s).round_ties_even() as usize,
    )
}

/// `cv2.resize(g, None, fx=s, fy=s, interpolation=interp)`.
pub fn resize(g: &Gray8, s: f64, interp: Interp) -> Gray8 {
    match interp {
        Interp::Area => resize_area(g, s),
        Interp::Cubic => resize_cubic(g, s),
    }
}

/// One entry of `computeResizeAreaTab`: destination index, source index,
/// weight.
#[derive(Clone, Copy, Debug)]
struct Decimate {
    di: usize,
    si: usize,
    alpha: f32,
}

fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<Decimate> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push(Decimate {
                di: dx,
                si: (sx1 - 1) as usize,
                alpha: ((sx1 as f64 - fsx1) / cell) as f32,
            });
        }
        for sx in sx1..sx2 {
            tab.push(Decimate {
                di: dx,
                si: sx as usize,
                alpha: (1.0 / cell) as f32,
            });
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push(Decimate {
                di: dx,
                si: sx2 as usize,
                alpha: ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32,
            });
        }
    }
    tab
}

/// `INTER_AREA` with factor `s` < 1.
pub fn resize_area(g: &Gray8, s: f64) -> Gray8 {
    let (dw, dh) = scaled_size(g.width, g.height, s);
    let scale = 1.0 / s;
    let iscale = scale.round_ties_even() as i64;
    if (scale - iscale as f64).abs() < f64::EPSILON {
        return resize_area_fast(g, dw, dh, iscale as usize);
    }
    resize_area_tab(g, dw, dh, scale)
}

fma_dispatch!(
    /// `resizeArea_`: the weighted sums are fused like the reference build.
    fn resize_area_tab(g: &Gray8, dw: usize, dh: usize, scale: f64) -> Gray8 => resize_area_impl
);

#[inline(always)]
fn resize_area_impl<const FMA: bool>(g: &Gray8, dw: usize, dh: usize, scale: f64) -> Gray8 {
    let xtab = area_tab(g.width, dw, scale);
    let ytab = area_tab(g.height, dh, scale);
    let mut dst = Image::new(dw, dh);
    let mut buf = vec![0f32; dw];
    let mut sum = vec![0f32; dw];
    let store = |sum: &[f32], d: &mut [u8]| {
        for (o, &v) in d.iter_mut().zip(sum) {
            *o = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    };
    let mut prev_dy = ytab[0].di;
    for t in &ytab {
        let row = g.row(t.si);
        buf.iter_mut().for_each(|b| *b = 0.0);
        for x in &xtab {
            buf[x.di] = madd::<FMA>(row[x.si] as f32, x.alpha, buf[x.di]);
        }
        let beta = t.alpha;
        if t.di != prev_dy {
            store(&sum, dst.row_mut(prev_dy));
            for (s, &b) in sum.iter_mut().zip(&buf) {
                *s = beta * b;
            }
            prev_dy = t.di;
        } else {
            for (s, &b) in sum.iter_mut().zip(&buf) {
                *s = madd::<FMA>(beta, b, *s);
            }
        }
    }
    store(&sum, dst.row_mut(prev_dy));
    dst
}

/// `resizeAreaFast_`: integer factor `k`, mean of k x k cells. Factor 2 rounds
/// `(sum + 2) >> 2` (SIMD helper), other factors and partial cells round the
/// f32 mean half to even.
fn resize_area_fast(g: &Gray8, dw: usize, dh: usize, k: usize) -> Gray8 {
    let (sw, sh) = (g.width, g.height);
    let area = k * k;
    let scale = 1.0f32 / area as f32;
    let full_w = sw / k;
    let mut dst = Image::new(dw, dh);
    for dy in 0..dh {
        let sy0 = dy * k;
        if sy0 >= sh {
            continue;
        }
        let w = if sy0 + k <= sh { full_w } else { 0 };
        for dx in 0..dw {
            let sx0 = dx * k;
            let v = if dx < w {
                let mut sum = 0u32;
                for sy in 0..k {
                    let row = g.row(sy0 + sy);
                    for sx in 0..k {
                        sum += row[sx0 + sx] as u32;
                    }
                }
                if k == 2 {
                    ((sum + 2) >> 2) as u8
                } else {
                    (sum as f32 * scale).round_ties_even().min(255.0) as u8
                }
            } else if sx0 >= sw {
                0
            } else {
                let (mut sum, mut count) = (0u32, 0u32);
                for sy in 0..k.min(sh - sy0) {
                    let row = g.row(sy0 + sy);
                    for sx in 0..k.min(sw - sx0) {
                        sum += row[sx0 + sx] as u32;
                        count += 1;
                    }
                }
                (sum as f32 / count as f32).round_ties_even().min(255.0) as u8
            };
            dst.set(dx, dy, v);
        }
    }
    dst
}

/// Catmull-Rom weights (a = -0.75) for fraction `f` of taps -1, 0, 1, 2.
fn cubic_weights(f: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let near = |t: f64| ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    let far = |t: f64| ((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A;
    [far(f + 1.0), near(f), near(1.0 - f), far(2.0 - f)]
}

/// Source taps and weights along one axis (replicated border).
fn cubic_axis(n_src: usize, n_dst: usize, s: f64) -> Vec<([usize; 4], [f64; 4])> {
    (0..n_dst)
        .map(|d| {
            let x = (d as f64 + 0.5) / s - 0.5;
            let x0 = x.floor();
            let i0 = x0 as i64;
            let idx = [-1i64, 0, 1, 2].map(|o| (i0 + o).clamp(0, n_src as i64 - 1) as usize);
            (idx, cubic_weights(x - x0))
        })
        .collect()
}

/// `INTER_CUBIC` with factor `s` > 1 (IPP model, see the module docs).
pub fn resize_cubic(g: &Gray8, s: f64) -> Gray8 {
    let (dw, dh) = scaled_size(g.width, g.height, s);
    let xa = cubic_axis(g.width, dw, s);
    let ya = cubic_axis(g.height, dh, s);
    let mut tmp = vec![0f64; g.height * dw];
    for y in 0..g.height {
        let row = g.row(y);
        for (x, (idx, w)) in xa.iter().enumerate() {
            tmp[y * dw + x] = row[idx[0]] as f64 * w[0]
                + row[idx[1]] as f64 * w[1]
                + row[idx[2]] as f64 * w[2]
                + row[idx[3]] as f64 * w[3];
        }
    }
    let mut dst = Image::new(dw, dh);
    for (y, (idx, w)) in ya.iter().enumerate() {
        let r = |k: usize| &tmp[idx[k] * dw..(idx[k] + 1) * dw];
        let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
        for (x, d) in dst.row_mut(y).iter_mut().enumerate() {
            let v = r0[x] * w[0] + r1[x] * w[1] + r2[x] * w[2] + r3[x] * w[3];
            *d = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}
