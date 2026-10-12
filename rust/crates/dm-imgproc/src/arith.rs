// Derived from OpenCV 5.0 (core addWeighted, LUT, normalize),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! Point operations: `addWeighted`, `LUT`, `normalize(NORM_MINMAX)`.

use crate::image::{Gray8, GrayF32, Image};

/// `cv2.addWeighted(a, alpha, b, beta, gamma)` for 8-bit images: f32 maths,
/// round half to even, saturate.
pub fn add_weighted_u8(a: &Gray8, alpha: f64, b: &Gray8, beta: f64, gamma: f64) -> Gray8 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (wa, wb, g) = (alpha as f32, beta as f32, gamma as f32);
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(&x, &y)| {
            (x as f32)
                .mul_add(wa, (y as f32).mul_add(wb, g))
                .round_ties_even()
                .clamp(0.0, 255.0) as u8
        })
        .collect();
    Image::from_vec(a.width, a.height, data)
}

/// `cv2.LUT(g, lut)`.
pub fn lut(g: &Gray8, table: &[u8; 256]) -> Gray8 {
    g.map(|v| table[v as usize])
}

/// `cv2.normalize(img, None, a, b, cv2.NORM_MINMAX)` for a float image
/// (f32 output): scale and shift rounded to f32, applied with one FMA.
pub fn normalize_minmax_f32(img: &GrayF32, a: f64, b: f64) -> GrayF32 {
    let (dmin, dmax) = (a.min(b), a.max(b));
    let mut smin = f64::INFINITY;
    let mut smax = f64::NEG_INFINITY;
    for &v in &img.data {
        smin = smin.min(v as f64);
        smax = smax.max(v as f64);
    }
    let scale = (dmax - dmin)
        * if smax - smin > f64::EPSILON {
            1.0 / (smax - smin)
        } else {
            0.0
        };
    let scale = scale as f32;
    let shift = dmin as f32 - (smin * scale as f64) as f32;
    img.map(|v| v.mul_add(scale, shift))
}
