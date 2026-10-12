// Derived from OpenCV 5.0 (thresh.cpp (getThreshVal_Otsu_8u), adaptiveThreshold),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! Thresholds: Otsu (`THRESH_BINARY + THRESH_OTSU`) and
//! `adaptiveThreshold(ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY)`.

use crate::border::{Border, border_table};
use crate::image::{Gray8, Image};

/// Otsu level of `getThreshVal_Otsu_8u` (first maximum of the between-class
/// variance).
pub fn otsu_level(g: &Gray8) -> u8 {
    let mut hist = [0u64; 256];
    for &v in &g.data {
        hist[v as usize] += 1;
    }
    let scale = 1.0 / g.data.len() as f64;
    let mut mu = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        mu += i as f64 * h as f64;
    }
    mu *= scale;
    let (mut mu1, mut q1) = (0.0f64, 0.0f64);
    let (mut max_sigma, mut max_val) = (0.0f64, 0u8);
    for (i, &h) in hist.iter().enumerate() {
        let p_i = h as f64 * scale;
        mu1 *= q1;
        q1 += p_i;
        let q2 = 1.0 - q1;
        if q1.min(q2) < f32::EPSILON as f64 || q1.max(q2) > 1.0 - f32::EPSILON as f64 {
            continue;
        }
        mu1 = (mu1 + i as f64 * p_i) / q1;
        let mu2 = (mu - q1 * mu1) / q2;
        let sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if sigma > max_sigma {
            max_sigma = sigma;
            max_val = i as u8;
        }
    }
    max_val
}

/// `cv2.threshold(g, 0, 255, THRESH_BINARY + THRESH_OTSU)[1]`.
pub fn otsu(g: &Gray8) -> Gray8 {
    let t = otsu_level(g);
    g.map(|v| if v > t { 255 } else { 0 })
}

/// `cv2.boxFilter(g, -1, (k, k), normalize=True,
/// borderType=BORDER_REPLICATE)` for 8-bit images. Window sums are exact;
/// the scaling rounds like OpenCV's `ColumnSum<int, uchar>` (f32 in the
/// 8-lane SIMD part, f64 in the scalar tail).
pub fn box_mean_u8(g: &Gray8, k: usize) -> Gray8 {
    let (w, h) = (g.width, g.height);
    let r = k / 2;
    let xt = border_table(w, r, k - 1 - r, Border::Replicate);
    let yt = border_table(h, r, k - 1 - r, Border::Replicate);
    // horizontal window sums (running)
    let mut rows = vec![0i32; w * h];
    for y in 0..h {
        let src = g.row(y);
        let out = &mut rows[y * w..(y + 1) * w];
        let mut s: i32 = xt[..k].iter().map(|&i| src[i] as i32).sum();
        out[0] = s;
        for x in 1..w {
            s += src[xt[x + k - 1]] as i32 - src[xt[x - 1]] as i32;
            out[x] = s;
        }
    }
    let scale = 1.0 / (k * k) as f64;
    let scale_f = scale as f32;
    let vec_end = w - w % 8;
    let mut dst = Image::new(w, h);
    let mut sum = vec![0i32; w];
    for &j in &yt[..k] {
        for (s, &v) in sum.iter_mut().zip(&rows[j * w..(j + 1) * w]) {
            *s += v;
        }
    }
    for y in 0..h {
        if y > 0 {
            let (add, sub) = (yt[y + k - 1], yt[y - 1]);
            for x in 0..w {
                sum[x] += rows[add * w + x] - rows[sub * w + x];
            }
        }
        for (x, (d, &s)) in dst.row_mut(y).iter_mut().zip(&sum).enumerate() {
            let v = if x < vec_end {
                (s as f32 * scale_f).round_ties_even() as f64
            } else {
                (s as f64 * scale).round_ties_even()
            };
            *d = v.clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// `cv2.adaptiveThreshold(g, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
/// block, c)`: 255 where `g - mean > -ceil(c)`.
pub fn adaptive_threshold_mean(g: &Gray8, block: usize, c: f64) -> Gray8 {
    let mean = box_mean_u8(g, block);
    let idelta = c.ceil() as i32;
    let data = g
        .data
        .iter()
        .zip(&mean.data)
        .map(|(&v, &m)| {
            if v as i32 - m as i32 > -idelta {
                255
            } else {
                0
            }
        })
        .collect();
    Image::from_vec(g.width, g.height, data)
}
