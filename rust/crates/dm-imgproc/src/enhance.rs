//! The enhancement building blocks of `datamatrix_reader.py` (`channels`,
//! `clahe`, `flatten_background`, `merge_dots`, `remove_texture`,
//! `sharpen`, `otsu`, `adaptive`, `gamma`, `normalise_size`).

use crate::arith::{add_weighted_u8, lut, normalize_minmax_f32};
use crate::clahe::clahe_apply;
use crate::color::{bgr_to_gray, bgr_to_hsv_s, bgr_to_lab_l};
use crate::filter::{gaussian_blur_f32, gaussian_blur_u8};
use crate::image::{Bgr8, Gray8, Image};
use crate::morph::{close, erode};
use crate::resize::{Interp, resize};
use crate::thresh::adaptive_threshold_mean;

/// Grayscale candidates in Python's dict order: gray, sat, blue, lab-L
/// (each prefixed `inv-` and inverted with `invert`).
pub fn channels(bgr: &Bgr8, invert: bool) -> Vec<(String, Gray8)> {
    (0..CHANNELS).map(|i| channel(bgr, invert, i)).collect()
}

/// Number of entries of `channels`.
pub const CHANNELS: usize = 4;

/// Entry `i` of `channels` alone (for building the chains in parallel).
pub fn channel(bgr: &Bgr8, invert: bool, i: usize) -> (String, Gray8) {
    let (k, v) = match i {
        0 => ("gray", bgr_to_gray(bgr)),
        1 => ("sat", bgr_to_hsv_s(bgr).inverted()),
        2 => ("blue", bgr.channel(0)),
        3 => ("lab-L", bgr_to_lab_l(bgr)),
        _ => panic!("channel index {i} out of range"),
    };
    if invert {
        (format!("inv-{k}"), v.inverted())
    } else {
        (k.to_string(), v)
    }
}

/// `flatten_background`: divide by a heavily blurred copy (removes lighting
/// gradients), stretch to 0..255.
pub fn flatten_background(g: &Gray8, sigma: f64) -> Gray8 {
    let f = g.to_f32();
    let bg = gaussian_blur_f32(&f, sigma);
    let flat = Image::from_vec(
        f.width,
        f.height,
        f.data
            .iter()
            .zip(&bg.data)
            .map(|(&v, &b)| v / (b + 1.0))
            .collect(),
    );
    normalize_minmax_f32(&flat, 0.0, 255.0).to_u8_trunc()
}

/// `sharpen`: unsharp mask `g*(1+amount) - blur*amount`.
pub fn sharpen(g: &Gray8, sigma: f64, amount: f64) -> Gray8 {
    let blur = gaussian_blur_u8(g, sigma);
    add_weighted_u8(g, 1.0 + amount, &blur, -amount, 0.0)
}

/// `gamma`: LUT of `((i / 255) ** gm) * 255`, truncated.
pub fn gamma(g: &Gray8, gm: f64) -> Gray8 {
    let mut table = [0u8; 256];
    for (i, t) in table.iter_mut().enumerate() {
        *t = ((i as f64 / 255.0).powf(gm) * 255.0) as u8;
    }
    lut(g, &table)
}

/// `merge_dots`: grey erosion grows dark dots so dot-peened modules touch.
pub fn merge_dots(g: &Gray8, k: usize) -> Gray8 {
    erode(g, k)
}

/// `remove_texture`: grey closing removes thin dark grain lines.
pub fn remove_texture(g: &Gray8, k: usize) -> Gray8 {
    close(g, k)
}

/// `_odd`: nearest odd integer >= 3 (Python `round`: half to even).
pub fn odd(n: f64) -> usize {
    let n = (n.round_ties_even() as i64).max(3) as usize;
    if n % 2 == 1 { n } else { n + 1 }
}

/// `adaptive`: mean-C threshold over an odd block of about `block` px.
pub fn adaptive(g: &Gray8, block: f64) -> Gray8 {
    adaptive_threshold_mean(g, odd(block), 5.0)
}

/// `normalise_size`: bring the longer side near `target` px (INTER_AREA
/// down, INTER_CUBIC up); returns the image and the scale (1.0 if within
/// 0.8–1.25).
pub fn normalise_size(g: &Gray8, target: usize) -> (Gray8, f64) {
    let s = target as f64 / g.max_side() as f64;
    if 0.8 < s && s < 1.25 {
        return (g.clone(), 1.0);
    }
    let interp = if s < 1.0 { Interp::Area } else { Interp::Cubic };
    (resize(g, s, interp), s)
}

/// `clahe(g, clip, tiles)` (reader defaults: clip 3.0, 8 tiles).
pub fn clahe(g: &Gray8, clip: f64, tiles: usize) -> Gray8 {
    clahe_apply(g, clip, tiles)
}
