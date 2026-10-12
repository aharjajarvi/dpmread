//! Module classification and decoding of a fitted grid: `GridFit`,
//! `classify`, `codewords_from_bits`, `try_fit`, `plausible`, `_zscore`
//! (reader 951–961, 1369–1450).

use crate::np::{Order, clip, lstsq, mean_f32};
use crate::refine::tracked_axes;
use crate::sampler::{Quad, Sampler};
use crate::timing::{Stage, timed};
use dm_codec::{Decoded, decode_codewords};
use dm_ecc::{RSResult, rs_decode_symbol};
use dm_imgproc::{Border, GrayF32, Image, box_blur_f32};
use dm_symbol::Geometry;

/// A module grid laid over the symbol, with the last classification.
#[derive(Clone, Debug)]
pub struct GridFit {
    pub geom: &'static Geometry,
    /// Corners tl, tr, br, bl in variant px.
    pub quad: Quad,
    /// Finder / timing score of the fit (`refine_quad`).
    pub contrast: f64,
    pub values: Option<GrayF32>,
    /// Row-major module bits (1 = dark) and their confidence.
    pub bits: Option<Vec<u8>>,
    pub conf: Option<Vec<f64>>,
    pub threshold: f64,
    pub fixed_errors: usize,
    /// (column, row) centres in modules of a warped grid (`warp_fit`);
    /// `None` = the plain homography grid.
    pub axes: Option<(Vec<f64>, Vec<f64>)>,
}

impl GridFit {
    pub fn new(geom: &'static Geometry, quad: Quad, contrast: f64) -> Self {
        GridFit {
            geom,
            quad,
            contrast,
            values: None,
            bits: None,
            conf: None,
            threshold: 0.0,
            fixed_errors: 0,
            axes: None,
        }
    }
}

/// Threshold surface of `classify`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Plane fitted to the finder / timing modules.
    Plane,
    /// 5x5 module neighbourhood mean, clipped to the plane.
    Local,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Plane => "plane",
            Mode::Local => "local",
        }
    }
}

/// Means of the dark and light fixed-pattern modules (float32, pairwise).
fn fixed_means(vals: &[f32], g: &Geometry) -> (f32, f32) {
    let (mut d, mut l) = (Vec::new(), Vec::new());
    for ((&v, &m), &f) in vals.iter().zip(&g.fixed_mask).zip(&g.fixed_val) {
        if m {
            if f == 1 { d.push(v) } else { l.push(v) }
        }
    }
    (mean_f32(&d, Order::Pairwise), mean_f32(&l, Order::Pairwise))
}

/// Threshold module values (rows x cols): "plane" fits a smooth threshold
/// surface to the known finder / timing modules, "local" follows the 5x5
/// module neighbourhood within +-0.3 contrast of it. Returns bits (1 =
/// dark), confidence (distance to the threshold in contrast units) and the
/// global threshold.
pub fn classify(vals: &GrayF32, g: &Geometry, mode: Mode) -> (Vec<u8>, Vec<f64>, f64) {
    let (rows, cols) = (g.rows, g.cols);
    let v = &vals.data;
    let (d_mean, l_mean) = fixed_means(v, g);
    let thr_global = (d_mean + l_mean) / 2.0;
    // plane fit on midpoint estimates from fixed modules: value -+ half
    let half = (d_mean - l_mean) / 2.0;
    let (mut a, mut target) = (Vec::new(), Vec::new());
    for (i, &x) in v.iter().enumerate() {
        if g.fixed_mask[i] {
            let (r, c) = ((i / cols) as f64, (i % cols) as f64);
            a.push([1.0, r / rows as f64, c / cols as f64]);
            target.push(if g.fixed_val[i] == 1 {
                x - half
            } else {
                x + half
            } as f64);
        }
    }
    let span = d_mean - l_mean;
    let lo = (l_mean + 0.2 * span) as f64;
    let hi = (d_mean - 0.2 * span) as f64;
    let mut thr: Vec<f64> = match lstsq(&a, &target) {
        Some(k) => (0..rows * cols)
            .map(|i| {
                let (r, c) = ((i / cols) as f64, (i % cols) as f64);
                // keep the plane sane
                clip(
                    k[0] + k[1] * r / rows as f64 + k[2] * c / cols as f64,
                    lo,
                    hi,
                )
            })
            .collect(),
        None => vec![thr_global as f64; rows * cols],
    };
    if mode == Mode::Local {
        let loc = box_blur_f32(vals, 5, Border::Reflect);
        let sp = (0.3 * span) as f64;
        for (t, &l) in thr.iter_mut().zip(&loc.data) {
            *t = clip(l as f64, *t - sp, *t + sp);
        }
    }
    let bits = v
        .iter()
        .zip(&thr)
        .map(|(&x, &t)| (x as f64 > t) as u8)
        .collect();
    // Python `max(1e-3, span)`: the float32 span when it is larger
    let div = if span > 1e-3 { span as f64 } else { 1e-3 };
    let conf = v
        .iter()
        .zip(&thr)
        .map(|(&x, &t)| (x as f64 - t).abs() / div)
        .collect();
    (bits, conf, thr_global as f64)
}

/// Codewords (bit 1 = MSB) and per-codeword confidence (weakest bit).
pub fn codewords_from_bits(bits: &[u8], conf: &[f64], g: &Geometry) -> (Vec<u8>, Vec<f64>) {
    let cols = g.cols;
    g.cw_rows
        .iter()
        .zip(&g.cw_cols)
        .map(|(rr, cc)| {
            let mut cw = 0u8;
            let mut cf = f64::INFINITY;
            for (&r, &c) in rr.iter().zip(cc) {
                cw = cw << 1 | bits[r * cols + c];
                cf = cf.min(conf[r * cols + c]);
            }
            (cw, cf)
        })
        .unzip()
}

/// Reject text with control bytes other than GS/RS/EOT/CR/LF/TAB outside
/// BASE256 segments - typical of a miscorrected symbol.
pub fn plausible(dec: &Decoded) -> bool {
    const OK: [u8; 6] = [4, 9, 10, 13, 29, 30];
    dec.segments
        .iter()
        .filter(|s| s.mode != "BASE256")
        .all(|s| {
            s.data
                .iter()
                .all(|&b| (b >= 32 || OK.contains(&b)) && b != 127)
        })
}

/// Sample the fit (even grid, or timing-tracked centres) and decode.
pub fn try_fit(
    sampler: &Sampler,
    fit: &mut GridFit,
    mode: Mode,
    tracked: bool,
) -> Option<(RSResult, Decoded)> {
    let g = fit.geom;
    let vals = if tracked {
        let (ucol, vrow) = tracked_axes(sampler, &fit.quad, g);
        sampler.sample_grid(&fit.quad, g.rows, g.cols, &ucol, &vrow)
    } else {
        sampler.sample(&fit.quad, g.rows, g.cols, 0)
    };
    try_fit_values(fit, mode, vals)
}

/// Classify given module values, record them in `fit` (also on failure:
/// the near-miss report reads them), RS-correct and decode.
pub fn try_fit_values(fit: &mut GridFit, mode: Mode, vals: GrayF32) -> Option<(RSResult, Decoded)> {
    let g = fit.geom;
    let (bits, conf, thr) = classify(&vals, g, mode);
    fit.fixed_errors = bits
        .iter()
        .zip(&g.fixed_mask)
        .zip(&g.fixed_val)
        .filter(|((b, m), f)| **m && b != f)
        .count();
    let (cw, cconf) = codewords_from_bits(&bits, &conf, g);
    fit.values = Some(vals);
    fit.bits = Some(bits);
    fit.conf = Some(conf);
    fit.threshold = thr;
    let rs = timed(Stage::Rs, || rs_decode_symbol(&cw, &cconf, g))?;
    let dec = decode_codewords(&rs.data_cw).ok()?;
    if dec.data.is_empty() || !plausible(&dec) {
        return None;
    }
    Some((rs, dec))
}

/// Module values as contrast z-scores (x100) against the fixed pattern.
pub fn zscore(v: &GrayF32, g: &Geometry) -> GrayF32 {
    let (d, l) = fixed_means(&v.data, g);
    let mid = (d + l) / 2.0;
    let span = d - l;
    let div = if span > 1e-3 { span } else { 1e-3 };
    Image::from_vec(
        v.width,
        v.height,
        v.data.iter().map(|&x| (x - mid) / div * 100.0).collect(),
    )
}
