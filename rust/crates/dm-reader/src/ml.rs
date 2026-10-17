//! Module classifier hook (cargo feature `ml`): when every grid job has
//! failed, the best failed fits are classified again with the trained
//! module classifier (`dm-ml`) in place of the threshold of `classify`, and
//! RS-decoded with the same soft-erasure rule.

use std::time::Instant;

use dm_codec::decode_codewords;
use dm_ecc::rs_decode_symbol;
use dm_grid::{
    FitMap, GridFit, GridRead, NearMiss, anti_transposed, codewords_from_bits,
    fixed_pattern_too_damaged, near_miss_key, plausible, symbol_input,
};
use dm_imgproc::{GrayF32, Image};
use dm_ml::Classifier;
use dm_symbol::{Geometry, all_symbols, geometry};

use crate::variants::Variant;

/// Failed fits classified per image, best by `near_miss_key`.
pub const ML_FITS: usize = 3;
/// Largest share of wrong finder / timing modules (by `classify`) of a fit
/// the classifier is run on; above it the grid is wrong. On the real `val`
/// fits this skips 75 of 92 fits whose grid matches no label and 1 of 39
/// whose grid does.
pub const ML_MAX_FIXED_ERRORS: f64 = 0.25;

/// Keep the best `ML_FITS` failed fits, best first.
pub(crate) fn keep<'a>(fits: &mut Vec<(&'a Variant, NearMiss)>, v: &'a Variant, d: &NearMiss) {
    fits.push((v, d.clone()));
    fits.sort_by(|a, b| {
        near_miss_key(Some(&b.1))
            .partial_cmp(&near_miss_key(Some(&a.1)))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fits.truncate(ML_FITS);
}

fn geometry_of(size: &str) -> Option<&'static Geometry> {
    all_symbols().find(|s| s.name() == size).map(geometry)
}

/// Classify and decode the fits in order until one reads or `deadline`
/// passes. Returns the read and the number of classifier calls.
pub(crate) fn read_fits<'a>(
    fits: &[(&'a Variant, NearMiss)],
    deadline: Instant,
) -> (Option<(&'a Variant, GridRead)>, usize) {
    let mut calls = 0;
    for (v, d) in fits {
        if Instant::now() >= deadline {
            break;
        }
        let Some(g) = geometry_of(&d.size) else {
            continue;
        };
        let n_fixed = g.fixed_mask.iter().filter(|&&m| m).count();
        if d.fixed_pattern_module_errors as f64 > ML_MAX_FIXED_ERRORS * n_fixed as f64 {
            continue;
        }
        calls += 1;
        if let Some(r) = read_fit(v, d, g) {
            return (Some((v, r)), calls);
        }
    }
    (None, calls)
}

fn read_fit(v: &Variant, d: &NearMiss, g: &'static Geometry) -> Option<GridRead> {
    let map = FitMap::new(&d.quad, g.rows, g.cols, d.axes.as_ref())?;
    let input = symbol_input(&v.img, &map, g.rows, g.cols);
    let p = Classifier::embedded().p_dark(&input, g.rows, g.cols)?;
    read_probs(d, g, Image::from_vec(g.cols, g.rows, p))
}

/// Decode the classifier's dark probabilities (rows x cols) in the fit's
/// frame, then, for a square symbol, in its anti-transposed frame
/// (`dm_grid::anti_transposed`).
fn read_probs(d: &NearMiss, g: &'static Geometry, p: GrayF32) -> Option<GridRead> {
    let mut fit = GridFit::new(g, d.quad, d.score);
    fit.axes = d.axes.clone();
    if let Some(r) = decode(fit.clone(), &p, d.mirrored, d.qz) {
        return Some(r);
    }
    let (t, p) = anti_transposed(&fit, &p)?;
    decode(t, &p, !d.mirrored, d.qz)
}

/// Bit = p > 0.5, confidence = abs(p - 0.5) x 2; the weak-RS and
/// `plausible` checks of `try_fit_values`.
fn decode(mut fit: GridFit, p: &GrayF32, mirrored: bool, qz: bool) -> Option<GridRead> {
    let g = fit.geom;
    let bits: Vec<u8> = p.data.iter().map(|&p| u8::from(p > 0.5)).collect();
    let conf: Vec<f64> = p
        .data
        .iter()
        .map(|&p| (p as f64 - 0.5).abs() * 2.0)
        .collect();
    let fixed_errors = bits
        .iter()
        .zip(&g.fixed_mask)
        .zip(&g.fixed_val)
        .filter(|((b, m), f)| **m && b != f)
        .count();
    if fixed_pattern_too_damaged(g, fixed_errors) {
        return None;
    }
    let (cw, cconf) = codewords_from_bits(&bits, &conf, g);
    let rs = rs_decode_symbol(&cw, &cconf, g)?;
    let dec = decode_codewords(&rs.data_cw).ok()?;
    if dec.data.is_empty() || !plausible(&dec) {
        return None;
    }
    fit.bits = Some(bits);
    fit.conf = Some(conf);
    fit.fixed_errors = fixed_errors;
    let qz = if qz { "" } else { ", no-quiet-zone" };
    Some(GridRead {
        fit,
        rs,
        dec,
        mirrored,
        note: format!("module classifier{qz}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dm_symbol::build_matrix;

    /// "88482" as a 12x12 (`datamatrix_gen.py`).
    const CODEWORDS: [u8; 12] = [
        0xDA, 0xB2, 0x33, 0x81, 0x73, 0xB6, 0xF1, 0x9A, 0xCE, 0x96, 0x70, 0xCB,
    ];

    fn near_miss(g: &Geometry) -> NearMiss {
        NearMiss {
            score: 3.0,
            size: g.name(),
            qz: true,
            modules_px: 5.0,
            fixed_pattern_module_errors: 0,
            size_agree: 2,
            outside_dark: 0.0,
            measured_size: g.name(),
            pitch_drift_modules: None,
            quad: [[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [0.0, 60.0]],
            axes: None,
            mirrored: false,
            bits: None,
            conf: None,
        }
    }

    /// Dark probabilities of the symbol, read in the fit's frame or in the
    /// anti-transposed one.
    fn probs(g: &Geometry, anti: bool) -> GrayF32 {
        let n = g.rows;
        let m = build_matrix(&CODEWORDS, &g.sym);
        let data = (0..n * n)
            .map(|i| {
                if anti {
                    (n - 1 - i % n) * n + (n - 1 - i / n)
                } else {
                    i
                }
            })
            .map(|k| if m[k] == 1 { 0.9 } else { 0.1 })
            .collect();
        Image::from_vec(n, n, data)
    }

    #[test]
    fn reads_the_fit_frame_and_the_anti_transposed_frame() {
        let g = geometry_of("12x12").unwrap();
        let d = near_miss(g);
        let r = read_probs(&d, g, probs(g, false)).unwrap();
        assert_eq!(r.dec.data, b"88482");
        assert!(!r.mirrored);
        assert_eq!(r.fit.quad, d.quad);
        let r = read_probs(&d, g, probs(g, true)).unwrap();
        assert_eq!(r.dec.data, b"88482");
        assert!(r.mirrored);
        assert_eq!(r.fit.quad[0], d.quad[2]);
    }
}
