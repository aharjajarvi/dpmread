//! Module classifier hook (cargo feature `ml`): when every grid job has
//! failed, the best failed fits are classified again with the trained
//! module classifier (`dm-ml`) in place of the threshold of `classify`, and
//! RS-decoded with the same soft-erasure rule.

use std::time::Instant;

use dm_codec::decode_codewords;
use dm_ecc::rs_decode_symbol;
use dm_grid::{
    FitMap, GridFit, GridRead, NearMiss, codewords_from_bits, fixed_pattern_too_damaged,
    near_miss_key, plausible, symbol_input,
};
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

/// A fit turned about its anti-diagonal: module (r, c) becomes
/// (cols - 1 - c, rows - 1 - r) of a square symbol. The finder L and the
/// timing pattern map onto themselves, so a fit read in this frame has a
/// perfect fixed pattern but mirrored data; the corner order becomes
/// br, tr, tl, bl and the mirrored flag flips.
fn anti_transpose(
    d: &NearMiss,
    n: usize,
    bits: &[u8],
    conf: &[f64],
) -> (NearMiss, Vec<u8>, Vec<f64>) {
    let at = |r: usize, c: usize| (n - 1 - c) * n + (n - 1 - r);
    let idx: Vec<usize> = (0..n * n).map(|i| at(i / n, i % n)).collect();
    let mut t = d.clone();
    let q = d.quad;
    t.quad = [q[2], q[1], q[0], q[3]];
    t.axes = d.axes.as_ref().map(|(u, v)| {
        let nu = (0..n).map(|i| n as f64 - v[n - 1 - i]).collect();
        let nv = (0..n).map(|j| n as f64 - u[n - 1 - j]).collect();
        (nu, nv)
    });
    t.mirrored = !d.mirrored;
    (
        t,
        idx.iter().map(|&k| bits[k]).collect(),
        idx.iter().map(|&k| conf[k]).collect(),
    )
}

fn read_fit(v: &Variant, d: &NearMiss, g: &'static Geometry) -> Option<GridRead> {
    let map = FitMap::new(&d.quad, g.rows, g.cols, d.axes.as_ref())?;
    let input = symbol_input(&v.img, &map, g.rows, g.cols);
    let p = Classifier::embedded().p_dark(&input, g.rows, g.cols)?;
    let bits: Vec<u8> = p.iter().map(|&p| u8::from(p > 0.5)).collect();
    let conf: Vec<f64> = p.iter().map(|&p| (p as f64 - 0.5).abs() * 2.0).collect();
    if let Some(r) = decode(d, g, bits.clone(), conf.clone()) {
        return Some(r);
    }
    if g.rows != g.cols {
        return None;
    }
    let (t, bits, conf) = anti_transpose(d, g.rows, &bits, &conf);
    decode(&t, g, bits, conf)
}

fn decode(d: &NearMiss, g: &'static Geometry, bits: Vec<u8>, conf: Vec<f64>) -> Option<GridRead> {
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
    let mut fit = GridFit::new(g, d.quad, d.score);
    fit.axes = d.axes.clone();
    fit.bits = Some(bits);
    fit.conf = Some(conf);
    fit.fixed_errors = fixed_errors;
    let qz = if d.qz { "" } else { ", no-quiet-zone" };
    Some(GridRead {
        fit,
        rs,
        dec,
        mirrored: d.mirrored,
        note: format!("module classifier{qz}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anti_transpose_keeps_the_fixed_pattern_and_undoes_itself() {
        let g = geometry_of("12x12").unwrap();
        let n = g.rows;
        let d = NearMiss {
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
            axes: Some((
                (0..n).map(|c| c as f64 + 0.4).collect(),
                (0..n).map(|r| r as f64 + 0.6).collect(),
            )),
            mirrored: false,
            bits: None,
            conf: None,
        };
        let conf: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let (t, bits, c2) = anti_transpose(&d, n, &g.fixed_val, &conf);
        let fixed_ok = |b: &[u8]| (0..n * n).all(|i| !g.fixed_mask[i] || b[i] == g.fixed_val[i]);
        assert!(fixed_ok(&bits));
        assert!(t.mirrored);
        assert_eq!(t.quad[0], d.quad[2]);
        // module (r, c) moves to (n - 1 - c, n - 1 - r)
        assert_eq!(c2[0], conf[n * n - 1]);
        assert_eq!(c2[1], conf[(n - 2) * n + n - 1]);
        let (u, v) = t.axes.as_ref().unwrap();
        assert!((u[0] - 0.4).abs() < 1e-9 && (v[0] - 0.6).abs() < 1e-9);
        let (back, _, c3) = anti_transpose(&t, n, &bits, &c2);
        assert_eq!(back.quad, d.quad);
        let (bu, bv) = back.axes.as_ref().unwrap();
        let (du, dv) = d.axes.as_ref().unwrap();
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9);
        assert!(close(bu, du) && close(bv, dv));
        assert_eq!(c3, conf);
    }
}
