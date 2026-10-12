//! Print quality estimates of a read (in the spirit of ISO/IEC 15415 /
//! TR 29158, from a single uncalibrated photo; indicative only): `quality`,
//! `grade` (reader 1612–1677, v2 1833–1878).

use crate::fit::GridFit;
use crate::np::{Order, mean_f32, norm2_f32, percentile_f32, round_to};
use crate::sampler::{Quad, Sampler};
use crate::warp::pitch_drift;
use dm_ecc::RSResult;
use dm_imgproc::Gray8;

/// Letter for `v` against descending limits of A..D, else F.
pub fn grade(v: f64, limits: [f64; 4]) -> char {
    for (g, lim) in "ABCD".chars().zip(limits) {
        if v >= lim {
            return g;
        }
    }
    'F'
}

/// Quality metrics of a decoded fit (Python's `quality` dict).
#[derive(Clone, Debug, PartialEq)]
pub struct Quality {
    pub symbol_contrast_pct: f64,
    pub symbol_contrast_grade: char,
    pub modulation_p5: f64,
    pub modulation_grade: char,
    pub axial_nonuniformity: f64,
    pub axial_nonuniformity_grade: char,
    pub unused_error_correction: f64,
    pub unused_error_correction_grade: char,
    pub fixed_pattern_damage_pct: f64,
    pub fixed_pattern_damage_grade: char,
    pub module_size_px: f64,
    pub symbol_size_px: (i64, i64),
    pub rotation_deg: f64,
    pub skew_deg: f64,
    /// `pitch_drift` of the fit's axes (0 for an even grid).
    pub pitch_drift_modules: f64,
    pub dark_reflectance_mean: f64,
    pub light_reflectance_mean: f64,
    pub corners_px: [[f64; 2]; 4],
    /// Worst letter of the grades.
    pub overall_grade_estimate: char,
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Metrics of `fit` (made on a variant at `scale` / `offset`) measured on
/// the original gray image.
pub fn quality(
    gray_orig: &Gray8,
    fit: &GridFit,
    scale: f64,
    rs: &RSResult,
    offset: (f64, f64),
) -> Quality {
    let g = fit.geom;
    let (s, ox, oy) = (scale as f32, offset.0 as f32, offset.1 as f32);
    let quad: Quad = fit.quad.map(|p| [p[0] / s + ox, p[1] / s + oy]);
    let smp = Sampler::new(gray_orig, 0.22);
    let vals = match &fit.axes {
        Some((u, v)) => smp.sample_grid(&quad, g.rows, g.cols, u, v),
        None => smp.sample(&quad, g.rows, g.cols, 0),
    };
    let bits = fit.bits.as_deref().unwrap_or(&[]);
    let (mut dark, mut light) = (Vec::new(), Vec::new());
    for (&v, &b) in vals.data.iter().zip(bits) {
        let refl = 255.0 - v; // back to reflectance
        if b == 1 {
            dark.push(refl)
        } else {
            light.push(refl)
        }
    }
    let rmax = percentile_f32(&light, 95.0) as f64;
    let rmin = percentile_f32(&dark, 5.0) as f64;
    let sc = (rmax - rmin) / 255.0 * 100.0;
    let gt = ((rmax + rmin) / 2.0) as f32;
    // modulation: worst module margin relative to symbol contrast
    let half = (rmax - rmin) / 2.0;
    let div = if half > 1e-3 { half } else { 1e-3 } as f32;
    let margin: Vec<f32> = vals
        .data
        .iter()
        .map(|&v| ((255.0 - v) - gt).abs() / div)
        .collect();
    let md = percentile_f32(&margin, 5.0) as f64;
    let w_ =
        (norm2_f32(sub(quad[1], quad[0])) + norm2_f32(sub(quad[2], quad[3]))) / 2.0 / g.cols as f32;
    let h_ =
        (norm2_f32(sub(quad[3], quad[0])) + norm2_f32(sub(quad[2], quad[1]))) / 2.0 / g.rows as f32;
    let anu = (w_ - h_).abs() / ((w_ + h_) / 2.0);
    let n_fixed = g.fixed_mask.iter().filter(|&&m| m).count().max(1);
    let fpd = fit.fixed_errors as f64 / n_fixed as f64;
    let top = sub(quad[1], quad[0]);
    let angle = (top[1] as f64).atan2(top[0] as f64).to_degrees();
    let left = sub(quad[3], quad[0]);
    let cos = (top[0] * left[0] + top[1] * left[1]) / (norm2_f32(top) * norm2_f32(left));
    let cos = if cos.is_nan() {
        cos
    } else {
        cos.clamp(-1.0, 1.0)
    };
    let skew = (cos as f64).acos().to_degrees() - 90.0;
    let r1 = |v: f64| round_to(v, 1);
    let mut q = Quality {
        symbol_contrast_pct: r1(sc),
        symbol_contrast_grade: grade(sc, [70.0, 55.0, 40.0, 20.0]),
        modulation_p5: round_to(md, 2),
        modulation_grade: grade(md, [0.5, 0.4, 0.3, 0.2]),
        axial_nonuniformity: round_to(anu as f64, 3),
        // float32 against the limits cast to float32
        axial_nonuniformity_grade: grade_f32(-anu, [-0.06, -0.08, -0.10, -0.12]),
        unused_error_correction: round_to(rs.uec, 2),
        unused_error_correction_grade: grade(rs.uec, [0.62, 0.5, 0.37, 0.25]),
        fixed_pattern_damage_pct: r1(fpd * 100.0),
        fixed_pattern_damage_grade: grade(-fpd, [0.0, -0.02, -0.05, -0.10]),
        module_size_px: round_to((w_ + h_) as f64 / 2.0, 2),
        symbol_size_px: (
            (w_ * g.cols as f32).round_ties_even() as i64,
            (h_ * g.rows as f32).round_ties_even() as i64,
        ),
        rotation_deg: r1(angle),
        skew_deg: r1(skew),
        pitch_drift_modules: round_to(pitch_drift(fit.axes.as_ref()), 2),
        dark_reflectance_mean: r1(mean_f32(&dark, Order::Pairwise) as f64),
        light_reflectance_mean: r1(mean_f32(&light, Order::Pairwise) as f64),
        corners_px: quad.map(|p| [r1(p[0] as f64), r1(p[1] as f64)]),
        overall_grade_estimate: 'A',
    };
    q.overall_grade_estimate = [
        q.symbol_contrast_grade,
        q.modulation_grade,
        q.axial_nonuniformity_grade,
        q.unused_error_correction_grade,
        q.fixed_pattern_damage_grade,
    ]
    .into_iter()
    .max()
    .unwrap_or('F');
    q
}

fn grade_f32(v: f32, limits: [f64; 4]) -> char {
    for (g, lim) in "ABCD".chars().zip(limits) {
        if v >= lim as f32 {
            return g;
        }
    }
    'F'
}
