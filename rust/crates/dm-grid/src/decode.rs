//! The grid search: `grid_decode`, `_grid_pass` (reader 1501–1611) and
//! v2's `_warp_pass` (v2 1767–1817).

use crate::deadline::Deadline;
use crate::fit::{GridFit, Mode, try_fit, try_fit_values};
use crate::nearmiss::{
    NearMiss, OUTSIDE_BUSY, interior_modules, near_miss_key, outside_dark, round2, size_agree,
};
use crate::np::norm2_f32;
use crate::quads::{candidate_quads_until, expand_quad, side_combos, snap_quad};
use crate::refine::refine_quad;
use crate::regions::region_values;
use crate::rescue::rescue_fit;
use crate::sampler::{Quad, Sampler, orientations, score_quads};
use crate::timing::{Span, Stage, timed};
use crate::warp::{WARP_MIN_PX, pitch_drift, warp_fit};
use dm_codec::Decoded;
use dm_ecc::RSResult;
use dm_imgproc::Gray8;
use dm_symbol::{Geometry, Symbol, all_symbols, geometry};

/// Finder / timing Fisher score above which the grid is trusted (rescue).
pub const STRONG_FIT: f64 = 3.5;
/// Same bar for the ring-less template (scores are not comparable).
pub const NO_QZ_STRONG_FACTOR: f64 = 1.3;
/// Rough quads from `candidate_quads` per image.
pub const MAX_QUADS: usize = 10;

/// A grid read: the fit, RS result, decoded data, mirrored flag and how it
/// was sampled.
#[derive(Clone, Debug)]
pub struct GridRead {
    pub fit: GridFit,
    pub rs: RSResult,
    pub dec: Decoded,
    pub mirrored: bool,
    pub note: String,
}

fn side(a: [f32; 2], b: [f32; 2]) -> f32 {
    norm2_f32([b[0] - a[0], b[1] - a[1]])
}

/// Locate, fit, sample and decode. `extra_quads` (zxing seeds) are tried
/// before the candidates of `candidate_quads`. `qz_modes`: `true` = the
/// template with the quiet-zone ring (always first), `false` = ring-less
/// retry with side combinations. `symbols`: expected sizes only (`None`:
/// all). `max_aspect`: blob aspect limit of `candidate_quads`. `diag`
/// receives the best failed fit.
#[allow(clippy::too_many_arguments)]
pub fn grid_decode(
    gray: &Gray8,
    deadline: &Deadline,
    extra_quads: &[Quad],
    alt_images: &[(String, &Gray8)],
    min_area: f64,
    max_aspect: f64,
    qz_modes: &[bool],
    symbols: Option<&[Symbol]>,
    mut diag: Option<&mut Option<NearMiss>>,
) -> Option<GridRead> {
    let setup = Span::new(Stage::Setup);
    let sampler = Sampler::new(gray, 0.22);
    // an expired deadline ends the job here as it would in the first
    // `grid_pass` check: same result, without finishing detection (CORE-111)
    let mut rough: Vec<Quad> = extra_quads.to_vec();
    rough.extend(candidate_quads_until(
        gray, MAX_QUADS, min_area, max_aspect, deadline,
    )?);
    let mut quads = Vec::with_capacity(2 * rough.len());
    for q in &rough {
        if deadline.expired() {
            return None;
        }
        quads.push(snap_quad(gray, q, 2, false));
        quads.push(*q);
    }
    drop(setup);
    if qz_modes.contains(&true) {
        let tier = Span::new(Stage::QzPass);
        let mut r = grid_pass(
            gray,
            &sampler,
            &quads,
            deadline,
            alt_images,
            true,
            symbols,
            diag.as_deref_mut(),
        );
        drop(tier);
        if r.is_none() && !deadline.expired() {
            let _tier = Span::new(Stage::Warp);
            r = warp_pass(
                gray,
                &sampler,
                &quads,
                deadline,
                alt_images,
                symbols,
                diag.as_deref_mut(),
            );
        }
        if r.is_some() || deadline.expired() || !qz_modes.contains(&false) {
            return r;
        }
    }
    let setup = Span::new(Stage::Setup);
    let mut quads2 = Vec::new();
    for (q, sq) in rough.iter().zip(quads.iter().step_by(2)) {
        if deadline.expired() {
            return None;
        }
        quads2.extend(side_combos(sq, &snap_quad(gray, sq, 2, true), 3.0));
        quads2.push(snap_quad(gray, q, 2, true));
        quads2.push(*q);
    }
    drop(setup);
    let _tier = Span::new(Stage::NoQzPass);
    grid_pass(
        gray, &sampler, &quads2, deadline, alt_images, false, symbols, diag,
    )
}

#[allow(clippy::too_many_arguments)]
fn grid_pass(
    gray: &Gray8,
    sampler: &Sampler,
    quads: &[Quad],
    deadline: &Deadline,
    alt_images: &[(String, &Gray8)],
    qz: bool,
    symbols: Option<&[Symbol]>,
    mut diag: Option<&mut Option<NearMiss>>,
) -> Option<GridRead> {
    let note = if qz { "" } else { ", no-quiet-zone" };
    // ring-less scores run higher on noise: demand more before the rescue
    let strong = if qz {
        STRONG_FIT
    } else {
        STRONG_FIT * NO_QZ_STRONG_FACTOR
    };
    let sym_list = symbol_list(symbols);
    for quad in quads {
        if deadline.expired() {
            return None;
        }
        if side(quad[0], quad[1]).min(side(quad[1], quad[2])) < 10.0 {
            continue;
        }
        // sym -> [(quad, mirror)], in first-seen order
        let mut by_sym: Vec<(&'static Symbol, Vec<(Quad, bool)>)> = Vec::new();
        for (oq, _, mirror) in orientations(quad) {
            let w_ = side(oq[0], oq[1]);
            let h_ = side(oq[0], oq[3]);
            for &sym in &sym_list {
                let (mw, mh) = (w_ / sym.cols as f32, h_ / sym.rows as f32);
                let ratio = mw / mh;
                if !(0.7 < ratio && ratio < 1.4) {
                    continue;
                }
                if mw.min(mh) < 1.6 {
                    continue; // modules under ~2 px
                }
                let slot = match by_sym.iter().position(|(s, _)| *s == sym) {
                    Some(i) => i,
                    None => {
                        by_sym.push((sym, Vec::new()));
                        by_sym.len() - 1
                    }
                };
                // without a quiet zone the side fit may stop at the finder's
                // inner edge: also try the quad grown by one module
                by_sym[slot].1.push((oq, mirror));
                if !qz {
                    by_sym[slot]
                        .1
                        .push((expand_quad(&oq, geometry(sym), 1.0), mirror));
                }
            }
        }
        let mut scored: Vec<(f64, &'static Geometry, Quad, bool)> = Vec::new();
        for (sym, cands) in &by_sym {
            if deadline.expired() {
                return None;
            }
            let gm = geometry(sym);
            let qs: Vec<Quad> = cands.iter().map(|c| c.0).collect();
            let s = timed(Stage::Score, || score_quads(sampler, &qs, gm, qz));
            scored.extend(
                s.iter()
                    .zip(cands)
                    .map(|(&sc, &(cq, m))| (sc as f64, gm, cq, m)),
            );
        }
        // the right size often scores low before alignment: coarse-align the
        // 16 best candidates first, then fully refine and decode the top 4
        sort_desc(&mut scored);
        let mut coarse = Vec::with_capacity(16);
        for &(_, gm, oq, mirror) in scored.iter().take(16) {
            if deadline.expired() {
                return None;
            }
            let (cq, cs) = timed(Stage::Refine, || {
                refine_quad(sampler, &oq, gm, 2, &[1.0, 0.5], qz)
            });
            coarse.push((cs, gm, cq, mirror));
        }
        sort_desc(&mut coarse);
        for &(_, gm, oq, mirror) in coarse.iter().take(4) {
            if deadline.expired() {
                return None;
            }
            let (rq, rs_) = timed(Stage::Refine, || {
                refine_quad(sampler, &oq, gm, 4, &[0.5, 0.25, 0.12], qz)
            });
            let mut fit = GridFit::new(gm, rq, rs_);
            for (mode, tr) in [
                (Mode::Plane, false),
                (Mode::Local, false),
                (Mode::Plane, true),
                (Mode::Local, true),
            ] {
                if let Some((rs, dec)) = try_fit(sampler, &mut fit, mode, tr) {
                    let how = if (mode, tr) == (Mode::Plane, false) {
                        "direct".to_string()
                    } else {
                        format!(
                            "{} threshold{}",
                            mode.name(),
                            if tr { ", timing-tracked grid" } else { "" }
                        )
                    };
                    return Some(GridRead {
                        fit,
                        rs,
                        dec,
                        mirrored: mirror,
                        note: how + note,
                    });
                }
            }
            if let Some(d) = diag.as_deref_mut() {
                record_near_miss(d, sampler, &fit, rs_, qz);
            }
            if let Some(r) = region_read(sampler, &fit, deadline, mirror, note) {
                return Some(r);
            }
            if rs_ >= strong {
                let mut imgs: Vec<(String, &Gray8)> = vec![("same image".to_string(), gray)];
                imgs.extend(alt_images.iter().map(|(n, i)| (n.clone(), *i)));
                if let Some((rs, dec, how)) = rescue_fit(&mut fit, &imgs, deadline) {
                    return Some(GridRead {
                        fit,
                        rs,
                        dec,
                        mirrored: mirror,
                        note: how + note,
                    });
                }
            }
        }
    }
    None
}

/// The symbols to try: all, or the expected sizes only.
fn symbol_list(symbols: Option<&[Symbol]>) -> Vec<&'static Symbol> {
    all_symbols()
        .filter(|s| {
            symbols.is_none_or(|want| {
                want.is_empty() || want.iter().any(|t| (t.rows, t.cols) == (s.rows, s.cols))
            })
        })
        .collect()
}

/// Last grid tier (v2 `_warp_pass`): every size / orientation whose timing
/// dash counts match both timing edges is sampled on a warped grid
/// (`global_track`).
fn warp_pass(
    gray: &Gray8,
    sampler: &Sampler,
    quads: &[Quad],
    deadline: &Deadline,
    alt_images: &[(String, &Gray8)],
    symbols: Option<&[Symbol]>,
    mut diag: Option<&mut Option<NearMiss>>,
) -> Option<GridRead> {
    let sym_list = symbol_list(symbols);
    for quad in quads {
        if deadline.expired() {
            return None;
        }
        if side(quad[0], quad[1]).min(side(quad[1], quad[2])) < 10.0 {
            continue;
        }
        for (oq, _, mirror) in orientations(quad) {
            let w_ = side(oq[0], oq[1]);
            let h_ = side(oq[0], oq[3]);
            for &sym in &sym_list {
                let (mw, mh) = (w_ / sym.cols as f32, h_ / sym.rows as f32);
                let ratio = mw / mh;
                if !(0.7 < ratio && ratio < 1.4) {
                    continue;
                }
                if mw.min(mh) < WARP_MIN_PX {
                    continue;
                }
                if deadline.expired() {
                    return None;
                }
                let gm = geometry(sym);
                let Some(mut fit) = timed(Stage::WarpFit, || warp_fit(sampler, &oq, gm, deadline))
                else {
                    continue;
                };
                let drift = pitch_drift(fit.axes.as_ref());
                let note = format!("warped grid (global timing tracking, drift {drift:.1} module)");
                let vals = {
                    let (u, v) = fit.axes.as_ref()?;
                    sampler.sample_grid(&fit.quad, gm.rows, gm.cols, u, v)
                };
                for mode in [Mode::Plane, Mode::Local] {
                    if let Some((rs, dec)) = try_fit_values(&mut fit, mode, vals.clone()) {
                        let note = format!("{} threshold, {note}", mode.name());
                        return Some(GridRead {
                            fit,
                            rs,
                            dec,
                            mirrored: mirror,
                            note,
                        });
                    }
                }
                if let Some(d) = diag.as_deref_mut()
                    && (2, true, fit.contrast) > near_miss_key(d.as_ref())
                {
                    // both timing dash counts matched: the size is confirmed
                    let rq = &fit.quad;
                    let (top, right) = (side(rq[0], rq[1]), side(rq[1], rq[2]));
                    *d = Some(NearMiss {
                        score: fit.contrast,
                        size: gm.name(),
                        qz: true,
                        modules_px: ((top / gm.cols as f32 + right / gm.rows as f32) / 2.0) as f64,
                        fixed_pattern_module_errors: fit.fixed_errors,
                        size_agree: 2,
                        outside_dark: 0.0,
                        measured_size: gm.name(),
                        pitch_drift_modules: Some(round2(drift)),
                    });
                }
                if let Some(r) = region_read(sampler, &fit, deadline, mirror, &format!(", {note}"))
                {
                    return Some(r);
                }
                if fit.contrast >= STRONG_FIT {
                    let mut imgs: Vec<(String, &Gray8)> = vec![("same image".to_string(), gray)];
                    imgs.extend(alt_images.iter().map(|(n, i)| (n.clone(), *i)));
                    if let Some((rs, dec, how)) = rescue_fit(&mut fit, &imgs, deadline) {
                        return Some(GridRead {
                            fit,
                            rs,
                            dec,
                            mirrored: mirror,
                            note: format!("{how}, {note}"),
                        });
                    }
                }
            }
        }
    }
    None
}

/// One more candidate before the rescue: every data region re-fitted on
/// its own alignment frame (CORE-103). `None` for single-region symbols.
fn region_read(
    sampler: &Sampler,
    fit: &GridFit,
    deadline: &Deadline,
    mirrored: bool,
    note: &str,
) -> Option<GridRead> {
    let vals = timed(Stage::Refine, || {
        region_values(sampler, &fit.quad, fit.axes.as_ref(), fit.geom, deadline)
    })?;
    let mut rf = fit.clone();
    for mode in [Mode::Plane, Mode::Local] {
        if let Some((rs, dec)) = try_fit_values(&mut rf, mode, vals.clone()) {
            return Some(GridRead {
                fit: rf,
                rs,
                dec,
                mirrored,
                note: format!("region anchors, {} threshold{note}", mode.name()),
            });
        }
    }
    None
}

/// Stable sort by descending score (Python `sort(key=lambda t: -t[0])`).
fn sort_desc<T>(v: &mut [(f64, &'static Geometry, Quad, T)]) {
    v.sort_by(|a, b| {
        (-a.0)
            .partial_cmp(&-b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Near-miss bookkeeping (the usability layer of industrial readers:
/// report what the best failed attempt looked like).
fn record_near_miss(
    diag: &mut Option<NearMiss>,
    sampler: &Sampler,
    fit: &GridFit,
    score: f64,
    qz: bool,
) {
    let (gm, rq) = (fit.geom, &fit.quad);
    let est = interior_modules(sampler, rq);
    let agree = size_agree(est, gm);
    let (top, right) = (side(rq[0], rq[1]), side(rq[1], rq[2]));
    // px/module from the measured count when it contradicts the size
    let modpx = if agree > 0 {
        (top / gm.cols as f32 + right / gm.rows as f32) / 2.0
    } else {
        (top / est.0 as f32 + right / est.1 as f32) / 2.0
    };
    let out = outside_dark(sampler, rq, gm, fit.threshold);
    if (agree, out <= OUTSIDE_BUSY, score) > near_miss_key(diag.as_ref()) {
        *diag = Some(NearMiss {
            score,
            size: gm.name(),
            qz,
            modules_px: modpx as f64,
            fixed_pattern_module_errors: fit.fixed_errors,
            size_agree: agree,
            outside_dark: round2(out),
            measured_size: format!(
                "{}x{}",
                est.1.round_ties_even() as i64,
                est.0.round_ties_even() as i64
            ),
            pitch_drift_modules: None,
        });
    }
}
