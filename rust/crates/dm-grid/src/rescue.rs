//! `rescue_fit` (reader 1451–1491, v2 1604–1643): the grid fits but the data does not
//! decode; re-sample the same grid with other images, module-sized
//! smoothing, sampling footprints and thresholds, then vote.

use crate::deadline::Deadline;
use crate::fit::{GridFit, Mode, try_fit_values, zscore};
use crate::refine::{module_px, tracked_axes};
use crate::sampler::Sampler;
use dm_codec::Decoded;
use dm_ecc::RSResult;
use dm_imgproc::{Gray8, GrayF32, Image, gamma, gaussian_blur_u8};

/// Gammas of the multi-exposure bracket added after the given images.
pub const RESCUE_GAMMAS: [f64; 2] = [0.6, 1.5];
const RESCUE_BLURS: [f32; 2] = [0.0, 0.25];
const RESCUE_RADII: [f64; 3] = [0.22, 0.1, 0.32];
const MODES: [Mode; 2] = [Mode::Plane, Mode::Local];

/// Python `{:g}` of the few constants used in the notes.
fn fmt_g(v: f64) -> String {
    format!("{v}")
}

/// Try every (image, blur, footprint, even / tracked grid, threshold)
/// combination on `fit`'s grid, then the average of all their z-scores.
/// `images[0]` is the image the fit was made on. Returns the decode and a
/// note on how it was sampled; `None` when nothing decodes or the deadline
/// passes.
pub fn rescue_fit(
    fit: &mut GridFit,
    images: &[(String, &Gray8)],
    deadline: &Deadline,
) -> Option<(RSResult, Decoded, String)> {
    let _span = crate::timing::Span::new(crate::timing::Stage::Rescue);
    let g = fit.geom;
    let md = module_px(&fit.quad, g);
    // multi-exposure bracket (HDR-style): the first image at two gammas
    let gammas: Vec<(String, Gray8)> = RESCUE_GAMMAS
        .iter()
        .map(|&gm| {
            (
                format!("{}, gamma {}", images[0].0, fmt_g(gm)),
                gamma(images[0].1, gm),
            )
        })
        .collect();
    let imgs = images
        .iter()
        .map(|(n, i)| (n, *i))
        .chain(gammas.iter().map(|(n, i)| (n, i)));
    let mut stack: Vec<GrayF32> = Vec::new();
    for (name, img) in imgs {
        for sig in RESCUE_BLURS {
            let blurred;
            let src = if sig != 0.0 {
                blurred = gaussian_blur_u8(img, (sig * md) as f64);
                &blurred
            } else {
                img
            };
            let base = Sampler::new(src, 0.0);
            let axes = match &fit.axes {
                Some(a) => a.clone(),
                None => tracked_axes(&base, &fit.quad, g),
            };
            // a warped fit is only sampled on its own axes
            let trs: &[bool] = if fit.axes.is_some() {
                &[true]
            } else {
                &[false, true]
            };
            for rad in RESCUE_RADII {
                let smp = base.with_radius(rad);
                for &tr in trs {
                    if deadline.expired() {
                        return None;
                    }
                    let vals = if tr {
                        smp.sample_grid(&fit.quad, g.rows, g.cols, &axes.0, &axes.1)
                    } else {
                        smp.sample(&fit.quad, g.rows, g.cols, 0)
                    };
                    stack.push(zscore(&vals, g));
                    for mode in MODES {
                        if let Some((rs, dec)) = try_fit_values(fit, mode, vals.clone()) {
                            let how = format!(
                                "{name}, blur {} module, footprint {}, {} threshold{}",
                                fmt_g(sig as f64),
                                fmt_g(rad),
                                mode.name(),
                                if tr && fit.axes.is_none() {
                                    ", timing-tracked grid"
                                } else {
                                    ""
                                }
                            );
                            return Some((rs, dec, how));
                        }
                    }
                }
            }
        }
    }
    if stack.is_empty() {
        return None;
    }
    // np.mean(stack, 0): float32, summed in stack order
    let n = stack[0].data.len();
    let mut acc = vec![0f32; n];
    for z in &stack {
        for (a, &v) in acc.iter_mut().zip(&z.data) {
            *a += v;
        }
    }
    let k = stack.len() as f32;
    let avg = Image::from_vec(g.cols, g.rows, acc.iter().map(|a| a / k).collect());
    for mode in MODES {
        if let Some((rs, dec)) = try_fit_values(fit, mode, avg.clone()) {
            let how = format!(
                "average of {} samplings, {} threshold",
                stack.len(),
                mode.name()
            );
            return Some((rs, dec, how));
        }
    }
    None
}
