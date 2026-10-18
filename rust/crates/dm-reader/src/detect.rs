//! Detector first try (cargo feature `ml`), unless one zxing call on the
//! gray image already reads (`read::quick_zxing`): the trained symbol detector
//! (`dm_ml::Detector`) gives up to `DETECTIONS` symbol quads with size
//! candidates and polarity; each quad goes to `grid_decode_seeds` on the
//! grid variants of a crop around it, quiet-zone template first. The first
//! read ends the image; otherwise the classic search runs as before.

use std::time::{Duration, Instant};

use dm_grid::{Deadline, GridRead, grid_decode_seeds};
use dm_imgproc::{Bgr8, Gray8};
use dm_ml::Detector;

use crate::read::{Polarity, ReadOptions, parse_sizes};
use crate::runner::Runner;
use crate::variants::{Variant, grid_variants};

/// Detections tried per image, best finder score first.
pub const DETECTIONS: usize = 3;
/// Finder score a detection needs (sigmoid of the centre logit).
pub const MIN_SCORE: f32 = 0.3;
/// Time the first try may take; the classic search gets the rest.
pub const FIRST_TRY_S: f64 = 3.0;
/// Polarity probability above which only that polarity is tried.
pub const SURE_POLARITY: f32 = 0.9;

/// Copy of `[x0, x0 + w) x [y0, y0 + h)`.
fn crop_bgr(bgr: &Bgr8, x0: usize, y0: usize, w: usize, h: usize) -> Bgr8 {
    let mut data = Vec::with_capacity(w * h * 3);
    for y in y0..y0 + h {
        let row = (y * bgr.width + x0) * 3;
        data.extend_from_slice(&bgr.data[row..row + w * 3]);
    }
    Bgr8::from_vec(w, h, data)
}

/// The first try. Returns the variant that read (named `detector > …`,
/// offsets in original pixels) with its read, and the number of
/// `grid_decode_seeds` calls.
pub(crate) fn first_try(
    bgr: &Bgr8,
    gray: &Gray8,
    opts: &ReadOptions,
    runner: &Runner,
    deadline: Instant,
) -> (Option<(Variant, GridRead)>, usize) {
    let dl = Deadline {
        at: Some(deadline.min(Instant::now() + Duration::from_secs_f64(FIRST_TRY_S))),
        abort: Some(runner.abort_flag()),
    };
    let Some(dets) = Detector::embedded().detect(gray, MIN_SCORE, DETECTIONS) else {
        return (None, 0);
    };
    let mut tries = 0;
    for d in &dets {
        if dl.expired() {
            break;
        }
        let (invert, both) = match opts.polarity {
            Polarity::Dark => (false, false),
            Polarity::Light => (true, false),
            Polarity::Any if d.p_dark >= SURE_POLARITY => (false, false),
            Polarity::Any if d.p_dark <= 1.0 - SURE_POLARITY => (true, false),
            Polarity::Any => (false, true),
        };
        let mut sizes: Vec<String> = d.sizes.iter().map(|s| s.to_string()).collect();
        if let Some(exp) = &opts.expect {
            let names: Vec<String> = exp.iter().map(|s| s.name()).collect();
            sizes.retain(|s| names.contains(s));
            if sizes.is_empty() {
                sizes = names;
            }
        }
        let symbols = parse_sizes(&sizes);
        // crop: twice the quad's bounding square around its centre
        let xs = d.quad.map(|p| p[0]);
        let ys = d.quad.map(|p| p[1]);
        let (minx, maxx) = (
            xs.iter().copied().fold(f32::MAX, f32::min),
            xs.iter().copied().fold(f32::MIN, f32::max),
        );
        let (miny, maxy) = (
            ys.iter().copied().fold(f32::MAX, f32::min),
            ys.iter().copied().fold(f32::MIN, f32::max),
        );
        let side = (maxx - minx).max(maxy - miny);
        let (cx, cy) = ((minx + maxx) / 2.0, (miny + maxy) / 2.0);
        let clamp = |v: f32, hi: usize| v.round().clamp(0.0, hi as f32) as usize;
        let (x0, y0) = (clamp(cx - side, bgr.width), clamp(cy - side, bgr.height));
        let (x1, y1) = (clamp(cx + side, bgr.width), clamp(cy + side, bgr.height));
        if x1 <= x0 + 8 || y1 <= y0 + 8 {
            continue;
        }
        let crop = crop_bgr(bgr, x0, y0, x1 - x0, y1 - y0);
        let gvars = runner.install(|| grid_variants(&crop, invert, both));
        for gv in gvars {
            if dl.expired() {
                break;
            }
            let s = gv.scale as f32;
            let q = d
                .quad
                .map(|[x, y]| [(x - x0 as f32) * s, (y - y0 as f32) * s]);
            tries += 1;
            if let Some(r) =
                grid_decode_seeds(&gv.img, &dl, &[q], &[true, false], symbols.as_deref())
            {
                let v = Variant {
                    name: format!("detector > {}", gv.name),
                    img: gv.img,
                    scale: gv.scale,
                    ox: x0 as f64,
                    oy: y0 as f64,
                };
                return (Some((v, r)), tries);
            }
        }
    }
    (None, tries)
}
