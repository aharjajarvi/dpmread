//! Detector first try (cargo feature `ml`), unless one zxing call on the
//! gray image already reads (`read::quick_zxing`): the trained symbol detector
//! (`dm_ml::Detector`) gives up to `DETECTIONS` symbol quads with size
//! candidates and polarity. The main symbol (nearest the image centre
//! among the large ones) and the detections centred inside it go to `grid_decode_seeds` on the grid variants of a crop
//! around each, quiet-zone template first; other symbols are not tried,
//! so a photo of one part never reports a neighbouring part's code from
//! here. The first read ends the image; otherwise the classic search runs
//! as before.

use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use dm_grid::{Deadline, GridRead, grid_decode_seeds};
use dm_imgproc::{Bgr8, Gray8};
use dm_ml::Detector;
use dm_symbol::Symbol;

use crate::read::{Polarity, ReadOptions, parse_sizes};
use crate::runner::Runner;
use crate::variants::{GridMember, Variant, grid_member, grid_members};

/// Detections tried per image, best finder score first.
pub const DETECTIONS: usize = 3;
/// Finder score a detection needs (sigmoid of the centre logit).
pub const MIN_SCORE: f32 = 0.3;
/// Time the first try may take; the classic search gets the rest.
pub const FIRST_TRY_S: f64 = 3.0;
/// Smallest area, as a share of the largest detection's, a detection needs
/// to be the main symbol.
pub const MAIN_AREA: f32 = 0.25;
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

/// Load the models and run the finder once, so the first read does not
/// pay for it (the load alone takes ~0.1 s on a phone).
pub fn preload() {
    let _ = Detector::embedded().detect(&Gray8::new(64, 48), MIN_SCORE, 1);
}

/// What the first try did.
pub(crate) struct FirstTry {
    /// The variant that read (named `detector > …`, offsets in original
    /// pixels) with its read.
    pub hit: Option<(Variant, GridRead)>,
    /// `grid_decode_seeds` calls.
    pub tries: usize,
    /// Seconds in the models and in cutting the crops (their grid variants
    /// are built in the grid jobs).
    pub models: f64,
    pub crops: f64,
}

/// The first try.
pub(crate) fn first_try(
    bgr: &Bgr8,
    gray: &Gray8,
    opts: &ReadOptions,
    runner: &Runner,
    deadline: Instant,
) -> FirstTry {
    let dl = Deadline {
        at: Some(deadline.min(Instant::now() + Duration::from_secs_f64(FIRST_TRY_S))),
        abort: Some(runner.abort_flag()),
    };
    let mut out = FirstTry {
        hit: None,
        tries: 0,
        models: 0.0,
        crops: 0.0,
    };
    let t = Instant::now();
    let dets = Detector::embedded().detect(gray, MIN_SCORE, DETECTIONS);
    out.models = t.elapsed().as_secs_f64();
    let Some(mut dets) = dets else {
        return out;
    };
    // only the main symbol: in a photo of one part it is that part's code,
    // and a read of another would report a neighbouring part's code. Main =
    // the one nearest the image centre among those of at least
    // `MAIN_AREA` x the largest area; detections centred inside it are the
    // same symbol
    let largest = dets.iter().map(|d| area(&d.quad)).fold(0.0, f32::max);
    let (w, h) = (gray.width as f32, gray.height as f32);
    let off_centre = |q: &[[f32; 2]; 4]| {
        let (cx, cy) = (
            q.iter().map(|p| p[0]).sum::<f32>() / 4.0,
            q.iter().map(|p| p[1]).sum::<f32>() / 4.0,
        );
        ((cx / w - 0.5).powi(2) + (cy / h - 0.5).powi(2)).sqrt()
    };
    let main = dets
        .iter()
        .filter(|d| area(&d.quad) >= MAIN_AREA * largest)
        .min_by(|a, b| off_centre(&a.quad).total_cmp(&off_centre(&b.quad)))
        .map(|d| d.quad);
    dets.sort_by_key(|d| !main.is_some_and(|m| m == d.quad));
    // one job per (detection, grid variant member), in that order; each
    // job builds its member's variants of the crop, so the first job (the
    // variant that reads most often) starts at once. The first read in job
    // order wins
    let mut crops: Vec<Bgr8> = Vec::new();
    let mut jobs: Vec<Job> = Vec::new();
    let t = Instant::now();
    for d in &dets {
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
        if !main.is_some_and(|m| inside(&m, (cx, cy))) {
            continue;
        }
        crops.push(crop_bgr(bgr, x0, y0, x1 - x0, y1 - y0));
        let quad = d.quad.map(|[x, y]| [x - x0 as f32, y - y0 as f32]);
        for member in grid_members(invert, both) {
            jobs.push(Job {
                crop: crops.len() - 1,
                invert,
                member,
                quad,
                ox: x0 as f64,
                oy: y0 as f64,
                symbols: symbols.clone(),
            });
        }
    }
    out.crops = t.elapsed().as_secs_f64();
    // the first job alone (it reads most often; on a phone its siblings
    // slow it down by more than they gain), then the rest in parallel
    let attempt = |j: &Job| {
        let mut calls = 0;
        for gv in grid_member(&crops[j.crop], j.invert, j.member) {
            if dl.at.is_some_and(|at| Instant::now() > at) {
                break;
            }
            let s = gv.scale as f32;
            let quad = j.quad.map(|[x, y]| [x * s, y * s]);
            calls += 1;
            let r = grid_decode_seeds(&gv.img, &dl, &[quad], &[true, false], j.symbols.as_deref());
            if let Some(r) = r {
                let v = Variant {
                    name: format!("detector > {}", gv.name),
                    img: gv.img,
                    scale: gv.scale,
                    ox: j.ox,
                    oy: j.oy,
                };
                return (Some((v, r)), calls);
            }
        }
        (None, calls)
    };
    let (mut tries, mut hit) = (0, None);
    let split = jobs.len().min(1);
    for part in [&jobs[..split], &jobs[split..]] {
        if hit.is_some() {
            break;
        }
        runner.run(part, dl.at, attempt, |_, (r, calls)| {
            tries += calls;
            match r {
                Some(r) => {
                    hit = Some(r);
                    ControlFlow::Break(())
                }
                None => ControlFlow::Continue(()),
            }
        });
    }
    out.hit = hit;
    out.tries = tries;
    out
}

/// One first-try job: a grid variant member of a crop (`crops` index,
/// offsets in original pixels), the detection's quad in crop pixels and
/// the sizes to try.
struct Job {
    crop: usize,
    invert: bool,
    member: GridMember,
    quad: [[f32; 2]; 4],
    ox: f64,
    oy: f64,
    symbols: Option<Vec<Symbol>>,
}

fn area(q: &[[f32; 2]; 4]) -> f32 {
    0.5 * (0..4)
        .map(|i| {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f32>()
        .abs()
}

/// Point in a convex quad (either winding).
fn inside(q: &[[f32; 2]; 4], p: (f32, f32)) -> bool {
    let s: Vec<f32> = (0..4)
        .map(|i| {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            (b[0] - a[0]) * (p.1 - a[1]) - (b[1] - a[1]) * (p.0 - a[0])
        })
        .collect();
    s.iter().all(|&v| v >= 0.0) || s.iter().all(|&v| v <= 0.0)
}
