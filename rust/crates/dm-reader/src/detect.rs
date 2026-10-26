//! Detector first try (cargo feature `ml`), unless one zxing call on the
//! gray image already reads (`read::quick_zxing`): the trained symbol detector
//! (`dm_ml::Detector`) gives up to `DETECTIONS` symbol quads with size
//! candidates and polarity. The main symbol (nearest the image centre
//! among the large ones) and the detections centred inside it go to `grid_decode_seeds` on the grid variants of a crop
//! around each, quiet-zone template first; then the other large
//! detections, on `FURTHER_MEMBERS` only; last the main symbol's quad
//! grown and shrunk by `NUDGE_MODULES`. Small detections off the main
//! symbol are not tried, so a photo of one part does not report a
//! neighbouring part's code from here. The first read ends the image;
//! otherwise the classic search runs as before.
//!
//! `--multi` (unless thorough): `symbol_aims` lists the symbols the
//! detector sees on the whole frame and on tiles; after the first read the
//! passes aim at them (`read::read_image`).

use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use dm_grid::{Deadline, GridRead, grid_decode_seeds};
use dm_imgproc::{Bgr8, Gray8};
use dm_ml::{Detection, Detector};
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
/// to be the main symbol or a further one.
pub const MAIN_AREA: f32 = 0.25;
/// Modules a side the main symbol's quad is grown and shrunk by in the
/// last jobs: a quad a pixel or two off can decide a read.
pub const NUDGE_MODULES: f32 = 0.5;
/// Polarity probability above which only that polarity is tried.
pub const SURE_POLARITY: f32 = 0.9;

/// The grid members tried on further detections and on the nudged quads:
/// the ones that read there (the other channels and `inv-gray` only cost
/// time).
fn few_member(m: GridMember) -> bool {
    matches!(m, GridMember::Flat(0) | GridMember::Gray)
}

/// `--multi`: the frame is also searched in `AIM_TILES` x `AIM_TILES`
/// tiles (half a tile overlap), so that a small symbol in a large photo
/// reaches the finder at a usable size.
pub const AIM_TILES: usize = 3;

/// Copy of `[x0, x0 + w) x [y0, y0 + h)`.
pub(crate) fn crop_bgr(bgr: &Bgr8, x0: usize, y0: usize, w: usize, h: usize) -> Bgr8 {
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

/// A symbol the detector sees, for the `--multi` passes.
pub(crate) struct Aim {
    pub det: Detection,
    /// Found on a tile only. Tiles give many false detections, so these
    /// get the first try but no classic search on a crop.
    pub tile: bool,
    /// The first try has run on it.
    pub tried: bool,
}

impl Aim {
    pub fn centre(&self) -> (f32, f32) {
        centre(&self.det.quad)
    }
}

/// `--multi`: the symbols the detector sees on the whole frame, then on
/// the `AIM_TILES` x `AIM_TILES` tiles, up to `max` per call; a detection
/// centred inside one kept before is the same symbol. `None` when a model
/// fails.
pub(crate) fn symbol_aims(gray: &Gray8, max: usize) -> Option<Vec<Aim>> {
    let det = Detector::embedded();
    let mut all: Vec<Aim> = det
        .detect(gray, MIN_SCORE, max)?
        .into_iter()
        .map(|det| Aim {
            det,
            tile: false,
            tried: false,
        })
        .collect();
    let n = AIM_TILES;
    let (tw, th) = (2 * gray.width / (n + 1), 2 * gray.height / (n + 1));
    if tw >= 8 && th >= 8 {
        for ty in 0..n {
            for tx in 0..n {
                let (x0, y0) = (tx * tw / 2, ty * th / 2);
                let mut data = Vec::with_capacity(tw * th);
                for y in y0..y0 + th {
                    let row = y * gray.width + x0;
                    data.extend_from_slice(&gray.data[row..row + tw]);
                }
                for mut d in det.detect(&Gray8::from_vec(tw, th, data), MIN_SCORE, max)? {
                    d.quad = d.quad.map(|[x, y]| [x + x0 as f32, y + y0 as f32]);
                    all.push(Aim {
                        det: d,
                        tile: true,
                        tried: false,
                    });
                }
            }
        }
    }
    let mut kept: Vec<Aim> = Vec::new();
    for a in all {
        if !kept.iter().any(|k| inside(&k.det.quad, a.centre())) {
            kept.push(a);
        }
    }
    Some(kept)
}

/// `2 k` times the bounding square of `q` around its centre, clipped to a
/// `w` x `h` image: `(x0, y0, x1, y1)`, `None` when 8 px or less remain.
pub(crate) fn crop_box(q: &[[f32; 2]; 4], w: usize, h: usize, k: f32) -> Option<[usize; 4]> {
    let xs = q.map(|p| p[0]);
    let ys = q.map(|p| p[1]);
    let (minx, maxx) = (
        xs.iter().copied().fold(f32::MAX, f32::min),
        xs.iter().copied().fold(f32::MIN, f32::max),
    );
    let (miny, maxy) = (
        ys.iter().copied().fold(f32::MAX, f32::min),
        ys.iter().copied().fold(f32::MIN, f32::max),
    );
    let side = (maxx - minx).max(maxy - miny) * k;
    let (cx, cy) = ((minx + maxx) / 2.0, (miny + maxy) / 2.0);
    let clamp = |v: f32, hi: usize| v.round().clamp(0.0, hi as f32) as usize;
    let (x0, y0) = (clamp(cx - side, w), clamp(cy - side, h));
    let (x1, y1) = (clamp(cx + side, w), clamp(cy + side, h));
    (x1 > x0 + 8 && y1 > y0 + 8).then_some([x0, y0, x1, y1])
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

/// The first try; with `aim`, on that detection only (no models run), its
/// crop `k` times the usual size.
pub(crate) fn first_try(
    bgr: &Bgr8,
    gray: &Gray8,
    opts: &ReadOptions,
    runner: &Runner,
    deadline: Instant,
    aim: Option<(&Detection, f32)>,
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
    let dets = match aim {
        Some((d, _)) => Some(vec![d.clone()]),
        None => Detector::embedded().detect(gray, MIN_SCORE, DETECTIONS),
    };
    out.models = t.elapsed().as_secs_f64();
    let Some(dets) = dets else {
        return out;
    };
    // one job per (detection, grid variant member), in `try_order`; each
    // job builds its member's variants of the crop, so the first job (the
    // variant that reads most often) starts at once. The nudged quads of
    // the main group come last. The first read in job order wins
    let mut crops: Vec<Bgr8> = Vec::new();
    let mut jobs: Vec<Job> = Vec::new();
    let mut nudge: Vec<Job> = Vec::new();
    let t = Instant::now();
    for (i, further) in try_order(&dets, gray.width as f32, gray.height as f32) {
        let d = &dets[i];
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
        let k = aim.map_or(1.0, |a| a.1);
        let Some([x0, y0, x1, y1]) = crop_box(&d.quad, bgr.width, bgr.height, k) else {
            continue;
        };
        crops.push(crop_bgr(bgr, x0, y0, x1 - x0, y1 - y0));
        let quad = d.quad.map(|[x, y]| [x - x0 as f32, y - y0 as f32]);
        for member in grid_members(invert, both) {
            if further && !few_member(member) {
                continue;
            }
            let job = |seeds: Vec<[[f32; 2]; 4]>| Job {
                crop: crops.len() - 1,
                invert,
                member,
                seeds,
                ox: x0 as f64,
                oy: y0 as f64,
                symbols: symbols.clone(),
            };
            jobs.push(job(vec![quad]));
            if !further && few_member(member) {
                nudge.push(job(nudged(&quad, d.sizes[0]).to_vec()));
            }
        }
    }
    jobs.extend(nudge);
    out.crops = t.elapsed().as_secs_f64();
    // the first job alone (it reads most often; on a phone its siblings
    // slow it down by more than they gain), then the rest in parallel. A
    // variant is built and tried only with at least `left` of the budget
    // to go: building one cannot be stopped, so for the later jobs `left`
    // is what the first job took
    let attempt = |j: &Job, left: Duration| {
        let mut calls = 0;
        if dl.at.is_some_and(|at| Instant::now() + left > at) {
            return (None, 0);
        }
        for gv in grid_member(&crops[j.crop], j.invert, j.member) {
            if dl.at.is_some_and(|at| Instant::now() + left > at) {
                break;
            }
            let s = gv.scale as f32;
            let seeds: Vec<_> = j
                .seeds
                .iter()
                .map(|q| q.map(|[x, y]| [x * s, y * s]))
                .collect();
            calls += 1;
            let r = grid_decode_seeds(&gv.img, &dl, &seeds, &[true, false], j.symbols.as_deref());
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
    let mut left = Duration::ZERO;
    for part in [&jobs[..split], &jobs[split..]] {
        if hit.is_some() {
            break;
        }
        let t = Instant::now();
        runner.run(
            part,
            dl.at,
            |j| attempt(j, left),
            |_, (r, calls)| {
                tries += calls;
                match r {
                    Some(r) => {
                        hit = Some(r);
                        ControlFlow::Break(())
                    }
                    None => ControlFlow::Continue(()),
                }
            },
        );
        left = t.elapsed();
    }
    out.hit = hit;
    out.tries = tries;
    out
}

/// One first-try job: a grid variant member of a crop (`crops` index,
/// offsets in original pixels), the seed quads in crop pixels and the
/// sizes to try.
struct Job {
    crop: usize,
    invert: bool,
    member: GridMember,
    seeds: Vec<[[f32; 2]; 4]>,
    ox: f64,
    oy: f64,
    symbols: Option<Vec<Symbol>>,
}

/// The detections the first try uses, in order, `true` for a further one:
/// the main symbol, the detections centred inside it, then the further
/// ones. Main = the one nearest the image centre among those of at least
/// `MAIN_AREA` x the largest area (in a photo of one part it is that
/// part's code); further = the others of that area not centred inside one
/// before, in `dets` order (best score first). A neighbouring part's code
/// in a photo of one part is small or cut, so it is not tried.
fn try_order(dets: &[Detection], w: f32, h: f32) -> Vec<(usize, bool)> {
    let largest = dets.iter().map(|d| area(&d.quad)).fold(0.0, f32::max);
    let large = |i: &usize| area(&dets[*i].quad) >= MAIN_AREA * largest;
    let off_centre = |i: usize| {
        let (cx, cy) = centre(&dets[i].quad);
        ((cx / w - 0.5).powi(2) + (cy / h - 0.5).powi(2)).sqrt()
    };
    let Some(main) = (0..dets.len())
        .filter(large)
        .min_by(|&a, &b| off_centre(a).total_cmp(&off_centre(b)))
    else {
        return Vec::new();
    };
    let in_main = |i: &usize| inside(&dets[main].quad, box_centre(&dets[*i].quad));
    let mut order = vec![(main, false)];
    order.extend(
        (0..dets.len())
            .filter(|i| *i != main && in_main(i))
            .map(|i| (i, false)),
    );
    let mut tried = vec![dets[main].quad];
    for i in (0..dets.len()).filter(|i| large(i) && !in_main(i)) {
        if !tried.iter().any(|q| inside(q, box_centre(&dets[i].quad))) {
            tried.push(dets[i].quad);
            order.push((i, true));
        }
    }
    order
}

/// `q` scaled about its centre so that a side grows and shrinks by
/// `NUDGE_MODULES` modules, for a symbol of size `size` (`RxC`, the larger
/// side counted, at least 10).
fn nudged(q: &[[f32; 2]; 4], size: &str) -> [[[f32; 2]; 4]; 2] {
    let n = size
        .split('x')
        .filter_map(|v| v.parse::<f32>().ok())
        .fold(10.0, f32::max);
    let (cx, cy) = centre(q);
    [1.0, -1.0].map(|s| {
        let f = 1.0 + s * 2.0 * NUDGE_MODULES / n;
        q.map(|[x, y]| [cx + (x - cx) * f, cy + (y - cy) * f])
    })
}

fn centre(q: &[[f32; 2]; 4]) -> (f32, f32) {
    (
        q.iter().map(|p| p[0]).sum::<f32>() / 4.0,
        q.iter().map(|p| p[1]).sum::<f32>() / 4.0,
    )
}

/// Centre of the quad's bounding box.
fn box_centre(q: &[[f32; 2]; 4]) -> (f32, f32) {
    let (xs, ys) = (q.map(|p| p[0]), q.map(|p| p[1]));
    let mid = |v: [f32; 4]| {
        (v.iter().copied().fold(f32::MAX, f32::min) + v.iter().copied().fold(f32::MIN, f32::max))
            / 2.0
    };
    (mid(xs), mid(ys))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn det(score: f32, x: f32, y: f32, side: f32) -> Detection {
        let quad = [[x, y], [x + side, y], [x + side, y + side], [x, y + side]];
        Detection {
            score,
            quad,
            finder_quad: quad,
            p_dark: 1.0,
            sizes: ["18x18", "16x16", "20x20"],
        }
    }

    #[test]
    fn main_symbol_first_then_large_detections() {
        // 1000 x 800 image: a 0.45 code at the centre is main, the 0.90 one
        // of the same size beside it is tried next
        let dets = [
            det(0.90, 600.0, 300.0, 200.0),
            det(0.45, 400.0, 300.0, 200.0),
        ];
        assert_eq!(try_order(&dets, 1000.0, 800.0), [(1, false), (0, true)]);
    }

    #[test]
    fn small_neighbour_is_not_tried() {
        // a phone photo of one part: the main code, a neighbour's code of a
        // tenth of its area in a corner
        let dets = [det(0.66, 350.0, 250.0, 300.0), det(0.34, 20.0, 20.0, 95.0)];
        assert_eq!(try_order(&dets, 1000.0, 800.0), [(0, false)]);
    }

    #[test]
    fn detections_inside_are_one_symbol() {
        // a second quad of the main code stays in the main group, one of a
        // further code is not tried twice
        let dets = [
            det(0.9, 450.0, 350.0, 100.0),
            det(0.8, 100.0, 100.0, 100.0),
            det(0.7, 455.0, 355.0, 95.0),
            det(0.6, 105.0, 105.0, 95.0),
        ];
        assert_eq!(
            try_order(&dets, 1000.0, 800.0),
            [(0, false), (2, false), (1, true)]
        );
    }

    #[test]
    fn nudged_quads_move_half_a_module_a_side() {
        let q = det(1.0, 0.0, 0.0, 200.0).quad;
        // 20x20 symbol of 10 px modules: sides of 210 and 190 px
        let [grown, shrunk] = nudged(&q, "20x20");
        assert!((grown[1][0] - grown[0][0] - 210.0).abs() < 1e-3);
        assert!((shrunk[1][0] - shrunk[0][0] - 190.0).abs() < 1e-3);
        assert_eq!(centre(&grown), centre(&q));
        // the larger side counts; small symbols as 10 modules
        let [g, _] = nudged(&q, "16x36");
        assert!((g[1][0] - g[0][0] - 200.0 * (1.0 + 1.0 / 36.0)).abs() < 1e-3);
        let [g, _] = nudged(&q, "8x8");
        assert!((g[1][0] - g[0][0] - 220.0).abs() < 1e-3);
    }
}
