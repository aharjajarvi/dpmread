//! `ReadResult`, `read_image`, `analyse_known`, `to_text`, `parse_sizes`,
//! `collect_inputs` (reader 101–140, 1678–2018).

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dm_codec::Decoded;
use dm_ecc::RSResult;
use dm_grid::timing::{self, Acc};
use dm_grid::{
    Deadline, GridFit, GridRead, NearMiss, Quality, grid_decode, max_aspect, near_miss_key, quality,
};
use dm_imgproc::{Bgr8, Gray8, LoadError, bgr_to_gray, load_image};
use dm_symbol::{Symbol, all_symbols};

use crate::annotate::{save_annotated, save_variants};
use crate::charset::to_text;
use crate::multi::{MULTI_FALLBACK_MODULES, MULTI_MAX, blank_symbol, inside};
use crate::runner::Runner;
use crate::variants::{
    Quad, Variant, enhancement_variants, grid_variants, map_quad, tile_variants,
};
use crate::zxing::{ZXING_AVAILABLE, ZxInfo, ZxRead};

/// File extensions read from folders (lower case, with the dot).
pub const IMAGE_EXT: [&str; 8] = [
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif",
];

/// Grid time cap per job on a whole-image variant and on a tile.
const GRID_CAP_S: f64 = 25.0;
const TILE_CAP_S: f64 = 10.0;
/// `analyse_known` time per grid attempt.
const ANALYSE_S: f64 = 8.0;

/// Mark polarity (`--polarity`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Polarity {
    /// Both, with a small inverted variant chain.
    #[default]
    Any,
    /// Dark on light only.
    Dark,
    /// Light on dark only.
    Light,
}

/// Decoders to use (`--engines`). libdmtx is not part of the port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Engines {
    pub zxing: bool,
    pub grid: bool,
}

impl Default for Engines {
    fn default() -> Self {
        Engines {
            zxing: true,
            grid: true,
        }
    }
}

/// Options of `read_image` (the CLI flags that affect one image).
#[derive(Clone, Debug)]
pub struct ReadOptions {
    pub engines: Engines,
    /// Try every method and keep the majority read.
    pub exhaustive: bool,
    /// Seconds per image.
    pub budget_s: f64,
    /// Expected symbol sizes (`None`: all).
    pub expect: Option<Vec<Symbol>>,
    pub polarity: Polarity,
    /// `--debug`: write the variant PNGs and the annotated result here.
    pub debug_dir: Option<PathBuf>,
    /// `--multi`: read every symbol of the image (`ReadResult.more`).
    pub multi: bool,
    /// Called after each consumed job (GUI progress); does not change the read.
    pub progress: Option<Progress>,
}

/// Progress callback of `read_image`: `(stage, jobs done, jobs in stage)`
/// with stage `"zxing"` or `"grid"`, called on the calling thread after
/// each consumed job (in every `multi` pass again).
#[derive(Clone)]
pub struct Progress(pub Arc<ProgressFn>);

pub type ProgressFn = dyn Fn(&str, usize, usize) + Send + Sync;

impl Progress {
    fn report(opts: &ReadOptions, stage: &str, done: usize, total: usize) {
        if let Some(p) = &opts.progress {
            (p.0)(stage, done, total);
        }
    }
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Progress")
    }
}

impl Default for ReadOptions {
    fn default() -> Self {
        ReadOptions {
            engines: Engines::default(),
            exhaustive: false,
            budget_s: 40.0,
            expect: None,
            polarity: Polarity::Any,
            debug_dir: None,
            multi: false,
            progress: None,
        }
    }
}

/// The best failed grid fit of an image: the grid engine's near miss plus
/// the module size in original pixels and the variant it was found on.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadNearMiss {
    pub nm: NearMiss,
    pub modules_px_orig: f64,
    pub variant: String,
}

/// Where the time of one image went (`DPMREAD_TIMINGS`, P4b.7). Seconds;
/// `*_wall` is elapsed time of a stage, `*_busy` the sum of its job times
/// over all workers (also jobs whose result was not used), so
/// `workers * wall - busy` is idle worker time.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub workers: usize,
    /// Load, gray conversion, enhancement and tile variants.
    pub variants: f64,
    pub zxing_wall: f64,
    pub zxing_busy: f64,
    pub zxing_jobs: usize,
    /// `grid_variants`.
    pub grid_variants: f64,
    pub grid_wall: f64,
    pub grid_busy: f64,
    pub grid_jobs: usize,
    /// Grid stage counters summed over the consumed grid jobs (`grid_jobs`).
    pub grid: Acc,
    /// `analyse_known` (after a zxing read) or `quality` (grid read).
    pub finish: f64,
}

/// Everything known about one image after `read_image`.
#[derive(Clone, Debug, Default)]
pub struct ReadResult {
    pub path: PathBuf,
    /// "zxing" or "grid"; empty if nothing decoded.
    pub engine: String,
    pub variant: String,
    pub data: Vec<u8>,
    pub text: String,
    pub elapsed: f64,
    pub attempts: usize,
    pub fit: Option<GridFit>,
    pub rs: Option<RSResult>,
    pub dec: Option<Decoded>,
    pub mirrored: bool,
    /// Of the variant `fit.quad` refers to.
    pub scale: f64,
    pub ox: f64,
    pub oy: f64,
    pub zx: Option<ZxInfo>,
    pub quality: Option<Quality>,
    pub agreeing: Vec<String>,
    pub sampling: String,
    pub near_miss: Option<ReadNearMiss>,
    /// Symbol outline in original image pixels (tl, tr, br, bl as found).
    pub quad: Option<Quad>,
    /// `--multi`: the further symbols, in reading order (each with its own
    /// `elapsed`; this result's `elapsed` is the whole image).
    pub more: Vec<ReadResult>,
    /// Only with `DPMREAD_TIMINGS`; not part of the JSON.
    pub timings: Option<Timings>,
}

impl ReadResult {
    fn new(path: &Path) -> Self {
        ReadResult {
            path: path.to_path_buf(),
            scale: 1.0,
            ..Default::default()
        }
    }

    /// The decode that sets the text's charsets: the grid decoder's, else
    /// the ECIs zxing reported.
    pub fn text_dec(&self) -> Option<&Decoded> {
        self.dec
            .as_ref()
            .or_else(|| self.zx.as_ref()?.eci.as_deref())
    }
}

/// `collect_inputs`: folders expand to their image files (not recursive,
/// sorted by name); other arguments are taken as they are.
pub fn collect_inputs(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut names: Vec<_> = std::fs::read_dir(p)
                .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.file_name())).collect())
                .unwrap_or_default();
            names.sort();
            for name in names {
                let ext = Path::new(&name)
                    .extension()
                    .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()));
                if ext.is_some_and(|e| IMAGE_EXT.contains(&e.as_str())) {
                    out.push(p.join(name));
                }
            }
        } else {
            out.push(p.clone());
        }
    }
    out
}

/// `parse_sizes`: `--expect` values (e.g. `24x24`, `12x26`, comma
/// separated) -> symbols sorted by size, or `None` (all). Malformed and
/// unknown sizes are ignored with a warning on stderr.
pub fn parse_sizes(specs: &[String]) -> Option<Vec<Symbol>> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for spec in specs {
        let spec = spec.replace(['×', 'X'], "x");
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let rc = part.split_once('x').and_then(|(r, c)| {
                (!c.contains('x')).then_some(())?;
                Some((parse_py_int(r)?, parse_py_int(c)?))
            });
            match rc {
                Some(rc) => out.push(rc),
                None => eprintln!("warning: ignoring malformed size hint '{part}'"),
            }
        }
    }
    out.sort();
    out.dedup();
    if out.is_empty() {
        return None;
    }
    let known = |rc: &(usize, usize)| all_symbols().any(|s| (s.rows, s.cols) == *rc);
    let bad: Vec<_> = out.iter().filter(|rc| !known(rc)).collect();
    if !bad.is_empty() {
        let list: Vec<String> = bad.iter().map(|(r, c)| format!("({r}, {c})")).collect();
        eprintln!(
            "warning: unknown symbol size(s) [{}] ignored",
            list.join(", ")
        );
    }
    let syms: Vec<Symbol> = out
        .iter()
        .filter_map(|rc| all_symbols().find(|s| (s.rows, s.cols) == *rc).copied())
        .collect();
    (!syms.is_empty()).then_some(syms)
}

/// Python `int(s)`: surrounding whitespace allowed, optional sign.
fn parse_py_int(s: &str) -> Option<usize> {
    let s = s.trim();
    s.strip_prefix('+').unwrap_or(s).parse().ok()
}

/// `_min_area`: tiles are wide shots, a symbol covers a small part of them.
fn min_area(v: &Variant) -> f64 {
    if v.tile() { 0.004 } else { 0.02 }
}

fn grid_quad(q: &Quad) -> dm_grid::Quad {
    q.map(|(x, y)| [x, y])
}

fn reader_quad(q: &dm_grid::Quad) -> Quad {
    q.map(|[x, y]| (x, y))
}

#[cfg(feature = "zxing")]
fn zx(v: &Variant, try_invert: bool) -> (Option<ZxRead>, Vec<Quad>) {
    crate::zxing::zxing_read(v, try_invert)
}

#[cfg(not(feature = "zxing"))]
fn zx(_v: &Variant, _try_invert: bool) -> (Option<ZxRead>, Vec<Quad>) {
    (None, Vec::new())
}

/// Name -> variant, in insertion order; a later insert of a known name
/// replaces the value in place (a Python dict).
struct ByName<'a>(Vec<(&'a str, &'a Variant)>);

impl<'a> ByName<'a> {
    fn insert(&mut self, v: &'a Variant) {
        match self.0.iter_mut().find(|(n, _)| *n == v.name) {
            Some(slot) => slot.1 = v,
            None => self.0.push((&v.name, v)),
        }
    }

    fn get(&self, name: &str) -> Option<&'a Variant> {
        self.0.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }
}

enum Hit<'a> {
    Zxing(Quad, ZxInfo),
    Grid(&'a Variant, Box<GridRead>),
}

/// One successful read: engine, variant name, data, details.
struct Found<'a> {
    engine: &'static str,
    variant: String,
    data: Vec<u8>,
    hit: Hit<'a>,
}

/// A grid job: variant, time cap, the variant group whose other members
/// of the same shape are the rescue's alternative images, and the template.
struct GridJob<'a> {
    v: &'a Variant,
    cap_s: f64,
    group: &'a [Variant],
    qz: bool,
}

/// Read one image: zxing on every variant (cheap), then the grid decoder,
/// within `opts.budget_s`. Stops at the first read unless exhaustive; the
/// majority content wins. With `opts.multi` the read symbol is covered with
/// the local background and the image read again (same budget), until
/// nothing more reads; the further symbols go to `more`.
pub fn read_image(
    path: &Path,
    opts: &ReadOptions,
    runner: &Runner,
) -> Result<ReadResult, LoadError> {
    let t0 = Instant::now();
    let bgr = load_image(path)?;
    let deadline = t0 + Duration::from_secs_f64(opts.budget_s);
    let mut res = read_bgr(path, &bgr, opts, runner, t0, deadline, true);
    if !opts.multi || res.data.is_empty() {
        return Ok(res);
    }
    let mut img = bgr;
    let mut covered: Vec<Quad> = Vec::new();
    let (mut quad, mut modules) = (res.quad, symbol_modules(&res));
    while 1 + res.more.len() < MULTI_MAX && Instant::now() < deadline && !runner.stopped() {
        let Some(q) = quad else { break };
        blank_symbol(&mut img, &q, modules);
        covered.push(q);
        let r = read_bgr(path, &img, opts, runner, Instant::now(), deadline, false);
        // nothing left, or a read inside a covered symbol (cover failed)
        let Some(rq) = r.quad.filter(|_| !r.data.is_empty()) else {
            break;
        };
        let (cx, cy) = rq
            .iter()
            .fold((0.0, 0.0), |a, p| (a.0 + p.0 / 4.0, a.1 + p.1 / 4.0));
        if covered.iter().any(|c| inside(c, cx, cy)) {
            break;
        }
        (quad, modules) = (r.quad, symbol_modules(&r));
        res.more.push(r);
    }
    res.elapsed = t0.elapsed().as_secs_f64();
    Ok(res)
}

/// Modules across a read symbol (mean of rows and columns) for `--multi`.
fn symbol_modules(res: &ReadResult) -> f32 {
    res.fit.as_ref().map_or(MULTI_FALLBACK_MODULES, |f| {
        (f.geom.rows + f.geom.cols) as f32 / 2.0
    })
}

/// A quad in variant pixels -> original image pixels.
fn orig_quad(q: &Quad, scale: f64, ox: f64, oy: f64) -> Quad {
    let (s, ox, oy) = (scale as f32, ox as f32, oy as f32);
    q.map(|(x, y)| (x / s + ox, y / s + oy))
}

/// `read_image` on a loaded image. `t0`: start of this pass (for `elapsed`
/// and the timings); `debug`: write `opts.debug_dir` files.
fn read_bgr(
    path: &Path,
    bgr: &Bgr8,
    opts: &ReadOptions,
    runner: &Runner,
    t0: Instant,
    deadline: Instant,
    debug: bool,
) -> ReadResult {
    let mut res = ReadResult::new(path);
    let mut tm = Timings {
        workers: runner.workers(),
        ..Default::default()
    };
    let lap = |t: &mut Instant| {
        let now = Instant::now();
        let d = now.duration_since(*t).as_secs_f64();
        *t = now;
        d
    };
    let mut t = t0;
    let busy = AtomicU64::new(0);
    let gray_orig = bgr_to_gray(bgr);
    let invert = opts.polarity == Polarity::Light;
    let inverted_extra = opts.polarity == Polarity::Any;
    let expect = opts.expect.as_deref();
    // variant chains and tiles are rayon tasks on the workers
    let (variants, tiles) = runner.install(|| {
        rayon::join(
            || enhancement_variants(bgr, invert, inverted_extra),
            || tile_variants(bgr, invert),
        )
    });
    let gvars: Vec<Variant>;
    let mut by_name = ByName(Vec::new());
    variants.iter().for_each(|v| by_name.insert(v));
    tiles.iter().flatten().for_each(|v| by_name.insert(v));
    if let Some(dir) = opts.debug_dir.as_ref().filter(|_| debug) {
        save_variants(dir, path, &variants, &tiles);
    }
    tm.variants = lap(&mut t);
    let exhaustive = opts.exhaustive;
    let mut found: Vec<Found> = Vec::new();
    let done = |found: &Vec<Found>| !found.is_empty() && !exhaustive;

    let mut seeds: Vec<(&Variant, Quad)> = Vec::new();
    let mut tile_seeds: Vec<(&Variant, Quad)> = Vec::new();
    let mut tile_hit: Vec<&str> = Vec::new();
    // 1. zxing on all variants (cheap), whole image first, then the tiles
    if opts.engines.zxing && ZXING_AVAILABLE {
        let ladder: Vec<&Variant> = variants.iter().chain(tiles.iter().flatten()).collect();
        // with a known polarity the ladder already has it: no inverted retries
        let try_invert = opts.polarity == Polarity::Any;
        runner.run(
            &ladder,
            None,
            |v| job_timed(&busy, || zx(v, try_invert)),
            |i, ((r, failed), _)| {
                let v = ladder[i];
                res.attempts += 1;
                tm.zxing_jobs += 1;
                Progress::report(opts, "zxing", i + 1, ladder.len());
                if let Some(r) = r {
                    found.push(Found {
                        engine: "zxing",
                        variant: v.name.clone(),
                        data: r.data,
                        hit: Hit::Zxing(r.quad, r.info),
                    });
                    tile_hit.push(v.name.split(" > ").next().unwrap_or(""));
                    if done(&found) {
                        return ControlFlow::Break(());
                    }
                } else if !v.tile() && seeds.len() < 6 {
                    seeds.extend(failed.into_iter().map(|q| (v, q)));
                } else if v.tile() && tile_seeds.len() < 12 {
                    tile_seeds.extend(failed.into_iter().map(|q| (v, q)));
                }
                ControlFlow::Continue(())
            },
        );
        tm.zxing_wall = lap(&mut t);
        tm.zxing_busy = busy.swap(0, Ordering::Relaxed) as f64 * 1e-9;
    }
    // 2. built-in grid decoder on analogue variants, then on the tiles that
    //    zxing did not read (their gray / flatten variants, not binarised);
    //    all of that with the quiet-zone template, then the ring-less retry
    if !done(&found) && opts.engines.grid {
        gvars = runner.install(|| grid_variants(bgr, invert, inverted_extra));
        tm.grid_variants = lap(&mut t);
        let mut stage: Vec<(&Variant, f64, &[Variant])> =
            gvars.iter().map(|v| (v, GRID_CAP_S, &gvars[..])).collect();
        for tv in &tiles {
            let head = tv[0].name.split(" > ").next().unwrap_or("");
            if !tile_hit.contains(&head) {
                let two = &tv[..tv.len().min(2)];
                stage.extend(two.iter().map(|v| (v, TILE_CAP_S, two)));
            }
        }
        let ladder: Vec<GridJob> = [true, false]
            .iter()
            .flat_map(|&qz| {
                stage.iter().map(move |&(v, cap_s, group)| GridJob {
                    v,
                    cap_s,
                    group,
                    qz,
                })
            })
            .collect();
        let all_seeds: Vec<&(&Variant, Quad)> = seeds.iter().chain(&tile_seeds).collect();
        let base = Deadline {
            at: Some(deadline),
            abort: Some(runner.abort_flag()),
        };
        let grid_job = |job: &GridJob| {
            let v = job.v;
            let (w, h) = (v.img.width as f32, v.img.height as f32);
            let extra: Vec<dm_grid::Quad> = all_seeds
                .iter()
                .map(|(sv, q)| map_quad(q, sv, v))
                // tiles: only seeds that lie in this tile
                .filter(|q| {
                    let mx = q.iter().fold(0f32, |a, p| a + p.0) / 4.0;
                    let my = q.iter().fold(0f32, |a, p| a + p.1) / 4.0;
                    !v.tile() || ((0.0..w).contains(&mx) && (0.0..h).contains(&my))
                })
                .map(|q| grid_quad(&q))
                .collect();
            let alts: Vec<(String, &Gray8)> = job
                .group
                .iter()
                .filter(|o| {
                    !ptr::eq(*o, v) && o.img.width == v.img.width && o.img.height == v.img.height
                })
                .map(|o| (o.name.clone(), &o.img))
                .collect();
            let dl = base.capped(Duration::from_secs_f64(job.cap_s));
            job_timed(&busy, || {
                let mut diag = None;
                let r = grid_decode(
                    &v.img,
                    &dl,
                    &extra,
                    &alts,
                    min_area(v),
                    max_aspect(expect),
                    &[job.qz],
                    expect,
                    Some(&mut diag),
                );
                (r, diag)
            })
        };
        runner.run(&ladder, Some(deadline), grid_job, |i, ((r, diag), acc)| {
            let v = ladder[i].v;
            res.attempts += 1;
            tm.grid_jobs += 1;
            Progress::report(opts, "grid", i + 1, ladder.len());
            tm.grid.iter_mut().zip(acc).for_each(|(a, b)| *a += b);
            if let Some(r) = r {
                by_name.insert(v);
                found.push(Found {
                    engine: "grid",
                    variant: v.name.clone(),
                    data: r.dec.data.clone(),
                    hit: Hit::Grid(v, Box::new(r)),
                });
                if done(&found) {
                    return ControlFlow::Break(());
                }
            } else if let Some(d) = diag
                && near_miss_key(Some(&d)) > near_miss_key(res.near_miss.as_ref().map(|n| &n.nm))
            {
                res.near_miss = Some(ReadNearMiss {
                    modules_px_orig: d.modules_px / v.scale,
                    nm: d,
                    variant: v.name.clone(),
                });
            }
            ControlFlow::Continue(())
        });
        tm.grid_wall = lap(&mut t);
        tm.grid_busy = busy.swap(0, Ordering::Relaxed) as f64 * 1e-9;
    }

    res.elapsed = t0.elapsed().as_secs_f64();
    if found.is_empty() {
        res.timings = timing::enabled().then_some(tm);
        return res;
    }
    // majority content wins, earliest engine as the reported one
    let mut counts: Vec<(&[u8], usize)> = Vec::new();
    for f in &found {
        match counts.iter_mut().find(|(d, _)| *d == f.data.as_slice()) {
            Some(c) => c.1 += 1,
            None => counts.push((&f.data, 1)),
        }
    }
    let top = counts.iter().map(|c| c.1).max().unwrap_or(0);
    let best_data = counts.iter().find(|c| c.1 == top).map(|c| c.0.to_vec());
    let best_data = best_data.unwrap_or_default();
    res.agreeing = found
        .iter()
        .filter(|f| f.data == best_data)
        .map(|f| format!("{}: {}", f.engine, f.variant))
        .collect();
    let first = found
        .into_iter()
        .find(|f| f.data == best_data)
        .expect("best data comes from a read");
    res.engine = first.engine.to_string();
    res.variant = first.variant;
    res.data = first.data;
    match first.hit {
        Hit::Grid(v, r) => {
            let GridRead {
                fit,
                rs,
                dec,
                mirrored,
                note,
            } = *r;
            res.quality = Some(quality(&gray_orig, &fit, v.scale, &rs, (v.ox, v.oy)));
            res.fit = Some(fit);
            res.rs = Some(rs);
            res.dec = Some(dec);
            res.mirrored = mirrored;
            res.scale = v.scale;
            res.ox = v.ox;
            res.oy = v.oy;
            res.sampling = note;
        }
        Hit::Zxing(quad, info) => {
            res.zx = Some(info);
            if let Some(v) = by_name.get(&res.variant) {
                res.quad = Some(orig_quad(&quad, v.scale, v.ox, v.oy));
            }
            if opts.engines.grid {
                analyse_known(&mut res, &by_name, &quad, &gray_orig, expect);
            }
        }
    }
    if let Some(fit) = &res.fit {
        res.quad = Some(orig_quad(&fit_quad(fit), res.scale, res.ox, res.oy));
    }
    res.text = to_text(&res.data, res.text_dec()).0;
    res.elapsed = t0.elapsed().as_secs_f64();
    tm.finish = lap(&mut t);
    res.timings = timing::enabled().then_some(tm);
    if let Some(dir) = opts.debug_dir.as_ref().filter(|_| debug) {
        save_annotated(dir, bgr, &res);
    }
    res
}

/// Run a job; when profiling, add its time (ns) to `busy` and return its
/// grid counters.
fn job_timed<R>(busy: &AtomicU64, f: impl FnOnce() -> R) -> (R, Acc) {
    if !timing::enabled() {
        return (f(), Acc::default());
    }
    timing::take();
    let t = Instant::now();
    let r = f();
    busy.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    (r, timing::take())
}

/// A library decoded the symbol: run the grid decoder at its position to
/// recover module matrix, codewords and RS statistics for the report.
fn analyse_known(
    res: &mut ReadResult,
    by_name: &ByName,
    quad: &Quad,
    gray_orig: &Gray8,
    expect: Option<&[Symbol]>,
) {
    let Some(v) = by_name.get(&res.variant) else {
        return;
    };
    let mut others: Vec<&Variant> = by_name
        .0
        .iter()
        .map(|(_, x)| *x)
        .filter(|x| !ptr::eq(*x, v))
        .collect();
    if v.tile() {
        // same tile first
        let prefix = format!("{} > ", v.name.split(" > ").next().unwrap_or(""));
        others.sort_by_key(|x| !x.name.starts_with(&prefix));
    }
    for gv in std::iter::once(v).chain(others.into_iter().take(6)) {
        let q = grid_quad(&map_quad(quad, v, gv));
        let dl = Deadline::at(Instant::now() + Duration::from_secs_f64(ANALYSE_S));
        let r = grid_decode(
            &gv.img,
            &dl,
            &[q],
            &[],
            min_area(gv),
            max_aspect(expect),
            &[true, false],
            expect,
            None,
        );
        if let Some(r) = r
            && r.dec.data == res.data
        {
            res.quality = Some(quality(gray_orig, &r.fit, gv.scale, &r.rs, (gv.ox, gv.oy)));
            res.fit = Some(r.fit);
            res.rs = Some(r.rs);
            res.dec = Some(r.dec);
            res.mirrored = r.mirrored;
            res.sampling = r.note;
            res.scale = gv.scale;
            res.ox = gv.ox;
            res.oy = gv.oy;
            return;
        }
    }
}

/// Quad of a grid fit in variant pixels, as the reader's quad type.
pub fn fit_quad(fit: &GridFit) -> Quad {
    reader_quad(&fit.quad)
}
