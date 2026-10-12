//! Profiling counters (P4b.7): seconds spent per grid stage, accumulated
//! per thread. Only active when the `DPMREAD_TIMINGS` environment variable is
//! set; otherwise a span costs one relaxed load. Never affects results.

use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

/// Grid stages. The first four partition `grid_decode`; the rest are nested
/// inside them (and inside each other: RS runs inside rescue too).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `Sampler::new`, `candidate_quads`, `snap_quad`, `side_combos`.
    Setup,
    /// `grid_pass` with the quiet-zone template.
    QzPass,
    /// `warp_pass`.
    Warp,
    /// `grid_pass` without the quiet-zone ring.
    NoQzPass,
    /// `score_quads` in `grid_pass`.
    Score,
    /// `refine_quad` in `grid_pass` (16 coarse + 4 fine per quad).
    Refine,
    /// `warp_fit` (global tracking + tracked refinement).
    WarpFit,
    /// `rescue_fit`.
    Rescue,
    /// `rs_decode_symbol`.
    Rs,
    /// `global_track` (inside `warp_fit`).
    Track,
    /// `Sampler::sample_uv` / `sample_uv_many`: projection, bilinear remap,
    /// footprint mean (inside score, refine, warp fit and rescue).
    Sample,
}

pub const STAGES: usize = 11;
pub const STAGE_NAMES: [&str; STAGES] = [
    "setup", "qz", "warp", "noqz", "score", "refine", "warpfit", "rescue", "rs", "track", "sample",
];

pub type Acc = [f64; STAGES];

thread_local! {
    static ACC: Cell<Acc> = const { Cell::new([0.0; STAGES]) };
}

/// `DPMREAD_TIMINGS` is set (read once).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DPMREAD_TIMINGS").is_some())
}

/// Counters of this thread since the last call; resets them.
pub fn take() -> Acc {
    ACC.with(|a| a.replace([0.0; STAGES]))
}

/// Adds the time until drop to `stage` on this thread.
pub struct Span(Option<(Stage, Instant)>);

impl Span {
    pub fn new(stage: Stage) -> Self {
        Span(enabled().then(|| (stage, Instant::now())))
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((s, t)) = self.0 {
            let dt = t.elapsed().as_secs_f64();
            ACC.with(|a| {
                let mut v = a.get();
                v[s as usize] += dt;
                a.set(v);
            });
        }
    }
}

/// Run `f` inside a span of `stage`.
pub fn timed<R>(stage: Stage, f: impl FnOnce() -> R) -> R {
    let _s = Span::new(stage);
    f()
}
