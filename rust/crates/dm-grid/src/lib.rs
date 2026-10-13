//! Grid engine: candidate quads, sampler, fit and rescue.
//!
//! Port of the grid decoder of `datamatrix_reader_v2.py` (v1 plus the
//! warped grid of `warp`). Arithmetic follows
//! numpy's dtypes: float32 where Python computes in float32 (quads, the
//! darkness image, scores), float64 where it uses float64 (homographies of
//! `sample_uv`, module coordinates).

pub mod deadline;
pub mod decode;
pub mod dots;
pub mod fit;
pub mod nearmiss;
pub mod np;
pub mod quads;
pub mod quality;
pub mod refine;
pub mod regions;
pub mod rescue;
pub mod sampler;
pub mod timing;
pub mod warp;

pub use crate::deadline::Deadline;
pub use crate::decode::{
    GridRead, MAX_QUADS, NO_QZ_STRONG_FACTOR, STRONG_FIT, grid_decode, grid_decode_seeds,
};
pub use crate::fit::{
    GridFit, Mode, classify, codewords_from_bits, plausible, try_fit, try_fit_values, zscore,
};
pub use crate::nearmiss::{
    NearMiss, OUTSIDE_BUSY, SIZE_CHECK_PX, interior_modules, near_miss_key, outside_dark,
    size_agree,
};
pub use crate::quads::{
    PY_MAX_ASPECT, candidate_quads, expand_quad, line_intersect, max_aspect, percentile_u8,
    side_combos, snap_quad,
};
pub use crate::quality::{Quality, grade, quality};
pub use crate::refine::{Axis, REFINE_STEPS, module_px, refine_quad, track_axis, tracked_axes};
pub use crate::rescue::{RESCUE_GAMMAS, rescue_fit};
pub use crate::sampler::{Quad, Sampler, fisher_scores, order_quad, orientations, score_quads};
pub use crate::warp::{
    WARP_MAX_JERK, WARP_MIN_PX, WARP_MIN_SCORE, WARP_ROUNDS, WARP_STEP, WarpAxis, dash_centres,
    global_track, pitch_drift, refine_tracked, tracked_score, warp_fit, warp_shifts,
};
