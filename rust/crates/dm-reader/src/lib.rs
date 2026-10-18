//! Enhancement ladder, zxing wrapper, job runner and read results.

pub mod annotate;
pub mod charset;
#[cfg(feature = "ml")]
pub mod detect;
pub mod fitdump;
#[cfg(feature = "ml")]
pub mod ml;
pub mod multi;
pub mod read;
pub mod report;
pub mod runner;
pub mod variants;
pub mod zxing;

pub use dm_grid::timing;
/// The image type of `read_frame`.
pub use dm_imgproc::Gray8;

pub use annotate::{annotated, save_annotated, save_variants};
pub use charset::to_text;
pub use fitdump::{write_fit_dump, write_npy};
pub use read::{
    Engines, FitInput, IMAGE_EXT, MAX_FAILED_FITS, Polarity, Progress, ReadNearMiss, ReadOptions,
    ReadResult, Timings, collect_inputs, fit_quad, parse_sizes, read_frame, read_frame_file,
    read_image,
};
pub use report::{
    hex_spaced, hexrows, matrix_art, printable, py_float, report_text, to_dict, to_json,
};
pub use runner::{JOB_GRACE_S, Runner, StopHandle, default_jobs};
pub use variants::{
    Quad, Variant, enhancement_variants, grid_variants, inverted_chain, map_quad, tile_axis,
    tile_variants,
};
#[cfg(feature = "zxing")]
pub use zxing::zxing_read;
pub use zxing::{ZXING_AVAILABLE, ZxInfo, ZxRead, binary};
