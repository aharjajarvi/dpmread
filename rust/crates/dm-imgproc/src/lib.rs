//! Gray image type and the OpenCV operations the reader needs.
//!
//! Every op follows the OpenCV 5.0 implementation that the Python reference
//! (`datamatrix_reader.py`) runs, including its rounding, so the enhancement
//! variants stay within a grey level or two of Python's.

pub mod arith;
pub mod border;
pub mod clahe;
pub mod color;
pub mod contours;
pub mod enhance;
pub mod filter;
mod fma;
pub mod geom;
pub mod image;
pub mod load;
pub mod morph;
pub mod resize;
pub mod sample;
pub mod thresh;

pub use crate::arith::{add_weighted_u8, lut, normalize_minmax_f32};
pub use crate::border::{Border, border_interpolate};
pub use crate::color::{bgr_to_gray, bgr_to_hsv_s, bgr_to_lab_l};
pub use crate::contours::{
    RotatedRect, box_points, contour_area, convex_hull, find_contours_external, min_area_rect,
};
pub use crate::enhance::*;
pub use crate::filter::{
    box_blur_f32, box_blur_row_f32, gaussian_blur_f32, gaussian_blur_f32_ksize, gaussian_blur_u8,
    gaussian_ksize, median_blur5,
};
pub use crate::geom::{
    Mat3, get_perspective_transform, homography, invert3, perspective_transform, warp_perspective,
};
pub use crate::image::{Bgr8, Gray8, GrayF32, Image};
pub use crate::load::{LoadError, decode_image, load_image};
pub use crate::morph::{close, dilate, ellipse_mask, erode, open};
pub use crate::resize::{Interp, resize, resize_area, resize_cubic, scaled_size};
pub use crate::sample::{
    remap_points, remap_points_scalar, remap_project_f32, remap_project_f32_scalar,
    remap_project_f64, remap_project_f64_scalar,
};
pub use crate::thresh::{adaptive_threshold_mean, box_mean_u8, otsu, otsu_level};
