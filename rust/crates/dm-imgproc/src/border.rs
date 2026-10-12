// Derived from OpenCV 5.0 (core borderInterpolate),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! OpenCV border modes (`borderInterpolate`).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// `aaaaaa|abcdefgh|hhhhhhh`
    Replicate,
    /// `fedcba|abcdefgh|hgfedcb`
    Reflect,
    /// `gfedcb|abcdefgh|gfedcba` (`BORDER_DEFAULT`)
    Reflect101,
}

/// Source index for a possibly out-of-range coordinate `p` along an axis of
/// length `len` (> 0).
#[inline]
pub fn border_interpolate(p: isize, len: usize, border: Border) -> usize {
    let n = len as isize;
    if (0..n).contains(&p) {
        return p as usize;
    }
    match border {
        Border::Replicate => p.clamp(0, n - 1) as usize,
        Border::Reflect | Border::Reflect101 => {
            if n == 1 {
                return 0;
            }
            let delta = (border == Border::Reflect101) as isize;
            let mut p = p;
            loop {
                if p < 0 {
                    p = -p - 1 + delta;
                } else {
                    p = n - 1 - (p - n) - delta;
                }
                if (0..n).contains(&p) {
                    return p as usize;
                }
            }
        }
    }
}

/// Index table for `[-pre, len + post)`: entry `i` is the source index of
/// coordinate `i - pre`.
pub fn border_table(len: usize, pre: usize, post: usize, border: Border) -> Vec<usize> {
    (-(pre as isize)..(len + post) as isize)
        .map(|p| border_interpolate(p, len, border))
        .collect()
}
