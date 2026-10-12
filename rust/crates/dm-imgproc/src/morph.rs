// Derived from OpenCV 5.0 (morph, getStructuringElement (MORPH_ELLIPSE)),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! Grey morphology with OpenCV's ellipse structuring element: `erode`,
//! `dilate`, `MORPH_CLOSE`, `MORPH_OPEN`. Pixels outside the image are
//! ignored (OpenCV's default constant border `morphologyDefaultBorderValue`).
//!
//! The ellipse is a stack of symmetric horizontal runs: a row pass grows a
//! running min/max one pixel per side up to the widest run, a column pass
//! combines the run widths over the ellipse rows (vectorised slice ops).

use crate::fma::avx2_dispatch;
use crate::image::{Gray8, Image};

/// `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))` as a row-major
/// 0/1 mask.
pub fn ellipse_mask(k: usize) -> Vec<u8> {
    let mut m = vec![0u8; k * k];
    for (dy, lo, hi) in ellipse_runs(k) {
        let i = (dy + (k / 2) as isize) as usize;
        for j in lo..=hi {
            m[i * k + (j + (k / 2) as isize) as usize] = 1;
        }
    }
    m
}

/// Runs of the ellipse element: (row offset, first, last column offset)
/// relative to the anchor (centre).
pub fn ellipse_runs(k: usize) -> Vec<(isize, isize, isize)> {
    let r = (k / 2) as i64;
    let c = (k / 2) as i64;
    let inv_r2 = if r != 0 { 1.0 / (r * r) as f64 } else { 0.0 };
    let mut out = Vec::new();
    for i in 0..k as i64 {
        let dy = i - r;
        if dy.abs() > r {
            continue;
        }
        let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round_ties_even() as i64;
        let j1 = (c - dx).max(0);
        let j2 = (c + dx + 1).min(k as i64);
        if j1 < j2 {
            out.push((dy as isize, (j1 - c) as isize, (j2 - 1 - c) as isize));
        }
    }
    out
}

/// Running min (`MAX = false`) or max of `a` and `b`.
#[inline(always)]
fn op<const MAX: bool>(a: u8, b: u8) -> u8 {
    if MAX { a.max(b) } else { a.min(b) }
}

/// Value that leaves the other operand unchanged.
const fn identity<const MAX: bool>() -> u8 {
    if MAX { 0 } else { 255 }
}

avx2_dispatch!(
    /// Min / max filter with the ellipse (see `morph_impl`).
    fn morph<const MAX: bool>(src: &Gray8, k: usize) -> Gray8 => morph_impl
);

/// Min / max filter with the ellipse; out-of-image pixels are ignored
/// (OpenCV's default border for morphology). Each ellipse row is a
/// symmetric span `[-a, a]`, so the row pass grows the window one pixel
/// per side at a time, `m_a[x] = op(m_(a-1)[x], row[x - a], row[x + a])`,
/// keeping the widths the ellipse needs; the column pass combines them.
#[inline(always)]
fn morph_impl<const MAX: bool>(src: &Gray8, k: usize) -> Gray8 {
    let (w, h) = (src.width, src.height);
    let runs = ellipse_runs(k);
    debug_assert!(runs.iter().all(|r| r.1 == -r.2));
    let max_a = runs.iter().map(|r| r.2 as usize).max().unwrap_or(0);
    let mut slot = vec![usize::MAX; max_a + 1];
    let mut n_slots = 0;
    for r in &runs {
        let a = r.2 as usize;
        if slot[a] == usize::MAX {
            slot[a] = n_slots;
            n_slots += 1;
        }
    }
    let id = identity::<MAX>();
    let mut bufs = vec![id; n_slots * w * h];
    let mut pad = vec![id; w + 2 * max_a];
    let mut cur = vec![id; w];
    for y in 0..h {
        pad[max_a..max_a + w].copy_from_slice(src.row(y));
        cur.copy_from_slice(src.row(y));
        for a in 0..=max_a {
            if a > 0 {
                let left = &pad[max_a - a..max_a - a + w];
                let right = &pad[max_a + a..max_a + a + w];
                for ((c, &l), &r) in cur.iter_mut().zip(left).zip(right) {
                    *c = op::<MAX>(op::<MAX>(*c, l), r);
                }
            }
            if slot[a] != usize::MAX {
                let o = (slot[a] * h + y) * w;
                bufs[o..o + w].copy_from_slice(&cur);
            }
        }
    }
    let mut dst = Image::filled(w, h, id);
    for &(dy, _, a) in &runs {
        let base = slot[a as usize] * h * w;
        for y in 0..h {
            let sy = y as isize + dy;
            if !(0..h as isize).contains(&sy) {
                continue;
            }
            let o = base + sy as usize * w;
            for (d, &v) in dst.row_mut(y).iter_mut().zip(&bufs[o..o + w]) {
                *d = op::<MAX>(*d, v);
            }
        }
    }
    dst
}

/// `cv2.erode(g, ellipse(k))`.
pub fn erode(src: &Gray8, k: usize) -> Gray8 {
    morph::<false>(src, k)
}

/// `cv2.dilate(g, ellipse(k))`.
pub fn dilate(src: &Gray8, k: usize) -> Gray8 {
    morph::<true>(src, k)
}

/// `cv2.morphologyEx(g, cv2.MORPH_CLOSE, ellipse(k))`.
pub fn close(src: &Gray8, k: usize) -> Gray8 {
    erode(&dilate(src, k), k)
}

/// `cv2.morphologyEx(g, cv2.MORPH_OPEN, ellipse(k))`.
pub fn open(src: &Gray8, k: usize) -> Gray8 {
    dilate(&erode(src, k), k)
}
