// Derived from OpenCV 5.0 (imgwarp (getPerspectiveTransform, warp kernels), core LU),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! Perspective geometry: `getPerspectiveTransform` (8x8 LU, f64), the batched
//! `homographies` of the reader, `perspectiveTransform`, 3x3 `invert`,
//! bilinear `remap` with `BORDER_REPLICATE` on float images and
//! `warpPerspective`.
//!
//! OpenCV 5 interpolates at the exact float coordinates (`warp_kernels`),
//! `v = p00 + fx*(p01 - p00)` etc. with FMA; the reference build also fuses
//! the `a*b + c*d` style sums in the scalar C++ code (see `fma`).

use crate::fma::{fma_dispatch, madd};
use crate::image::{GrayF32, Image};

/// Row-major 3x3 matrix.
pub type Mat3 = [f64; 9];

/// `a*b + c*d` in f64 with the first product fused (GCC contraction).
#[inline]
fn dot2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a.mul_add(b, c * d)
}

/// `a*b - c*d` in f64 with the first product fused.
#[inline]
fn cross2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a.mul_add(b, -(c * d))
}

/// OpenCV `LUImpl<double>` (partial pivoting, fused updates); solves
/// `a x = b` in place, false if singular.
fn lu_solve(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> bool {
    let m = 8;
    for i in 0..m {
        let mut k = i;
        for j in i + 1..m {
            if a[j][i].abs() > a[k][i].abs() {
                k = j;
            }
        }
        if a[k][i].abs() < f64::EPSILON * 100.0 {
            return false;
        }
        if k != i {
            a.swap(i, k);
            b.swap(i, k);
        }
        let d = -1.0 / a[i][i];
        let (top, rest) = a.split_at_mut(i + 1);
        let pivot = &top[i];
        for (j, row) in rest.iter_mut().enumerate() {
            let alpha = row[i] * d;
            for (r, &p) in row[i + 1..].iter_mut().zip(&pivot[i + 1..]) {
                *r = alpha.mul_add(p, *r);
            }
            b[i + 1 + j] = alpha.mul_add(b[i], b[i + 1 + j]);
        }
    }
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s = (-a[i][k]).mul_add(b[k], s);
        }
        b[i] = s / a[i][i];
    }
    true
}

/// `cv2.getPerspectiveTransform(src, dst)` for f32 corner arrays. `None`
/// when the LU solve fails (OpenCV then falls back to an SVD solution of a
/// degenerate quad; not needed by the reader).
pub fn get_perspective_transform(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Option<Mat3> {
    let mut a = [[0f64; 8]; 8];
    let mut b = [0f64; 8];
    for i in 0..4 {
        let (sx, sy) = src[i];
        let (dx, dy) = dst[i];
        a[i][0] = sx as f64;
        a[i + 4][3] = sx as f64;
        a[i][1] = sy as f64;
        a[i + 4][4] = sy as f64;
        a[i][2] = 1.0;
        a[i + 4][5] = 1.0;
        // float products, as in OpenCV
        a[i][6] = (-sx * dx) as f64;
        a[i][7] = (-sy * dx) as f64;
        a[i + 4][6] = (-sx * dy) as f64;
        a[i + 4][7] = (-sy * dy) as f64;
        b[i] = dx as f64;
        b[i + 4] = dy as f64;
    }
    let a0 = a;
    let b0 = b;
    if !lu_solve(&mut a, &mut b) {
        return None;
    }
    // OpenCV checks norm(A x - b) < 1e-8 before accepting the solution
    let mut r2 = 0.0;
    for i in 0..8 {
        let s: f64 = (0..8).map(|j| a0[i][j] * b[j]).sum();
        r2 += (s - b0[i]).powi(2);
    }
    if r2.sqrt() >= 1e-8 {
        return None;
    }
    Some([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.0])
}

/// `homographies(src, quads)`: transform `src` (4 corners, f64) to each quad
/// (numpy `linalg.solve` in Python; the same LU here). `None` for a
/// degenerate quad.
pub fn homography(src: &[(f64, f64); 4], quad: &[(f64, f64); 4]) -> Option<Mat3> {
    let mut a = [[0f64; 8]; 8];
    let mut b = [0f64; 8];
    for i in 0..4 {
        let (x, y) = src[i];
        let (u, v) = quad[i];
        a[2 * i][0] = x;
        a[2 * i][1] = y;
        a[2 * i][2] = 1.0;
        a[2 * i][6] = -x * u;
        a[2 * i][7] = -y * u;
        a[2 * i + 1][3] = x;
        a[2 * i + 1][4] = y;
        a[2 * i + 1][5] = 1.0;
        a[2 * i + 1][6] = -x * v;
        a[2 * i + 1][7] = -y * v;
        b[2 * i] = u;
        b[2 * i + 1] = v;
    }
    if !lu_solve(&mut a, &mut b) {
        return None;
    }
    Some([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.0])
}

/// `cv2.perspectiveTransform(pts, m)` for f32 points (f64 maths).
pub fn perspective_transform(pts: &[(f32, f32)], m: &Mat3) -> Vec<(f32, f32)> {
    pts.iter()
        .map(|&(x, y)| {
            let (x, y) = (x as f64, y as f64);
            let w = dot2(x, m[6], y, m[7]) + m[8];
            if w.abs() > f32::EPSILON as f64 {
                let w = 1.0 / w;
                (
                    ((dot2(x, m[0], y, m[1]) + m[2]) * w) as f32,
                    ((dot2(x, m[3], y, m[4]) + m[5]) * w) as f32,
                )
            } else {
                (0.0, 0.0)
            }
        })
        .collect()
}

/// `cv2.invert(m)` of a 3x3 f64 matrix (closed form, `DECOMP_LU`).
pub fn invert3(m: &Mat3) -> Option<Mat3> {
    let s = |r: usize, c: usize| m[r * 3 + c];
    let x = cross2(s(1, 1), s(2, 2), s(1, 2), s(2, 1));
    let y = cross2(s(1, 0), s(2, 2), s(1, 2), s(2, 0));
    let z = cross2(s(1, 0), s(2, 1), s(1, 1), s(2, 0));
    let d = s(0, 2).mul_add(z, s(0, 0).mul_add(x, -(s(0, 1) * y)));
    if d == 0.0 {
        return None;
    }
    let d = 1.0 / d;
    let t = |a: f64, b: f64, c: f64, e: f64| cross2(a, b, c, e) * d;
    Some([
        t(s(1, 1), s(2, 2), s(1, 2), s(2, 1)),
        t(s(0, 2), s(2, 1), s(0, 1), s(2, 2)),
        t(s(0, 1), s(1, 2), s(0, 2), s(1, 1)),
        // shares both products with `y`: GCC fuses the other one here
        (-s(1, 0)).mul_add(s(2, 2), s(1, 2) * s(2, 0)) * d,
        t(s(0, 0), s(2, 2), s(0, 2), s(2, 0)),
        t(s(0, 2), s(1, 0), s(0, 0), s(1, 2)),
        t(s(1, 0), s(2, 1), s(1, 1), s(2, 0)),
        t(s(0, 1), s(2, 0), s(0, 0), s(2, 1)),
        t(s(0, 0), s(1, 1), s(0, 1), s(1, 0)),
    ])
}

/// `cvFloor` of a float as the x86 conversion does it (NaN and out of range
/// give `i32::MIN`).
#[inline(always)]
fn floor_i32(v: f32) -> i64 {
    let f = v.floor();
    if f.is_nan() || f >= 2147483648.0 || f < -2147483648.0 {
        i32::MIN as i64
    } else {
        f as i64
    }
}

/// Bilinear sample at (`sx`, `sy`) with replicated border.
#[inline(always)]
pub(crate) fn bilinear<const FMA: bool>(img: &GrayF32, sx: f32, sy: f32) -> f32 {
    let (w, h) = (img.width as i64, img.height as i64);
    let ix = floor_i32(sx);
    let iy = floor_i32(sy);
    let fx = sx - ix as i32 as f32;
    let fy = sy - iy as i32 as f32;
    let (p00, p01, p10, p11);
    if (0..w - 1).contains(&ix) && (0..h - 1).contains(&iy) {
        let o = (iy * w + ix) as usize;
        let d = &img.data;
        (p00, p01, p10, p11) = (d[o], d[o + 1], d[o + w as usize], d[o + w as usize + 1]);
    } else {
        let px = |dx: i64, dy: i64| {
            let x = (ix + dx).clamp(0, w - 1);
            let y = (iy + dy).clamp(0, h - 1);
            img.data[(y * w + x) as usize]
        };
        (p00, p01, p10, p11) = (px(0, 0), px(1, 0), px(0, 1), px(1, 1));
    }
    let v0 = madd::<FMA>(fx, p01 - p00, p00);
    let v1 = madd::<FMA>(fx, p11 - p10, p10);
    madd::<FMA>(fy, v1 - v0, v0)
}

fma_dispatch!(
    /// `cv2.warpPerspective(img, m, (w, h), borderMode=BORDER_REPLICATE)`
    /// (INTER_LINEAR, `m` maps source to destination).
    pub fn warp_perspective(img: &GrayF32, m: &Mat3, w: usize, h: usize) -> GrayF32 => warp_impl
);

#[inline(always)]
fn warp_impl<const FMA: bool>(img: &GrayF32, m: &Mat3, w: usize, h: usize) -> GrayF32 {
    let mut dst = Image::new(w, h);
    let Some(inv) = invert3(m) else { return dst };
    let mf: Vec<f32> = inv.iter().map(|&v| v as f32).collect();
    // OpenCV's AVX2 kernel handles 16 pixels per step; the scalar tail
    // computes the coordinates in a different order
    let vec_end = w - w % 16;
    for y in 0..h {
        let yf = y as f32;
        // per-row offsets are not fused in the reference build
        let m_x = yf * mf[1] + mf[2];
        let m_y = yf * mf[4] + mf[5];
        let m_w = yf * mf[7] + mf[8];
        let row = dst.row_mut(y);
        for (x, d) in row.iter_mut().enumerate() {
            let xf = x as f32;
            let (sx, sy) = if x < vec_end {
                let ww = madd::<FMA>(mf[6], xf, m_w);
                (
                    madd::<FMA>(mf[0], xf, m_x) / ww,
                    madd::<FMA>(mf[3], xf, m_y) / ww,
                )
            } else {
                let ww = madd::<FMA>(xf, mf[6], yf * mf[7]) + mf[8];
                (
                    (madd::<FMA>(xf, mf[0], yf * mf[1]) + mf[2]) / ww,
                    (madd::<FMA>(xf, mf[3], yf * mf[4]) + mf[5]) / ww,
                )
            };
            *d = bilinear::<FMA>(img, sx, sy);
        }
    }
    dst
}
