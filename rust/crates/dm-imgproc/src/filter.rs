// Derived from OpenCV 5.0 (smooth (getGaussianKernelBitExact, fixed-point Gaussian), box and median filters),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! Linear and rank filters: `GaussianBlur` (8-bit bit-exact fixed point and
//! float), `blur` (box) and `medianBlur(5)`.

use crate::border::{Border, border_table};
use crate::fma::{avx2_dispatch, fma_dispatch, madd};
use crate::image::{Gray8, GrayF32, Image};

/// OpenCV kernel size rule for `ksize=(0,0)`: `round(sigma*3*2+1)|1` for
/// 8-bit images, `round(sigma*4*2+1)|1` for float images.
pub fn gaussian_ksize(sigma: f64, eight_bit: bool) -> usize {
    let r = if eight_bit { 3.0 } else { 4.0 };
    ((sigma * r * 2.0 + 1.0).round_ties_even() as usize) | 1
}

/// `getGaussianKernelBitExact` (softdouble = f64) for `sigma > 0`.
pub fn gaussian_kernel_f64(n: usize, sigma: f64) -> Vec<f64> {
    let scale2x = -0.125 / (sigma * sigma);
    let n2 = (n - 1) / 2;
    let mut values = Vec::with_capacity(n2);
    let mut sum = 0.0f64;
    let mut x = 1 - n as i64;
    for _ in 0..n2 {
        let t = ((x * x) as f64 * scale2x).exp();
        values.push(t);
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if n.is_multiple_of(2) {
        sum += 1.0;
    }
    let mul1 = 1.0 / sum;
    let mut k = vec![0.0; n];
    for (i, v) in values.iter().enumerate() {
        k[i] = v * mul1;
        k[n - 1 - i] = v * mul1;
    }
    k[n2] = mul1;
    if n.is_multiple_of(2) {
        k[n2 + 1] = k[n2];
    }
    k
}

/// `getGaussianKernelFixedPoint_ED` with 8 fraction bits (`ufixedpoint16`):
/// rounding with error diffusion, centre tap takes the remainder.
fn gaussian_kernel_fixed(k: &[f64]) -> Vec<u32> {
    let n = k.len();
    let n2 = n / 2;
    let mut out = vec![0u32; n];
    let mut err = 0.0f64;
    let mut sum = 0i64;
    for i in 0..n2 {
        let adj = k[i] * 256.0 + err;
        let v = adj.round_ties_even() as i64;
        err = adj - v as f64;
        out[i] = v as u32;
        out[n - 1 - i] = v as u32;
        sum += v;
    }
    out[n2] = (256 - 2 * sum) as u32;
    out
}

avx2_dispatch!(
    /// `cv2.GaussianBlur(g, (0, 0), sigma)` on an 8-bit image (bit-exact
    /// fixed-point path, `BORDER_REFLECT_101`).
    pub fn gaussian_blur_u8(src: &Gray8, sigma: f64) -> Gray8 => gaussian_blur_u8_impl
);

#[inline(always)]
fn gaussian_blur_u8_impl(src: &Gray8, sigma: f64) -> Gray8 {
    let n = gaussian_ksize(sigma, true);
    let k = gaussian_kernel_fixed(&gaussian_kernel_f64(n, sigma));
    let (w, h) = (src.width, src.height);
    let r = n / 2;
    let xt = border_table(w, r, r, Border::Reflect101);
    let yt = border_table(h, r, r, Border::Reflect101);
    // horizontal: u8 x 8.8 -> 8.8 fixed point (exact in u32)
    // (tap-major over a padded row; integer sums, so the order is free)
    let mut tmp = vec![0u32; w * h];
    let mut ext = vec![0u32; w + 2 * r];
    for y in 0..h {
        let row = src.row(y);
        for (e, &i) in ext.iter_mut().zip(&xt) {
            *e = row[i] as u32;
        }
        let out = &mut tmp[y * w..(y + 1) * w];
        for (j, &kj) in k.iter().enumerate() {
            for (o, &e) in out.iter_mut().zip(&ext[j..j + w]) {
                *o += kj * e;
            }
        }
    }
    // vertical: 8.8 x 8.8 -> 16.16, rounded to u8
    let mut dst = Image::new(w, h);
    let mut acc = vec![0u32; w];
    for y in 0..h {
        acc.iter_mut().for_each(|a| *a = 0);
        for (j, &kj) in k.iter().enumerate() {
            let srow = &tmp[yt[y + j] * w..(yt[y + j] + 1) * w];
            for (a, &s) in acc.iter_mut().zip(srow) {
                *a += kj * s;
            }
        }
        for (d, &a) in dst.row_mut(y).iter_mut().zip(&acc) {
            *d = ((a + (1 << 15)) >> 16).min(255) as u8;
        }
    }
    dst
}

/// `cv2.GaussianBlur(g, (0, 0), sigma)` on a float image (`sepFilter2D`
/// with an f32 kernel, `BORDER_REFLECT_101`). Same summation order as
/// OpenCV's row filter and symmetric column filter.
pub fn gaussian_blur_f32(src: &GrayF32, sigma: f64) -> GrayF32 {
    let n = gaussian_ksize(sigma, false);
    let k: Vec<f32> = gaussian_kernel_f64(n, sigma)
        .iter()
        .map(|&v| v as f32)
        .collect();
    sep_filter_symm_f32(src, &k, &k, Border::Reflect101)
}

/// `cv2.GaussianBlur(g, (kw, kh), sigma)` on a float image with an explicit
/// kernel size (sigmaY = sigma; `kh = 1` is the identity along y).
pub fn gaussian_blur_f32_ksize(src: &GrayF32, kw: usize, kh: usize, sigma: f64) -> GrayF32 {
    let kern = |n| -> Vec<f32> {
        gaussian_kernel_f64(n, sigma)
            .iter()
            .map(|&v| v as f32)
            .collect()
    };
    sep_filter_symm_f32(src, &kern(kw), &kern(kh), Border::Reflect101)
}

fma_dispatch!(avx2
    /// Separable filter with symmetric kernels `kx` (rows) and `ky`
    /// (columns), fused like the reference build (see `fma`).
    fn sep_filter_symm_f32(src: &GrayF32, kx: &[f32], ky: &[f32], border: Border) -> GrayF32 => sep_filter_symm_impl
);

#[inline(always)]
fn sep_filter_symm_impl<const FMA: bool>(
    src: &GrayF32,
    kx: &[f32],
    ky: &[f32],
    border: Border,
) -> GrayF32 {
    let (w, h) = (src.width, src.height);
    let (k, n) = (kx, kx.len());
    let r = n / 2;
    let xt = border_table(w, r, r, border);
    let ry = ky.len() / 2;
    let yt = border_table(h, ry, ry, border);
    // OpenCV's SIMD loops cover whole 8-lane blocks; the scalar tail of each
    // row is not fused
    let vec_end = w - w % 8;
    // rows: s = sum_k kx[k] * x[k], k ascending (RowVec_32f)
    let mut tmp = vec![0f32; w * h];
    let mut ext = vec![0f32; w + 2 * r];
    for y in 0..h {
        let row = src.row(y);
        for (e, &i) in ext.iter_mut().zip(&xt) {
            *e = row[i];
        }
        let out = &mut tmp[y * w..(y + 1) * w];
        // tap-major loops vectorise over x; each output still sums its taps
        // in ascending order
        let (head, tail) = out.split_at_mut(vec_end);
        head.iter_mut().for_each(|o| *o = 0.0);
        for (j, &kj) in k.iter().enumerate() {
            for (o, &a) in head.iter_mut().zip(&ext[j..j + vec_end]) {
                *o = madd::<FMA>(a, kj, *o);
            }
        }
        for (i, o) in tail.iter_mut().enumerate() {
            let x = vec_end + i;
            let mut s = 0f32;
            for (&a, &b) in ext[x..x + n].iter().zip(k) {
                s += a * b;
            }
            *o = s;
        }
    }
    // columns: s = k0 * c + sum_j kj * (up_j + down_j) (SymmColumnVec_32f)
    let mut dst = Image::new(w, h);
    let mut acc = vec![0f32; w];
    let (k, r) = (ky, ry);
    for y in 0..h {
        let c = &tmp[yt[y + r] * w..(yt[y + r] + 1) * w];
        for (a, &v) in acc.iter_mut().zip(c) {
            *a = madd::<FMA>(v, k[r], 0.0);
        }
        for j in 1..=r {
            let p = &tmp[yt[y + r + j] * w..(yt[y + r + j] + 1) * w];
            let m = &tmp[yt[y + r - j] * w..(yt[y + r - j] + 1) * w];
            let kj = k[r + j];
            for (x, ((a, &u), &v)) in acc.iter_mut().zip(p).zip(m).enumerate() {
                *a = if x < vec_end {
                    madd::<FMA>(u + v, kj, *a)
                } else {
                    *a + kj * (u + v)
                };
            }
        }
        dst.row_mut(y).copy_from_slice(&acc);
    }
    dst
}

/// `cv2.blur(img, (k, k), borderType=border)` on a float image (normalised
/// box filter; sums in f64 as OpenCV's `RowSum<float, double>`).
pub fn box_blur_f32(src: &GrayF32, k: usize, border: Border) -> GrayF32 {
    let (w, h) = (src.width, src.height);
    let r = k / 2;
    let pre = r;
    let post = k - 1 - r;
    let xt = border_table(w, pre, post, border);
    let yt = border_table(h, pre, post, border);
    let mut tmp = vec![0f64; w * h];
    for y in 0..h {
        let row = src.row(y);
        for x in 0..w {
            tmp[y * w + x] = xt[x..x + k].iter().map(|&i| row[i] as f64).sum();
        }
    }
    let scale = 1.0 / (k * k) as f64;
    let mut dst = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let s: f64 = yt[y..y + k].iter().map(|&j| tmp[j * w + x]).sum();
            dst.set(x, y, (s * scale) as f32);
        }
    }
    dst
}

/// `cv2.blur(img, (k, 1), borderType=border)` on a float image: box mean
/// along x only, with OpenCV's running row sum in f64 (`RowSum<float,
/// double>`: first window summed, then `s += x[i + k] - x[i]`).
pub fn box_blur_row_f32(src: &GrayF32, k: usize, border: Border) -> GrayF32 {
    let w = src.width;
    let r = k / 2;
    let xt = border_table(w, r, k - 1 - r, border);
    let scale = 1.0 / k as f64;
    let mut dst = Image::new(w, src.height);
    let mut ext = vec![0f64; w + k - 1];
    for y in 0..src.height {
        let row = src.row(y);
        for (e, &i) in ext.iter_mut().zip(&xt) {
            *e = row[i] as f64;
        }
        let mut s: f64 = ext[..k].iter().sum();
        let out = dst.row_mut(y);
        out[0] = (s * scale) as f32;
        for x in 1..w {
            s += ext[x - 1 + k] - ext[x - 1];
            out[x] = (s * scale) as f32;
        }
    }
    dst
}

/// `cv2.medianBlur(g, 5)` (`BORDER_REPLICATE`).
pub fn median_blur5(src: &Gray8) -> Gray8 {
    let (w, h) = (src.width, src.height);
    let xt = border_table(w, 2, 2, Border::Replicate);
    let yt = border_table(h, 2, 2, Border::Replicate);
    let mut dst = Image::new(w, h);
    let mut win = [0u8; 25];
    for y in 0..h {
        for x in 0..w {
            let mut i = 0;
            for dy in 0..5 {
                let row = src.row(yt[y + dy]);
                for dx in 0..5 {
                    win[i] = row[xt[x + dx]];
                    i += 1;
                }
            }
            let (_, m, _) = win.select_nth_unstable(12);
            dst.set(x, y, *m);
        }
    }
    dst
}
