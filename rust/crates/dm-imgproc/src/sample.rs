//! Point sampling kernels of the grid sampler (P4b.9): bilinear `remap` at
//! arbitrary points, optionally after projecting module coordinates through
//! a homography. With AVX2 + FMA eight points are done per step (gathered
//! pixels, vector division); results are bit-identical to the scalar
//! versions, which stay as the fallback and the test reference. A vector
//! with a point outside the interior (border, NaN, huge coordinates) is
//! done lane by lane with the scalar code.

use crate::fma::fma_dispatch;
use crate::geom::{Mat3, bilinear};
use crate::image::GrayF32;

/// AVX2 and FMA available (x86-64 runtime check, cached by std).
#[cfg(target_arch = "x86_64")]
#[inline]
fn simd() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// `cv2.remap(img, x, y, INTER_LINEAR, borderMode=BORDER_REPLICATE)` at
/// arbitrary points (`remap_points`).
pub fn remap_points(img: &GrayF32, xs: &[f32], ys: &[f32]) -> Vec<f32> {
    assert_eq!(xs.len(), ys.len());
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the CPU supports AVX2 and FMA (checked above).
        return unsafe { x86::remap_points(img, xs, ys) };
    }
    remap_points_scalar(img, xs, ys)
}

fma_dispatch!(
    /// Scalar `remap_points` (fallback and test reference).
    #[doc(hidden)]
    pub fn remap_points_scalar(img: &GrayF32, xs: &[f32], ys: &[f32]) -> Vec<f32> => remap_impl
);

#[inline(always)]
fn remap_impl<const FMA: bool>(img: &GrayF32, xs: &[f32], ys: &[f32]) -> Vec<f32> {
    assert_eq!(xs.len(), ys.len());
    xs.iter()
        .zip(ys)
        .map(|(&x, &y)| bilinear::<FMA>(img, x, y))
        .collect()
}

/// Module coordinates `uv` plus each offset of `offs` (offset-major), mapped
/// through `h` in float64 (`x = (h0 u + h1 v + h2) / (h6 u + h7 v + h8)`,
/// cast to float32) and sampled bilinearly: the reader's `_sample_uv`.
pub fn remap_project_f64(img: &GrayF32, h: &Mat3, uv: &[[f64; 2]], offs: &[[f64; 2]]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the CPU supports AVX2 and FMA (checked above).
        return unsafe { x86::project_f64(img, h, uv, offs) };
    }
    remap_project_f64_scalar(img, h, uv, offs)
}

fma_dispatch!(
    /// Scalar `remap_project_f64` (fallback and test reference).
    #[doc(hidden)]
    pub fn remap_project_f64_scalar(
        img: &GrayF32,
        h: &Mat3,
        uv: &[[f64; 2]],
        offs: &[[f64; 2]],
    ) -> Vec<f32> => project_f64_impl
);

#[inline(always)]
fn project_f64_impl<const FMA: bool>(
    img: &GrayF32,
    h: &Mat3,
    uv: &[[f64; 2]],
    offs: &[[f64; 2]],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(offs.len() * uv.len());
    for o in offs {
        for p in uv {
            let (px, py) = (p[0] + o[0], p[1] + o[1]);
            let (x, y) = point_f64(h, px, py);
            out.push(bilinear::<FMA>(img, x, y));
        }
    }
    out
}

#[inline(always)]
fn point_f64(h: &Mat3, px: f64, py: f64) -> (f32, f32) {
    let den = h[6] * px + h[7] * py + h[8];
    (
        ((h[0] * px + h[1] * py + h[2]) / den) as f32,
        ((h[3] * px + h[4] * py + h[5]) / den) as f32,
    )
}

/// Points mapped through a float32 homography (denominator clamped to
/// 1e-9 in magnitude below it, as the batched numpy code) and sampled
/// bilinearly: the reader's `_sample_uv_many` for one quad.
pub fn remap_project_f32(img: &GrayF32, h: &[f32; 9], pts: &[[f32; 2]]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the CPU supports AVX2 and FMA (checked above).
        return unsafe { x86::project_f32(img, h, pts) };
    }
    remap_project_f32_scalar(img, h, pts)
}

fma_dispatch!(
    /// Scalar `remap_project_f32` (fallback and test reference).
    #[doc(hidden)]
    pub fn remap_project_f32_scalar(img: &GrayF32, h: &[f32; 9], pts: &[[f32; 2]]) -> Vec<f32>
        => project_f32_impl
);

#[inline(always)]
fn project_f32_impl<const FMA: bool>(img: &GrayF32, h: &[f32; 9], pts: &[[f32; 2]]) -> Vec<f32> {
    pts.iter()
        .map(|p| {
            let (x, y) = point_f32(h, p[0], p[1]);
            bilinear::<FMA>(img, x, y)
        })
        .collect()
}

#[inline(always)]
fn point_f32(h: &[f32; 9], px: f32, py: f32) -> (f32, f32) {
    let mut den = h[6] * px + h[7] * py + h[8];
    if den.abs() < 1e-9 {
        den = 1e-9;
    }
    (
        (h[0] * px + h[1] * py + h[2]) / den,
        (h[3] * px + h[4] * py + h[5]) / den,
    )
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{bilinear, point_f32, point_f64};
    use crate::geom::Mat3;
    use crate::image::GrayF32;
    use std::arch::x86_64::*;

    /// Bilinear samples at 8 points into `out[..8]`; gathers when all 8 are
    /// interior, the scalar code per lane otherwise.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn bilinear8(img: &GrayF32, sx: __m256, sy: __m256, out: &mut [f32]) {
        let (w, h) = (img.width as i32, img.height as i32);
        let ix = _mm256_cvttps_epi32(_mm256_floor_ps(sx));
        let iy = _mm256_cvttps_epi32(_mm256_floor_ps(sy));
        let m1 = _mm256_set1_epi32(-1);
        let inx = _mm256_and_si256(
            _mm256_cmpgt_epi32(ix, m1),
            _mm256_cmpgt_epi32(_mm256_set1_epi32(w - 1), ix),
        );
        let iny = _mm256_and_si256(
            _mm256_cmpgt_epi32(iy, m1),
            _mm256_cmpgt_epi32(_mm256_set1_epi32(h - 1), iy),
        );
        let inside = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(inx, iny)));
        let out: &mut [f32; 8] = (&mut out[..8]).try_into().unwrap();
        if inside != 0xFF {
            let (mut xs, mut ys) = ([0f32; 8], [0f32; 8]);
            // SAFETY: 8-float stores into 8-float arrays.
            unsafe {
                _mm256_storeu_ps(xs.as_mut_ptr(), sx);
                _mm256_storeu_ps(ys.as_mut_ptr(), sy);
            }
            for i in 0..8 {
                out[i] = bilinear::<true>(img, xs[i], ys[i]);
            }
            return;
        }
        let fx = _mm256_sub_ps(sx, _mm256_cvtepi32_ps(ix));
        let fy = _mm256_sub_ps(sy, _mm256_cvtepi32_ps(iy));
        let o = _mm256_add_epi32(_mm256_mullo_epi32(iy, _mm256_set1_epi32(w)), ix);
        let base = img.data.as_ptr();
        let wu = w as usize;
        // SAFETY: every lane has 0 <= ix < w - 1 and 0 <= iy < h - 1, so
        // o, o + 1, o + w and o + w + 1 are inside the w * h pixel buffer.
        let (p00, p01, p10, p11) = unsafe {
            (
                _mm256_i32gather_ps::<4>(base, o),
                _mm256_i32gather_ps::<4>(base.add(1), o),
                _mm256_i32gather_ps::<4>(base.add(wu), o),
                _mm256_i32gather_ps::<4>(base.add(wu + 1), o),
            )
        };
        let v0 = _mm256_fmadd_ps(fx, _mm256_sub_ps(p01, p00), p00);
        let v1 = _mm256_fmadd_ps(fx, _mm256_sub_ps(p11, p10), p10);
        let v = _mm256_fmadd_ps(fy, _mm256_sub_ps(v1, v0), v0);
        // SAFETY: `out` has 8 floats.
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn remap_points(img: &GrayF32, xs: &[f32], ys: &[f32]) -> Vec<f32> {
        let n = xs.len();
        let mut out = vec![0f32; n];
        let end = n - n % 8;
        for i in (0..end).step_by(8) {
            // SAFETY: i + 8 <= n for both inputs.
            let (sx, sy) = unsafe {
                (
                    _mm256_loadu_ps(xs.as_ptr().add(i)),
                    _mm256_loadu_ps(ys.as_ptr().add(i)),
                )
            };
            bilinear8(img, sx, sy, &mut out[i..]);
        }
        for i in end..n {
            out[i] = bilinear::<true>(img, xs[i], ys[i]);
        }
        out
    }

    /// 4 projected coordinates (x or y) in float64, cast to float32.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn proj4(h: &Mat3, row: usize, px: __m256d, py: __m256d, den: __m256d) -> __m128 {
        let a = _mm256_add_pd(
            _mm256_add_pd(
                _mm256_mul_pd(_mm256_set1_pd(h[row]), px),
                _mm256_mul_pd(_mm256_set1_pd(h[row + 1]), py),
            ),
            _mm256_set1_pd(h[row + 2]),
        );
        _mm256_cvtpd_ps(_mm256_div_pd(a, den))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn project_f64(
        img: &GrayF32,
        h: &Mat3,
        uv: &[[f64; 2]],
        offs: &[[f64; 2]],
    ) -> Vec<f32> {
        let n = uv.len();
        let mut out = vec![0f32; offs.len() * n];
        let end = n - n % 8;
        let flat = uv.as_flattened();
        for (k, o) in offs.iter().enumerate() {
            let dst = &mut out[k * n..(k + 1) * n];
            let (ox, oy) = (_mm256_set1_pd(o[0]), _mm256_set1_pd(o[1]));
            for i in (0..end).step_by(8) {
                let mut half = [_mm_setzero_ps(); 4];
                for j in 0..2 {
                    // SAFETY: points i + 4j .. i + 4j + 4 exist (i + 8 <= n),
                    // 8 f64 from index 2 (i + 4j).
                    let (a, b) = unsafe {
                        let p = flat.as_ptr().add(2 * (i + 4 * j));
                        (_mm256_loadu_pd(p), _mm256_loadu_pd(p.add(4)))
                    };
                    // [x0 y0 x1 y1], [x2 y2 x3 y3] -> [x0 x1 x2 x3], [y0 .. y3]
                    let xs = _mm256_permute4x64_pd::<0xD8>(_mm256_unpacklo_pd(a, b));
                    let ys = _mm256_permute4x64_pd::<0xD8>(_mm256_unpackhi_pd(a, b));
                    let px = _mm256_add_pd(xs, ox);
                    let py = _mm256_add_pd(ys, oy);
                    let den = _mm256_add_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_set1_pd(h[6]), px),
                            _mm256_mul_pd(_mm256_set1_pd(h[7]), py),
                        ),
                        _mm256_set1_pd(h[8]),
                    );
                    half[j] = proj4(h, 0, px, py, den);
                    half[2 + j] = proj4(h, 3, px, py, den);
                }
                let sx = _mm256_set_m128(half[1], half[0]);
                let sy = _mm256_set_m128(half[3], half[2]);
                bilinear8(img, sx, sy, &mut dst[i..]);
            }
            for i in end..n {
                let (x, y) = point_f64(h, uv[i][0] + o[0], uv[i][1] + o[1]);
                dst[i] = bilinear::<true>(img, x, y);
            }
        }
        out
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn project_f32(img: &GrayF32, h: &[f32; 9], pts: &[[f32; 2]]) -> Vec<f32> {
        let n = pts.len();
        let mut out = vec![0f32; n];
        let end = n - n % 8;
        let flat = pts.as_flattened();
        let hv = h.map(|v| _mm256_set1_ps(v));
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        let tiny = _mm256_set1_ps(1e-9);
        for i in (0..end).step_by(8) {
            // SAFETY: points i .. i + 8 exist: 16 floats from index 2 i.
            let (a, b) = unsafe {
                let p = flat.as_ptr().add(2 * i);
                (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)))
            };
            // per 128-bit lane [x0 x1 x4 x5 | x2 x3 x6 x7] -> x0 .. x7
            let px = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(
                _mm256_shuffle_ps::<0x88>(a, b),
            )));
            let py = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(
                _mm256_shuffle_ps::<0xDD>(a, b),
            )));
            let lin = |r: usize| {
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(hv[r], px), _mm256_mul_ps(hv[r + 1], py)),
                    hv[r + 2],
                )
            };
            let den = lin(6);
            let small = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(den, abs_mask), tiny);
            let den = _mm256_blendv_ps(den, tiny, small);
            let sx = _mm256_div_ps(lin(0), den);
            let sy = _mm256_div_ps(lin(3), den);
            bilinear8(img, sx, sy, &mut out[i..]);
        }
        for i in end..n {
            let (x, y) = point_f32(h, pts[i][0], pts[i][1]);
            out[i] = bilinear::<true>(img, x, y);
        }
        out
    }
}
