// Derived from OpenCV 5.0 (color_rgb, color_hsv, color_lab),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! `cv2.cvtColor` for 8-bit BGR input: gray, the S channel of HSV and the L
//! channel of L*a*b*, with OpenCV's fixed-point formulas and tables
//! (`color_rgb`, `color_hsv`, `color_lab` in OpenCV 5.0).

use crate::image::{Bgr8, Gray8, Image};
use std::sync::OnceLock;

/// `COLOR_BGR2GRAY`: `(b*3735 + g*19235 + r*9798 + 2^14) >> 15`.
pub fn bgr_to_gray(img: &Bgr8) -> Gray8 {
    const BY: u32 = 3735;
    const GY: u32 = 19235;
    const RY: u32 = 9798;
    let data = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| ((p[0] as u32 * BY + p[1] as u32 * GY + p[2] as u32 * RY + (1 << 14)) >> 15) as u8)
        .collect();
    Image::from_vec(img.width, img.height, data)
}

/// Channel 1 (saturation) of `COLOR_BGR2HSV`.
pub fn bgr_to_hsv_s(img: &Bgr8) -> Gray8 {
    static SDIV: OnceLock<[i32; 256]> = OnceLock::new();
    let sdiv = SDIV.get_or_init(|| {
        let mut t = [0i32; 256];
        for (i, v) in t.iter_mut().enumerate().skip(1) {
            *v = ((255 << 12) as f64 / i as f64).round_ties_even() as i32;
        }
        t
    });
    let data = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let v = p[0].max(p[1]).max(p[2]);
            let diff = (v - p[0].min(p[1]).min(p[2])) as i32;
            ((diff * sdiv[v as usize] + (1 << 11)) >> 12) as u8
        })
        .collect();
    Image::from_vec(img.width, img.height, data)
}

const GAMMA_SHIFT: u32 = 3;
const LAB_SHIFT: u32 = 12;
const LAB_SHIFT2: u32 = LAB_SHIFT + GAMMA_SHIFT;
const LAB_CBRT_TAB_SIZE_B: usize = 256 * 3 / 2 * (1 << GAMMA_SHIFT);

struct LabTabs {
    gamma: [i32; 256],
    cbrt: Vec<i32>,
}

/// `sRGBGammaTab_b` and `LabCbrtTab_b` (`initLabTabs`, softfloat = f32).
fn lab_tabs() -> &'static LabTabs {
    static TABS: OnceLock<LabTabs> = OnceLock::new();
    TABS.get_or_init(|| {
        let apply_gamma = |x: f32| -> f32 {
            let xd = x as f64;
            let v = if xd <= 809.0 / 20000.0 {
                xd / (323.0 / 25.0)
            } else {
                ((xd + 11.0 / 200.0) / (1.0 + 11.0 / 200.0)).powf(12.0 / 5.0)
            };
            v as f32
        };
        let mut gamma = [0i32; 256];
        let int_scale = (255 * (1 << GAMMA_SHIFT)) as f32;
        for (i, g) in gamma.iter_mut().enumerate() {
            let x = i as f32 / 255.0f32;
            *g = (int_scale * apply_gamma(x)).round_ties_even() as i32;
        }
        let lthresh = 216.0f32 / 24389.0f32;
        let lscale = 841.0f32 / 108.0f32;
        let lbias = 16.0f32 / 116.0f32;
        let cb_scale = 1.0f32 / (255.0f32 * (1 << GAMMA_SHIFT) as f32);
        let lshift2 = (1 << LAB_SHIFT2) as f32;
        let cbrt = (0..LAB_CBRT_TAB_SIZE_B)
            .map(|i| {
                let x = cb_scale * i as f32;
                let f = if x < lthresh {
                    x.mul_add(lscale, lbias)
                } else {
                    soft_cbrt(x)
                };
                (lshift2 * f).round_ties_even() as i32
            })
            .collect();
        LabTabs { gamma, cbrt }
    })
}

/// OpenCV's softfloat `f32_cbrt`: a rational approximation in f64 whose
/// mantissa is truncated, not rounded (differs from `f32::cbrt` by an ulp).
fn soft_cbrt(x: f32) -> f32 {
    let bits = x.to_bits();
    if bits & 0x7fff_ffff == 0 {
        return 0.0;
    }
    let ex = ((bits >> 23) & 0xff) as i32 - 127;
    let mut shx = ex % 3;
    if shx >= 0 {
        shx -= 3;
    }
    let ex = (ex - shx) / 3 - 1;
    let fr = f64::from_bits((((shx + 1023) as u64) << 52) | (((bits & 0x7f_ffff) as u64) << 29));
    let a = [
        f64::from_bits(0x4046a09e6653ba70),
        f64::from_bits(0x406808f46c6116e0),
        f64::from_bits(0x405dca97439cae14),
        f64::from_bits(0x402add70d2827500),
        f64::from_bits(0x3fc4f15f83f55d2d),
        f64::from_bits(0x402d9e20660edb21),
        f64::from_bits(0x4062ff15c0285815),
        f64::from_bits(0x406510d06a8112ce),
        f64::from_bits(0x4040fecbc9e2c375),
    ];
    let fr = ((((a[0] * fr + a[1]) * fr + a[2]) * fr + a[3]) * fr + a[4])
        / ((((a[5] * fr + a[6]) * fr + a[7]) * fr + a[8]) * fr + 1.0);
    let frac = ((fr.to_bits() & ((1u64 << 52) - 1)) >> 29) as u32;
    f32::from_bits((bits & 0x8000_0000) | (((ex + 127) as u32) << 23) | frac)
}

/// Channel 0 (L) of `COLOR_BGR2LAB` (sRGB gamma, D65).
pub fn bgr_to_lab_l(img: &Bgr8) -> Gray8 {
    let t = lab_tabs();
    // Y row of sRGB2XYZ_D65 scaled by 2^12: blue, green, red
    const CB: i32 = 296;
    const CG: i32 = 2929;
    const CR: i32 = 871;
    const LSCALE: i32 = (116 * 255 + 50) / 100;
    const LSHIFT: i32 = -((16 * 255 * (1 << LAB_SHIFT2) + 50) / 100);
    let data = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let (b, g, r) = (
                t.gamma[p[0] as usize],
                t.gamma[p[1] as usize],
                t.gamma[p[2] as usize],
            );
            let fy =
                t.cbrt[((b * CB + g * CG + r * CR + (1 << (LAB_SHIFT - 1))) >> LAB_SHIFT) as usize];
            let l = (LSCALE * fy + LSHIFT + (1 << (LAB_SHIFT2 - 1))) >> LAB_SHIFT2;
            l.clamp(0, 255) as u8
        })
        .collect();
    Image::from_vec(img.width, img.height, data)
}
