// Derived from OpenCV 5.0 (clahe.cpp (CLAHE_Impl)),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! `cv2.createCLAHE(clipLimit, (tiles, tiles)).apply(g)` for 8-bit images
//! (OpenCV's `CLAHE_Impl`: tile histograms with clipping and
//! redistribution, bilinear interpolation between tile LUTs).

use crate::border::{Border, border_table};
use crate::fma::{fma_dispatch, madd};
use crate::image::{Gray8, Image};

/// Tile LUTs: `tiles * tiles` tables of 256 entries.
fn tile_luts(g: &Gray8, clip: f64, tiles: usize) -> (Vec<[u8; 256]>, usize, usize) {
    let (w, h) = (g.width, g.height);
    // the source is padded (BORDER_REFLECT_101) on the right / bottom unless
    // both sides divide evenly
    let (ew, eh) = if w % tiles == 0 && h % tiles == 0 {
        (w, h)
    } else {
        (w + tiles - w % tiles, h + tiles - h % tiles)
    };
    let xt = border_table(w, 0, ew - w, Border::Reflect101);
    let yt = border_table(h, 0, eh - h, Border::Reflect101);
    let (tw, th) = (ew / tiles, eh / tiles);
    let total = tw * th;
    let lut_scale = 255.0f32 / total as f32;
    let clip_limit = if clip > 0.0 {
        ((clip * total as f64 / 256.0) as i64).max(1)
    } else {
        0
    };
    let mut luts = vec![[0u8; 256]; tiles * tiles];
    for (k, lut) in luts.iter_mut().enumerate() {
        let (ty, tx) = (k / tiles, k % tiles);
        let mut hist = [0i64; 256];
        for &sy in &yt[ty * th..(ty + 1) * th] {
            let row = g.row(sy);
            for &sx in &xt[tx * tw..(tx + 1) * tw] {
                hist[row[sx] as usize] += 1;
            }
        }
        if clip > 0.0 {
            let mut clipped = 0;
            for v in hist.iter_mut() {
                if *v > clip_limit {
                    clipped += *v - clip_limit;
                    *v = clip_limit;
                }
            }
            let batch = clipped / 256;
            let mut residual = clipped - batch * 256;
            hist.iter_mut().for_each(|v| *v += batch);
            if residual != 0 {
                let step = (256 / residual).max(1) as usize;
                let mut i = 0;
                while i < 256 && residual > 0 {
                    hist[i] += 1;
                    i += step;
                    residual -= 1;
                }
            }
        }
        let mut sum = 0i64;
        for (l, &v) in lut.iter_mut().zip(&hist) {
            sum += v;
            *l = (sum as f32 * lut_scale).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    (luts, tw, th)
}

fma_dispatch!(
    /// CLAHE with clip limit `clip` on a `tiles` x `tiles` grid.
    pub fn clahe_apply(g: &Gray8, clip: f64, tiles: usize) -> Gray8 => clahe_impl
);

#[inline(always)]
fn clahe_impl<const FMA: bool>(g: &Gray8, clip: f64, tiles: usize) -> Gray8 {
    let (luts, tw, th) = tile_luts(g, clip, tiles);
    let (w, h) = (g.width, g.height);
    let inv_tw = 1.0f32 / tw as f32;
    let inv_th = 1.0f32 / th as f32;
    let cols: Vec<(usize, usize, f32, f32)> = (0..w)
        .map(|x| {
            let txf = madd::<FMA>(x as f32, inv_tw, -0.5);
            let tx1 = txf.floor() as i64;
            let xa = txf - tx1 as f32;
            let tx2 = (tx1 + 1).min(tiles as i64 - 1);
            (tx1.max(0) as usize, tx2 as usize, xa, 1.0 - xa)
        })
        .collect();
    let mut dst = Image::new(w, h);
    for y in 0..h {
        let tyf = madd::<FMA>(y as f32, inv_th, -0.5);
        let ty1 = tyf.floor() as i64;
        let ya = tyf - ty1 as f32;
        let ya1 = 1.0 - ya;
        let ty2 = (ty1 + 1).min(tiles as i64 - 1) as usize;
        let ty1 = ty1.max(0) as usize;
        let (p1, p2) = (
            &luts[ty1 * tiles..(ty1 + 1) * tiles],
            &luts[ty2 * tiles..(ty2 + 1) * tiles],
        );
        let src = g.row(y);
        for (x, d) in dst.row_mut(y).iter_mut().enumerate() {
            let v = src[x] as usize;
            let (i1, i2, xa, xa1) = cols[x];
            // (l1a*xa1 + l1b*xa)*ya1 + (l2a*xa1 + l2b*xa)*ya with the first
            // product of each sum fused (GCC contraction in the reference)
            let (a, b) = (p1[i1][v] as f32, p1[i2][v] as f32);
            let (c, e) = (p2[i1][v] as f32, p2[i2][v] as f32);
            let top = madd::<FMA>(a, xa1, b * xa);
            let bot = madd::<FMA>(c, xa1, e * xa);
            let res = madd::<FMA>(top, ya1, bot * ya);
            *d = res.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}
