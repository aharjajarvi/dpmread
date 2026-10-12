//! `--debug` output: variant PNGs and the annotated result image
//! (`save_annotated`, reader 2019–2045). Debug output only: the drawing
//! follows OpenCV's look, not its exact pixels.

use std::path::Path;

use dm_imgproc::{Bgr8, Gray8, get_perspective_transform, perspective_transform};

use crate::read::ReadResult;
use crate::variants::Variant;

type Bgr = [u8; 3];

const CHANGED: Bgr = [0, 0, 255];
const DARK: Bgr = [0, 200, 0];
const LIGHT: Bgr = [255, 160, 0];
const OUTLINE: Bgr = [0, 255, 255];

fn put(img: &mut Bgr8, x: i64, y: i64, c: Bgr) {
    if x >= 0 && y >= 0 && (x as usize) < img.width && (y as usize) < img.height {
        let i = 3 * (y as usize * img.width + x as usize);
        img.data[i..i + 3].copy_from_slice(&c);
    }
}

/// Pixels at squared distance `lo2..=hi2` from (cx, cy).
fn ring(img: &mut Bgr8, (cx, cy): (i64, i64), r: i64, lo2: i64, hi2: i64, c: Bgr) {
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if (lo2..=hi2).contains(&d2) {
                put(img, cx + dx, cy + dy, c);
            }
        }
    }
}

/// `cv2.circle(..., -1)`: filled disk (OpenCV: d² ≤ r²).
fn disk(img: &mut Bgr8, centre: (i64, i64), r: i64, c: Bgr) {
    ring(img, centre, r, 0, r * r, c);
}

/// `cv2.circle(..., 2)`: circle line 2 px wide (OpenCV draws a polygon
/// about 3 px across; r ± 1.5 is close).
fn circle2(img: &mut Bgr8, centre: (i64, i64), r: i64, c: Bgr) {
    let (lo, hi) = (r as f64 - 1.5, r as f64 + 1.5);
    let lo2 = (lo.max(0.0) * lo.max(0.0)).ceil() as i64;
    ring(img, centre, r + 2, lo2, (hi * hi) as i64, c);
}

/// 1 px line (Bresenham).
fn line(img: &mut Bgr8, (mut x0, mut y0): (i64, i64), (x1, y1): (i64, i64), c: Bgr) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let mut err = dx + dy;
    loop {
        put(img, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// The original image with the decoded fit drawn on it: module centres
/// (green dark, blue light, red in corrected codewords), the symbol outline
/// and a ring at its first corner. `None` without a grid fit.
pub fn annotated(bgr: &Bgr8, res: &ReadResult) -> Option<Bgr8> {
    let fit = res.fit.as_ref()?;
    let g = fit.geom;
    let (s, ox, oy) = (res.scale as f32, res.ox as f32, res.oy as f32);
    let quad = fit.quad.map(|[x, y]| (x / s + ox, y / s + oy));
    let (gc, gr) = (g.cols as f32, g.rows as f32);
    let src = [(0.0, 0.0), (gc, 0.0), (gc, gr), (0.0, gr)];
    let hmat = get_perspective_transform(&src, &quad)?;
    // module centres: the warped grid's axes, else the even grid
    let centres: Vec<(f32, f32)> = match &fit.axes {
        Some((u, v)) => v
            .iter()
            .flat_map(|&r| u.iter().map(move |&c| (c as f32, r as f32)))
            .collect(),
        None => (0..g.rows)
            .flat_map(|r| (0..g.cols).map(move |c| (c as f32 + 0.5, r as f32 + 0.5)))
            .collect(),
    };
    let pts = perspective_transform(&centres, &hmat);
    let mpx = res.quality.as_ref().map_or(4.0, |q| q.module_size_px);
    let rad = ((mpx * 0.2) as i64).max(1);
    let mut changed = vec![false; g.rows * g.cols];
    if let Some(rs) = &res.rs {
        for &p in &rs.corrected_positions {
            for (&r, &c) in g.cw_rows[p].iter().zip(&g.cw_cols[p]) {
                changed[r * g.cols + c] = true;
            }
        }
    }
    let bits = fit.bits.as_deref().unwrap_or(&[]);
    let mut img = bgr.clone();
    for (i, &(x, y)) in pts.iter().enumerate() {
        let col = if changed[i] {
            CHANGED
        } else if bits.get(i).is_some_and(|&b| b != 0) {
            DARK
        } else {
            LIGHT
        };
        disk(&mut img, (x as i64, y as i64), rad, col);
    }
    let corners = quad.map(|(x, y)| (x as i64, y as i64));
    for k in 0..4 {
        line(&mut img, corners[k], corners[(k + 1) % 4], OUTLINE);
    }
    circle2(&mut img, corners[0], rad * 3, CHANGED);
    Some(img)
}

fn write_png(path: &Path, data: &[u8], w: usize, h: usize, color: image::ColorType) {
    if let Err(e) = image::save_buffer(path, data, w as u32, h as u32, color) {
        eprintln!("warning: cannot write {}: {e}", path.display());
    }
}

pub fn save_gray_png(path: &Path, img: &Gray8) {
    write_png(path, &img.data, img.width, img.height, image::ColorType::L8);
}

pub fn save_bgr_png(path: &Path, img: &Bgr8) {
    let rgb: Vec<u8> = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&[b, g, r]| [r, g, b])
        .collect();
    write_png(path, &rgb, img.width, img.height, image::ColorType::Rgb8);
}

/// `{stem}` of an input path, as Python's `splitext(basename(path))[0]`.
pub fn stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The variant images: `{stem}_vNN.png` (enhancement ladder) and
/// `{stem}_tNN.png` (tiles, flattened).
pub fn save_variants(dir: &Path, path: &Path, variants: &[Variant], tiles: &[Vec<Variant>]) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("warning: cannot create {}: {e}", dir.display());
        return;
    }
    let stem = stem(path);
    for (k, v) in variants.iter().enumerate() {
        save_gray_png(&dir.join(format!("{stem}_v{k:02}.png")), &v.img);
    }
    for (k, v) in tiles.iter().flatten().enumerate() {
        save_gray_png(&dir.join(format!("{stem}_t{k:02}.png")), &v.img);
    }
}

/// `save_annotated`: `{stem}_result.png` in `dir`.
pub fn save_annotated(dir: &Path, bgr: &Bgr8, res: &ReadResult) {
    if let Some(img) = annotated(bgr, res) {
        save_bgr_png(&dir.join(format!("{}_result.png", stem(&res.path))), &img);
    }
}
