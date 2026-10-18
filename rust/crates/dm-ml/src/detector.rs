//! Symbol detector (`ml/detector.py`): a finder on the whole frame gives
//! up to a few symbol quads, a measurer on a full-resolution crop of each
//! refines the corners and gives polarity and the three most likely
//! symbol sizes. Both are ONNX, run with tract; the weights are embedded.
//! The quads only seed the reader's own grid fit: nothing here places
//! modules.
//!
//! Resampling follows `ml/detector_data.py`: the finder frame is the gray
//! image turned 90 degrees clockwise when it is portrait, area-averaged to
//! fit `FINDER_W` x `FINDER_H` and padded with its edge pixels; a measurer
//! crop is the square of side `CROP_PAD` x the finder quad's longer
//! bounding-box side around its centre, sampled bilinearly at that size
//! (at least `CROP` px) and area-averaged to `CROP` px.

use std::sync::OnceLock;

use dm_imgproc::{Gray8, GrayF32, resize_area};
use tract_onnx::prelude::*;

/// The embedded models (`ml/train_detector.py`, fine-tuned on real images).
const FINDER: &[u8] = include_bytes!("../model/finder.onnx");
const MEASURER: &[u8] = include_bytes!("../model/measurer.onnx");

pub const FINDER_W: usize = 512;
pub const FINDER_H: usize = 384;
/// Finder output cell, in frame pixels.
pub const STRIDE: usize = 4;
/// Finder corner offsets are in cells / `OFFSET_SCALE`.
pub const OFFSET_SCALE: f32 = 16.0;
pub const CROP: usize = 192;
pub const CROP_PAD: f32 = 1.25;
/// Measurer size classes, in the model's output order.
pub const SIZES: [&str; 30] = [
    "10x10", "12x12", "14x14", "16x16", "18x18", "20x20", "22x22", "24x24", "26x26", "32x32",
    "36x36", "40x40", "44x44", "48x48", "52x52", "64x64", "72x72", "80x80", "88x88", "96x96",
    "104x104", "120x120", "132x132", "144x144", "8x18", "8x32", "12x26", "12x36", "16x36", "16x48",
];

type Plan = Arc<TypedRunnableModel>;

/// One detected symbol.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Finder centre score (sigmoid), 0..1.
    pub score: f32,
    /// Corners in image pixels, clockwise on screen, from the measurer.
    pub quad: [[f32; 2]; 4],
    /// The finder's own corners (image pixels).
    pub finder_quad: [[f32; 2]; 4],
    /// Probability that the symbol is dark on light.
    pub p_dark: f32,
    /// The three most likely sizes (`RxC`), best first.
    pub sizes: [&'static str; 3],
}

pub struct Detector {
    finder: Plan,
    measurer: Plan,
}

fn plan(bytes: &[u8], h: usize, w: usize) -> TractResult<Plan> {
    tract_onnx::onnx()
        .model_for_read(&mut &bytes[..])?
        .with_input_fact(0, f32::fact([1, 1, h, w]).into())?
        .into_optimized()?
        .into_runnable()
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Clockwise on screen (y down): positive shoelace sum.
fn clockwise(mut q: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let area: f32 = (0..4)
        .map(|i| {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    if area < 0.0 {
        q.reverse();
    }
    q
}

impl Detector {
    pub fn from_bytes(finder: &[u8], measurer: &[u8]) -> TractResult<Self> {
        Ok(Detector {
            finder: plan(finder, FINDER_H, FINDER_W)?,
            measurer: plan(measurer, CROP, CROP)?,
        })
    }

    /// The embedded models, loaded on first use.
    pub fn embedded() -> &'static Detector {
        static D: OnceLock<Detector> = OnceLock::new();
        D.get_or_init(|| Detector::from_bytes(FINDER, MEASURER).expect("embedded models load"))
    }

    /// Finder output for a `FINDER_H` x `FINDER_W` frame (gray 0..1):
    /// (centre logits, H/4 x W/4; corner offsets, 8 x H/4 x W/4).
    pub fn finder_raw(&self, frame: &GrayF32) -> Option<(Vec<f32>, Vec<f32>)> {
        let x = Tensor::from_shape(&[1, 1, FINDER_H, FINDER_W], &frame.data).ok()?;
        let out = self.finder.run(tvec![x.into()]).ok()?;
        let z = out[0].to_plain_array_view::<f32>().ok()?;
        let (h, w) = (FINDER_H / STRIDE, FINDER_W / STRIDE);
        if z.shape() != [1, 9, h, w] {
            return None;
        }
        let v: Vec<f32> = z.iter().copied().collect();
        let n = h * w;
        Some((v[..n].to_vec(), v[n..].to_vec()))
    }

    /// Measurer output for a `CROP` x `CROP` crop (gray 0..1): corners in
    /// crop pixels, p(dark), size logits.
    pub fn measurer_raw(&self, crop: &GrayF32) -> Option<([[f32; 2]; 4], f32, Vec<f32>)> {
        let x = Tensor::from_shape(&[1, 1, CROP, CROP], &crop.data).ok()?;
        let out = self.measurer.run(tvec![x.into()]).ok()?;
        let c = out[0].to_plain_array_view::<f32>().ok()?;
        let p = out[1].to_plain_array_view::<f32>().ok()?;
        let s = out[2].to_plain_array_view::<f32>().ok()?;
        if c.len() != 8 || p.len() != 2 || s.len() != SIZES.len() {
            return None;
        }
        let c: Vec<f32> = c.iter().map(|v| (v + 1.0) / 2.0 * CROP as f32).collect();
        let corners = [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]];
        let (p0, p1) = (p.iter().next()?, p.iter().nth(1)?);
        Some((corners, sigmoid(p0 - p1), s.iter().copied().collect()))
    }

    /// Up to `max` symbols with a finder score of at least `thresh`, best
    /// first. `None` when a model fails.
    pub fn detect(&self, gray: &Gray8, thresh: f32, max: usize) -> Option<Vec<Detection>> {
        let (frame, s, turned) = finder_frame(gray);
        let (heat, off) = self.finder_raw(&frame)?;
        let (h, w) = (FINDER_H / STRIDE, FINDER_W / STRIDE);
        let mut peaks: Vec<(f32, usize, usize)> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v = sigmoid(heat[y * w + x]);
                if v < thresh {
                    continue;
                }
                let local_max = (y.saturating_sub(1)..(y + 2).min(h)).all(|yy| {
                    (x.saturating_sub(1)..(x + 2).min(w)).all(|xx| sigmoid(heat[yy * w + xx]) <= v)
                });
                if local_max {
                    peaks.push((v, y, x));
                }
            }
        }
        peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
        peaks.truncate(max);
        let mut out = Vec::with_capacity(peaks.len());
        for (score, y, x) in peaks {
            let mut q = [[0f32; 2]; 4];
            for (k, c) in q.iter_mut().enumerate() {
                let ox = off[(2 * k) * h * w + y * w + x];
                let oy = off[(2 * k + 1) * h * w + y * w + x];
                let fx = (x as f32 + 0.5 + ox * OFFSET_SCALE) * STRIDE as f32;
                let fy = (y as f32 + 0.5 + oy * OFFSET_SCALE) * STRIDE as f32;
                // frame -> image pixels, undoing the turn
                let (ix, iy) = (fx / s, fy / s);
                *c = if turned {
                    [iy, gray.height as f32 - ix]
                } else {
                    [ix, iy]
                };
            }
            let finder_quad = clockwise(q);
            let (cx, cy, side) = crop_box(&finder_quad);
            let (crop, n, a) = measurer_crop(gray, cx, cy, side);
            let (c, p_dark, logits) = self.measurer_raw(&crop)?;
            let k = n as f32 / CROP as f32;
            let quad = clockwise(c.map(|[u, v]| {
                [
                    (u * k - n as f32 / 2.0) / a + cx,
                    (v * k - n as f32 / 2.0) / a + cy,
                ]
            }));
            let mut order: Vec<usize> = (0..SIZES.len()).collect();
            order.sort_by(|&i, &j| logits[j].total_cmp(&logits[i]));
            out.push(Detection {
                score,
                quad,
                finder_quad,
                p_dark,
                sizes: [SIZES[order[0]], SIZES[order[1]], SIZES[order[2]]],
            });
        }
        Some(out)
    }
}

/// The finder frame (gray 0..1, `FINDER_H` x `FINDER_W`), the scale from
/// image to frame pixels and whether the image was turned.
pub fn finder_frame(gray: &Gray8) -> (GrayF32, f32, bool) {
    let turned = gray.height > gray.width;
    let g = if turned {
        turn_clockwise(gray)
    } else {
        gray.clone()
    };
    let s = (FINDER_W as f64 / g.width as f64).min(FINDER_H as f64 / g.height as f64);
    let small = if s < 1.0 {
        resize_area(&g, s)
    } else {
        upscale_area(&g, s)
    };
    let (sw, sh) = (small.width.min(FINDER_W), small.height.min(FINDER_H));
    let mut data = vec![0f32; FINDER_W * FINDER_H];
    for y in 0..FINDER_H {
        let row = small.row(y.min(sh - 1));
        for x in 0..FINDER_W {
            data[y * FINDER_W + x] = row[x.min(sw - 1)] as f32 / 255.0;
        }
    }
    (
        GrayF32::from_vec(FINDER_W, FINDER_H, data),
        s as f32,
        turned,
    )
}

/// `cv2.resize(g, (round(w * s), round(h * s)), interpolation=INTER_AREA)`
/// for s >= 1. OpenCV computes it as a linear interpolation between source
/// pixels `i` and `i + 1`, `i = floor(d * n / m)`, with the fraction
/// `(d + 1) - (i + 1) * m / n` clipped to 0 (n source, m destination px).
fn upscale_area(g: &Gray8, s: f64) -> Gray8 {
    let dw = (g.width as f64 * s).round_ties_even() as usize;
    let dh = (g.height as f64 * s).round_ties_even() as usize;
    let tab = |m: usize, n: usize| -> Vec<(usize, usize, f32)> {
        let (scale, inv) = (n as f64 / m as f64, m as f64 / n as f64);
        (0..m)
            .map(|d| {
                let i = (d as f64 * scale).floor() as i64;
                let f = (d as f64 + 1.0) - (i as f64 + 1.0) * inv;
                let mut f = if f <= 0.0 { 0.0 } else { f - f.floor() };
                let i = if i >= n as i64 - 1 {
                    f = 0.0;
                    n - 1
                } else {
                    i.max(0) as usize
                };
                (i, (i + 1).min(n - 1), f as f32)
            })
            .collect()
    };
    let (tx, ty) = (tab(dw, g.width), tab(dh, g.height));
    let mut data = Vec::with_capacity(dw * dh);
    for &(y0, y1, fy) in &ty {
        let (r0, r1) = (g.row(y0), g.row(y1));
        for &(x0, x1, fx) in &tx {
            let top = r0[x0] as f32 * (1.0 - fx) + r0[x1] as f32 * fx;
            let bot = r1[x0] as f32 * (1.0 - fx) + r1[x1] as f32 * fx;
            data.push((top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8);
        }
    }
    Gray8::from_vec(dw, dh, data)
}

fn turn_clockwise(g: &Gray8) -> Gray8 {
    let (w, h) = (g.width, g.height);
    let mut data = vec![0u8; w * h];
    // out (x', y') = in (x = y', y = h - 1 - x'); out is h wide, w high
    for yo in 0..w {
        for xo in 0..h {
            data[yo * h + xo] = g.row(h - 1 - xo)[yo];
        }
    }
    Gray8::from_vec(h, w, data)
}

/// Centre and side of the measurer crop around a quad.
pub fn crop_box(q: &[[f32; 2]; 4]) -> (f32, f32, f32) {
    let (mut minx, mut maxx, mut miny, mut maxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in q {
        minx = minx.min(p[0]);
        maxx = maxx.max(p[0]);
        miny = miny.min(p[1]);
        maxy = maxy.max(p[1]);
    }
    let cx = q.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy = q.iter().map(|p| p[1]).sum::<f32>() / 4.0;
    (cx, cy, (CROP_PAD * (maxx - minx).max(maxy - miny)).max(8.0))
}

/// The `CROP` x `CROP` crop (gray 0..1), the side `n` it was sampled at
/// and the scale `a` = n / side: image (x, y) = (crop px * n / CROP - n /
/// 2) / a + (cx, cy).
pub fn measurer_crop(gray: &Gray8, cx: f32, cy: f32, side: f32) -> (GrayF32, usize, f32) {
    let n = CROP.max(side.ceil() as usize);
    let a = n as f32 / side;
    let (w, h) = (gray.width as i64, gray.height as i64);
    let px =
        |x: i64, y: i64| gray.row(y.clamp(0, h - 1) as usize)[x.clamp(0, w - 1) as usize] as f32;
    let mut sq = vec![0u8; n * n];
    for y in 0..n {
        let sy = (y as f32 - n as f32 / 2.0) / a + cy;
        let (y0, fy) = (sy.floor() as i64, sy - sy.floor());
        for x in 0..n {
            let sx = (x as f32 - n as f32 / 2.0) / a + cx;
            let (x0, fx) = (sx.floor() as i64, sx - sx.floor());
            let v = (px(x0, y0) * (1.0 - fx) + px(x0 + 1, y0) * fx) * (1.0 - fy)
                + (px(x0, y0 + 1) * (1.0 - fx) + px(x0 + 1, y0 + 1) * fx) * fy;
            sq[y * n + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let sq = Gray8::from_vec(n, n, sq);
    let c = if n > CROP {
        resize_area(&sq, CROP as f64 / n as f64)
    } else {
        sq
    };
    let data = c.data.iter().map(|&v| v as f32 / 255.0).collect();
    (GrayF32::from_vec(CROP, CROP, data), n, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gray 0..1: a smooth background with an n x n checker "symbol" of
    /// m px modules at (ox, oy), solid left column and bottom row.
    fn frame(h: usize, w: usize, ox: usize, oy: usize, n: usize, m: usize) -> GrayF32 {
        let mut data = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let mut v = 0.55 + 0.25 * (x as f32 * 0.05).sin() * (y as f32 * 0.07).cos();
                let inside = (ox..ox + n * m).contains(&x) && (oy..oy + n * m).contains(&y);
                if inside {
                    let (mx, my) = ((x - ox) / m, (y - oy) / m);
                    if (mx + my) % 2 == 0 || mx == 0 || my == n - 1 {
                        v = 0.1;
                    }
                }
                data.push(v);
            }
        }
        GrayF32::from_vec(w, h, data)
    }

    fn close(a: f32, b: f32, tol: f32, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
    }

    /// Outputs of the trained PyTorch models (`ml/detector.py`) on the
    /// same inputs, rounded.
    #[test]
    fn matches_the_training_framework() {
        let d = Detector::embedded();
        let (heat, off) = d
            .finder_raw(&frame(384, 512, 200, 150, 12, 6))
            .expect("finder runs");
        let w = FINDER_W / STRIDE;
        let (i, z) = heat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert_eq!((i / w, i % w), (46, 59));
        close(*z, 3.066, 0.01, "peak logit");
        let want = [0.585, 0.557, -0.578, 0.578, -0.607, -0.520, 0.558, -0.540];
        for (k, &v) in want.iter().enumerate() {
            close(off[k * heat.len() + i], v, 0.005, "offset");
        }
        close(heat.iter().sum::<f32>(), -62334.64, 2.0, "heat sum");

        let (c, p_dark, size) = d
            .measurer_raw(&frame(192, 192, 48, 48, 16, 6))
            .expect("measurer runs");
        let want = [-0.495, 0.475, -0.473, -0.484, 0.509, -0.457, 0.487, 0.507];
        for (k, &v) in want.iter().enumerate() {
            close(
                c[k / 2][k % 2],
                (v + 1.0) / 2.0 * CROP as f32,
                0.2,
                "corner",
            );
        }
        close(p_dark, sigmoid(3.740 + 3.035), 0.001, "p_dark");
        close(size[7], 8.41, 0.02, "24x24 logit");
        close(size[28], 5.06, 0.02, "16x36 logit");
    }

    #[test]
    fn finds_a_symbol() {
        let d = Detector::embedded();
        // a 1440 x 1080 gray image with the checker at (600, 450), 18 px modules
        let f = frame(1080, 1440, 600, 450, 12, 18);
        let g = Gray8::from_vec(
            1440,
            1080,
            f.data.iter().map(|v| (v * 255.0) as u8).collect(),
        );
        let dets = d.detect(&g, 0.3, 3).expect("models run");
        let best = dets.first().expect("a detection");
        let (cx, cy) = (
            best.quad.iter().map(|p| p[0]).sum::<f32>() / 4.0,
            best.quad.iter().map(|p| p[1]).sum::<f32>() / 4.0,
        );
        assert!(
            (cx - 708.0).abs() < 18.0 && (cy - 558.0).abs() < 18.0,
            "{best:?}"
        );
    }

    #[test]
    #[ignore = "timing, run with --ignored --nocapture"]
    fn timing() {
        let d = Detector::embedded();
        let f = frame(3000, 4000, 1600, 1200, 20, 30);
        let g = Gray8::from_vec(
            4000,
            3000,
            f.data.iter().map(|v| (v * 255.0) as u8).collect(),
        );
        d.detect(&g, 0.3, 3).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..10 {
            d.detect(&g, 0.3, 3).unwrap();
        }
        println!("4000 x 3000: {:.1} ms", t.elapsed().as_secs_f64() * 100.0);
    }
}
