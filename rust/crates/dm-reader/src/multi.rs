//! `--multi` (round 4 A5): after a read, cover the symbol with the local
//! background and read the rest of the image again.

use dm_imgproc::Bgr8;

use crate::variants::Quad;

/// Most symbols read from one image.
pub const MULTI_MAX: usize = 16;
/// Margin around a read symbol that is covered as well, in modules (quiet
/// zone plus a blurred edge).
pub const MULTI_MARGIN_MODULES: f32 = 2.0;
/// Module count assumed when a read has no grid fit (zxing only).
pub const MULTI_FALLBACK_MODULES: f32 = 12.0;

fn centre(q: &Quad) -> (f32, f32) {
    let (sx, sy) = q.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    (sx / 4.0, sy / 4.0)
}

fn mean_side(q: &Quad) -> f32 {
    (0..4)
        .map(|k| {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
        })
        .sum::<f32>()
        / 4.0
}

/// `q` scaled about its centre so that every side moves out by about
/// `margin` px.
pub fn grow_quad(q: &Quad, margin: f32) -> Quad {
    let (cx, cy) = centre(q);
    let k = 1.0 + 2.0 * margin / mean_side(q).max(1.0);
    q.map(|(x, y)| (cx + (x - cx) * k, cy + (y - cy) * k))
}

/// Point inside a convex quad (either winding).
pub fn inside(q: &Quad, x: f32, y: f32) -> bool {
    let mut sign = 0f32;
    for k in 0..4 {
        let (a, b) = (q[k], q[(k + 1) % 4]);
        let c = (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0);
        if c != 0.0 {
            if sign != 0.0 && c.signum() != sign {
                return false;
            }
            sign = c.signum();
        }
    }
    true
}

/// Cover a read symbol: `quad` (original pixels, `modules` across) grown by
/// `MULTI_MARGIN_MODULES` is filled with the per-channel median of a ring
/// of the same width around it.
pub fn blank_symbol(img: &mut Bgr8, quad: &Quad, modules: f32) {
    let module = mean_side(quad) / modules.max(1.0);
    let margin = MULTI_MARGIN_MODULES * module;
    let inner = grow_quad(quad, margin);
    let outer = grow_quad(quad, 2.0 * margin);
    let (w, h) = (img.width, img.height);
    let bbox = |q: &Quad| {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in q {
            (x0, y0, x1, y1) = (x0.min(p.0), y0.min(p.1), x1.max(p.0), y1.max(p.1));
        }
        let cl = |v: f32, n: usize| (v.max(0.0) as usize).min(n);
        (
            cl(x0.floor(), w),
            cl(y0.floor(), h),
            cl(x1.ceil() + 1.0, w),
            cl(y1.ceil() + 1.0, h),
        )
    };
    let (x0, y0, x1, y1) = bbox(&outer);
    let mut ring: [Vec<u8>; 3] = Default::default();
    let mut cover = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let (px, py) = (x as f32, y as f32);
            let i = 3 * (y * w + x);
            if inside(&inner, px, py) {
                cover.push(i);
            } else if inside(&outer, px, py) {
                for (c, r) in ring.iter_mut().enumerate() {
                    r.push(img.data[i + c]);
                }
            }
        }
    }
    let fill: Vec<u8> = ring
        .iter_mut()
        .map(|r| {
            if r.is_empty() {
                255
            } else {
                let m = r.len() / 2;
                *r.select_nth_unstable(m).1
            }
        })
        .collect();
    for i in cover {
        img.data[i..i + 3].copy_from_slice(&fill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_either_winding() {
        let q: Quad = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let r: Quad = [q[0], q[3], q[2], q[1]];
        for q in [q, r] {
            assert!(inside(&q, 5.0, 5.0));
            assert!(inside(&q, 0.0, 5.0));
            assert!(!inside(&q, 11.0, 5.0));
            assert!(!inside(&q, 5.0, -0.5));
        }
        let g = grow_quad(&q, 2.0);
        assert_eq!(g[0], (-2.0, -2.0));
        assert_eq!(g[2], (12.0, 12.0));
    }

    #[test]
    fn blank_covers_symbol_and_margin_only() {
        // gray 200 background, a dark 20x20 "symbol" of 10 modules at (40, 40),
        // a second dark square at (90, 40) that must stay
        let (w, h) = (120, 100);
        let mut data = vec![200u8; w * h * 3];
        for (x0, y0) in [(40, 40), (90, 40)] {
            for y in y0..y0 + 20 {
                for x in x0..x0 + 20 {
                    data[3 * (y * w + x)..3 * (y * w + x) + 3].fill(30);
                }
            }
        }
        let mut img = Bgr8::from_vec(w, h, data);
        let q: Quad = [(40.0, 40.0), (59.0, 40.0), (59.0, 59.0), (40.0, 59.0)];
        blank_symbol(&mut img, &q, 10.0);
        let px = |x: usize, y: usize| img.data[3 * (y * w + x)];
        assert!((40..60).all(|y| (40..60).all(|x| px(x, y) == 200)));
        assert!((40..60).all(|y| (90..110).all(|x| px(x, y) == 30)));
    }
}
