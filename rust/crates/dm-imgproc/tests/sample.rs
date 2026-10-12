//! P4b.9: the AVX2 point samplers equal the scalar ones bit for bit, on
//! interior points, borders, outside points, NaN / inf and lengths that are
//! not a multiple of 8. (On a CPU without AVX2 + FMA both sides are the
//! scalar code.)

use dm_imgproc::*;

/// Small deterministic generator (64-bit LCG, top bits).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn image(rng: &mut Rng, w: usize, h: usize) -> GrayF32 {
    Image::from_vec(
        w,
        h,
        (0..w * h).map(|_| rng.range(0.0, 255.0) as f32).collect(),
    )
}

/// A coordinate in [-3, size + 3] mostly, sometimes a special value.
fn coord(rng: &mut Rng, size: usize) -> f64 {
    match (rng.next() * 40.0) as usize {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => -1e12,
        3 => 3e9,
        4 => size as f64 - 1.0,
        5 => 0.0,
        _ => rng.range(-3.0, size as f64 + 3.0),
    }
}

fn same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            "{what}: point {i}: {x} vs {y}"
        );
    }
}

const SIZES: [(usize, usize); 5] = [(37, 23), (2, 2), (1, 9), (640, 480), (8, 1)];

#[test]
fn remap_points_equals_scalar() {
    let mut rng = Rng(1);
    for (w, h) in SIZES {
        let img = image(&mut rng, w, h);
        for n in [0, 1, 7, 8, 9, 64, 1003] {
            let xs: Vec<f32> = (0..n).map(|_| coord(&mut rng, w) as f32).collect();
            let ys: Vec<f32> = (0..n).map(|_| coord(&mut rng, h) as f32).collect();
            same(
                &remap_points(&img, &xs, &ys),
                &remap_points_scalar(&img, &xs, &ys),
                &format!("remap {w}x{h} n={n}"),
            );
        }
        // interior only: every vector takes the gather path
        let xs: Vec<f32> = (0..999)
            .map(|_| rng.range(0.0, w as f64 - 1.0) as f32)
            .collect();
        let ys: Vec<f32> = (0..999)
            .map(|_| rng.range(0.0, h as f64 - 1.0) as f32)
            .collect();
        same(
            &remap_points(&img, &xs, &ys),
            &remap_points_scalar(&img, &xs, &ys),
            &format!("remap interior {w}x{h}"),
        );
    }
}

/// Quad corners around the image (some degenerate: NaN homography).
fn quad(rng: &mut Rng, w: usize, h: usize) -> [(f64, f64); 4] {
    std::array::from_fn(|_| {
        (
            coord(rng, w).clamp(-50.0, 1e4),
            coord(rng, h).clamp(-50.0, 1e4),
        )
    })
}

#[test]
fn project_f64_equals_scalar() {
    let mut rng = Rng(2);
    let offs = [
        [0.0, 0.0],
        [-0.22, 0.0],
        [0.22, 0.0],
        [0.0, -0.22],
        [0.0, 0.22],
    ];
    for (w, h) in SIZES {
        let img = image(&mut rng, w, h);
        for n in [1, 7, 8, 13, 240, 1001] {
            let (rows, cols) = (24.0, 24.0);
            let q = quad(&mut rng, w, h);
            let src = [(0.0, 0.0), (cols, 0.0), (cols, rows), (0.0, rows)];
            let hm = homography(&src, &q).unwrap_or([f64::NAN; 9]);
            let uv: Vec<[f64; 2]> = (0..n)
                .map(|_| [rng.range(-2.0, cols + 2.0), rng.range(-2.0, rows + 2.0)])
                .collect();
            same(
                &remap_project_f64(&img, &hm, &uv, &offs),
                &remap_project_f64_scalar(&img, &hm, &uv, &offs),
                &format!("f64 {w}x{h} n={n}"),
            );
        }
    }
}

#[test]
fn project_f32_equals_scalar() {
    let mut rng = Rng(3);
    for (w, h) in SIZES {
        let img = image(&mut rng, w, h);
        for n in [1, 7, 8, 13, 240, 1001] {
            let (rows, cols) = (16.0, 36.0);
            let q = quad(&mut rng, w, h);
            let src = [(0.0, 0.0), (cols, 0.0), (cols, rows), (0.0, rows)];
            let mut hm = homography(&src, &q)
                .unwrap_or([f64::NAN; 9])
                .map(|v| v as f32);
            if n == 13 {
                // denominator zero on the line u = 0: the 1e-9 clamp
                hm[6] = 1.0;
                hm[7] = 0.0;
                hm[8] = 0.0;
            }
            let pts: Vec<[f32; 2]> = (0..n)
                .map(|i| {
                    let u = if i % 5 == 0 {
                        0.0
                    } else {
                        rng.range(-2.0, cols + 2.0)
                    };
                    [u as f32, rng.range(-2.0, rows + 2.0) as f32]
                })
                .collect();
            same(
                &remap_project_f32(&img, &hm, &pts),
                &remap_project_f32_scalar(&img, &hm, &pts),
                &format!("f32 {w}x{h} n={n}"),
            );
        }
    }
}
