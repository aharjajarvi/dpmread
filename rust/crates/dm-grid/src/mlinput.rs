//! Module classifier input: a grid fit's module centres and the symbol
//! resampled through the fit to a fixed module size. The same function
//! makes the training crops (`--dump-fits`) and, later, the model input.

use crate::sampler::Quad;
use dm_imgproc::{Gray8, GrayF32, Image, Mat3, homography};

/// Pixels per module of the classifier input.
pub const INPUT_PPM: usize = 6;
/// Modules of margin around the symbol in the classifier input.
pub const INPUT_MARGIN: usize = 2;

/// Module coordinates (column, row; the symbol spans 0..cols, 0..rows, so
/// a plain grid has its centres at c + 0.5) to image pixels for a fit.
/// With `axes` (tracked column / row centres of a warped grid) the
/// coordinates are first moved piecewise linearly through the tracked
/// centres.
pub struct FitMap<'a> {
    h: Mat3,
    axes: Option<(&'a [f64], &'a [f64])>,
}

impl<'a> FitMap<'a> {
    /// `None` for a degenerate quad.
    pub fn new(
        quad: &Quad,
        rows: usize,
        cols: usize,
        axes: Option<&'a (Vec<f64>, Vec<f64>)>,
    ) -> Option<Self> {
        let src = [
            (0.0, 0.0),
            (cols as f64, 0.0),
            (cols as f64, rows as f64),
            (0.0, rows as f64),
        ];
        let q = quad.map(|p| (p[0] as f64, p[1] as f64));
        let axes = axes
            .filter(|(u, v)| u.len() == cols && v.len() == rows)
            .map(|(u, v)| (u.as_slice(), v.as_slice()));
        Some(FitMap {
            h: homography(&src, &q)?,
            axes,
        })
    }

    /// Image pixel of module coordinate (u, v).
    pub fn px(&self, u: f64, v: f64) -> [f64; 2] {
        let (u, v) = match self.axes {
            Some((cu, rv)) => (through(cu, u), through(rv, v)),
            None => (u, v),
        };
        let h = &self.h;
        let w = h[6] * u + h[7] * v + h[8];
        let w = if w.abs() < 1e-12 { 1e-12 } else { w };
        [
            (h[0] * u + h[1] * v + h[2]) / w,
            (h[3] * u + h[4] * v + h[5]) / w,
        ]
    }

    /// Module centres, row-major (rows x cols), in image pixels.
    pub fn centres(&self, rows: usize, cols: usize) -> Vec<[f64; 2]> {
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .map(|(c, r)| self.px(c as f64 + 0.5, r as f64 + 0.5))
            .collect()
    }
}

/// Coordinate `x` moved through tracked centres: centre i sits at i + 0.5
/// on the plain grid and at `t[i]` on the tracked one; linear between
/// centres and beyond the end ones (slope of the end segment).
fn through(t: &[f64], x: f64) -> f64 {
    let n = t.len();
    if n < 2 {
        return x;
    }
    let p = x - 0.5;
    let i = (p.floor().max(0.0) as usize).min(n - 2);
    let f = p - i as f64;
    t[i] + f * (t[i + 1] - t[i])
}

/// Bilinear gray value at (x, y), edges replicated, scaled to 0..1.
fn bilinear(img: &Gray8, x: f64, y: f64) -> f32 {
    let (w, h) = (img.width as i64, img.height as i64);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |xi: i64, yi: i64| {
        let xi = xi.clamp(0, w - 1) as usize;
        let yi = yi.clamp(0, h - 1) as usize;
        img.data[yi * img.width + xi] as f64
    };
    let (xi, yi) = (x0 as i64, y0 as i64);
    let top = at(xi, yi) * (1.0 - fx) + at(xi + 1, yi) * fx;
    let bot = at(xi, yi + 1) * (1.0 - fx) + at(xi + 1, yi + 1) * fx;
    ((top * (1.0 - fy) + bot * fy) / 255.0) as f32
}

/// The symbol of a fit resampled to `INPUT_PPM` pixels per module with
/// `INPUT_MARGIN` modules on each side: ((cols + 2 m) * ppm) wide, gray
/// 0..1 of `img` (the variant the fit was made on). Pixel (x, y) is module
/// coordinate ((x + 0.5) / ppm - m, (y + 0.5) / ppm - m).
pub fn symbol_input(img: &Gray8, map: &FitMap, rows: usize, cols: usize) -> GrayF32 {
    let (ppm, m) = (INPUT_PPM as f64, INPUT_MARGIN as f64);
    let w = (cols + 2 * INPUT_MARGIN) * INPUT_PPM;
    let h = (rows + 2 * INPUT_MARGIN) * INPUT_PPM;
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        let v = (y as f64 + 0.5) / ppm - m;
        for x in 0..w {
            let u = (x as f64 + 0.5) / ppm - m;
            let [px, py] = map.px(u, v);
            data.push(bilinear(img, px, py));
        }
    }
    Image::from_vec(w, h, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_grid_centres() {
        let quad: Quad = [[10.0, 20.0], [110.0, 20.0], [110.0, 120.0], [10.0, 120.0]];
        let map = FitMap::new(&quad, 10, 10, None).unwrap();
        let c = map.centres(10, 10);
        assert!((c[0][0] - 15.0).abs() < 1e-9 && (c[0][1] - 25.0).abs() < 1e-9);
        assert!((c[99][0] - 105.0).abs() < 1e-9 && (c[99][1] - 115.0).abs() < 1e-9);
    }

    #[test]
    fn tracked_axes_move_centres() {
        let quad: Quad = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let u: Vec<f64> = (0..10).map(|i| i as f64 + 0.5 + 0.1 * i as f64).collect();
        let v: Vec<f64> = (0..10).map(|i| i as f64 + 0.5).collect();
        let axes = (u, v);
        let map = FitMap::new(&quad, 10, 10, Some(&axes)).unwrap();
        let c = map.centres(10, 10);
        assert!((c[9][0] - (9.5 + 0.9)).abs() < 1e-9);
        // beyond the last centre: slope of the end segment (1.1 per module)
        assert!((map.px(10.5, 0.5)[0] - (10.4 + 1.1)).abs() < 1e-9);
    }

    #[test]
    fn input_size_and_values() {
        let mut img = Gray8::new(40, 40);
        img.data.iter_mut().for_each(|p| *p = 200);
        let quad: Quad = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]];
        let map = FitMap::new(&quad, 10, 10, None).unwrap();
        let inp = symbol_input(&img, &map, 10, 10);
        assert_eq!((inp.width, inp.height), (84, 84));
        assert!(inp.data.iter().all(|&v| (v - 200.0 / 255.0).abs() < 1e-6));
    }
}
