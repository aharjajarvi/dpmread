//! `Sampler` (module values of a darkness image for a quad), `score_quads`
//! (Fisher score of the finder / timing pattern), `order_quad` and
//! `orientations` (reader 1007–1014, 1128–1201).

use crate::np::{Order, mean_f32, var_f32};
use dm_imgproc::{
    Gray8, GrayF32, Image, Mat3, get_perspective_transform, homography, perspective_transform,
    remap_points, remap_project_f32, remap_project_f64,
};
use dm_symbol::Geometry;
use std::sync::Arc;

/// Four corners (x, y), float32 like the numpy `(4, 2)` arrays; oriented
/// quads are tl, tr, br, bl.
pub type Quad = [[f32; 2]; 4];

/// Samples a darkness image (`255 - gray`) at module centres for a quad,
/// averaging a 5-point footprint of `radius` modules.
#[derive(Clone, Debug)]
pub struct Sampler {
    /// Shared: samplers with other footprints reuse it.
    pub dark: Arc<GrayF32>,
    offs: [[f64; 2]; 5],
}

fn src_f32(rows: usize, cols: usize) -> [(f32, f32); 4] {
    let (c, r) = (cols as f32, rows as f32);
    [(0.0, 0.0), (c, 0.0), (c, r), (0.0, r)]
}

fn tuples(q: &Quad) -> [(f32, f32); 4] {
    q.map(|p| (p[0], p[1]))
}

/// Mean over the 5 footprint offsets: `vals` is offset-major (5 x n),
/// summed in order and divided in float32 as `mean(0)`.
fn footprint_mean(vals: &[f32], n: usize) -> Vec<f32> {
    let mut acc = vec![0f32; n];
    for o in 0..5 {
        for (a, &v) in acc.iter_mut().zip(&vals[o * n..(o + 1) * n]) {
            *a += v;
        }
    }
    acc.iter_mut().for_each(|a| *a /= 5.0);
    acc
}
use crate::timing::{Span, Stage};

impl Sampler {
    pub fn new(gray: &Gray8, radius: f64) -> Self {
        Self::from_dark(gray.map(|v| (255 - v) as f32), radius)
    }

    /// Sampler over an existing darkness image.
    pub fn from_dark(dark: impl Into<Arc<GrayF32>>, radius: f64) -> Self {
        let r = radius;
        Sampler {
            dark: dark.into(),
            offs: [[0.0, 0.0], [-r, 0.0], [r, 0.0], [0.0, -r], [0.0, r]],
        }
    }

    /// Same image, another footprint radius.
    pub fn with_radius(&self, radius: f64) -> Self {
        Sampler::from_dark(Arc::clone(&self.dark), radius)
    }

    /// Module coordinates (x = col, y = row) sampled in one remap; the
    /// homography from `getPerspectiveTransform`, projection in float64.
    /// All NaN if the quad is degenerate (OpenCV would fall back to SVD).
    pub fn sample_uv(&self, quad: &Quad, rows: usize, cols: usize, uv: &[[f64; 2]]) -> Vec<f32> {
        let _span = Span::new(Stage::Sample);
        let n = uv.len();
        let Some(h) = get_perspective_transform(&src_f32(rows, cols), &tuples(quad)) else {
            return vec![f32::NAN; n];
        };
        footprint_mean(&remap_project_f64(&self.dark, &h, uv, &self.offs), n)
    }

    /// `sample_uv` for many quads (K x len(uv), row-major): batched float64
    /// homographies cast to float32, projection in float32.
    pub fn sample_uv_many(
        &self,
        quads: &[Quad],
        rows: usize,
        cols: usize,
        uv: &[[f64; 2]],
    ) -> Vec<f32> {
        let _span = Span::new(Stage::Sample);
        let n = uv.len();
        let src = [
            (0.0, 0.0),
            (cols as f64, 0.0),
            (cols as f64, rows as f64),
            (0.0, rows as f64),
        ];
        let pts: Vec<[f32; 2]> = self
            .offs
            .iter()
            .flat_map(|o| {
                uv.iter()
                    .map(|p| [(p[0] + o[0]) as f32, (p[1] + o[1]) as f32])
            })
            .collect();
        let mut out = Vec::with_capacity(quads.len() * n);
        for q in quads {
            let q64 = q.map(|p| (p[0] as f64, p[1] as f64));
            // numpy falls back to lstsq for a singular batch member
            let h: [f32; 9] = homography(&src, &q64)
                .unwrap_or([f64::NAN; 9])
                .map(|v| v as f32);
            // projection clamps |den| < 1e-9 (the numpy code)
            out.extend(footprint_mean(&remap_project_f32(&self.dark, &h, &pts), n));
        }
        out
    }

    /// Module values at tracked column / row centres (rows x cols).
    pub fn sample_grid(
        &self,
        quad: &Quad,
        rows: usize,
        cols: usize,
        ucol: &[f64],
        vrow: &[f64],
    ) -> GrayF32 {
        let uv: Vec<[f64; 2]> = vrow
            .iter()
            .flat_map(|&v| ucol.iter().map(move |&u| [u, v]))
            .collect();
        Image::from_vec(cols, rows, self.sample_uv(quad, rows, cols, &uv))
    }

    /// Module values on the even grid, `pad` modules beyond each side;
    /// `perspectiveTransform` of float32 points, one remap per offset.
    pub fn sample(&self, quad: &Quad, rows: usize, cols: usize, pad: usize) -> GrayF32 {
        let (w, hgt) = (cols + 2 * pad, rows + 2 * pad);
        let mut acc = vec![0f32; w * hgt];
        let Some(m) = get_perspective_transform(&src_f32(rows, cols), &tuples(quad)) else {
            return Image::from_vec(w, hgt, vec![f32::NAN; w * hgt]);
        };
        self.sample_offsets(&m, w, hgt, pad, &mut acc);
        acc.iter_mut().for_each(|a| *a /= 5.0);
        Image::from_vec(w, hgt, acc)
    }

    fn sample_offsets(&self, m: &Mat3, w: usize, hgt: usize, pad: usize, acc: &mut [f32]) {
        for &[dx, dy] in &self.offs {
            let pts: Vec<(f32, f32)> = (0..hgt)
                .flat_map(|r| {
                    (0..w).map(move |c| {
                        let cc = c as f64 - pad as f64 + 0.5;
                        let rr = r as f64 - pad as f64 + 0.5;
                        ((cc + dx) as f32, (rr + dy) as f32)
                    })
                })
                .collect();
            let p = perspective_transform(&pts, m);
            let (xs, ys): (Vec<f32>, Vec<f32>) = p.into_iter().unzip();
            for (a, v) in acc.iter_mut().zip(remap_points(&self.dark, &xs, &ys)) {
                *a += v;
            }
        }
    }
}

/// Fisher score of each quad (finder / timing modules, with the quiet-zone
/// ring when `qz`); NaN / inf become `-inf`. Scores with and without the
/// ring are not comparable.
pub fn score_quads(sampler: &Sampler, quads: &[Quad], g: &Geometry, qz: bool) -> Vec<f32> {
    let (uv, dk) = if qz {
        (&g.fp_uv_qz, &g.fp_dark_qz)
    } else {
        (&g.fp_uv_in, &g.fp_dark_in)
    };
    fisher_scores(sampler, quads, g.rows, g.cols, uv, dk)
}

/// Fisher score `(mean_d - mean_l) / sqrt((var_d + var_l) / 2 + 4)` of the
/// module coordinates `uv` (dark where `dk`) for each quad; NaN / inf
/// become `-inf`.
pub fn fisher_scores(
    sampler: &Sampler,
    quads: &[Quad],
    rows: usize,
    cols: usize,
    uv: &[[f64; 2]],
    dk: &[bool],
) -> Vec<f32> {
    let n = uv.len();
    let v = sampler.sample_uv_many(quads, rows, cols, uv);
    // `v[:, dk]` is F-ordered: its rows reduce sequentially, except for a
    // single quad, where the row is contiguous (pairwise)
    let o = if quads.len() == 1 {
        Order::Pairwise
    } else {
        Order::Sequential
    };
    let (mut d, mut l) = (Vec::with_capacity(n), Vec::with_capacity(n));
    v.chunks_exact(n.max(1))
        .take(quads.len())
        .map(|row| {
            d.clear();
            l.clear();
            for (&x, &k) in row.iter().zip(dk) {
                if k { d.push(x) } else { l.push(x) }
            }
            let s = (mean_f32(&d, o) - mean_f32(&l, o))
                / ((var_f32(&d, o) + var_f32(&l, o)) / 2.0 + 4.0).sqrt();
            if s.is_finite() { s } else { f32::NEG_INFINITY }
        })
        .collect()
}

/// Clockwise in image coordinates starting top-left-most (angles around
/// the float32 centroid, stable sort, first minimum of x + y).
pub fn order_quad(q: &Quad) -> Quad {
    let mut c = [0f32; 2];
    for p in q {
        c[0] += p[0];
        c[1] += p[1];
    }
    let c = [c[0] / 4.0, c[1] / 4.0];
    let ang: Vec<f32> = q.iter().map(|p| (p[1] - c[1]).atan2(p[0] - c[0])).collect();
    let mut idx = [0usize, 1, 2, 3];
    // numpy sorts NaN last and treats -0.0 == 0.0
    idx.sort_by(|&a, &b| {
        ang[a]
            .partial_cmp(&ang[b])
            .unwrap_or_else(|| ang[a].is_nan().cmp(&ang[b].is_nan()))
    });
    let s: Vec<[f32; 2]> = idx.iter().map(|&i| q[i]).collect();
    let mut start = 0;
    for k in 1..4 {
        if s[k][0] + s[k][1] < s[start][0] + s[start][1] {
            start = k;
        }
    }
    std::array::from_fn(|k| s[(k + start) % 4])
}

/// The 8 corner assignments (4 rotations x mirror) as (quad tl, tr, br,
/// bl, rotation, mirrored).
pub fn orientations(quad: &Quad) -> Vec<(Quad, usize, bool)> {
    let q = order_quad(quad);
    let mut out = Vec::with_capacity(8);
    for mirror in [false, true] {
        let base = if mirror { [q[0], q[3], q[2], q[1]] } else { q };
        for r in 0..4 {
            out.push((std::array::from_fn(|k| base[(k + r) % 4]), r, mirror));
        }
    }
    out
}
