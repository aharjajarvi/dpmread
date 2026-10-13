//! Round-dot matched filter (CORE-104): dot-peen and laser-dot marks put
//! one disc per dark module with gaps between them. The point footprints
//! average whatever lies under them; here each module is correlated with a
//! disc of the measured dot size instead. Only used when the solid finder
//! line of the fit shows a dot texture, so solid marks are untouched.

use crate::sampler::{Quad, Sampler};
use dm_imgproc::{GrayF32, Image};
use dm_symbol::Geometry;

/// Samples per module along the finder line for the texture check.
const PROFILE_STEPS: usize = 10;
/// Dip at the module edges, relative to the dark / light contrast, above
/// which the finder line counts as dotted.
pub const DOT_TEXTURE_MIN: f64 = 0.2;
/// Sample grid inside a module for the disc correlation (n x n).
const GRID_N: usize = 7;

/// Dot texture of a fit: (edge dip relative to contrast, dot radius in
/// modules), measured on the solid finder L (left column, bottom row). On
/// a solid line the darkness is flat; on dots it dips between modules.
pub fn dot_texture(sampler: &Sampler, quad: &Quad, g: &Geometry) -> (f64, f64) {
    let (rows, cols) = (g.rows, g.cols);
    let step = |k: usize| k as f64 / PROFILE_STEPS as f64;
    // module centre line of the left column and of the bottom row
    let mut uv = Vec::new();
    for r in 1..rows - 1 {
        uv.extend((0..PROFILE_STEPS).map(|k| [0.5, r as f64 + step(k)]));
    }
    for c in 1..cols - 1 {
        uv.extend((0..PROFILE_STEPS).map(|k| [c as f64 + step(k), rows as f64 - 0.5]));
    }
    let prof = sampler.sample_uv(quad, rows, cols, &uv);
    // light reference: the light modules of the top timing row
    let light_uv: Vec<[f64; 2]> = (0..cols)
        .filter(|c| g.fixed_val[*c] == 0)
        .map(|c| [c as f64 + 0.5, 0.5])
        .collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let light: Vec<f64> = sampler
        .sample_uv(quad, rows, cols, &light_uv)
        .iter()
        .map(|&x| x as f64)
        .collect();
    let l = mean(&light);
    // per module: sample 0 is the module edge, PROFILE_STEPS / 2 the centre
    let (mut centre, mut edge, mut dark_len) = (Vec::new(), Vec::new(), Vec::new());
    for m in prof.as_chunks::<PROFILE_STEPS>().0 {
        let pk = m[PROFILE_STEPS / 2] as f64;
        centre.push(pk);
        edge.push(m[0] as f64);
        let mid = (pk + l) / 2.0;
        dark_len.push(step(m.iter().filter(|&&x| x as f64 > mid).count()));
    }
    let (c, e) = (mean(&centre), mean(&edge));
    let dip = (c - e) / (c - l).max(1e-3);
    let radius = (mean(&dark_len) / 2.0).clamp(0.2, 0.5);
    (dip, radius)
}

/// Module values from a disc correlation: mean darkness inside a disc of
/// `radius` modules minus the mean of the rest of the module cell, for every
/// module of the grid given by its column / row centres.
pub fn disc_values(
    sampler: &Sampler,
    quad: &Quad,
    rows: usize,
    cols: usize,
    ucol: &[f64],
    vrow: &[f64],
    radius: f64,
) -> GrayF32 {
    let offs: Vec<([f64; 2], bool)> = (0..GRID_N)
        .flat_map(|i| (0..GRID_N).map(move |j| (i, j)))
        .map(|(i, j)| {
            let o = [
                (i as f64 + 0.5) / GRID_N as f64 - 0.5,
                (j as f64 + 0.5) / GRID_N as f64 - 0.5,
            ];
            (o, (o[0] * o[0] + o[1] * o[1]).sqrt() <= radius)
        })
        .collect();
    let mut uv = Vec::with_capacity(rows * cols * offs.len());
    for &v in vrow {
        for &u in ucol {
            uv.extend(offs.iter().map(|(o, _)| [u + o[0], v + o[1]]));
        }
    }
    let s = sampler.sample_uv(quad, rows, cols, &uv);
    let n_in = offs.iter().filter(|o| o.1).count().max(1) as f32;
    let n_out = (offs.len() as f32 - n_in).max(1.0);
    let vals = s
        .chunks_exact(offs.len())
        .map(|m| {
            let (mut a, mut b) = (0f32, 0f32);
            for (x, (_, inside)) in m.iter().zip(&offs) {
                if *inside { a += x } else { b += x }
            }
            a / n_in - b / n_out
        })
        .collect();
    Image::from_vec(cols, rows, vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dm_imgproc::Gray8;
    use dm_symbol::{all_symbols, geometry};

    const PX: usize = 12;
    const QZ: usize = 2;

    /// The fixed pattern of 16x16 rendered at `PX` px per module, dark
    /// modules as discs of `radius` modules or (`None`) as solid squares.
    fn render(radius: Option<f64>) -> (Gray8, Quad, &'static Geometry) {
        let sym = all_symbols()
            .find(|s| s.rows == 16 && s.cols == 16)
            .unwrap();
        let g = geometry(sym);
        let n = (g.cols + 2 * QZ) * PX;
        let mut data = vec![220u8; n * n];
        for r in 0..g.rows {
            for c in 0..g.cols {
                let i = r * g.cols + c;
                if !(g.fixed_mask[i] && g.fixed_val[i] == 1) {
                    continue;
                }
                for y in 0..PX {
                    for x in 0..PX {
                        let (dx, dy) = (
                            (x as f64 + 0.5) / PX as f64 - 0.5,
                            (y as f64 + 0.5) / PX as f64 - 0.5,
                        );
                        if radius.is_none_or(|rr| (dx * dx + dy * dy).sqrt() <= rr) {
                            data[((QZ + r) * PX + y) * n + (QZ + c) * PX + x] = 40;
                        }
                    }
                }
            }
        }
        let (a, b) = ((QZ * PX) as f32, ((QZ + g.cols) * PX) as f32);
        (
            Gray8::from_vec(n, n, data),
            [[a, a], [b, a], [b, b], [a, b]],
            g,
        )
    }

    #[test]
    fn dots_are_detected_and_sized() {
        let (img, q, g) = render(Some(0.32));
        let (dip, radius) = dot_texture(&Sampler::new(&img, 0.0), &q, g);
        assert!(dip > DOT_TEXTURE_MIN, "dip {dip}");
        assert!((radius - 0.32).abs() < 0.08, "radius {radius}");
    }

    #[test]
    fn solid_finder_is_not_dotted() {
        let (img, q, g) = render(None);
        let (dip, _) = dot_texture(&Sampler::new(&img, 0.0), &q, g);
        assert!(dip < DOT_TEXTURE_MIN, "dip {dip}");
    }

    #[test]
    fn disc_values_separate_dark_and_light() {
        let (img, q, g) = render(Some(0.32));
        let even = |n: usize| (0..n).map(|i| i as f64 + 0.5).collect::<Vec<_>>();
        let v = disc_values(
            &Sampler::new(&img, 0.0),
            &q,
            g.rows,
            g.cols,
            &even(g.cols),
            &even(g.rows),
            0.32,
        );
        // bottom-left corner module is dark, the top-right one light
        let (dark, light) = (v.data[(g.rows - 1) * g.cols], v.data[g.cols - 1]);
        assert!(dark > light + 50.0, "dark {dark} light {light}");
    }
}
