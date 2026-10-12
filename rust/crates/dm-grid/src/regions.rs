//! Alignment-pattern anchors (CORE-103): on a symbol with several data
//! regions every region is framed by its own solid and dashed alignment
//! lines. A global fit only extrapolates the grid inside the symbol; here
//! each region is re-fitted on its own frame and sampled with its own
//! homography, which follows a bend or pitch drift inside the symbol.

use crate::deadline::Deadline;
use crate::refine::{module_px, tracked_axes};
use crate::sampler::{Quad, Sampler, fisher_scores};
use dm_imgproc::{GrayF32, Image, get_perspective_transform, perspective_transform};
use dm_symbol::Geometry;

/// Step schedule (modules) of the region corner search.
const STEPS: [f32; 3] = [0.25, 0.12, 0.06];
/// Rounds per step.
const ITERS: usize = 4;
/// A region corner may move at most this far (modules) from where the
/// global fit puts it; beyond that the region frame was not found.
const MAX_SHIFT: f32 = 0.6;

/// One data region with its frame: module rectangle in the symbol and the
/// frame modules (centres in region coordinates, expected dark).
struct Region {
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
    uv: Vec<[f64; 2]>,
    dark: Vec<bool>,
}

fn regions(g: &Geometry) -> Vec<Region> {
    let s = &g.sym;
    let (rows, cols) = (s.dregr + 2, s.dregc + 2);
    let mut out = Vec::with_capacity(s.regr * s.regc);
    for br in 0..s.regr {
        for bc in 0..s.regc {
            let (r0, c0) = (br * rows, bc * cols);
            let (mut uv, mut dark) = (Vec::new(), Vec::new());
            for r in 0..rows {
                for c in 0..cols {
                    if r == 0 || c == 0 || r == rows - 1 || c == cols - 1 {
                        let i = (r0 + r) * g.cols + c0 + c;
                        uv.push([c as f64 + 0.5, r as f64 + 0.5]);
                        dark.push(g.fixed_val[i] == 1);
                    }
                }
            }
            out.push(Region {
                r0,
                c0,
                rows,
                cols,
                uv,
                dark,
            });
        }
    }
    out
}

fn score(sampler: &Sampler, q: &Quad, rg: &Region) -> f32 {
    fisher_scores(sampler, &[*q], rg.rows, rg.cols, &rg.uv, &rg.dark)[0]
}

/// Coordinate ascent on the 8 corner coordinates of one region quad,
/// maximising its frame contrast; corners stay within `MAX_SHIFT` modules
/// of `start`.
fn refine_region(sampler: &Sampler, start: &Quad, rg: &Region, md: f32) -> (Quad, f32) {
    let mut q = *start;
    let mut best = score(sampler, &q, rg);
    let lim = MAX_SHIFT * md;
    for &f in &STEPS {
        let step = f * md;
        for _ in 0..ITERS {
            let mut improved = false;
            for k in 0..4 {
                for axis in 0..2 {
                    for sgn in [1.0f32, -1.0] {
                        let mut t = q;
                        t[k][axis] += sgn * step;
                        if (t[k][axis] - start[k][axis]).abs() > lim {
                            continue;
                        }
                        let s = score(sampler, &t, rg);
                        if s > best + 1e-4 {
                            (best, q, improved) = (s, t, true);
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
    (q, best)
}

/// Module values (rows x cols) of a multi-region symbol, every region
/// sampled with its own re-fitted quad; `None` for a single-region symbol,
/// a degenerate quad or an expired deadline. `quad` is the global fit;
/// region corners start on its (warped or timing-tracked) module grid.
/// Module coordinate of the edge before module `k` (`k == n`: after the
/// last) from module centres `axis` (length `n`).
fn edge(axis: &[f64], k: usize) -> f64 {
    let n = axis.len();
    if k == 0 {
        axis[0] - (axis[1] - axis[0]) / 2.0
    } else if k >= n {
        axis[n - 1] + (axis[n - 1] - axis[n - 2]) / 2.0
    } else {
        (axis[k - 1] + axis[k]) / 2.0
    }
}

pub fn region_values(
    sampler: &Sampler,
    quad: &Quad,
    axes: Option<&(Vec<f64>, Vec<f64>)>,
    g: &'static Geometry,
    deadline: &Deadline,
) -> Option<GrayF32> {
    let s = &g.sym;
    if s.regr * s.regc < 2 {
        return None;
    }
    let src = [
        (0.0, 0.0),
        (g.cols as f32, 0.0),
        (g.cols as f32, g.rows as f32),
        (0.0, g.rows as f32),
    ];
    let h = get_perspective_transform(&src, &quad.map(|p| (p[0], p[1])))?;
    let md = module_px(quad, g);
    let tracked;
    let (ucol, vrow) = match axes {
        Some(a) => (&a.0, &a.1),
        None => {
            tracked = tracked_axes(sampler, quad, g);
            (&tracked.0, &tracked.1)
        }
    };
    let mut vals = vec![0f32; g.rows * g.cols];
    for rg in regions(g) {
        if deadline.expired() {
            return None;
        }
        let (r0, c0) = (edge(vrow, rg.r0) as f32, edge(ucol, rg.c0) as f32);
        let r1 = edge(vrow, rg.r0 + rg.rows) as f32;
        let c1 = edge(ucol, rg.c0 + rg.cols) as f32;
        let p = perspective_transform(&[(c0, r0), (c1, r0), (c1, r1), (c0, r1)], &h);
        let start: Quad = std::array::from_fn(|k| [p[k].0, p[k].1]);
        let (q, _) = refine_region(sampler, &start, &rg, md);
        let v = sampler.sample(&q, rg.rows, rg.cols, 0);
        for r in 0..rg.rows {
            for c in 0..rg.cols {
                vals[(rg.r0 + r) * g.cols + rg.c0 + c] = v.data[r * rg.cols + c];
            }
        }
    }
    if vals.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(Image::from_vec(g.cols, g.rows, vals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dm_symbol::{all_symbols, geometry};

    /// Every region frame: solid left column and bottom row, alternating
    /// top row and right column, as a single-region symbol of that size.
    #[test]
    fn region_frames_are_finder_and_timing() {
        for sym in all_symbols().filter(|s| s.regr * s.regc > 1) {
            let g = geometry(sym);
            let rs = regions(g);
            assert_eq!(rs.len(), sym.regr * sym.regc, "{}", sym.name());
            for rg in &rs {
                assert_eq!(rg.uv.len(), 2 * (rg.rows + rg.cols) - 4);
                for (p, &d) in rg.uv.iter().zip(&rg.dark) {
                    let (c, r) = ((p[0] - 0.5) as usize, (p[1] - 0.5) as usize);
                    let want = if c == 0 || r == rg.rows - 1 {
                        true
                    } else if r == 0 {
                        c % 2 == 0
                    } else {
                        // right column, counted from the bottom
                        (rg.rows - 1 - r) % 2 == 0
                    };
                    assert_eq!(d, want, "{} region at ({}, {}): ({r}, {c})", sym.name(), rg.r0, rg.c0);
                }
            }
        }
    }
}
