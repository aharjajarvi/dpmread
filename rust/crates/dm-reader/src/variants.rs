//! Enhancement variants (`Variant`, `map_quad`, `enhancement_variants`,
//! `inverted_chain`, `grid_variants`, `tile_variants`, `_tile_axis`).
//! Variant names are identical to Python's: they appear in the report, the
//! JSON output and the parity comparison.

use dm_imgproc::{
    Bgr8, CHANNELS, Gray8, Interp, adaptive, bgr_to_gray, channel, clahe, flatten_background,
    gamma, gaussian_blur_u8, median_blur5, merge_dots, normalise_size, otsu, remove_texture,
    resize, sharpen,
};
use rayon::prelude::*;

/// One preprocessed grayscale image handed to the detectors.
#[derive(Clone, Debug)]
pub struct Variant {
    /// Processing chain, e.g. `gray > flatten > clahe`.
    pub name: String,
    /// 8-bit grayscale, symbol expected dark on light.
    pub img: Gray8,
    /// original px = variant px / scale + (ox, oy)
    pub scale: f64,
    /// Crop offset in original pixels (tiles).
    pub ox: f64,
    pub oy: f64,
}

impl Variant {
    pub fn new(name: impl Into<String>, img: Gray8, scale: f64) -> Self {
        Variant {
            name: name.into(),
            img,
            scale,
            ox: 0.0,
            oy: 0.0,
        }
    }

    /// A tile of a large photo.
    pub fn tile(&self) -> bool {
        self.name.starts_with("tile")
    }
}

/// Quad corners (tl, tr, br, bl) as f32, like the numpy arrays in Python.
pub type Quad = [(f32, f32); 4];

/// `map_quad`: quad in `src` variant pixels -> `dst` variant pixels (numpy
/// float32 maths: the Python scalars are rounded to f32 first).
pub fn map_quad(q: &Quad, src: &Variant, dst: &Variant) -> Quad {
    if src.ox == 0.0 && src.oy == 0.0 && dst.ox == 0.0 && dst.oy == 0.0 {
        let k = (dst.scale / src.scale) as f32;
        return q.map(|(x, y)| (x * k, y * k));
    }
    let (ss, ds) = (src.scale as f32, dst.scale as f32);
    let (sox, soy, dox, doy) = (src.ox as f32, src.oy as f32, dst.ox as f32, dst.oy as f32);
    q.map(|(x, y)| ((x / ss + sox - dox) * ds, (y / ss + soy - doy) * ds))
}

/// Python `f"{sig:g}"` for the blur sigmas used here (1.0 -> "1").
fn fmt_g(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// `enhancement_variants`: ordered from cheap/faithful to aggressive; the
/// name records the processing chain. `inverted_extra` appends the small
/// inverted chain for polarity "any". The chains (one per channel, the
/// two extra scales, the inverted chain) are built as rayon tasks and
/// concatenated in order.
pub fn enhancement_variants(bgr: &Bgr8, invert: bool, inverted_extra: bool) -> Vec<Variant> {
    #[derive(Clone, Copy)]
    enum Chain {
        Channel(usize),
        Scale(usize),
        Inverted,
    }
    let mut chains: Vec<Chain> = (0..CHANNELS).map(Chain::Channel).collect();
    chains.extend([350, 1100].map(Chain::Scale));
    if !invert && inverted_extra {
        chains.push(Chain::Inverted);
    }
    let parts: Vec<Vec<Variant>> = chains
        .into_par_iter()
        .map(|c| match c {
            Chain::Channel(i) => {
                let (cname, g0) = channel(bgr, invert, i);
                channel_chain(&cname, &g0)
            }
            Chain::Scale(tgt) => scale_chain(bgr, tgt),
            Chain::Inverted => inverted_chain(bgr),
        })
        .collect();
    parts.into_iter().flatten().collect()
}

/// The full chain of one channel at 700 px.
fn channel_chain(cname: &str, g0: &Gray8) -> Vec<Variant> {
    let mut out = Vec::new();
    let (g, s) = normalise_size(g0, 700);
    let m = g.min_side() as f64;
    let cl = clahe(&g, 3.0, 8);
    let flat = clahe(&flatten_background(&g, m / 8.0), 3.0, 8);
    out.push(Variant::new(cname, g.clone(), s));
    out.push(Variant::new(format!("{cname} > clahe"), cl.clone(), s));
    out.push(Variant::new(
        format!("{cname} > flatten > clahe"),
        flat.clone(),
        s,
    ));
    for sig in [1.0, 2.0] {
        let b = gaussian_blur_u8(&flat, sig);
        let bin = otsu(&b);
        let sg = fmt_g(sig);
        out.push(Variant::new(
            format!("{cname} > flatten > clahe > blur{sg}"),
            b,
            s,
        ));
        out.push(Variant::new(
            format!("{cname} > flatten > clahe > blur{sg} > otsu"),
            bin,
            s,
        ));
    }
    out.push(Variant::new(
        format!("{cname} > median5 > sharpen"),
        sharpen(&median_blur5(&g), 2.0, 1.5),
        s,
    ));
    let fb = gaussian_blur_u8(&flat, 1.0);
    for k in [3, 5, 7, 9] {
        let md = merge_dots(&fb, k);
        let bin = otsu(&md);
        out.push(Variant::new(
            format!("{cname} > flatten > clahe > merge-dots{k}"),
            md,
            s,
        ));
        out.push(Variant::new(
            format!("{cname} > flatten > clahe > merge-dots{k} > otsu"),
            bin,
            s,
        ));
    }
    let tx = remove_texture(&flat, 3);
    out.push(Variant::new(
        format!("{cname} > flatten > clahe > de-grain3 > merge-dots5 > otsu"),
        otsu(&merge_dots(&tx, 5)),
        s,
    ));
    out.push(Variant::new(
        format!("{cname} > clahe > adaptive"),
        adaptive(&gaussian_blur_u8(&cl, 1.5), m / 6.0),
        s,
    ));
    out.push(Variant::new(
        format!("{cname} > gamma2 > clahe"),
        clahe(&gamma(&g, 2.0), 3.0, 8),
        s,
    ));
    out
}

/// Extra scales of the most useful chain: module size matters for
/// detectors.
fn scale_chain(bgr: &Bgr8, tgt: usize) -> Vec<Variant> {
    let mut out = Vec::new();
    let (g, s) = normalise_size(&bgr_to_gray(bgr), tgt);
    let flat = clahe(&flatten_background(&g, g.min_side() as f64 / 8.0), 3.0, 8);
    let fb = gaussian_blur_u8(&flat, 1.0);
    out.push(Variant::new(
        format!("gray@{tgt}px > flatten > clahe"),
        flat,
        s,
    ));
    for k in [3, 5, 9] {
        out.push(Variant::new(
            format!("gray@{tgt}px > flatten > clahe > merge-dots{k} > otsu"),
            otsu(&merge_dots(&fb, k)),
            s,
        ));
    }
    out
}

/// `inverted_chain`: light-on-dark marks, the useful gray chain inverted.
pub fn inverted_chain(bgr: &Bgr8) -> Vec<Variant> {
    let g0 = bgr_to_gray(bgr).inverted();
    let (g, s) = normalise_size(&g0, 700);
    let m = g.min_side() as f64;
    let flat = clahe(&flatten_background(&g, m / 8.0), 3.0, 8);
    let md = otsu(&merge_dots(&gaussian_blur_u8(&flat, 1.0), 5));
    vec![
        Variant::new("inv-gray", g, s),
        Variant::new("inv-gray > flatten > clahe", flat, s),
        Variant::new("inv-gray > flatten > clahe > merge-dots5 > otsu", md, s),
    ]
}

/// One member of `grid_variants`: one or two variants built together.
#[derive(Clone, Copy, Debug)]
pub enum GridMember {
    /// `<channel> > flatten` of the first three channels.
    Flat(usize),
    /// Gray and `gray > flatten > clahe`.
    Gray,
    Inverted,
}

/// The members of `grid_variants`, in order.
pub fn grid_members(invert: bool, inverted_extra: bool) -> Vec<GridMember> {
    let mut members: Vec<GridMember> = (0..3).map(GridMember::Flat).collect();
    members.push(GridMember::Gray);
    if !invert && inverted_extra {
        members.push(GridMember::Inverted);
    }
    members
}

/// The variants of one `grid_variants` member.
pub fn grid_member(bgr: &Bgr8, invert: bool, m: GridMember) -> Vec<Variant> {
    let flat6 = |name: String, c: &Gray8| {
        let (g, s) = normalise_size(c, 600);
        let sigma = g.min_side() as f64 / 6.0;
        Variant::new(name, flatten_background(&g, sigma), s)
    };
    match m {
        GridMember::Flat(i) => {
            let (cname, c) = channel(bgr, invert, i);
            vec![flat6(format!("{cname} > flatten"), &c)]
        }
        GridMember::Gray => {
            let (gk, c) = channel(bgr, invert, 0);
            let (g, s) = normalise_size(&c, 600);
            let sigma = g.min_side() as f64 / 6.0;
            let fc = clahe(&flatten_background(&g, sigma), 2.0, 4);
            vec![
                Variant::new(gk.clone(), g, s),
                Variant::new(format!("{gk} > flatten > clahe"), fc, s),
            ]
        }
        GridMember::Inverted => vec![flat6(
            "inv-gray > flatten".to_string(),
            &bgr_to_gray(bgr).inverted(),
        )],
    }
}

/// `grid_variants`: grayscale variants for the grid decoder (no
/// binarisation); each member is a rayon task, in order.
pub fn grid_variants(bgr: &Bgr8, invert: bool, inverted_extra: bool) -> Vec<Variant> {
    let parts: Vec<Vec<Variant>> = grid_members(invert, inverted_extra)
        .into_par_iter()
        .map(|m| grid_member(bgr, invert, m))
        .collect();
    parts.into_iter().flatten().collect()
}

/// Longer side above which tiles are scanned.
pub const TILE_TRIGGER: usize = 1400;
/// Tile size handed to the detectors (px).
pub const TILE_SIZE: usize = 900;
/// A symbol on a tile border is whole in a neighbour.
pub const TILE_OVERLAP: f64 = 0.3;
/// Tiles per axis.
pub const TILE_MAX: usize = 3;

/// `_tile_axis`: (tile count, tile length in original px) along one axis.
pub fn tile_axis(length: usize) -> (usize, usize) {
    let ts = TILE_SIZE as f64;
    let n = ((length as f64 - ts) / (ts * (1.0 - TILE_OVERLAP))).ceil() as i64 + 1;
    let n = n.clamp(1, TILE_MAX as i64) as f64;
    let t = (TILE_SIZE as f64).max((length as f64 / (n - (n - 1.0) * TILE_OVERLAP)).ceil());
    (n as usize, length.min(t as usize))
}

/// `tile_variants`: overlapping tiles of a large photo, one small variant set
/// (gray, flatten > clahe, merge-dots5 > otsu) per tile, one rayon task per
/// tile; empty unless the longer side exceeds `TILE_TRIGGER`.
pub fn tile_variants(bgr: &Bgr8, invert: bool) -> Vec<Vec<Variant>> {
    let (h, w) = (bgr.height, bgr.width);
    if h.max(w) <= TILE_TRIGGER {
        return Vec::new();
    }
    let mut gray = bgr_to_gray(bgr);
    if invert {
        gray = gray.inverted();
    }
    let ((ny, th), (nx, tw)) = (tile_axis(h), tile_axis(w));
    let s = (TILE_SIZE as f64 / th.max(tw) as f64).min(1.0);
    let pos = |n: usize, len: usize, t: usize| -> Vec<usize> {
        (0..n)
            .map(|i| {
                (i as f64 * (len - t) as f64 / (n - 1).max(1) as f64).round_ties_even() as usize
            })
            .collect()
    };
    let (ys, xs) = (pos(ny, h, th), pos(nx, w, tw));
    let cells: Vec<(usize, usize, usize, usize)> = ys
        .iter()
        .enumerate()
        .flat_map(|(r, &y0)| xs.iter().enumerate().map(move |(c, &x0)| (r, c, y0, x0)))
        .collect();
    cells
        .into_par_iter()
        .map(|(r, c, y0, x0)| {
            let mut g = gray.crop(x0, y0, tw, th);
            if s < 1.0 {
                g = resize(&g, s, Interp::Area);
            }
            let name = format!("tile{r}x{c}");
            let flat = clahe(&flatten_background(&g, g.min_side() as f64 / 8.0), 3.0, 8);
            let md = otsu(&merge_dots(&gaussian_blur_u8(&flat, 1.0), 5));
            let (ox, oy) = (x0 as f64, y0 as f64);
            vec![
                Variant {
                    name: format!("{name} > gray"),
                    img: g,
                    scale: s,
                    ox,
                    oy,
                },
                Variant {
                    name: format!("{name} > gray > flatten > clahe"),
                    img: flat,
                    scale: s,
                    ox,
                    oy,
                },
                Variant {
                    name: format!("{name} > gray > flatten > clahe > merge-dots5 > otsu"),
                    img: md,
                    scale: s,
                    ox,
                    oy,
                },
            ]
        })
        .collect()
}
