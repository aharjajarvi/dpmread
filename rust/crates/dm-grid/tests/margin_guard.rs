//! A read that looks wrong must leave error correction unused: a 10x10
//! with all of its correction used is not accepted on a grid far from
//! square, nor a small symbol with a damaged finder / timing pattern and
//! erasures (CORE-133).

use dm_ecc::RSResult;
use dm_grid::*;
use dm_imgproc::Image;
use dm_symbol::{Geometry, all_symbols, build_matrix, geometry};

/// "88482" as a 10x10 (`datamatrix_gen.py`).
const CODEWORDS: [u8; 8] = [0xDA, 0xB2, 0x33, 0xEF, 0xCE, 0xAD, 0xA4, 0x1F];

fn sym(rows: usize, cols: usize) -> &'static Geometry {
    geometry(
        all_symbols()
            .find(|s| (s.rows, s.cols) == (rows, cols))
            .unwrap(),
    )
}

/// Module values (dark = high) of the symbol with the first `wrong`
/// codewords inverted.
fn values(g: &Geometry, wrong: usize) -> Image<f32> {
    let mut cw = CODEWORDS;
    for c in cw.iter_mut().take(wrong) {
        *c ^= 0xFF;
    }
    let data = build_matrix(&cw, &g.sym)
        .iter()
        .map(|&b| if b == 1 { 200.0 } else { 50.0 })
        .collect();
    Image::from_vec(g.cols, g.rows, data)
}

/// Read on a quad `w` x `h` px.
fn read(wrong: usize, w: f32, h: f32) -> Option<String> {
    let g = sym(10, 10);
    let mut fit = GridFit::new(g, [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], 0.0);
    try_fit_values(&mut fit, Mode::Plane, values(g, wrong))
        .map(|(_, dec)| String::from_utf8(dec.data).unwrap())
}

#[test]
fn square_grid_may_use_all_correction() {
    assert_eq!(read(2, 100.0, 100.0).as_deref(), Some("88482"));
    // 100 x 70.2: 0.35, the highest true read seen
    assert_eq!(read(2, 100.0, 70.2).as_deref(), Some("88482"));
}

#[test]
fn non_square_grid_needs_a_spare_correction() {
    // 100 x 200: axial nonuniformity 0.67
    assert_eq!(read(1, 100.0, 200.0).as_deref(), Some("88482"));
    assert_eq!(read(2, 100.0, 200.0), None);
}

fn rs(errors: usize, erasures: usize, uec: f64) -> RSResult {
    RSResult {
        data_cw: vec![],
        all_cw: vec![],
        block_errors: vec![errors],
        block_erasures: vec![erasures],
        corrected_positions: vec![],
        uec,
    }
}

#[test]
fn damaged_small_symbol_may_not_use_erasures() {
    let g = sym(14, 14);
    let sq = [[0.0, 0.0], [35.0, 0.0], [35.0, 35.0], [0.0, 35.0]];
    let n_fixed = g.fixed_mask.iter().filter(|&&m| m).count();
    let damaged = (0.22 * n_fixed as f64) as usize;
    let light = (0.15 * n_fixed as f64) as usize;
    // the false 14x14: 6 erasures, 1 error, 2 of 10 check codewords left
    assert!(suspect_without_margin(g, &sq, damaged, &rs(7, 6, 0.2)));
    assert!(!suspect_without_margin(g, &sq, damaged, &rs(3, 0, 0.4)));
    assert!(!suspect_without_margin(g, &sq, light, &rs(7, 6, 0.2)));
    // larger symbols are gated on shape only
    let g18 = sym(18, 18);
    assert!(!suspect_without_margin(g18, &sq, 30, &rs(7, 6, 0.0)));
}

#[test]
fn axial_nonuniformity_matches_the_quality_report() {
    let g = sym(10, 10);
    let q = [[0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]];
    assert!((axial_nonuniformity(&q, g) - 2.0 / 3.0).abs() < 1e-9);
}
