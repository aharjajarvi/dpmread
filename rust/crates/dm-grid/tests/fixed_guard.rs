//! Weak-RS symbols are not accepted when their finder / timing pattern does
//! not match the fit: the codewords of a 10x10 that RS accepted inside a
//! 20x20 (2 of 2 errors corrected), with fixed modules flipped (CORE-125).

use dm_grid::*;
use dm_imgproc::Image;
use dm_symbol::{Geometry, all_symbols, build_matrix, geometry};

const CODEWORDS: [u8; 8] = [0xDA, 0xB2, 0x33, 0xEF, 0xCE, 0xAD, 0xA4, 0x1F];

fn g10() -> &'static Geometry {
    geometry(
        all_symbols()
            .find(|s| (s.rows, s.cols) == (10, 10))
            .unwrap(),
    )
}

/// The symbol's module values (dark = high) with the first `flips` fixed
/// modules inverted.
fn values(g: &Geometry, flips: usize) -> Image<f32> {
    let mut m = build_matrix(&CODEWORDS, &g.sym);
    let fixed: Vec<usize> = (0..m.len()).filter(|&i| g.fixed_mask[i]).collect();
    for &i in fixed.iter().take(flips) {
        m[i] ^= 1;
    }
    let data = m
        .iter()
        .map(|&b| if b == 1 { 200.0 } else { 50.0 })
        .collect();
    Image::from_vec(g.cols, g.rows, data)
}

fn read(flips: usize) -> Option<String> {
    let g = g10();
    let mut fit = GridFit::new(g, [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], 0.0);
    let got = try_fit_values(&mut fit, Mode::Plane, values(g, flips));
    assert_eq!(fit.fixed_errors, flips);
    got.map(|(_, dec)| String::from_utf8(dec.data).unwrap())
}

#[test]
fn intact_and_lightly_damaged_read() {
    assert_eq!(read(0).as_deref(), Some("88482"));
    // 7 / 36 = 19.4 %
    assert_eq!(read(7).as_deref(), Some("88482"));
}

#[test]
fn alias_damage_is_rejected() {
    // 8 / 36 = 22.2 %; 11 / 36 = 30.6 % as in CORE-125
    assert_eq!(read(8), None);
    assert_eq!(read(11), None);
}

#[test]
fn only_weak_rs_symbols_are_gated() {
    let gated: Vec<String> = all_symbols()
        .map(geometry)
        .filter(|g| g.n_ecc <= WEAK_RS_ECC)
        .map(|g| g.name())
        .collect();
    assert_eq!(gated, ["10x10", "12x12", "8x18"]);
    let g14 = geometry(
        all_symbols()
            .find(|s| (s.rows, s.cols) == (14, 14))
            .unwrap(),
    );
    assert!(!fixed_pattern_too_damaged(g14, 52));
}
