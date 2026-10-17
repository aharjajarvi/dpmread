//! A square symbol read in its anti-transposed frame (module (r, c) ->
//! (n - 1 - c, n - 1 - r)) keeps a perfect finder / timing pattern but
//! fails RS; `anti_transposed` maps it back (CORE-128).

use dm_grid::*;
use dm_imgproc::Image;
use dm_symbol::{Geometry, all_symbols, build_matrix, geometry};

/// "88482" as a 12x12 (`datamatrix_gen.py`).
const CODEWORDS: [u8; 12] = [
    0xDA, 0xB2, 0x33, 0x81, 0x73, 0xB6, 0xF1, 0x9A, 0xCE, 0x96, 0x70, 0xCB,
];

fn g12() -> &'static Geometry {
    geometry(
        all_symbols()
            .find(|s| (s.rows, s.cols) == (12, 12))
            .unwrap(),
    )
}

/// The symbol's module values (dark = high) in the anti-transposed frame.
fn mirrored_values(g: &Geometry) -> Image<f32> {
    let n = g.rows;
    let m = build_matrix(&CODEWORDS, &g.sym);
    let data = (0..n * n)
        .map(|i| m[(n - 1 - i % n) * n + (n - 1 - i / n)])
        .map(|b| if b == 1 { 200.0 } else { 50.0 })
        .collect();
    Image::from_vec(n, n, data)
}

fn fit(g: &'static Geometry) -> GridFit {
    let n = g.rows;
    let mut f = GridFit::new(g, [[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [0.0, 60.0]], 0.0);
    f.axes = Some((
        (0..n).map(|c| c as f64 + 0.4).collect(),
        (0..n).map(|r| r as f64 + 0.6).collect(),
    ));
    f
}

#[test]
fn mirrored_frame_has_a_perfect_fixed_pattern_and_fails() {
    let g = g12();
    let mut f = fit(g);
    assert!(try_fit_values(&mut f, Mode::Plane, mirrored_values(g)).is_none());
    assert_eq!(f.fixed_errors, 0);
}

#[test]
fn anti_transposed_frame_reads() {
    let g = g12();
    let f = fit(g);
    let (mut at, vals) = anti_transposed(&f, &mirrored_values(g)).unwrap();
    let (_, dec) = try_fit_values(&mut at, Mode::Plane, vals).unwrap();
    assert_eq!(String::from_utf8(dec.data).unwrap(), "88482");
    assert_eq!(at.quad[0], f.quad[2]);
    assert_eq!(at.quad[1], f.quad[1]);
    // column centres of the new frame are the reversed row centres
    let (u, v) = at.axes.as_ref().unwrap();
    assert!((u[0] - 0.4).abs() < 1e-9 && (v[0] - 0.6).abs() < 1e-9);
}

#[test]
fn anti_transpose_undoes_itself() {
    let g = g12();
    let f = fit(g);
    let vals = mirrored_values(g);
    let (at, at_vals) = anti_transposed(&f, &vals).unwrap();
    let (back, back_vals) = anti_transposed(&at, &at_vals).unwrap();
    assert_eq!(back.quad, f.quad);
    assert_eq!(back_vals.data, vals.data);
    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9);
    let ((bu, bv), (fu, fv)) = (back.axes.unwrap(), f.axes.unwrap());
    assert!(close(&bu, &fu) && close(&bv, &fv));
}

#[test]
fn rectangular_symbols_have_no_anti_transposed_frame() {
    let g = geometry(all_symbols().find(|s| (s.rows, s.cols) == (8, 18)).unwrap());
    let f = GridFit::new(g, [[0.0, 0.0]; 4], 0.0);
    let vals = Image::from_vec(g.cols, g.rows, vec![0.0; g.rows * g.cols]);
    assert!(anti_transposed(&f, &vals).is_none());
}
