//! P2.3: Gaussian blur (8-bit exact, float), box blur, median, addWeighted,
//! normalize, LUT and the enhancement wrappers built on them.

mod common;

use common::{assert_close_f32, assert_close_u8, cases, f32img, gray, name};
use dm_imgproc::*;

fn p(c: &common::Case, k: &str) -> f64 {
    c.params[k].as_f64().unwrap()
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn gaussian_u8_exact() {
    for c in cases("gauss_u8") {
        let got = gaussian_blur_u8(&gray(&c.input), p(&c, "sigma"));
        assert_close_u8(&name(&c), &got, &gray(&c.output), 0.0, 0);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn gaussian_f32() {
    for c in cases("gauss_f32") {
        let got = gaussian_blur_f32(&f32img(&c.input), p(&c, "sigma"));
        assert_close_f32(&name(&c), &got, &f32img(&c.output), 1e-3);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn box_f32() {
    for c in cases("box_f32") {
        let got = box_blur_f32(&f32img(&c.input), 5, Border::Reflect);
        assert_close_f32(&name(&c), &got, &f32img(&c.output), 1e-4);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn median5_exact() {
    for c in cases("median5") {
        assert_close_u8(
            &name(&c),
            &median_blur5(&gray(&c.input)),
            &gray(&c.output),
            0.0,
            0,
        );
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn add_weighted_exact() {
    for c in cases("add_weighted") {
        let i = c.input.as_array().unwrap();
        let got = add_weighted_u8(
            &gray(&i[0]),
            p(&c, "alpha"),
            &gray(&i[1]),
            p(&c, "beta"),
            p(&c, "gamma"),
        );
        assert_close_u8(&name(&c), &got, &gray(&c.output), 0.0, 0);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn normalize_f32() {
    for c in cases("normalize_f32") {
        let got = normalize_minmax_f32(&f32img(&c.input), p(&c, "a"), p(&c, "b"));
        assert_close_f32(&name(&c), &got, &f32img(&c.output), 1e-4);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn flatten() {
    for c in cases("flatten_background") {
        let got = flatten_background(&gray(&c.input), p(&c, "sigma"));
        assert_close_u8(&name(&c), &got, &gray(&c.output), 0.01, 1);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn sharpen_gamma_exact() {
    for c in cases("sharpen_median5") {
        let got = sharpen(&median_blur5(&gray(&c.input)), 2.0, 1.5);
        assert_close_u8(&name(&c), &got, &gray(&c.output), 0.0, 0);
    }
    for c in cases("gamma") {
        assert_close_u8(
            &name(&c),
            &gamma(&gray(&c.input), 2.0),
            &gray(&c.output),
            0.0,
            0,
        );
    }
}
