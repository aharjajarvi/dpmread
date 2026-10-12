//! P2.6: resize INTER_AREA (exact) and INTER_CUBIC (IPP model), and
//! `normalise_size` at all targets the reader uses (350, 600, 700, 1100 px,
//! tiles).

mod common;

use common::{assert_close_u8, cases, gray, name};
use dm_imgproc::*;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn resize_area_exact() {
    for c in cases("resize") {
        let s = c.params["s"].as_f64().unwrap();
        assert_close_u8(
            &name(&c),
            &resize_area(&gray(&c.input), s),
            &gray(&c.output),
            0.0,
            0,
        );
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn normalise_size_close() {
    let (mut n_diff, mut n_px) = (0usize, 0usize);
    for c in cases("normalise_size") {
        let target = c.params["target"].as_u64().unwrap() as usize;
        let (got, s) = normalise_size(&gray(&c.input), target);
        assert_eq!(s, c.params["scale"].as_f64().unwrap(), "{}", name(&c));
        let want = gray(&c.output);
        // area: exact; cubic (IPP): off by one on a few pixels per million,
        // more for integer factors of square images (IPP's non-affine path)
        let max_mean = if s < 1.0 { 0.0 } else { 0.01 };
        let max_diff = if s < 1.0 { 0 } else { 1 };
        assert_close_u8(&name(&c), &got, &want, max_mean, max_diff);
        n_diff += common::diff_u8(&got.data, &want.data).2;
        n_px += got.data.len();
    }
    eprintln!("normalise_size: {n_diff} of {n_px} px differ");
}
