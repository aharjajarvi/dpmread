//! P3.8: `quality` / `grade` on the fit of every read of the test images
//! (grid reads and zxing reads with `analyse_known` fits), measured on the
//! original gray image: all fields equal to Python's dict.

mod common;

use common::*;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn quality_matches() {
    let all = cases("quality");
    for c in &all {
        let o = &c.output;
        let q = quality_of(c);
        // v1 has no pitch drift key
        let mut got = quality_json(&q);
        got.as_object_mut().unwrap().remove("pitch_drift_modules");
        assert_eq!(got, o["quality"], "{}", c.params["image"]);
    }
    eprintln!("quality: {} reads, all fields equal", all.len());
}

#[test]
fn round_like_python() {
    for (x, d, want) in [
        (0.25, 1, 0.2),
        (0.35, 1, 0.3),
        (2.675, 2, 2.67),
        (0.125, 2, 0.12),
        (1.5, 0, 2.0),
    ] {
        assert_eq!(dm_grid::np::round_to(x, d), want, "{x} {d}");
    }
}
