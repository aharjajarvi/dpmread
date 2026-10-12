//! P2.4: ellipse structuring element and grey morphology, exact.

mod common;

use common::{assert_close_u8, cases, gray, name};
use dm_imgproc::*;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn ellipse_masks_exact() {
    let c = &cases("ellipse")[0];
    for (k, want) in c.output.as_object().unwrap() {
        let want: Vec<u8> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(ellipse_mask(k.parse().unwrap()), want, "k = {k}");
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn morph_exact() {
    for c in cases("morph") {
        let k = c.params["k"].as_u64().unwrap() as usize;
        let src = gray(&c.input);
        let got = match c.params["op"].as_str().unwrap() {
            "erode" => erode(&src, k),
            "dilate" => dilate(&src, k),
            "close" => close(&src, k),
            "open" => open(&src, k),
            _ => unreachable!(),
        };
        assert_close_u8(&name(&c), &got, &gray(&c.output), 0.0, 0);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn wrappers_exact() {
    for c in cases("merge_dots") {
        let k = c.params["k"].as_u64().unwrap() as usize;
        assert_close_u8(
            &name(&c),
            &merge_dots(&gray(&c.input), k),
            &gray(&c.output),
            0.0,
            0,
        );
    }
    for c in cases("remove_texture") {
        assert_close_u8(
            &name(&c),
            &remove_texture(&gray(&c.input), 3),
            &gray(&c.output),
            0.0,
            0,
        );
    }
}
