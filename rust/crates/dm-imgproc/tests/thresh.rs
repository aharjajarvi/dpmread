//! P2.5: Otsu level equal, adaptive threshold (plan: < 0.1 % of pixels differ).

mod common;

use common::{assert_close_u8, cases, gray, name};
use dm_imgproc::*;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn otsu_equal() {
    for c in cases("otsu") {
        let g = gray(&c.input);
        assert_eq!(
            otsu_level(&g) as f64,
            c.params["level"].as_f64().unwrap(),
            "{}",
            name(&c)
        );
        assert_close_u8(&name(&c), &otsu(&g), &gray(&c.output), 0.0, 0);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn adaptive_close() {
    for c in cases("adaptive") {
        let g = gray(&c.input);
        let block = c.params["block"].as_f64().unwrap();
        assert_eq!(odd(block) as u64, c.params["k"].as_u64().unwrap());
        assert_close_u8(&name(&c), &adaptive(&g, block), &gray(&c.output), 0.0, 0);
    }
    for c in cases("box_mean") {
        let k = c.params["k"].as_u64().unwrap() as usize;
        assert_close_u8(
            &name(&c),
            &box_mean_u8(&gray(&c.input), k),
            &gray(&c.output),
            0.0,
            0,
        );
    }
}
