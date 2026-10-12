//! P2.7: CLAHE (clip 3.0 / 8 tiles and clip 2.0 / 4 tiles).

mod common;

use common::{assert_close_u8, cases, gray, name};
use dm_imgproc::*;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn clahe_matches() {
    for c in cases("clahe") {
        let clip = c.params["clip"].as_f64().unwrap();
        let tiles = c.params["tiles"].as_u64().unwrap() as usize;
        assert_close_u8(
            &name(&c),
            &clahe(&gray(&c.input), clip, tiles),
            &gray(&c.output),
            0.0,
            0,
        );
    }
}
