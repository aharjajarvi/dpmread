//! P2.1: `load_image` gives the same pixels as `cv2.imdecode(IMREAD_COLOR)`
//! for every test image and for synthetic EXIF / alpha / 16-bit / gray /
//! palette files written by the fixture script. Alpha is composited over
//! white (round 4), so the alpha file is checked against OpenCV's colour
//! composited with the file's alpha.

mod common;

use dm_imgproc::load_image;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn load_matches_opencv() {
    let mut bad = Vec::new();
    for c in common::cases("load") {
        let path = c.input["path"].as_str().unwrap();
        let got = load_image(&common::data(path)).unwrap();
        let mut want = common::bgr(&c.output);
        if path.ends_with("rgba.png") {
            let a = image::open(common::data(path)).unwrap().to_rgba8();
            for (px, p) in want.data.chunks_mut(3).zip(a.pixels()) {
                let a = p[3] as u32;
                for v in px {
                    *v = ((*v as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
                }
            }
        }
        let (mean, max, n) = if (got.width, got.height) == (want.width, want.height) {
            common::diff_u8(&got.data, &want.data)
        } else {
            (f64::INFINITY, 255, 0)
        };
        eprintln!(
            "{path}: {}x{} mean {mean:.4} max {max} n {n}",
            got.width, got.height
        );
        if max != 0 {
            bad.push(path.to_string());
        }
    }
    assert!(bad.is_empty(), "differ: {bad:?}");
}
