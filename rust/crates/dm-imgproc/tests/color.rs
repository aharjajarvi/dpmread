//! P2.2: colour conversions and `channels()` against OpenCV, including every
//! one of the 2^24 colours.

mod common;

use dm_imgproc::{Bgr8, bgr_to_gray, bgr_to_hsv_s, bgr_to_lab_l, channels};

fn all_colours() -> Bgr8 {
    let data = (0u32..1 << 24)
        .flat_map(|i| [(i & 255) as u8, ((i >> 8) & 255) as u8, (i >> 16) as u8])
        .collect();
    Bgr8::from_vec(4096, 4096, data)
}

fn input(c: &common::Case) -> Bgr8 {
    if c.input.get("synthetic").is_some() {
        all_colours()
    } else {
        common::bgr(&c.input)
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn cvt_color() {
    for c in common::cases("cvt") {
        let img = input(&c);
        let code = c.params["code"].as_str().unwrap();
        let got = match code {
            "gray" => bgr_to_gray(&img),
            "hsv_s" => bgr_to_hsv_s(&img),
            "lab_l" => bgr_to_lab_l(&img),
            _ => unreachable!(),
        };
        let want = common::gray(&c.output);
        common::assert_close_u8(&common::name(&c), &got, &want, 0.0, 0); // exact (plan: max diff <= 1)
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn channels_match() {
    for c in common::cases("channels") {
        let img = input(&c);
        let ch = channels(&img, c.params["invert"].as_bool().unwrap());
        let names: Vec<&str> = c.params["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            ch.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            names
        );
        for ((n, got), want) in ch.iter().zip(c.output.as_array().unwrap()) {
            common::assert_close_u8(n, got, &common::gray(want), 0.0, 0);
        }
    }
}
