//! P2.9: getPerspectiveTransform, homographies, perspectiveTransform,
//! invert, remap (bilinear, replicate) and warpPerspective.

mod common;

use common::{assert_close_f32, cases, f32img, name};
use dm_imgproc::*;
use serde_json::Value;

fn v64(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn pts32(v: &Value) -> Vec<(f32, f32)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32))
        .collect()
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn homographies() {
    let c = &cases("homography")[0];
    let (mut worst_h, mut worst_h64, mut worst_inv, mut worst_pt) = (0f64, 0f64, 0f64, 0f32);
    let mut exact_h = 0;
    let all = c.output.as_array().unwrap();
    for k in all {
        let src: [(f32, f32); 4] = pts32(&k["src"]).try_into().unwrap();
        let quad: [(f32, f32); 4] = pts32(&k["quad"]).try_into().unwrap();
        let h = get_perspective_transform(&src, &quad).expect("solve");
        let want = v64(&k["h"]);
        worst_h = worst_h.max(max_rel(&h, &want));
        exact_h += (h.as_slice() == want.as_slice()) as usize;
        let s64 = src.map(|(x, y)| (x as f64, y as f64));
        let q64 = quad.map(|(x, y)| (x as f64, y as f64));
        worst_h64 = worst_h64.max(max_rel(&homography(&s64, &q64).unwrap(), &v64(&k["h64"])));
        let wh: Mat3 = want.clone().try_into().unwrap();
        worst_inv = worst_inv.max(max_rel(&invert3(&wh).unwrap(), &v64(&k["inv"])));
        let pt = perspective_transform(&pts32(&k["pts"]), &wh);
        for (a, b) in pt.iter().zip(pts32(&k["pt"])) {
            worst_pt = worst_pt.max((a.0 - b.0).abs().max((a.1 - b.1).abs()));
        }
    }
    eprintln!(
        "getPerspectiveTransform: {exact_h}/{} exact, max rel {worst_h:e}; homographies max rel {worst_h64:e}; \
         invert max rel {worst_inv:e}; perspectiveTransform max abs {worst_pt:e}",
        all.len()
    );
    assert!(worst_h < 1e-9 && worst_h64 < 1e-9 && worst_inv == 0.0 && worst_pt == 0.0);
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn remap() {
    for c in cases("remap") {
        let i = c.input.as_array().unwrap();
        let img = f32img(&i[0]);
        let (xs, ys) = (f32img(&i[1]), f32img(&i[2]));
        let got = remap_points(&img, &xs.data, &ys.data);
        let want = f32img(&c.output);
        let got = Image::from_vec(got.len(), 1, got);
        assert_close_f32(&name(&c), &got, &want, 0.0);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn warp() {
    for c in cases("warp") {
        let m: Mat3 = v64(&c.params["m"]).try_into().unwrap();
        let size = c.params["size"].as_array().unwrap();
        let (w, h) = (
            size[0].as_u64().unwrap() as usize,
            size[1].as_u64().unwrap() as usize,
        );
        let got = warp_perspective(&f32img(&c.input), &m, w, h);
        assert_close_f32(&name(&c), &got, &f32img(&c.output), 0.0);
    }
}
