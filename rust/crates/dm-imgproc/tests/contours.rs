//! P2.8: findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE), contourArea,
//! convexHull, minAreaRect, boxPoints on the `candidate_quads` blob images.

mod common;

use common::{cases, gray, name};
use dm_imgproc::contours::to_f32;
use dm_imgproc::*;
use serde_json::Value;

fn pts(v: &Value) -> Vec<(i32, i32)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32))
        .collect()
}

fn f(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn contours_match() {
    let (mut n_cnt, mut n_box_exact, mut max_box) = (0, 0, 0f32);
    for c in cases("contours") {
        let bw = gray(&c.input);
        let got = find_contours_external(&bw);
        let want = c.output.as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{}: contour count", name(&c));
        for (g, wv) in got.iter().zip(want) {
            assert_eq!(g, &pts(&wv["pts"]), "{}: contour points", name(&c));
            assert_eq!(contour_area(&to_f32(g)), wv["area"].as_f64().unwrap());
            assert_eq!(convex_hull(g), pts(&wv["hull"]), "{}: hull", name(&c));
            let r = min_area_rect(g);
            let wr = &wv["rect"];
            let want_rect = RotatedRect {
                center: (f(&wr[0][0]), f(&wr[0][1])),
                size: (f(&wr[1][0]), f(&wr[1][1])),
                angle: f(&wr[2]),
            };
            let bx = box_points(&r);
            let wb: Vec<(f32, f32)> = wv["box"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (f(&p[0]), f(&p[1])))
                .collect();
            let d = bx
                .iter()
                .zip(&wb)
                .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
                .fold(0f32, f32::max);
            // plan: corners within 0.5 px; achieved: bit-exact
            assert!(
                r == want_rect && d == 0.0,
                "{}: box {bx:?} vs {wb:?} (rect {r:?} vs {want_rect:?})",
                name(&c)
            );
            max_box = max_box.max(d);
            n_box_exact += (r == want_rect && d == 0.0) as usize;
            let ba = contour_area(&bx);
            assert!((ba - wv["box_area"].as_f64().unwrap()).abs() < 1e-3 * ba.max(1.0));
            n_cnt += 1;
        }
    }
    eprintln!(
        "{n_cnt} contours, {n_box_exact} rects/boxes bit-exact, max box corner diff {max_box}"
    );
}
