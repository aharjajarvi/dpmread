//! P3.1: `Sampler`, `score_quads`, `orientations` against Python on the
//! `gray > flatten` grid variants of 6 test images.

mod common;

use common::*;
use dm_grid::*;
use std::collections::HashMap;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn orientations_match() {
    for c in cases("orientations") {
        let got = orientations(&quad(&c.input["quad"]));
        let want = c.output.as_array().unwrap();
        assert_eq!(got.len(), want.len());
        for ((q, r, m), w) in got.iter().zip(want) {
            assert_eq!(*q, quad(&w["quad"]), "{}", c.input);
            assert_eq!(*r as u64, w["rot"].as_u64().unwrap());
            assert_eq!(*m, w["mirror"].as_bool().unwrap());
        }
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn score_quads_match() {
    let mut images = HashMap::new();
    let (mut n, mut exact, mut worst) = (0usize, 0usize, 0f32);
    for c in cases("score_quads") {
        let file = c.input["img"]["file"].as_str().unwrap().to_string();
        let smp = images
            .entry(file)
            .or_insert_with(|| Sampler::new(&gray(&c.input["img"]), 0.22));
        let qs = quads(&c.input["quads"]);
        for s in c.output.as_array().unwrap() {
            let got = score_quads(smp, &qs, geom(s), s["qz"].as_bool().unwrap());
            let want = f32_or_ninf(&s["s"]);
            for (a, b) in got.iter().zip(&want) {
                n += 1;
                exact += (a == b) as usize;
            }
            let d = max_diff(&got, &want);
            assert!(
                d < 1e-3,
                "{} {}x{}: {got:?} vs {want:?}",
                c.input["img"],
                s["rows"],
                s["cols"]
            );
            worst = worst.max(d);
        }
    }
    eprintln!("score_quads: {exact}/{n} bit-exact, max abs diff {worst:e}");
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn samplers_match() {
    let (mut n, mut worst) = (0usize, 0f32);
    for c in cases("sample") {
        let p = &c.params;
        let (rows, cols) = (
            p["rows"].as_u64().unwrap() as usize,
            p["cols"].as_u64().unwrap() as usize,
        );
        let smp = Sampler::new(&gray(&c.input["img"]), p["radius"].as_f64().unwrap());
        let q = quad(&c.input["quad"]);
        let got = smp.sample(&q, rows, cols, 0);
        let d1 = max_diff(&got.data, &f32s(&c.output["sample"]));
        let got = smp.sample_uv(&q, rows, cols, &uv(&c.input["uv"]));
        let d2 = max_diff(&got, &f32s(&c.output["sample_uv"]));
        let got = smp.sample_grid(
            &q,
            rows,
            cols,
            &f64s(&c.input["ucol"]),
            &f64s(&c.input["vrow"]),
        );
        let d3 = max_diff(&got.data, &f32s(&c.output["sample_grid"]));
        assert!(
            d1 == 0.0 && d2 == 0.0 && d3 == 0.0,
            "{rows}x{cols} r={}: {d1} {d2} {d3}",
            p["radius"]
        );
        n += 1;
        worst = worst.max(d1.max(d2).max(d3));
    }
    eprintln!("sample / sample_uv / sample_grid: {n} cases bit-exact");
}
