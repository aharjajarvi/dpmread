//! P3.3: `refine_quad` (coarse, fine and default schedules) and
//! `_track_axis` against Python on the best scored candidates of
//! `_grid_pass` (6 images, qz on/off). Plan: refined quads within 0.05
//! module.

mod common;

use common::*;
use dm_grid::*;

fn quad_diff(a: &Quad, b: &Quad) -> f32 {
    let mut d = 0f32;
    for k in 0..4 {
        for ax in 0..2 {
            d = d.max((a[k][ax] - b[k][ax]).abs());
        }
    }
    d
}

fn axis_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn refine_and_track_match() {
    let all = cases("refine");
    let res: Vec<(f32, usize, f64, usize)> = std::thread::scope(|sc| {
        let hs: Vec<_> = all.iter().map(|c| sc.spawn(move || check(c))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let worst_q = res.iter().map(|r| r.0).fold(0f32, f32::max);
    let exact_q: usize = res.iter().map(|r| r.1).sum();
    let worst_a = res.iter().map(|r| r.2).fold(0f64, f64::max);
    let exact_a: usize = res.iter().map(|r| r.3).sum();
    eprintln!(
        "refine: {exact_q}/{} quads bit-exact, max diff {worst_q:e} module; \
         track_axis: {exact_a}/{} bit-exact, max diff {worst_a:e} module",
        3 * all.len(),
        4 * all.len()
    );
    assert!(worst_q < 0.05 && worst_a < 0.05);
}

fn check(c: &Case) -> (f32, usize, f64, usize) {
    let g = geom(&c.params);
    let qz = c.params["qz"].as_bool().unwrap();
    let smp = Sampler::new(&gray(&c.input["img"]), 0.22);
    let oq = quad(&c.input["quad"]);
    let o = &c.output;
    let mut worst = 0f32;
    let mut exact = 0;
    let mut cmp = |got: (Quad, f64), q: &str, s: &str| {
        let want = quad(&o[q]);
        let d = quad_diff(&got.0, &want) / module_px(&want, g);
        exact += (d == 0.0 && got.1 == o[s].as_f64().unwrap()) as usize;
        worst = worst.max(d);
    };
    let coarse = refine_quad(&smp, &oq, g, 2, &[1.0, 0.5], qz);
    cmp(coarse, "coarse", "coarse_s");
    // the fine pass starts from Python's coarse quad (no cascading)
    let fine = refine_quad(&smp, &quad(&o["coarse"]), g, 4, &[0.5, 0.25, 0.12], qz);
    cmp(fine, "fine", "fine_s");
    cmp(
        refine_quad(&smp, &oq, g, 4, &REFINE_STEPS, qz),
        "full",
        "full_s",
    );
    let (mut wa, mut ea) = (0f64, 0);
    for (q, ku, kv) in [(quad(&o["fine"]), "u", "v"), (oq, "u0", "v0")] {
        let (u, v) = tracked_axes(&smp, &q, g);
        for (got, k) in [(u, ku), (v, kv)] {
            let d = axis_diff(&got, &f64s(&o[k]));
            ea += (d == 0.0) as usize;
            wa = wa.max(d);
        }
    }
    (worst, exact, wa, ea)
}
