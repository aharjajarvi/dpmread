//! P3.2: `candidate_quads`, `snap_quad`, `side_combos`, `expand_quad`,
//! `_line_intersect`, `np.percentile` against Python on the grid variants
//! (and two tiles) of 6 test images. Plan: same quad list, corners within
//! 0.5 px.

mod common;

use common::*;
use dm_grid::*;
use dm_imgproc::Image;

/// Largest corner difference of two quad lists of equal length.
fn quad_diff(what: &str, got: &[Quad], want: &[Quad]) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: {got:?} vs {want:?}");
    let mut d = 0f32;
    for (a, b) in got.iter().zip(want) {
        for k in 0..4 {
            for ax in 0..2 {
                d = d.max((a[k][ax] - b[k][ax]).abs());
            }
        }
    }
    d
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn quads_match() {
    let all = cases("quads");
    let stats: Vec<(usize, usize, f32)> = std::thread::scope(|sc| {
        let hs: Vec<_> = all.iter().map(|c| sc.spawn(move || check(c))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (n, exact, worst) = stats
        .iter()
        .fold((0, 0, 0f32), |a, s| (a.0 + s.0, a.1 + s.1, a.2.max(s.2)));
    eprintln!("quads: {n} quad lists, {exact} bit-exact, max corner diff {worst:e} px");
    assert!(worst < 0.5);
}

fn check(c: &Case) -> (usize, usize, f32) {
    let what = format!("{} {}", c.params["image"], c.params["variant"]);
    let g = gray(&c.input["img"]);
    let min_area = c.params["min_area"].as_f64().unwrap();
    let cq = candidate_quads(&g, 10, min_area, PY_MAX_ASPECT);
    let want = quads(&c.output["candidates"]);
    let mut worst = quad_diff(&format!("{what} candidates"), &cq, &want);
    let (mut n, mut exact) = (1, (worst == 0.0) as usize);
    let mut add = |name: &str, got: &[Quad], want: &[Quad]| {
        let d = quad_diff(&format!("{what} {name}"), got, want);
        n += 1;
        exact += (d == 0.0) as usize;
        worst = worst.max(d);
    };
    for (q, s) in want.iter().zip(c.output["snaps"].as_array().unwrap()) {
        let sq = snap_quad(&g, q, 2, false);
        add("snap", &[sq], &[quad(&s["snap"])]);
        let si = snap_quad(&g, &sq, 2, true);
        add("snap_in", &[si], &[quad(&s["snap_in"])]);
        add(
            "rough_in",
            &[snap_quad(&g, q, 2, true)],
            &[quad(&s["rough_in"])],
        );
        // from Python's snapped quads, so differences do not cascade
        let combos = side_combos(&quad(&s["snap"]), &quad(&s["snap_in"]), 3.0);
        add("combos", &combos, &quads(&s["combos"]));
        let oq = quad(&s["oq"]);
        for (name, e) in s["expand"].as_object().unwrap() {
            let (r, cl) = name.split_once('x').unwrap();
            let gm = dm_symbol::geometry(symbol(r.parse().unwrap(), cl.parse().unwrap()));
            add("expand", &[expand_quad(&oq, gm, 1.0)], &[quad(e)]);
        }
    }
    (n, exact, worst)
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn percentile_matches() {
    for p in cases("percentile")[0].output.as_array().unwrap() {
        let a: Vec<u8> = p["a"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let img = Image::from_vec(a.len(), 1, a);
        for (pct, want) in [65.0, 75.0, 85.0].iter().zip(f64s(&p["p"])) {
            assert_eq!(percentile_u8(&img, *pct), want);
        }
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn line_intersect_matches() {
    let mut exact = 0;
    let all = cases("line_intersect")[0]
        .output
        .as_array()
        .unwrap()
        .clone();
    for c in &all {
        let v: Vec<Vec<f64>> = c["in"].as_array().unwrap().iter().map(f64s).collect();
        let p = |i: usize| [v[i][0], v[i][1]];
        let got = line_intersect(p(0), p(1), p(2), p(3));
        let want = f64s(&c["out"]);
        exact += (got[0] == want[0] && got[1] == want[1]) as usize;
        assert!(
            (got[0] - want[0]).abs() <= 1e-9 * want[0].abs().max(1.0),
            "{got:?} {want:?}"
        );
    }
    eprintln!("line_intersect: {exact}/{} bit-exact", all.len());
    assert_eq!(exact, all.len());
}
