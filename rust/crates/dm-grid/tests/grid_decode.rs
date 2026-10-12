//! P3.6 / P3.7, P4b.5: every grid job Python's `read_image` ran on the test
//! images (variant, zxing seeds, alt images, qz mode), replayed without a
//! deadline: same reads (data, sampling note, size, mirror) and the same
//! near miss of each failed call. Also prints the time against Python.
//! Reference: `datamatrix_reader_v2.py` (`grid_v2` fixtures, v2 baseline);
//! also `quality` of every v2 read, warped fits included.

mod common;

use common::*;
use dm_grid::*;
use dm_imgproc::Gray8;
use serde_json::Value;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Outcome {
    what: String,
    ok: bool,
    near_equal: bool,
    baseline: bool,
    rust_s: f64,
    py_s: f64,
    errors: Vec<String>,
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn grid_decode_replay() {
    use_dir("grid_v2");
    let all = cases("grid_decode");
    let next = AtomicUsize::new(0);
    let out = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(c) = all.get(i) else { break };
                    let o = check(c);
                    out.lock().unwrap().push(o);
                }
            });
        }
    });
    let out = out.into_inner().unwrap();
    let (mut rust, mut py) = (0.0, 0.0);
    let mut errors = Vec::new();
    for o in &out {
        rust += o.rust_s;
        py += o.py_s;
        for e in &o.errors {
            errors.push(format!("{}: {e}", o.what));
        }
    }
    let reads = out.iter().filter(|o| o.ok).count();
    let near = out.iter().filter(|o| !o.ok && o.near_equal).count();
    let base = out
        .iter()
        .filter(|o| o.baseline && o.errors.is_empty())
        .count();
    let n_base = out.iter().filter(|o| o.baseline).count();
    eprintln!("baseline near misses reproduced: {base}/{n_base}");
    eprintln!(
        "grid_decode: {} calls, {reads} reads, {near}/{} near misses equal; \
         grid time Rust {rust:.1} s (sum over calls) vs Python {py:.1} s ({:.0}x)",
        out.len(),
        out.len() - reads,
        py / rust
    );
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

fn baseline_near_miss(image: &str) -> Value {
    let path = private_dir().join("baseline/python_v2_baseline.json");
    let all: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    all.as_array()
        .unwrap()
        .iter()
        .find(|e| e["file"].as_str().unwrap() == format!("testimages/{image}"))
        .map(|e| e["near_miss"].clone())
        .unwrap()
}

fn near_miss_json(d: &Option<NearMiss>) -> Value {
    match d {
        None => serde_json::json!({}),
        Some(d) => {
            let mut v = serde_json::json!({
                "score": d.score, "size": d.size, "qz": d.qz, "modules_px": d.modules_px,
                "fixed_pattern_module_errors": d.fixed_pattern_module_errors,
                "size_agree": d.size_agree, "outside_dark": d.outside_dark,
                "measured_size": d.measured_size,
            });
            if let Some(p) = d.pitch_drift_modules {
                v["pitch_drift_modules"] = p.into();
            }
            v
        }
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn quality_v2() {
    use_dir("grid_v2");
    let all = cases("quality");
    let mut warped = 0;
    for c in &all {
        warped += c.output["axes"].is_array() as usize;
        assert_eq!(
            quality_json(&quality_of(c)),
            c.output["quality"],
            "{}",
            c.params["image"]
        );
    }
    eprintln!(
        "quality: {} reads ({warped} warped), all fields equal",
        all.len()
    );
}

fn check(c: &Case) -> Outcome {
    let p = &c.params;
    let what = format!("{} [{}] qz={}", p["image"], p["variant"], p["qz"]);
    let gray = gray(&c.input["img"]);
    let alts: Vec<(String, Gray8)> = c.input["alts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| (a[0].as_str().unwrap().to_string(), common::gray(&a[1])))
        .collect();
    let alt_refs: Vec<(String, &Gray8)> = alts.iter().map(|(n, i)| (n.clone(), i)).collect();
    let extra = quads(&c.input["extra"]);
    let mut diag = None;
    let t = Instant::now();
    let r = grid_decode(
        &gray,
        &Deadline::never(),
        &extra,
        &alt_refs,
        p["min_area"].as_f64().unwrap(),
        PY_MAX_ASPECT,
        &[p["qz"].as_bool().unwrap()],
        None,
        Some(&mut diag),
    );
    let rust_s = t.elapsed().as_secs_f64();
    let o = &c.output;
    let mut errors = Vec::new();
    let ok = o["ok"].as_bool().unwrap();
    if r.is_some() != ok {
        errors.push(format!("decoded {} vs Python {ok}", r.is_some()));
    }
    if let Some(r) = &r {
        let hex: String = r.dec.data.iter().map(|b| format!("{b:02x}")).collect();
        let got = (hex, r.note.clone(), r.mirrored, r.fit.geom.name());
        let want = (
            o["data"].as_str().unwrap_or("").to_string(),
            o["note"].as_str().unwrap_or("").to_string(),
            o["mirrored"].as_bool().unwrap_or(false),
            o["size"].as_str().unwrap_or("").to_string(),
        );
        if got != want {
            errors.push(format!("read {got:?} vs Python {want:?}"));
        } else if r.fit.quad != quad(&o["quad"]) {
            errors.push(format!("quad {:?} vs {}", r.fit.quad, o["quad"]));
        }
    }
    let got = near_miss_json(&diag);
    let near_equal = got == o["diag"];
    if !near_equal {
        errors.push(format!("near miss {got} vs Python {}", o["diag"]));
    }
    // P3.7: the job that gave a baseline near miss gives the same one
    if p["baseline_near_miss"].as_bool().unwrap() {
        let base = baseline_near_miss(p["image"].as_str().unwrap());
        for k in ["size", "measured_size"] {
            if got[k] != base[k] {
                errors.push(format!("{k} {} vs baseline {}", got[k], base[k]));
            }
        }
    }
    Outcome {
        what,
        ok,
        near_equal,
        baseline: p["baseline_near_miss"].as_bool().unwrap(),
        rust_s,
        py_s: o["py_s"].as_f64().unwrap(),
        errors,
    }
}
