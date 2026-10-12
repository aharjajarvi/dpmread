//! Single-threaded timing of `grid_decode` on recorded grid jobs
//! (`fixtures/grid` in the private data): `cargo run --release -p dm-grid --example
//! grid_bench [FILTER]` runs every job whose "image [variant] qz" contains
//! FILTER. The Python column is the time recorded by the fixture script
//! (12 processes in parallel, so inflated); for a fair reference run
//! `python rust/tools/grid_bench_ref.py FILTER`.

use dm_grid::{Deadline, PY_MAX_ASPECT, grid_decode};
use dm_imgproc::{Gray8, decode_image};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

/// Private test data, not part of this repo: `DM_PRIVATE_DIR`.
fn private_dir() -> PathBuf {
    std::env::var_os("DM_PRIVATE_DIR")
        .map(PathBuf::from)
        .expect("DM_PRIVATE_DIR not set: it points at the private test data")
}

fn main() {
    let dir = private_dir().join("fixtures/grid");
    let m: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let filter = std::env::args().nth(1).unwrap_or_default();
    let load = |v: &Value| -> Gray8 {
        decode_image(&std::fs::read(dir.join(v["file"].as_str().unwrap())).unwrap())
            .unwrap()
            .channel(0)
    };
    let (mut rt, mut pt) = (0.0, 0.0);
    for c in m["cases"].as_array().unwrap() {
        if c["op"] != "grid_decode" {
            continue;
        }
        let p = &c["params"];
        let what = format!(
            "{} [{}] qz={}",
            p["image"].as_str().unwrap(),
            p["variant"].as_str().unwrap(),
            p["qz"]
        );
        if !what.contains(&filter) {
            continue;
        }
        let gray = load(&c["in"]["img"]);
        let alts: Vec<(String, Gray8)> = c["in"]["alts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| (a[0].as_str().unwrap().to_string(), load(&a[1])))
            .collect();
        let alt_refs: Vec<(String, &Gray8)> = alts.iter().map(|(n, i)| (n.clone(), i)).collect();
        let extra: Vec<[[f32; 2]; 4]> = c["in"]["extra"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                std::array::from_fn(|k| {
                    [
                        q[k][0].as_f64().unwrap() as f32,
                        q[k][1].as_f64().unwrap() as f32,
                    ]
                })
            })
            .collect();
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
        let s = t.elapsed().as_secs_f64();
        let py = c["out"]["py_s"].as_f64().unwrap();
        rt += s;
        pt += py;
        println!(
            "{s:7.3} s  python {py:7.3} s  {:5.1}x  {}  {what}",
            py / s,
            if r.is_some() { "read" } else { "-" }
        );
    }
    println!("total {rt:.2} s  (recorded python {pt:.2} s)");
}
