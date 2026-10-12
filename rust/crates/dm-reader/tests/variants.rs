//! P2.10: variant ladders against Python (`enhancement_variants`,
//! `grid_variants`, `tile_variants`, `map_quad`, `_tile_axis`) for 4 images
//! x 3 polarities: same names in the same order, same scales and offsets,
//! pixels within tolerance (fixtures from `rust/tools/dump_fixtures_img.py`,
//! in the private data).

use dm_imgproc::{Gray8, decode_image, load_image};
use dm_reader::*;
use serde_json::Value;
use std::path::PathBuf;

/// Private test data (fixtures, test images, baselines): `DM_PRIVATE_DIR`
/// (not part of this repo). Tests that need it are
/// `#[ignore]`d; `rust/tools/ci.sh` runs them when the data is there.
fn private_dir() -> PathBuf {
    std::env::var_os("DM_PRIVATE_DIR")
        .map(PathBuf::from)
        .expect("DM_PRIVATE_DIR not set: it points at the private test data")
}

fn cases(op: &str) -> Vec<Value> {
    let path = private_dir().join("fixtures/img/manifest.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; run `python rust/tools/dump_fixtures_img.py`",
            path.display()
        )
    });
    let m: Value = serde_json::from_str(&text).unwrap();
    m["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["op"] == op)
        .cloned()
        .collect()
}

fn png(v: &Value) -> Gray8 {
    let file = private_dir()
        .join("fixtures/img")
        .join(v["file"].as_str().unwrap());
    decode_image(&std::fs::read(file).unwrap())
        .unwrap()
        .channel(0)
}

/// Per case: (variants compared, exact, worst mean, differing px, all px).
type Stats = (usize, usize, f64, usize, usize);

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn variant_ladders() {
    let all = cases("variants");
    let stats: Vec<Stats> = std::thread::scope(|sc| {
        let handles: Vec<_> = all
            .iter()
            .map(|c| sc.spawn(move || check_case(c)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut n_var, mut n_exact, mut worst_mean, mut px_diff, mut px_all) =
        (0, 0, 0f64, 0usize, 0usize);
    for (a, b, c, d, e) in stats {
        n_var += a;
        n_exact += b;
        worst_mean = worst_mean.max(c);
        px_diff += d;
        px_all += e;
    }
    eprintln!(
        "{n_var} variants compared, {n_exact} exact, worst mean {worst_mean:.5}, {px_diff} of {px_all} px differ"
    );
}

fn check_case(c: &Value) -> Stats {
    let (mut n_var, mut n_exact, mut worst_mean, mut px_diff, mut px_all) =
        (0, 0, 0f64, 0usize, 0usize);
    {
        let path = c["in"]["path"].as_str().unwrap();
        let bgr = load_image(&private_dir().join(path)).unwrap();
        let kind = c["params"]["kind"].as_str().unwrap();
        let (invert, extra) = match c["params"]["polarity"].as_str().unwrap() {
            "any" => (false, true),
            "dark" => (false, false),
            _ => (true, false),
        };
        let got: Vec<Variant> = match kind {
            "enhancement" => enhancement_variants(&bgr, invert, extra),
            "grid" => grid_variants(&bgr, invert, extra),
            _ => tile_variants(&bgr, invert).into_iter().flatten().collect(),
        };
        let want = c["out"].as_array().unwrap();
        let names: Vec<&str> = got.iter().map(|v| v.name.as_str()).collect();
        let want_names: Vec<&str> = want.iter().map(|v| v["name"].as_str().unwrap()).collect();
        assert_eq!(names, want_names, "{path} {kind}");
        for (g, w) in got.iter().zip(want) {
            let what = format!("{path} {}", g.name);
            assert_eq!(g.scale, w["scale"].as_f64().unwrap(), "{what}: scale");
            assert_eq!(
                (g.ox, g.oy),
                (w["ox"].as_f64().unwrap(), w["oy"].as_f64().unwrap()),
                "{what}: offset"
            );
            let shape: Vec<usize> = w["img"]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(
                (g.img.height, g.img.width),
                (shape[0], shape[1]),
                "{what}: size"
            );
            if !c["params"]["pixels"].as_bool().unwrap() {
                continue;
            }
            let wi = png(&w["img"]);
            let mut sum = 0u64;
            let mut n = 0usize;
            let mut max = 0u8;
            for (&a, &b) in g.img.data.iter().zip(&wi.data) {
                let d = a.abs_diff(b);
                sum += d as u64;
                n += (d != 0) as usize;
                max = max.max(d);
            }
            let mean = sum as f64 / wi.data.len() as f64;
            n_var += 1;
            n_exact += (n == 0) as usize;
            px_diff += n;
            px_all += wi.data.len();
            worst_mean = worst_mean.max(mean);
            if n != 0 {
                eprintln!(
                    "{what}: mean {mean:.5} max {max} differing {n}/{}",
                    wi.data.len()
                );
            }
            // Only upscaled (INTER_CUBIC / IPP) images differ: a grey level
            // off by one can move a CLAHE tile LUT (a few hundred pixels off
            // by up to ~7) or flip a binarised pixel (255); area-downscaled
            // photos are exact. Plan: mean < 0.5.
            assert!(mean < 0.05, "{what}: mean {mean}");
            assert!(
                (n as f64) < 0.01 * wi.data.len() as f64,
                "{what}: {n} px differ"
            );
            if g.scale < 1.0 {
                assert_eq!(n, 0, "{what}: downscaled variants are exact");
            }
        }
    }
    (n_var, n_exact, worst_mean, px_diff, px_all)
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn map_quad_exact() {
    for k in cases("map_quad")[0]["out"].as_array().unwrap() {
        let quad = |v: &Value| -> Quad {
            let p: Vec<(f32, f32)> = v
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32))
                .collect();
            p.try_into().unwrap()
        };
        let var = |v: &Value| {
            let f = |i: usize| v[i].as_f64().unwrap();
            Variant {
                name: String::new(),
                img: Gray8::new(1, 1),
                scale: f(0),
                ox: f(1),
                oy: f(2),
            }
        };
        assert_eq!(
            map_quad(&quad(&k["q"]), &var(&k["src"]), &var(&k["dst"])),
            quad(&k["out"])
        );
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn tile_axis_exact() {
    for (n, v) in cases("tile_axis")[0]["out"].as_object().unwrap() {
        let want = (
            v[0].as_u64().unwrap() as usize,
            v[1].as_u64().unwrap() as usize,
        );
        assert_eq!(tile_axis(n.parse().unwrap()), want, "length {n}");
    }
}

/// A test image by role from `testimages/picks.json` in the private data
/// (the names stay out of this repo).
fn pick(key: &str) -> serde_json::Value {
    let path = private_dir().join("testimages/picks.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    v[key].clone()
}

/// P4b.8: the variant builders run their chains as rayon tasks; on 8
/// workers they must give exactly what one worker gives (names, scales,
/// offsets, pixels, order).
#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn parallel_variants_exact() {
    let seq = Runner::new(1);
    let par = Runner::new(8);
    let images = pick("parallel_variants");
    for rel in images.as_array().unwrap() {
        let path = rel.as_str().unwrap();
        let bgr = load_image(&private_dir().join("testimages").join(path)).unwrap();
        for (invert, extra) in [(false, true), (true, false)] {
            let build = |r: &Runner| {
                r.install(|| {
                    let mut all = enhancement_variants(&bgr, invert, extra);
                    all.extend(tile_variants(&bgr, invert).into_iter().flatten());
                    all.extend(grid_variants(&bgr, invert, extra));
                    all
                })
            };
            let (a, b) = (build(&seq), build(&par));
            assert_eq!(a.len(), b.len(), "{path}");
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.name, y.name, "{path}");
                assert_eq!(
                    (x.scale, x.ox, x.oy),
                    (y.scale, y.ox, y.oy),
                    "{path} {}",
                    x.name
                );
                assert_eq!(
                    (x.img.width, x.img.height),
                    (y.img.width, y.img.height),
                    "{path} {}",
                    x.name
                );
                assert!(x.img.data == y.img.data, "{path} {}: pixels differ", x.name);
            }
        }
    }
}
