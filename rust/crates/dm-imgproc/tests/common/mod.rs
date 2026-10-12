//! Fixture access for the image tests (`fixtures/img` in the private data, written by
//! `python rust/tools/dump_fixtures_img.py`).
#![allow(dead_code)]

use dm_imgproc::{Bgr8, Gray8, GrayF32, Image, decode_image};
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Deserialize, Clone, Debug)]
pub struct Arr {
    pub file: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Case {
    pub op: String,
    pub params: Value,
    #[serde(rename = "in")]
    pub input: Value,
    #[serde(rename = "out")]
    pub output: Value,
}

/// Private test data (fixtures, test images, baselines): `DM_PRIVATE_DIR`
/// (not part of this repo). Tests that need it are
/// `#[ignore]`d; `rust/tools/ci.sh` runs them when the data is there.
pub fn private_dir() -> PathBuf {
    std::env::var_os("DM_PRIVATE_DIR")
        .map(PathBuf::from)
        .expect("DM_PRIVATE_DIR not set: it points at the private test data")
}

/// A path from a fixture manifest (`rust/fixtures/...` as written by the
/// dump scripts) inside the private data.
pub fn data(path: &str) -> PathBuf {
    private_dir().join(path.strip_prefix("rust/").unwrap_or(path))
}

pub fn dir() -> PathBuf {
    private_dir().join("fixtures/img")
}

pub fn cases(op: &str) -> Vec<Case> {
    static ALL: OnceLock<Vec<Case>> = OnceLock::new();
    let all = ALL.get_or_init(|| {
        let path = dir().join("manifest.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}; run `python rust/tools/dump_fixtures_img.py`",
                path.display()
            )
        });
        #[derive(Deserialize)]
        struct Manifest {
            cases: Vec<Case>,
        }
        serde_json::from_str::<Manifest>(&text).unwrap().cases
    });
    let v: Vec<Case> = all.iter().filter(|c| c.op == op).cloned().collect();
    assert!(!v.is_empty(), "no fixture cases for {op}");
    v
}

pub fn arr(v: &Value) -> Arr {
    serde_json::from_value(v.clone()).unwrap()
}

pub fn bgr(v: &Value) -> Bgr8 {
    let a = arr(v);
    assert_eq!(a.shape.len(), 3, "{}", a.file);
    let img = decode_image(&std::fs::read(dir().join(&a.file)).unwrap()).unwrap();
    assert_eq!((img.height, img.width), (a.shape[0], a.shape[1]));
    img
}

pub fn gray(v: &Value) -> Gray8 {
    let a = arr(v);
    assert_eq!(a.shape.len(), 2, "{}", a.file);
    let img = decode_image(&std::fs::read(dir().join(&a.file)).unwrap()).unwrap();
    assert_eq!((img.height, img.width), (a.shape[0], a.shape[1]));
    img.channel(0)
}

pub fn f32img(v: &Value) -> GrayF32 {
    let a = arr(v);
    assert_eq!(a.dtype, "float32");
    let bytes = std::fs::read(dir().join(&a.file)).unwrap();
    let data: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let (h, w) = if a.shape.len() == 1 {
        (1, a.shape[0])
    } else {
        (a.shape[0], a.shape[1])
    };
    Image::from_vec(w, h, data)
}

/// Mean and max absolute difference of two equally sized u8 buffers.
pub fn diff_u8(a: &[u8], b: &[u8]) -> (f64, u8, usize) {
    assert_eq!(a.len(), b.len(), "size");
    let mut sum = 0u64;
    let mut max = 0u8;
    let mut n = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        let d = x.abs_diff(y);
        sum += d as u64;
        max = max.max(d);
        n += (d != 0) as usize;
    }
    (sum as f64 / a.len().max(1) as f64, max, n)
}

/// Plan tolerance: mean abs diff < 0.5, max < 3 grey levels.
pub fn assert_close_u8(what: &str, got: &Gray8, want: &Gray8, mean_tol: f64, max_tol: u8) {
    assert_eq!(
        (got.width, got.height),
        (want.width, want.height),
        "{what}: size"
    );
    let (mean, max, n) = diff_u8(&got.data, &want.data);
    eprintln!(
        "{what}: mean {mean:.4} max {max} differing {n}/{}",
        got.data.len()
    );
    assert!(
        mean <= mean_tol && max <= max_tol,
        "{what}: mean {mean} max {max} ({n} px differ)"
    );
}

pub fn assert_close_f32(what: &str, got: &GrayF32, want: &GrayF32, tol: f32) {
    assert_eq!(
        (got.width, got.height),
        (want.width, want.height),
        "{what}: size"
    );
    let mut max = 0f32;
    for (&x, &y) in got.data.iter().zip(&want.data) {
        let d = if x.is_nan() && y.is_nan() {
            0.0
        } else {
            (x - y).abs()
        };
        max = max.max(if d.is_nan() { f32::INFINITY } else { d });
    }
    let bad: Vec<(usize, usize)> = got
        .data
        .iter()
        .zip(&want.data)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| (i % got.width, i / got.width))
        .collect();
    let xs: std::collections::BTreeSet<usize> = bad.iter().map(|p| p.0).collect();
    eprintln!(
        "{what}: max abs diff {max:e}, {} differ, columns {:?}",
        bad.len(),
        xs.iter().take(12).collect::<Vec<_>>()
    );
    assert!(max <= tol, "{what}: max abs diff {max}");
}

pub fn name(c: &Case) -> String {
    format!("{} {} {}", c.op, c.params, c.input)
}
