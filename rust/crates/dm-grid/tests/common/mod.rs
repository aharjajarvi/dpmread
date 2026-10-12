//! Fixture access for the grid tests (`fixtures/grid` in the private data, written by
//! `python rust/tools/dump_fixtures_grid.py`).
#![allow(dead_code)]

use dm_ecc::RSResult;
use dm_grid::{GridFit, Quad, Quality, quality};
use dm_imgproc::{Gray8, GrayF32, Image, decode_image};
use dm_symbol::{Geometry, Symbol, all_symbols, geometry};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::OnceLock;

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

static SUB: OnceLock<&'static str> = OnceLock::new();

/// Read fixtures from `fixtures/<sub>` instead of `grid` (call before
/// any other access; `grid_v2` is written by `dump_fixtures_grid.py --v2`).
pub fn use_dir(sub: &'static str) {
    let got = *SUB.get_or_init(|| sub);
    assert_eq!(got, sub, "fixture dir already chosen");
}

pub fn dir() -> PathBuf {
    private_dir()
        .join("fixtures")
        .join(SUB.get_or_init(|| "grid"))
}

pub fn cases(op: &str) -> Vec<Case> {
    static ALL: OnceLock<Vec<Case>> = OnceLock::new();
    let all = ALL.get_or_init(|| {
        let path = dir().join("manifest.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}; run `python rust/tools/dump_fixtures_grid.py` (--v2 for grid_v2)",
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

pub fn gray(v: &Value) -> Gray8 {
    let file = dir().join(v["file"].as_str().unwrap());
    decode_image(&std::fs::read(file).unwrap())
        .unwrap()
        .channel(0)
}

/// Raw little-endian float32 array (`.f32`), any shape, flat.
pub fn f32s(v: &Value) -> Vec<f32> {
    let bytes = std::fs::read(dir().join(v["file"].as_str().unwrap())).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

pub fn f32img(v: &Value) -> GrayF32 {
    let shape = v["shape"].as_array().unwrap();
    let (h, w) = (
        shape[0].as_u64().unwrap() as usize,
        shape[1].as_u64().unwrap() as usize,
    );
    Image::from_vec(w, h, f32s(v))
}

pub fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// JSON f32 list with `null` for -inf / NaN.
pub fn f32_or_ninf(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().map_or(f32::NEG_INFINITY, |f| f as f32))
        .collect()
}

pub fn quad(v: &Value) -> Quad {
    let a = v.as_array().unwrap();
    std::array::from_fn(|k| {
        [
            a[k][0].as_f64().unwrap() as f32,
            a[k][1].as_f64().unwrap() as f32,
        ]
    })
}

pub fn quads(v: &Value) -> Vec<Quad> {
    v.as_array().unwrap().iter().map(quad).collect()
}

pub fn uv(v: &Value) -> Vec<[f64; 2]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
        .collect()
}

pub fn symbol(rows: usize, cols: usize) -> &'static Symbol {
    all_symbols()
        .find(|s| s.rows == rows && s.cols == cols)
        .unwrap_or_else(|| panic!("no symbol {rows}x{cols}"))
}

pub fn geom(p: &Value) -> &'static Geometry {
    geometry(symbol(
        p["rows"].as_u64().unwrap() as usize,
        p["cols"].as_u64().unwrap() as usize,
    ))
}

/// Largest absolute difference; -inf == -inf, NaN == NaN.
pub fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            if x == y || (x.is_nan() && y.is_nan()) {
                0.0
            } else {
                let d = (x - y).abs();
                if d.is_nan() { f32::INFINITY } else { d }
            }
        })
        .fold(0.0, f32::max)
}

/// Python's `quality` dict of `q` (v2 key order).
pub fn quality_json(q: &Quality) -> Value {
    json!({
        "symbol_contrast_pct": q.symbol_contrast_pct,
        "symbol_contrast_grade": q.symbol_contrast_grade.to_string(),
        "modulation_p5": q.modulation_p5,
        "modulation_grade": q.modulation_grade.to_string(),
        "axial_nonuniformity": q.axial_nonuniformity,
        "axial_nonuniformity_grade": q.axial_nonuniformity_grade.to_string(),
        "unused_error_correction": q.unused_error_correction,
        "unused_error_correction_grade": q.unused_error_correction_grade.to_string(),
        "fixed_pattern_damage_pct": q.fixed_pattern_damage_pct,
        "fixed_pattern_damage_grade": q.fixed_pattern_damage_grade.to_string(),
        "module_size_px": q.module_size_px,
        "symbol_size_px": [q.symbol_size_px.0, q.symbol_size_px.1],
        "rotation_deg": q.rotation_deg,
        "skew_deg": q.skew_deg,
        "pitch_drift_modules": q.pitch_drift_modules,
        "dark_reflectance_mean": q.dark_reflectance_mean,
        "light_reflectance_mean": q.light_reflectance_mean,
        "corners_px": q.corners_px,
        "overall_grade_estimate": q.overall_grade_estimate.to_string(),
    })
}

/// `quality` of a `quality` fixture case (fit rebuilt from its fields,
/// with the warped grid's axes when the case has them).
pub fn quality_of(c: &Case) -> Quality {
    let o = &c.output;
    let (r, cl) = o["size"].as_str().unwrap().split_once('x').unwrap();
    let g = geometry(symbol(r.parse().unwrap(), cl.parse().unwrap()));
    let mut fit = GridFit::new(g, quad(&o["quad"]), o["contrast"].as_f64().unwrap());
    fit.bits = Some(
        o["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect(),
    );
    fit.fixed_errors = o["fixed_errors"].as_u64().unwrap() as usize;
    if let Some(ax) = o.get("axes").and_then(|a| a.as_array()) {
        fit.axes = Some((f64s(&ax[0]), f64s(&ax[1])));
    }
    let rs = RSResult {
        data_cw: vec![],
        all_cw: vec![],
        block_errors: vec![],
        block_erasures: vec![],
        corrected_positions: vec![],
        uec: o["uec"].as_f64().unwrap(),
    };
    let off = f64s(&o["offset"]);
    quality(
        &gray(&c.input["gray"]),
        &fit,
        o["scale"].as_f64().unwrap(),
        &rs,
        (off[0], off[1]),
    )
}
