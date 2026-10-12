//! P1.3: `rs_correct` and `rs_decode_symbol` equal to Python on the fixtures
//! (`rust/fixtures/rs_*.json`, written by `python rust/tools/dump_fixtures.py`).

use dm_ecc::{GF_EXP, GF_LOG, gf_div, gf_mul, rs_correct, rs_decode_symbol, rs_syndromes};
use dm_symbol::{Symbol, all_symbols, geometry};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Corrected {
    out: Vec<u8>,
    changed: Vec<usize>,
}

#[derive(Deserialize)]
struct CorrectCase {
    msg: Vec<u8>,
    nsym: usize,
    erasures: Vec<usize>,
    result: Option<Corrected>,
}

#[derive(Deserialize)]
struct SymResult {
    data_cw: Vec<u8>,
    all_cw: Vec<u8>,
    block_errors: Vec<usize>,
    block_erasures: Vec<usize>,
    corrected_positions: Vec<usize>,
    uec: f64,
}

#[derive(Deserialize)]
struct SymbolCase {
    sym: [usize; 9],
    cw: Vec<u8>,
    conf: Vec<f64>,
    result: Option<SymResult>,
}

/// Private test data (fixtures, test images, baselines): `DM_PRIVATE_DIR`
/// (not part of this repo). Tests that need it are
/// `#[ignore]`d; `rust/tools/ci.sh` runs them when the data is there.
fn private_dir() -> PathBuf {
    std::env::var_os("DM_PRIVATE_DIR")
        .map(PathBuf::from)
        .expect("DM_PRIVATE_DIR not set: it points at the private test data")
}

fn load<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = private_dir().join("fixtures").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; run `python rust/tools/dump_fixtures.py`",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap()
}

fn symbol(t: [usize; 9]) -> &'static Symbol {
    all_symbols().find(|s| s.as_tuple() == t).unwrap()
}

#[test]
fn gf_tables() {
    assert_eq!(GF_EXP[0], 1);
    assert_eq!(GF_EXP[8], 0x2D);
    assert_eq!(GF_EXP[255], 1);
    for a in 1..=255u8 {
        assert_eq!(GF_EXP[GF_LOG[a as usize] as usize], a);
        assert_eq!(gf_mul(a, gf_div(1, a)), 1);
    }
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn rs_correct_matches_python() {
    let cases: Vec<CorrectCase> = load("rs_correct.json");
    assert_eq!(cases.len(), 4000);
    let mut fixed = 0;
    for (i, c) in cases.iter().enumerate() {
        let got = rs_correct(&c.msg, c.nsym, &c.erasures, None);
        match (&got, &c.result) {
            (None, None) => {}
            (Some((out, changed)), Some(want)) => {
                assert_eq!(*out, want.out, "case {i} output");
                assert_eq!(*changed, want.changed, "case {i} changed");
                fixed += 1;
            }
            _ => panic!(
                "case {i}: rust {:?}, python {}",
                got.is_some(),
                c.result.is_some()
            ),
        }
        // precomputed syndromes give the same answer
        let synd = rs_syndromes(&c.msg, c.nsym);
        assert_eq!(rs_correct(&c.msg, c.nsym, &c.erasures, Some(&synd)), got);
    }
    assert_eq!(fixed, cases.iter().filter(|c| c.result.is_some()).count());
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn rs_decode_symbol_matches_python() {
    let cases: Vec<SymbolCase> = load("rs_symbol.json");
    for (i, c) in cases.iter().enumerate() {
        let g = geometry(symbol(c.sym));
        let got = rs_decode_symbol(&c.cw, &c.conf, g);
        match (got, &c.result) {
            (None, None) => {}
            (Some(r), Some(w)) => {
                assert_eq!(r.data_cw, w.data_cw, "case {i}");
                assert_eq!(r.all_cw, w.all_cw, "case {i}");
                assert_eq!(r.block_errors, w.block_errors, "case {i}");
                assert_eq!(r.block_erasures, w.block_erasures, "case {i}");
                assert_eq!(r.corrected_positions, w.corrected_positions, "case {i}");
                assert_eq!(r.uec, w.uec, "case {i}");
            }
            (r, w) => panic!(
                "case {i} {}: rust {:?}, python {}",
                g.name(),
                r.is_some(),
                w.is_some()
            ),
        }
    }
}
