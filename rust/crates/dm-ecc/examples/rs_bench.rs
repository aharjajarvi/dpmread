//! P1.5 RS micro-benchmark: time per call of `rs_correct` (4 000 fixture
//! blocks) and `rs_decode_symbol` (fixture symbols). Python counterpart:
//! `python rust/tools/rs_bench.py`.
//!
//! cargo run --release -p dm-ecc --example rs_bench

use dm_ecc::{rs_correct, rs_decode_symbol};
use dm_symbol::{all_symbols, geometry};
use serde::Deserialize;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct CorrectCase {
    msg: Vec<u8>,
    nsym: usize,
    erasures: Vec<usize>,
}

#[derive(Deserialize)]
struct SymbolCase {
    sym: [usize; 9],
    cw: Vec<u8>,
    conf: Vec<f64>,
}

/// Private test data, not part of this repo: `DM_PRIVATE_DIR`.
fn private_dir() -> PathBuf {
    std::env::var_os("DM_PRIVATE_DIR")
        .map(PathBuf::from)
        .expect("DM_PRIVATE_DIR not set: it points at the private test data")
}

fn load<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = private_dir().join("fixtures").join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}; run tools/dump_fixtures.py", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// Mean microseconds per call over `rounds` passes of `f` over `n` items.
fn time(n: usize, rounds: usize, mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    for _ in 0..rounds {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / (n * rounds) as f64
}

fn main() {
    let correct: Vec<CorrectCase> = load("rs_correct.json");
    let symbols: Vec<SymbolCase> = load("rs_symbol.json");
    let us = time(correct.len(), 20, || {
        for c in &correct {
            black_box(rs_correct(&c.msg, c.nsym, &c.erasures, None));
        }
    });
    println!(
        "rs_correct        {:>5} cases  {us:>9.2} us/call",
        correct.len()
    );

    let geoms: Vec<_> = symbols
        .iter()
        .map(|c| geometry(all_symbols().find(|s| s.as_tuple() == c.sym).unwrap()))
        .collect();
    let us = time(symbols.len(), 20, || {
        for (c, g) in symbols.iter().zip(&geoms) {
            black_box(rs_decode_symbol(&c.cw, &c.conf, g));
        }
    });
    println!(
        "rs_decode_symbol  {:>5} cases  {us:>9.2} us/call",
        symbols.len()
    );
}
