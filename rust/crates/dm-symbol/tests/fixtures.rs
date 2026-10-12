//! P1.2: exact equality with the Python fixtures (`rust/fixtures/symbols.json`,
//! written by `python rust/tools/dump_fixtures.py`).

use dm_symbol::{all_symbols, build_matrix, deinterleave, geometry, placement_map};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    sym: [usize; 9],
    placement: Vec<Option<(usize, u8)>>,
    codewords: Vec<u8>,
    matrix: Vec<u8>,
    fixed_mask: Vec<u8>,
    fixed_val: Vec<u8>,
    cw_rows: Vec<[usize; 8]>,
    cw_cols: Vec<[usize; 8]>,
    pad_mask: Vec<u8>,
    pad_val: Vec<u8>,
    fp_uv_qz: Vec<[f64; 2]>,
    fp_dark_qz: Vec<u8>,
    fp_uv_in: Vec<[f64; 2]>,
    fp_dark_in: Vec<u8>,
    deinterleave: Vec<Vec<usize>>,
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

fn bools(v: &[bool]) -> Vec<u8> {
    v.iter().map(|&b| u8::from(b)).collect()
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn symbols_match_python() {
    let cases: Vec<Case> = load("symbols.json");
    let syms: Vec<_> = all_symbols().collect();
    assert_eq!(cases.len(), syms.len());
    for (case, sym) in cases.iter().zip(syms) {
        let name = sym.name();
        assert_eq!(case.sym, sym.as_tuple(), "symbol table order");
        let place = placement_map(sym.dregr * sym.regr, sym.dregc * sym.regc);
        assert_eq!(place, case.placement, "{name} placement");
        assert_eq!(
            build_matrix(&case.codewords, sym),
            case.matrix,
            "{name} matrix"
        );

        let g = geometry(sym);
        assert_eq!(g.name(), name);
        assert_eq!(bools(&g.fixed_mask), case.fixed_mask, "{name} fixed_mask");
        assert_eq!(g.fixed_val, case.fixed_val, "{name} fixed_val");
        assert_eq!(g.cw_rows, case.cw_rows, "{name} cw_rows");
        assert_eq!(g.cw_cols, case.cw_cols, "{name} cw_cols");
        assert_eq!(bools(&g.pad_mask), case.pad_mask, "{name} pad_mask");
        assert_eq!(g.pad_val, case.pad_val, "{name} pad_val");
        assert_eq!(g.fp_uv_qz, case.fp_uv_qz, "{name} fp_uv_qz");
        assert_eq!(bools(&g.fp_dark_qz), case.fp_dark_qz, "{name} fp_dark_qz");
        assert_eq!(g.fp_uv_in, case.fp_uv_in, "{name} fp_uv_in");
        assert_eq!(bools(&g.fp_dark_in), case.fp_dark_in, "{name} fp_dark_in");

        let blocks = deinterleave(&case.codewords, g);
        let pos: Vec<Vec<usize>> = blocks.iter().map(|(p, _)| p.clone()).collect();
        assert_eq!(pos, case.deinterleave, "{name} deinterleave");
        for (p, b) in &blocks {
            let want: Vec<u8> = p.iter().map(|&i| case.codewords[i]).collect();
            assert_eq!(*b, want);
        }
        assert!(std::ptr::eq(g, geometry(sym)), "geometry is cached");
    }
}
