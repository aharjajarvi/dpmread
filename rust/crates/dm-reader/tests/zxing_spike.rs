//! P0.3 / P0.6: the zxing-cpp crate reads the fixture variants exactly as
//! Python's zxingcpp does (`testdata/zxing/expected.tsv`, checked against
//! `rust/tools/zxing_spike_ref.py`). Also covers non-ASCII file names.
#![cfg(feature = "zxing")]

#[allow(dead_code)]
#[path = "../examples/zxing_spike.rs"]
mod spike;

use std::path::PathBuf;

/// Private test data (fixtures, test images, baselines): `DM_PRIVATE_DIR`
/// (not part of this repo). Tests that need it are
/// `#[ignore]`d; `rust/tools/ci.sh` runs them when the data is there.
fn private_dir() -> PathBuf {
    std::env::var_os("DM_PRIVATE_DIR")
        .map(PathBuf::from)
        .expect("DM_PRIVATE_DIR not set: it points at the private test data")
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn fixtures_match_python() {
    let dir = private_dir().join("testdata/zxing");
    let expected = std::fs::read_to_string(dir.join("expected.tsv")).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    files.sort();
    let got: Vec<String> = files
        .iter()
        .flat_map(|f| spike::result_lines(f).unwrap())
        .collect();
    let want: Vec<&str> = expected.lines().collect();
    assert_eq!(got, want);
    assert!(files.iter().any(|f| !f.to_string_lossy().is_ascii()));
}
