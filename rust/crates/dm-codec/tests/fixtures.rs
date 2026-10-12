//! P1.4: every `Decoded` field (or the error) equal to Python on the fixtures
//! (`rust/fixtures/codec.json`, `gs1.json`, written by
//! `python rust/tools/dump_fixtures.py`).

use dm_codec::{DataError, decode_codewords, parse_gs1};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Want {
    data: Vec<u8>,
    segments: Vec<(String, usize, usize, Vec<u8>, String)>,
    payload_cw: usize,
    pad_cw: usize,
    gs1: bool,
    eci: Vec<i64>,
    #[serde(rename = "macro")]
    macro_kind: String,
    structured_append: String,
    reader_programming: bool,
    warnings: Vec<String>,
}

#[derive(Deserialize)]
struct CodecCase {
    cw: Vec<u8>,
    ok: Option<Want>,
    err: Option<(String, String)>,
}

#[derive(Deserialize)]
struct Gs1Case {
    text: String,
    elements: Vec<(String, String, String)>,
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

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn decode_codewords_matches_python() {
    let cases: Vec<CodecCase> = load("codec.json");
    let (mut ok, mut err) = (0, 0);
    for (i, c) in cases.iter().enumerate() {
        let got = decode_codewords(&c.cw);
        match (&got, &c.ok, &c.err) {
            (Ok(d), Some(w), None) => {
                let ctx = format!("case {i} {:?}", c.cw);
                assert_eq!(d.data, w.data, "{ctx} data");
                let segs: Vec<_> = d
                    .segments
                    .iter()
                    .map(|s| {
                        (
                            s.mode.to_string(),
                            s.cw_start,
                            s.cw_end,
                            s.data.clone(),
                            s.note.clone(),
                        )
                    })
                    .collect();
                assert_eq!(segs, w.segments, "{ctx} segments");
                assert_eq!(d.payload_cw, w.payload_cw, "{ctx} payload_cw");
                assert_eq!(d.pad_cw, w.pad_cw, "{ctx} pad_cw");
                assert_eq!(d.gs1, w.gs1, "{ctx} gs1");
                assert_eq!(d.eci, w.eci, "{ctx} eci");
                assert_eq!(d.macro_kind, w.macro_kind, "{ctx} macro");
                assert_eq!(d.structured_append, w.structured_append, "{ctx} sa");
                assert_eq!(d.reader_programming, w.reader_programming, "{ctx} rp");
                assert_eq!(d.warnings, w.warnings, "{ctx} warnings");
                ok += 1;
            }
            (Err(DataError::Invalid(msg)), None, Some((kind, want))) if kind == "DataError" => {
                assert_eq!(msg, want, "case {i} {:?}", c.cw);
                err += 1;
            }
            (Err(DataError::Index), None, Some((kind, _))) if kind == "IndexError" => err += 1,
            _ => panic!(
                "case {i} {:?}: rust {got:?}, python ok={} err={:?}",
                c.cw,
                c.ok.is_some(),
                c.err
            ),
        }
    }
    assert_eq!(ok + err, cases.len());
    assert!(
        ok > 500 && err > 500,
        "fixture covers both: {ok} ok, {err} errors"
    );
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn parse_gs1_matches_python() {
    let cases: Vec<Gs1Case> = load("gs1.json");
    for c in &cases {
        let got: Vec<(String, String, String)> = parse_gs1(&c.text)
            .into_iter()
            .map(|(a, t, v)| (a, t.to_string(), v))
            .collect();
        assert_eq!(got, c.elements, "{:?}", c.text);
    }
}
