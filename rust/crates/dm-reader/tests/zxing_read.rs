//! P4.1: `zxing_read` returns what Python's `zxing_read` returns
//! (`testdata/zxing/zxing_read.tsv` from `rust/tools/zxing_read_ref.py`):
//! the read, its quad and info, and the failed positions of the first
//! binarizer, for the plain and the binarised ("> otsu") path and both
//! `try_invert` values.
#![cfg(feature = "zxing")]

use std::path::PathBuf;

/// Private test data (fixtures, test images, baselines): `DM_PRIVATE_DIR`
/// (not part of this repo). Tests that need it are
/// `#[ignore]`d; `rust/tools/ci.sh` runs them when the data is there.
fn private_dir() -> PathBuf {
    std::env::var_os("DM_PRIVATE_DIR")
        .map(PathBuf::from)
        .expect("DM_PRIVATE_DIR not set: it points at the private test data")
}

use dm_imgproc::Gray8;
use dm_reader::{Quad, Variant, zxing_read};

fn quad(q: &Quad) -> String {
    q.iter()
        .map(|(x, y)| format!("{x},{y}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn zxing_read_matches_python() {
    let dir = private_dir().join("testdata/zxing");
    let want = std::fs::read_to_string(dir.join("zxing_read.tsv")).unwrap();
    let mut n = 0;
    for line in want.lines() {
        let cols: Vec<&str> = line.splitn(4, '\t').collect();
        let (file, name, inv) = (cols[0], cols[1], cols[2] == "true");
        let img = image::open(dir.join(file)).unwrap().to_luma8();
        let (w, h) = img.dimensions();
        let g = Gray8::from_vec(w as usize, h as usize, img.into_raw());
        let (r, failed) = zxing_read(&Variant::new(name, g, 1.0), inv);
        let res = match r {
            Some(r) => format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.data
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
                quad(&r.quad),
                r.info.binarizer,
                r.info.symbology_identifier,
                r.info.content_type,
                r.info.orientation_deg,
                r.info.ec_level
            ),
            None => "-".into(),
        };
        let fl = failed.iter().map(quad).collect::<Vec<_>>().join(";");
        assert_eq!(format!("{file}\t{name}\t{inv}\t{res}\t{fl}"), line);
        n += 1;
    }
    assert_eq!(n, 20);
}
