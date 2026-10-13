//! `Reader::read` on a rendered 10x10 symbol: the JSON is the CLI's object
//! (`to_dict`) of the same read, the symbol carries its outline and (grid
//! engine) module matrix; a missing file is a load error; `stop` ends a
//! read of an unreadable image soon, and with no read running does not
//! stop the next read.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dm_ecc::{gf_mul, gf_pow};
use dm_ffi::{ReadError, Reader, default_options};
use dm_reader::{ReadOptions, Runner, read_image, to_dict};
use dm_symbol::{all_symbols, build_matrix};

const TEXT: &str = "123456";
const PX: u32 = 8;
const QUIET: u32 = 4;

/// RS check codewords (generator roots a^1..a^n, highest power first).
fn rs_encode(data: &[u8], n: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for j in 1..=n {
        let r = gf_pow(2, j);
        let mut h = vec![0u8; g.len() + 1];
        for (i, &c) in g.iter().enumerate() {
            h[i] ^= c;
            h[i + 1] ^= gf_mul(c, r);
        }
        g = h;
    }
    let mut rem = vec![0u8; n];
    for &d in data {
        let f = d ^ rem[0];
        rem.remove(0);
        rem.push(0);
        for (r, &gc) in rem.iter_mut().zip(&g[1..]) {
            *r ^= gf_mul(gc, f);
        }
    }
    rem
}

/// `TEXT` as a 10x10 symbol (digit pairs: 130 + value; one RS block),
/// `PX` px per module with a `QUIET`-module quiet zone, saved as PNG.
fn render(name: &str) -> (PathBuf, Vec<u8>) {
    let sym = all_symbols()
        .find(|s| s.name() == "10x10")
        .expect("10x10 in the symbol table");
    let mut cw: Vec<u8> = TEXT
        .as_bytes()
        .chunks(2)
        .map(|p| 130 + (p[0] - b'0') * 10 + (p[1] - b'0'))
        .collect();
    let ecc = rs_encode(&cw, sym.n_codewords() - cw.len());
    cw.extend(ecc);
    let m = build_matrix(&cw, sym);
    let side = (sym.cols as u32 + 2 * QUIET) * PX;
    let img = image::GrayImage::from_fn(side, side, |x, y| {
        let (c, r) = (x / PX, y / PX);
        let dark = (QUIET..QUIET + sym.cols as u32).contains(&c)
            && (QUIET..QUIET + sym.rows as u32).contains(&r)
            && m[((r - QUIET) * sym.cols as u32 + c - QUIET) as usize] == 1;
        image::Luma([if dark { 0 } else { 255 }])
    });
    let path = std::env::temp_dir().join(format!("dm_ffi_{name}_{}.png", std::process::id()));
    img.save(&path).expect("write the test PNG");
    (path, m)
}

fn without_elapsed(mut v: serde_json::Value) -> serde_json::Value {
    v.as_object_mut().expect("JSON object").remove("elapsed_s");
    v
}

#[test]
fn read_matches_cli_json() {
    let (path, _) = render("json");
    let reader = Reader::new(2);
    reader.stop(); // no read running: must not stop the next one
    let out = reader
        .read(path.to_string_lossy().into_owned(), default_options())
        .expect("read");
    let cli = read_image(&path, &ReadOptions::default(), &Runner::new(2)).expect("read_image");
    let _ = std::fs::remove_file(&path);

    assert!(out.decoded && !out.stopped);
    let json: serde_json::Value = serde_json::from_str(&out.json).expect("valid JSON");
    assert_eq!(json["text"], TEXT);
    assert_eq!(without_elapsed(json), without_elapsed(to_dict(&cli)));
    assert!(out.report.contains(TEXT));

    assert_eq!(out.symbols.len(), 1);
    assert_eq!(out.symbols[0].text, TEXT);
    assert_eq!(out.symbols[0].outline.len(), 8);
}

#[test]
fn grid_read_has_module_matrix() {
    let (path, matrix) = render("grid");
    let opts = dm_ffi::Options {
        zxing: false,
        ..default_options()
    };
    let out = Reader::new(2)
        .read(path.to_string_lossy().into_owned(), opts)
        .expect("read");
    let _ = std::fs::remove_file(&path);
    assert!(out.decoded);
    let s = &out.symbols[0];
    assert_eq!((s.rows, s.cols), (10, 10));
    assert_eq!(s.modules, matrix);
    assert_eq!(s.outline.len(), 8);
}

#[test]
fn stop_ends_a_read_soon() {
    // noise: nothing reads, the grid engine searches until the budget
    let mut seed = 1u32;
    let img = image::GrayImage::from_fn(640, 480, |_, _| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        image::Luma([(seed >> 24) as u8])
    });
    let path = std::env::temp_dir().join(format!("dm_ffi_noise_{}.png", std::process::id()));
    img.save(&path).expect("write the test PNG");
    let reader = Arc::new(Reader::new(2));
    let r2 = reader.clone();
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        let t = Instant::now();
        r2.stop();
        t
    });
    let out = reader
        .read(path.to_string_lossy().into_owned(), default_options())
        .expect("read");
    let ended = Instant::now();
    let stopped_at = stopper.join().expect("stopper thread");
    let _ = std::fs::remove_file(&path);
    assert!(out.stopped && !out.decoded);
    assert!(ended.duration_since(stopped_at) < Duration::from_secs(1));
}

#[test]
fn missing_file_is_a_load_error() {
    let r = Reader::new(1).read("/nonexistent/dm_ffi.png".into(), default_options());
    assert!(matches!(r, Err(ReadError::Load { .. })));
}
