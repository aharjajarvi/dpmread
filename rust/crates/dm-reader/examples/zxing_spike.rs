//! P0.3 spike: read images with the `zxing-cpp` crate using the options of
//! `zxing_read` in datamatrix_reader.py and print one TSV line per result.
//!
//! cargo run --example zxing_spike -- IMAGE_OR_FOLDER...
//!
//! Columns: file, binarizer, index, valid, hex(bytes), symbology_identifier,
//! content_type, orientation, position, error. `rust/tools/zxing_spike_ref.py`
//! prints the same lines from Python for a direct diff;
//! `testdata/zxing/expected.tsv` is the output for `testdata/zxing`.

use std::path::{Path, PathBuf};

use zxingcpp::{BarcodeError, BarcodeFormat, Binarizer, ImageFormat, ImageView, TextMode};

const BINARIZERS: [Binarizer; 3] = [
    Binarizer::LocalAverage,
    Binarizer::GlobalHistogram,
    Binarizer::FixedThreshold,
];

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn error_kind(e: &BarcodeError) -> &'static str {
    match e {
        BarcodeError::None() => "",
        BarcodeError::Checksum(_) => "Checksum",
        BarcodeError::Format(_) => "Format",
        BarcodeError::Unsupported(_) => "Unsupported",
    }
}

/// One TSV line per zxing result, all binarizers in turn.
pub fn result_lines(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_luma8();
    let (w, h) = img.dimensions();
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let mut lines = Vec::new();
    for bz in BINARIZERS {
        let iv = ImageView::from_slice(img.as_raw(), w, h, ImageFormat::Lum)?;
        let results = zxingcpp::read()
            .formats([BarcodeFormat::DataMatrix])
            .binarizer(bz)
            .text_mode(TextMode::Plain)
            .try_invert(true)
            .return_errors(true)
            .from(iv)?;
        for (i, b) in results.iter().enumerate() {
            let p = b.position();
            let pos = [p.top_left, p.top_right, p.bottom_right, p.bottom_left]
                .iter()
                .map(|q| format!("{},{}", q.x, q.y))
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(format!(
                "{name}\t{bz:?}\t{i}\t{}\t{}\t{}\t{:?}\t{}\t{pos}\t{}",
                b.is_valid(),
                hex(&b.bytes()),
                b.symbology_identifier(),
                b.content_type(),
                b.orientation(),
                error_kind(&b.error()),
            ));
        }
    }
    Ok(lines)
}

/// Arguments as files; a folder expands to its .png / .jpg files, sorted.
fn collect_inputs() -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for arg in std::env::args_os().skip(1) {
        let p = PathBuf::from(arg);
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| {
                    f.extension().and_then(|x| x.to_str()).is_some_and(|x| {
                        matches!(x.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg")
                    })
                })
                .collect();
            files.sort();
            out.extend(files);
        } else {
            out.push(p);
        }
    }
    Ok(out)
}

fn main() {
    let inputs = collect_inputs().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let mut failed = false;
    for path in inputs {
        match result_lines(&path) {
            Ok(lines) => lines.iter().for_each(|l| println!("{l}")),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                failed = true;
            }
        }
    }
    std::process::exit(i32::from(failed));
}
