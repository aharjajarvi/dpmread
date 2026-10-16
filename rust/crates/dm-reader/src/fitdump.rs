//! `--dump-fits DIR`: the grid fits of an image and the module classifier
//! input for each, for training and checking a module classifier.
//!
//! Per image `STEM.fits.json` (file, decoded text, input geometry, one entry
//! per fit: `read`, `variant`, `size`, `mirrored`, `score`,
//! `fixed_pattern_module_errors`, `centres` in original image pixels,
//! row-major, and its input file) and `STEM.fitN.npy` (float32, gray 0..1 of
//! the fit's variant, `INPUT_PPM` pixels per module, `INPUT_MARGIN` modules
//! of margin; symbol dark on light).

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use dm_grid::{INPUT_MARGIN, INPUT_PPM};
use dm_imgproc::GrayF32;
use serde_json::json;

use crate::read::ReadResult;

/// A float32 image as a version 1.0 `.npy` file (shape height x width).
pub fn write_npy(path: &Path, img: &GrayF32) -> io::Result<()> {
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        img.height, img.width
    );
    // magic (6) + version (2) + length (2) + dict + padding + '\n', a
    // multiple of 64 bytes
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let header_len = dict.len() + pad + 1;
    let mut out = Vec::with_capacity(10 + header_len + 4 * img.data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad));
    out.push(b'\n');
    for v in &img.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::File::create(path)?.write_all(&out)
}

/// Write the fits of `res` (`ReadOptions.fit_inputs`) into `dir`.
pub fn write_fit_dump(dir: &Path, res: &ReadResult) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let stem = res
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut fits = Vec::with_capacity(res.fits.len());
    for (i, f) in res.fits.iter().enumerate() {
        let name = format!("{stem}.fit{i}.npy");
        write_npy(&dir.join(&name), &f.input)?;
        fits.push(json!({
            "read": f.read,
            "variant": f.variant,
            "size": f.size,
            "mirrored": f.mirrored,
            "score": f.score,
            "fixed_pattern_module_errors": f.fixed_pattern_module_errors,
            "centres": f.centres,
            "input": name,
        }));
    }
    let doc = json!({
        "file": res.path.to_string_lossy(),
        "decoded": !res.data.is_empty(),
        "text": res.text,
        "input_ppm": INPUT_PPM,
        "input_margin": INPUT_MARGIN,
        "fits": fits,
    });
    let text = serde_json::to_string_pretty(&doc).map_err(io::Error::other)?;
    fs::write(dir.join(format!("{stem}.fits.json")), text + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_header_is_aligned() {
        let dir = std::env::temp_dir().join(format!("dm-npy-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.npy");
        write_npy(
            &p,
            &GrayF32::from_vec(3, 2, vec![0.0, 0.5, 1.0, 0.25, 0.75, 1.0]),
        )
        .unwrap();
        let b = fs::read(&p).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!((10 + hl) % 64, 0);
        assert_eq!(b[10 + hl - 1], b'\n');
        assert_eq!(b.len(), 10 + hl + 6 * 4);
        let _ = fs::remove_dir_all(&dir);
    }
}
