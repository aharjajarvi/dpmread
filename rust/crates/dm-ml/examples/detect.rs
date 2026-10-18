//! Print the detector's detections of each image, one line each:
//! `file score p_dark sizes quad finder_quad` (corners x,y in image
//! pixels). `cargo run --release -p dm-ml --example detect -- IMAGE...`

use dm_imgproc::{bgr_to_gray, load_image};
use dm_ml::Detector;

fn quad(q: &[[f32; 2]; 4]) -> String {
    q.iter()
        .map(|p| format!("{:.1},{:.1}", p[0], p[1]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let d = Detector::embedded();
    for path in std::env::args().skip(1) {
        let gray = match load_image(path.as_ref()) {
            Ok(b) => bgr_to_gray(&b),
            Err(e) => {
                eprintln!("{path}: {e}");
                continue;
            }
        };
        for det in d.detect(&gray, 0.3, 3).unwrap_or_default() {
            println!(
                "{path}\t{:.3}\t{:.3}\t{}\t{}\t{}",
                det.score,
                det.p_dark,
                det.sizes.join(","),
                quad(&det.quad),
                quad(&det.finder_quad)
            );
        }
    }
}
