//! Module classifier: a small convolutional network that gives, for a grid
//! fit, the probability that each module is dark. Its input is
//! `dm_grid::symbol_input` (the symbol resampled through the fit at
//! `INPUT_PPM` pixels per module with `INPUT_MARGIN` modules of margin,
//! gray 0..1); its output one logit per module, row-major. The model is
//! ONNX, run with tract (pure Rust); the trained weights are embedded.
//! Training code: `ml/` at the repo root.

pub mod detector;

pub use detector::{Detection, Detector};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use dm_imgproc::GrayF32;
use tract_onnx::prelude::*;

/// The embedded model (`ml/train.py`, fine-tuned on real marks).
const MODEL: &[u8] = include_bytes!("../model/module_classifier.onnx");

type Plan = Arc<TypedRunnableModel>;

/// A loaded classifier. Runnable plans are made once per input shape
/// (one per symbol size) and cached.
pub struct Classifier {
    model: InferenceModel,
    plans: Mutex<HashMap<(usize, usize), Plan>>,
}

impl Classifier {
    /// Load an ONNX model with one input (1, 1, H, W) and one output
    /// (1, rows, cols).
    pub fn from_bytes(bytes: &[u8]) -> TractResult<Self> {
        let model = tract_onnx::onnx().model_for_read(&mut &bytes[..])?;
        Ok(Classifier {
            model,
            plans: Mutex::new(HashMap::new()),
        })
    }

    /// The embedded model, loaded on first use.
    pub fn embedded() -> &'static Classifier {
        static C: OnceLock<Classifier> = OnceLock::new();
        C.get_or_init(|| Classifier::from_bytes(MODEL).expect("embedded model loads"))
    }

    fn plan(&self, h: usize, w: usize) -> TractResult<Plan> {
        if let Some(p) = self.plans.lock().unwrap().get(&(h, w)) {
            return Ok(p.clone());
        }
        let p = self
            .model
            .clone()
            .with_input_fact(0, f32::fact([1, 1, h, w]).into())?
            .into_optimized()?
            .into_runnable()?;
        self.plans.lock().unwrap().insert((h, w), p.clone());
        Ok(p)
    }

    /// The model's logit for each module being dark, row-major (rows x
    /// cols), for a classifier input of a rows x cols symbol. `None` when
    /// the input does not fit the size or the model fails.
    pub fn logits(&self, input: &GrayF32, rows: usize, cols: usize) -> Option<Vec<f32>> {
        let (h, w) = (input.height, input.width);
        let plan = self.plan(h, w).ok()?;
        let x = Tensor::from_shape(&[1, 1, h, w], &input.data).ok()?;
        let out = plan.run(tvec![x.into()]).ok()?;
        let z = out[0].to_plain_array_view::<f32>().ok()?;
        (z.shape() == [1, rows, cols]).then(|| z.iter().copied().collect())
    }

    /// Probability that each module is dark (`logits` through a sigmoid).
    pub fn p_dark(&self, input: &GrayF32, rows: usize, cols: usize) -> Option<Vec<f32>> {
        let z = self.logits(input, rows, cols)?;
        Some(z.into_iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// (4 + n) * 6 pixels a side: a symbol of n x n modules with margin.
    fn checker(n: usize) -> GrayF32 {
        let s = (n + 4) * 6;
        let data = (0..s * s)
            .map(|i| {
                let (y, x) = (i / s / 6, i % s / 6);
                let inside = (2..n + 2).contains(&y) && (2..n + 2).contains(&x);
                if inside && (x + y) % 2 == 0 { 0.1 } else { 0.9 }
            })
            .collect();
        GrayF32::from_vec(s, s, data)
    }

    /// Logits of the trained model (PyTorch, `ml/model.py`) for
    /// `checker(10)`, rounded to 2 decimals.
    const CHECKER_10: [[f32; 10]; 10] = [
        [
            14.49, -11.77, 12.04, -12.69, 12.29, -12.68, 12.21, -12.34, 9.96, -17.98,
        ],
        [
            -0.51, 13.42, -3.81, 14.51, -4.17, 14.44, -4.18, 13.2, -4.94, 10.2,
        ],
        [
            17.2, -2.62, 17.67, -3.4, 18.29, -3.38, 18.25, -4.32, 11.51, -8.99,
        ],
        [
            1.54, 18.13, -1.98, 19.07, -2.6, 18.95, -2.62, 17.64, -4.35, 11.36,
        ],
        [
            17.14, -1.97, 18.09, -2.65, 18.73, -2.66, 18.75, -3.51, 11.86, -8.96,
        ],
        [
            1.52, 17.96, -2.02, 18.85, -2.66, 18.73, -2.68, 17.41, -4.44, 11.25,
        ],
        [
            17.16, -1.95, 18.11, -2.65, 18.75, -2.66, 18.76, -3.52, 11.87, -8.98,
        ],
        [
            2.03, 16.98, -2.4, 17.79, -3.08, 17.68, -3.09, 16.46, -4.55, 11.32,
        ],
        [
            14.88, -1.89, 13.59, -2.32, 14.17, -2.32, 14.19, -3.11, 10.46, -8.05,
        ],
        [7.32, 13.3, 5.19, 12.74, 3.66, 12.62, 3.67, 12.46, 2.5, 8.84],
    ];

    #[test]
    fn matches_the_training_framework() {
        let c = Classifier::embedded();
        let z = c.logits(&checker(10), 10, 10).expect("model runs");
        for (i, (&a, &b)) in z.iter().zip(CHECKER_10.as_flattened()).enumerate() {
            assert!((a - b).abs() < 0.02, "module {i}: {a} vs {b}");
        }
        let p = c.p_dark(&checker(24), 24, 24).expect("model runs");
        assert_eq!(p.len(), 24 * 24);
        assert!(c.p_dark(&checker(10), 12, 12).is_none());
    }

    #[test]
    #[ignore = "timing, run with --ignored --nocapture"]
    fn timing() {
        let c = Classifier::embedded();
        for n in [10, 24, 64, 144] {
            let x = checker(n);
            c.p_dark(&x, n, n).unwrap();
            let k = if n > 64 { 5 } else { 50 };
            let t = Instant::now();
            for _ in 0..k {
                c.p_dark(&x, n, n).unwrap();
            }
            println!(
                "{n}x{n}: {:.2} ms",
                t.elapsed().as_secs_f64() * 1e3 / k as f64
            );
        }
    }
}
