//! P3.4: `classify`, `codewords_from_bits`, `try_fit`, `_zscore` against
//! Python: the refined best candidates of 6 images (qz on/off), each with
//! plane / local thresholds on the even and the timing-tracked grid.

mod common;

use common::*;
use dm_grid::*;
use dm_imgproc::Image;

fn u8s(v: &serde_json::Value) -> Vec<u8> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u8)
        .collect()
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn fits_match() {
    let (mut runs, mut decoded, mut conf_exact, mut worst_conf) = (0, 0, 0, 0f64);
    for c in cases("fit") {
        let g = geom(&c.params);
        let smp = Sampler::new(&gray(&c.input["img"]), 0.22);
        let q = quad(&c.input["quad"]);
        for r in c.output.as_array().unwrap() {
            let what = format!(
                "{} {}x{} {} {}",
                c.params["image"], g.rows, g.cols, r["mode"], r["tracked"]
            );
            let mode = if r["mode"] == "plane" {
                Mode::Plane
            } else {
                Mode::Local
            };
            let mut fit = GridFit::new(g, q, c.input["score"].as_f64().unwrap());
            let got = try_fit(&smp, &mut fit, mode, r["tracked"].as_bool().unwrap());
            let vals = fit.values.clone().unwrap();
            assert_eq!(vals.data, f32s(&r["vals"]), "{what}: values");
            assert_eq!(
                fit.bits.as_deref().unwrap(),
                u8s(&r["bits"]),
                "{what}: bits"
            );
            assert_eq!(fit.threshold, r["threshold"].as_f64().unwrap(), "{what}");
            assert_eq!(fit.fixed_errors as u64, r["fixed_errors"].as_u64().unwrap());
            let want_conf = f64s(&r["conf"]);
            let conf = fit.conf.as_deref().unwrap();
            let d = conf
                .iter()
                .zip(&want_conf)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            worst_conf = worst_conf.max(d);
            conf_exact += (d == 0.0) as usize;
            assert!(d < 1e-9, "{what}: conf diff {d}");
            let (cw, cconf) = codewords_from_bits(fit.bits.as_deref().unwrap(), conf, g);
            assert_eq!(cw, u8s(&r["cw"]), "{what}: codewords");
            assert_eq!(cconf.len(), f64s(&r["cconf"]).len());
            assert_eq!(got.is_some(), r["ok"].as_bool().unwrap(), "{what}: decoded");
            if let Some((rs, dec)) = got {
                decoded += 1;
                let hex: String = dec.data.iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(hex, r["data"].as_str().unwrap(), "{what}");
                assert_eq!(rs.all_cw, u8s(&r["all_cw"]));
                assert_eq!(rs.uec, r["uec"].as_f64().unwrap());
            }
            // zscore on the same values
            let z = zscore(&Image::from_vec(g.cols, g.rows, f32s(&r["vals"])), g);
            assert_eq!(z.data, f32s(&r["zscore"]), "{what}: zscore");
            runs += 1;
        }
    }
    eprintln!(
        "fit: {runs} runs, {decoded} decoded; values, bits, codewords, zscore equal; \
         conf bit-exact in {conf_exact} runs, max diff {worst_conf:e}"
    );
}
