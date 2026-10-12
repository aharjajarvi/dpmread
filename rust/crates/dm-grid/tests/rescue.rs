//! P3.5: `rescue_fit` on every call Python's `grid_decode` made for the two
//! baseline rescue reads (poor/081252 `blue > flatten`, poor/094126
//! `gray > flatten > clahe`) and a failing variant: same success, data and
//! sampling note; on failure the same last classification.

mod common;

use common::*;
use dm_grid::*;

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn rescue_matches() {
    let all = cases("rescue");
    let res: Vec<bool> = std::thread::scope(|sc| {
        let hs: Vec<_> = all.iter().map(|c| sc.spawn(move || check(c))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ok = res.iter().filter(|&&r| r).count();
    eprintln!("rescue: {} calls, {ok} decoded, all equal", res.len());
    assert!(ok >= 2);
}

fn check(c: &Case) -> bool {
    let g = geom(&c.params);
    let what = format!(
        "{} {} {}x{}",
        c.params["image"], c.params["variant"], g.rows, g.cols
    );
    let imgs: Vec<(String, dm_imgproc::Gray8)> = c.input["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_str().unwrap().to_string(), gray(&p[1])))
        .collect();
    let refs: Vec<(String, &dm_imgproc::Gray8)> =
        imgs.iter().map(|(n, i)| (n.clone(), i)).collect();
    let mut fit = GridFit::new(
        g,
        quad(&c.input["quad"]),
        c.input["contrast"].as_f64().unwrap(),
    );
    let got = rescue_fit(&mut fit, &refs, &Deadline::never());
    let o = &c.output;
    assert_eq!(got.is_some(), o["ok"].as_bool().unwrap(), "{what}");
    let bits: Vec<u8> = o["bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    assert_eq!(fit.bits.as_deref().unwrap(), bits, "{what}: last bits");
    assert_eq!(fit.threshold, o["threshold"].as_f64().unwrap(), "{what}");
    assert_eq!(
        fit.fixed_errors as u64,
        o["fixed_errors"].as_u64().unwrap(),
        "{what}"
    );
    if let Some((_, dec, how)) = got {
        let hex: String = dec.data.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, o["data"].as_str().unwrap(), "{what}");
        assert_eq!(how, o["how"].as_str().unwrap(), "{what}");
        return true;
    }
    false
}
