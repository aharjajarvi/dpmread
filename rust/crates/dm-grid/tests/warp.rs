//! Warped grid against `datamatrix_reader_v2.py`.
//! P4b.2: `pitch_drift` on 3 axis pairs (random drifts, values from Python).
//! P4b.3 / P4b.4: every `warp_fit` call of the qz grid job on the three
//! `> flatten` variants of the `not_recognized` images (`grid_v2`
//! fixtures): `global_track` of both axes on the input quad, and the
//! `warp_fit` result (accept / reject, quad, axes, score).

mod common;

use common::*;
use dm_grid::*;
use serde_json::Value;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const TOL: f64 = 1e-6;

type DriftCase = (Vec<f64>, Vec<f64>, f64);

fn drift_cases() -> Vec<DriftCase> {
    vec![
        (
            vec![
                0.5000984122685986,
                1.5239980552692762,
                2.5020670268402987,
                3.430819679739717,
                4.394446016925979,
                5.315114292526262,
                6.319925780734057,
                7.42714300037842,
                8.387766478894314,
                9.338128486908719,
            ],
            vec![
                0.5394921025092599,
                1.562844350408003,
                2.580270712449895,
                3.5584765977645896,
                4.633537408876836,
                5.699765159722914,
                6.627789272635746,
                7.7021192119479895,
                8.659938863009959,
                9.720523113010751,
            ],
            0.0891455701689976,
        ),
        (
            vec![
                0.35266119697666143,
                1.333853906490687,
                2.232458187975191,
                3.2541593366809267,
                4.266699423610865,
                5.251744948040469,
                6.0504041711748275,
                7.0073087395070965,
                8.003428663875011,
                9.012493382755276,
                9.890082521514843,
                10.85186225943213,
                11.773580733187599,
                12.70887375403355,
                13.793745643904437,
                14.729142869877885,
                15.726541133482243,
                16.7972923228729,
                17.750604288253435,
                18.7416681322867,
                19.750505263746657,
                20.755607805687063,
                21.657603339573647,
                22.663694558003808,
                23.772400431743133,
                24.648628857492852,
                25.71737947253458,
                26.726927794590306,
                27.67561016306173,
                28.835643486769122,
                29.8966242637359,
                30.80068115156748,
                31.8066424498692,
                32.85277761656282,
                33.837675046534756,
                34.89230786791037,
            ],
            vec![
                0.5116741339925293,
                1.5783623780417164,
                2.6469261295828077,
                3.5712168874497174,
                4.64515693051948,
                5.641834621173079,
                6.701363420561292,
                7.665640273607494,
                8.726034920174866,
                9.775190201359775,
                10.85993819360502,
                11.902261100372707,
                12.808823610375788,
                13.865267881700648,
                14.967345171128672,
                15.865379010791274,
            ],
            0.2502329023962844,
        ),
        (
            vec![
                0.46294641080381066,
                1.4551634567502034,
                2.555724654933149,
                3.6108769669788097,
                4.584699893361034,
                5.555213821833037,
                6.535198189791603,
                7.657080541828096,
                8.622838546422267,
                9.598544115353087,
                10.62675124073591,
                11.617089605128992,
                12.601306866891735,
                13.51218149543965,
                14.511259777996568,
                15.475773280158613,
                16.56906350225383,
                17.621310582469924,
                18.619379093429128,
                19.672849575290517,
                20.645660011153463,
                21.72983011982762,
                22.72939815497389,
                23.776068743308326,
                24.672797283682446,
                25.70053168759272,
            ],
            vec![
                0.4305897941316729,
                1.4432335527530034,
                2.5597761561144283,
                3.5600036196200704,
                4.643202639785611,
                5.777237831324302,
                6.653413840929396,
                7.6938028206778055,
                8.765270197303236,
                9.809650664570619,
                10.806179696704712,
                11.83470348348734,
            ],
            0.14732122355102106,
        ),
    ]
}

#[test]
fn pitch_drift_matches_v2() {
    for (u, v, want) in drift_cases() {
        let got = pitch_drift(Some(&(u, v)));
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    assert_eq!(pitch_drift(None), 0.0);
}

fn axis(v: &Value) -> Option<Vec<f64>> {
    (!v.is_null()).then(|| f64s(v))
}

/// "" if equal within `TOL` (same `None`), else what differs.
fn cmp_axis(what: &str, got: &Option<Vec<f64>>, want: &Option<Vec<f64>>) -> String {
    match (got, want) {
        (None, None) => String::new(),
        (Some(a), Some(b)) if a.len() == b.len() => {
            let d = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            if d <= TOL {
                String::new()
            } else {
                format!("{what} max diff {d:e}")
            }
        }
        _ => format!(
            "{what} {} vs Python {}",
            got.as_ref()
                .map_or("None".into(), |a| format!("{} centres", a.len())),
            want.as_ref()
                .map_or("None".into(), |a| format!("{} centres", a.len()))
        ),
    }
}

/// Runs `check` on every call of every `warp` case (cases in parallel);
/// returns (calls, errors).
fn each_call(check: impl Fn(&Sampler, &Value) -> String + Sync) -> (usize, Vec<String>) {
    use_dir("grid_v2");
    let all = cases("warp");
    let next = AtomicUsize::new(0);
    let out = Mutex::new((0usize, Vec::new()));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(c) = all.get(i) else { break };
                    let smp = Sampler::new(&gray(&c.input["img"]), 0.22);
                    let calls = c.output.as_array().unwrap();
                    let mut errs = Vec::new();
                    for (k, call) in calls.iter().enumerate() {
                        let e = check(&smp, call);
                        if !e.is_empty() {
                            errs.push(format!(
                                "{} [{}] call {k} {}x{}: {e}",
                                c.params["image"], c.params["variant"], call["rows"], call["cols"]
                            ));
                        }
                    }
                    let mut o = out.lock().unwrap();
                    o.0 += calls.len();
                    o.1.extend(errs);
                }
            });
        }
    });
    out.into_inner().unwrap()
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn global_track_matches_v2() {
    let (n, errors) = each_call(|smp, call| {
        let (q, g) = (quad(&call["quad"]), geom(call));
        let u = global_track(smp, &q, g, WarpAxis::U);
        let v = global_track(smp, &q, g, WarpAxis::V);
        [
            cmp_axis("u", &u, &axis(&call["u"])),
            cmp_axis("v", &v, &axis(&call["v"])),
        ]
        .into_iter()
        .filter(|e| !e.is_empty())
        .collect::<Vec<_>>()
        .join("; ")
    });
    eprintln!("global_track: {n} quads x 2 axes, {} differ", errors.len());
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
#[ignore = "needs private test data (DM_PRIVATE_DIR)"]
fn warp_fit_matches_v2() {
    let (n, errors) = each_call(|smp, call| {
        let (q, g) = (quad(&call["quad"]), geom(call));
        let got = warp_fit(smp, &q, g, &Deadline::never());
        let want = &call["fit"];
        match (&got, want.is_null()) {
            (None, true) => String::new(),
            (Some(f), false) => {
                let wq = quad(&want["quad"]);
                let dq = f
                    .quad
                    .iter()
                    .flatten()
                    .zip(wq.iter().flatten())
                    .map(|(a, b)| (a - b).abs() as f64)
                    .fold(0.0, f64::max);
                let (u, v) = f.axes.clone().unwrap();
                let mut e = vec![
                    cmp_axis("u", &Some(u), &axis(&want["u"])),
                    cmp_axis("v", &Some(v), &axis(&want["v"])),
                ];
                if dq > TOL {
                    e.push(format!("quad max diff {dq:e}"));
                }
                let ds = (f.contrast - want["score"].as_f64().unwrap()).abs();
                if ds > TOL {
                    e.push(format!("score diff {ds:e}"));
                }
                e.retain(|x| !x.is_empty());
                e.join("; ")
            }
            (g, _) => format!("fit {} vs Python {}", g.is_some(), !want.is_null()),
        }
    });
    eprintln!("warp_fit: {n} calls, {} differ", errors.len());
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
