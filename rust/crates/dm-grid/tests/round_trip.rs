//! P3.9: synthetic round trip. Every symbol size gets random ASCII content
//! (data codewords + RS check codewords, interleaved), is rendered with
//! `build_matrix` at 4-8 px/module and read back with `grid_decode`: clean
//! renders (axis aligned, sharp, high contrast) and a degraded set
//! (rotation, perspective, blur, low contrast, noise; fixed seed).
//!
//! A wrong read fails the test, as does any clean failure outside
//! `CLEAN_FAILURES` or fewer degraded reads than `DEGRADED_READS`. Python's
//! `grid_decode` (blob aspect limit 4.5) misses 25 clean renders (the long
//! DMRE sizes and 144x144 @4 px) and reads 98 degraded ones; since P4b.12
//! (`max_aspect`) only 8x144 @7 px is left.
//! `ROUND_TRIP_DUMP=DIR` writes the renders for `rust/tools/round_trip_ref.py`.

use dm_ecc::{gf_mul, gf_pow, rs_syndromes};
use dm_grid::{Deadline, Quad, grid_decode, max_aspect};
use dm_imgproc::{Gray8, Image, gaussian_blur_u8, get_perspective_transform};
use dm_symbol::{Symbol, all_symbols, build_matrix, deinterleave, geometry};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

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

/// Random uppercase / digit text filling the data capacity, with valid
/// check codewords: (codeword stream, text).
fn symbol_codewords(sym: &Symbol, rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    let g = geometry(sym);
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let text: Vec<u8> = (0..g.n_data)
        .map(|_| CHARS[(rng.next() % CHARS.len() as u64) as usize])
        .collect();
    // ASCII encodation: one character per codeword (value + 1)
    let mut cw: Vec<u8> = text.iter().map(|c| c + 1).collect();
    cw.resize(sym.n_codewords(), 0);
    for (pos, block) in deinterleave(&cw, g) {
        let nd = block.len() - g.n_ecc;
        let ecc = rs_encode(&block[..nd], g.n_ecc);
        for (k, e) in ecc.into_iter().enumerate() {
            cw[pos[nd + k]] = e;
        }
    }
    for (_, block) in deinterleave(&cw, g) {
        assert!(rs_syndromes(&block, g.n_ecc).iter().all(|&s| s == 0));
    }
    (cw, text)
}

struct Look {
    px: f64,
    /// Symbol corners tl, tr, br, bl in px (before the margin offset).
    corners: Quad,
    dark: f64,
    light: f64,
    blur_modules: f64,
    noise: f64,
}

/// Supersampled render of the module matrix through the corner homography.
fn render(m: &[u8], rows: usize, cols: usize, look: &Look, rng: &mut Rng) -> Gray8 {
    let xs = look.corners.iter().map(|p| p[0]);
    let ys = look.corners.iter().map(|p| p[1]);
    let margin = 6.0 * look.px;
    let (x0, x1) = (
        xs.clone().fold(f32::MAX, f32::min),
        xs.fold(f32::MIN, f32::max),
    );
    let (y0, y1) = (
        ys.clone().fold(f32::MAX, f32::min),
        ys.fold(f32::MIN, f32::max),
    );
    let w = ((x1 - x0) as f64 + 2.0 * margin).ceil() as usize;
    let h = ((y1 - y0) as f64 + 2.0 * margin).ceil() as usize;
    let dst = look
        .corners
        .map(|p| (p[0] - x0 + margin as f32, p[1] - y0 + margin as f32));
    let src = [
        (0.0, 0.0),
        (cols as f32, 0.0),
        (cols as f32, rows as f32),
        (0.0, rows as f32),
    ];
    // image px -> module coordinates
    let hm = get_perspective_transform(&dst, &src).expect("homography");
    const SS: usize = 3;
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let px = x as f64 + (sx as f64 + 0.5) / SS as f64;
                    let py = y as f64 + (sy as f64 + 0.5) / SS as f64;
                    let d = hm[6] * px + hm[7] * py + hm[8];
                    let u = (hm[0] * px + hm[1] * py + hm[2]) / d;
                    let v = (hm[3] * px + hm[4] * py + hm[5]) / d;
                    let on = u >= 0.0
                        && v >= 0.0
                        && (u as usize) < cols
                        && (v as usize) < rows
                        && m[v as usize * cols + u as usize] == 1;
                    acc += if on { look.dark } else { look.light };
                }
            }
            img.set(x, y, (acc / (SS * SS) as f64) as u8);
        }
    }
    if look.blur_modules > 0.0 {
        img = gaussian_blur_u8(&img, look.blur_modules * look.px);
    }
    if look.noise > 0.0 {
        for v in img.data.iter_mut() {
            *v = (*v as f64 + rng.range(-look.noise, look.noise)).clamp(0.0, 255.0) as u8;
        }
    }
    img
}

fn axis_corners(rows: usize, cols: usize, px: f64) -> Quad {
    let (w, h) = ((cols as f64 * px) as f32, (rows as f64 * px) as f32);
    [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]
}

/// Rotated by `ang`, each corner jittered by up to `jit` of the side.
fn warped_corners(rows: usize, cols: usize, px: f64, ang: f64, jit: f64, rng: &mut Rng) -> Quad {
    let (w, h) = (cols as f64 * px, rows as f64 * px);
    let (c, s) = (ang.cos(), ang.sin());
    let base = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    base.map(|[x, y]| {
        let (x, y) = (
            x + rng.range(-jit, jit) * w.min(h),
            y + rng.range(-jit, jit) * w.min(h),
        );
        [(x * c - y * s) as f32, (x * s + y * c) as f32]
    })
}

/// Clean renders not read (P4b.12; Python: 25, `rust/tools/round_trip_ref.py`).
const CLEAN_FAILURES: [&str; 1] = ["8x144 @7 px"];
/// Degraded read count, 144 renders in all (P4b.12; Python: 98).
const DEGRADED_READS: usize = 126;

struct Job {
    sym: &'static Symbol,
    px: f64,
    degraded: bool,
    seed: u64,
}

fn run(job: &Job) -> (bool, Option<String>) {
    let mut rng = Rng(job.seed);
    let (rows, cols) = (job.sym.rows, job.sym.cols);
    let (cw, text) = symbol_codewords(job.sym, &mut rng);
    let m = build_matrix(&cw, job.sym);
    let look = if job.degraded {
        Look {
            px: job.px,
            corners: warped_corners(
                rows,
                cols,
                job.px,
                rng.range(0.0, std::f64::consts::TAU),
                0.03,
                &mut rng,
            ),
            dark: rng.range(50.0, 90.0),
            light: rng.range(150.0, 200.0),
            blur_modules: rng.range(0.15, 0.3),
            noise: 12.0,
        }
    } else {
        Look {
            px: job.px,
            corners: axis_corners(rows, cols, job.px),
            dark: 30.0,
            light: 220.0,
            blur_modules: 0.0,
            noise: 0.0,
        }
    };
    let img = render(&m, rows, cols, &look, &mut rng);
    // ROUND_TRIP_DUMP=DIR: write every render as PGM (for checks against Python)
    if let Ok(dir) = std::env::var("ROUND_TRIP_DUMP") {
        let name = format!(
            "{dir}/{rows}x{cols}_{}_{}.pgm",
            job.px,
            if job.degraded { "deg" } else { "clean" }
        );
        let mut f = format!("P5\n{} {}\n255\n", img.width, img.height).into_bytes();
        f.extend(&img.data);
        std::fs::write(name, f).unwrap();
        std::fs::write(
            format!(
                "{dir}/{rows}x{cols}_{}_{}.txt",
                job.px,
                if job.degraded { "deg" } else { "clean" }
            ),
            &text,
        )
        .unwrap();
    }
    let r = grid_decode(
        &img,
        &Deadline::never(),
        &[],
        &[],
        0.02,
        max_aspect(None),
        &[true, false],
        None,
        None,
    );
    match r {
        None => (false, None),
        Some(r) if r.dec.data == text => (true, None),
        Some(r) => (
            false,
            Some(format!(
                "{rows}x{cols} @{} px: wrong read {:?}",
                job.px,
                String::from_utf8_lossy(&r.dec.data)
            )),
        ),
    }
}

#[test]
fn round_trip() {
    let mut jobs = Vec::new();
    for (k, sym) in all_symbols().enumerate() {
        for px in [4.0, 5.0, 6.0, 7.0, 8.0] {
            jobs.push(Job {
                sym,
                px,
                degraded: false,
                seed: 1000 * k as u64 + px as u64,
            });
        }
        for px in [4.0, 6.0, 8.0] {
            jobs.push(Job {
                sym,
                px,
                degraded: true,
                seed: 7_000_000 + 1000 * k as u64 + px as u64,
            });
        }
    }
    let next = AtomicUsize::new(0);
    let res = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    let r = run(job);
                    res.lock().unwrap().push((i, r));
                }
            });
        }
    });
    let res = res.into_inner().unwrap();
    let mut clean_fail = Vec::new();
    let mut wrong = Vec::new();
    let (mut deg_n, mut deg_ok) = (0, 0);
    let mut deg_fail = Vec::new();
    for (i, (ok, err)) in res {
        let j = &jobs[i];
        let name = format!("{}x{} @{} px", j.sym.rows, j.sym.cols, j.px);
        if let Some(e) = err {
            wrong.push(e);
        }
        if j.degraded {
            deg_n += 1;
            deg_ok += ok as usize;
            if !ok {
                deg_fail.push(name);
            }
        } else if !ok {
            clean_fail.push(name);
        }
    }
    deg_fail.sort();
    clean_fail.sort();
    eprintln!(
        "round trip: clean {}/{} decoded, failed: {clean_fail:?}; degraded {deg_ok}/{deg_n} ({:.0} %), failed: {deg_fail:?}",
        jobs.iter().filter(|j| !j.degraded).count() - clean_fail.len(),
        jobs.iter().filter(|j| !j.degraded).count(),
        100.0 * deg_ok as f64 / deg_n as f64
    );
    assert!(wrong.is_empty(), "wrong reads: {wrong:?}");
    let new: Vec<&String> = clean_fail
        .iter()
        .filter(|n| !CLEAN_FAILURES.contains(&n.as_str()))
        .collect();
    assert!(new.is_empty(), "new clean failures: {new:?}");
    assert!(
        deg_ok >= DEGRADED_READS,
        "degraded reads {deg_ok} < {DEGRADED_READS}"
    );
}
