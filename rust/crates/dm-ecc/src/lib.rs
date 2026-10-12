//! GF(256) Reed-Solomon with soft erasures for ECC 200.
//!
//! Port of the reader's GF tables, `rs_syndromes`, `_solve_magnitudes`,
//! `rs_correct`, `RSResult` and `rs_decode_symbol`. Field polynomial 0x12D,
//! generator roots a^1 .. a^n, messages highest power first.

use dm_symbol::{Geometry, deinterleave};

/// Antilog table, doubled so that `GF_EXP[log a + log b]` needs no modulo.
pub const GF_EXP: [u8; 512] = gf_tables().0;
/// Log table; `GF_LOG[0]` is 0 (unused).
pub const GF_LOG: [u8; 256] = gf_tables().1;

const fn gf_tables() -> ([u8; 512], [u8; 256]) {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    let mut x: u32 = 1;
    let mut i = 0;
    while i < 255 {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= 0x12D;
        }
        i += 1;
    }
    while i < 512 {
        exp[i] = exp[i - 255];
        i += 1;
    }
    (exp, log)
}

#[inline]
pub fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    GF_EXP[GF_LOG[a as usize] as usize + GF_LOG[b as usize] as usize]
}

#[inline]
pub fn gf_div(a: u8, b: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    GF_EXP[(GF_LOG[a as usize] as usize + 255 - GF_LOG[b as usize] as usize) % 255]
}

/// `a^p`; like Python, `gf_pow(0, p)` is 1.
#[inline]
pub fn gf_pow(a: u8, p: usize) -> u8 {
    GF_EXP[(GF_LOG[a as usize] as usize * p) % 255]
}

#[inline]
pub fn gf_inv(a: u8) -> u8 {
    GF_EXP[255 - GF_LOG[a as usize] as usize]
}

/// `a^e` for any integer exponent.
#[inline]
fn alpha(e: isize) -> u8 {
    GF_EXP[e.rem_euclid(255) as usize]
}

/// Locator of stream position `p` in a message of length `n`: a^(n-1-p).
#[inline]
fn locator(n: usize, p: usize) -> u8 {
    alpha(n as isize - 1 - p as isize)
}

fn poly_mul(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut r = vec![0u8; p.len() + q.len() - 1];
    for (j, &b) in q.iter().enumerate() {
        if b != 0 {
            for (i, &a) in p.iter().enumerate() {
                if a != 0 {
                    r[i + j] ^= gf_mul(a, b);
                }
            }
        }
    }
    r
}

/// `S_j = msg(a^j)`, j = 1..nsym; `msg` highest power first.
pub fn rs_syndromes(msg: &[u8], nsym: usize) -> Vec<u8> {
    (1..=nsym)
        .map(|j| {
            let x = GF_EXP[j % 255];
            msg.iter().fold(0u8, |s, &c| gf_mul(s, x) ^ c)
        })
        .collect()
}

/// Gaussian elimination in GF(256): `S_j = sum e_k * X_k^j`, j = 1..v.
fn solve_magnitudes(synd: &[u8], positions: &[usize], n: usize) -> Option<Vec<u8>> {
    let v = positions.len();
    let xs: Vec<u8> = positions.iter().map(|&p| locator(n, p)).collect();
    let mut a: Vec<Vec<u8>> = (0..v)
        .map(|j| {
            let mut row: Vec<u8> = xs.iter().map(|&x| gf_pow(x, j + 1)).collect();
            row.push(synd[j]);
            row
        })
        .collect();
    for col in 0..v {
        let piv = (col..v).find(|&r| a[r][col] != 0)?;
        a.swap(col, piv);
        let inv = gf_inv(a[col][col]);
        for c in a[col].iter_mut() {
            *c = gf_mul(*c, inv);
        }
        for r in 0..v {
            if r != col && a[r][col] != 0 {
                let f = a[r][col];
                let (pivot, row) = if r < col {
                    let (lo, hi) = a.split_at_mut(col);
                    (&hi[0], &mut lo[r])
                } else {
                    let (lo, hi) = a.split_at_mut(r);
                    (&lo[col], &mut hi[0])
                };
                for (c, &d) in row.iter_mut().zip(pivot.iter()) {
                    *c ^= gf_mul(f, d);
                }
            }
        }
    }
    Some(a.iter().map(|row| row[v]).collect())
}

/// Correct errors and erasures. Returns `(corrected, error_positions)` where
/// `error_positions` excludes erasures that turned out to be right, or `None`.
/// `synd`: precomputed syndromes of `msg` (they do not depend on the erasures).
pub fn rs_correct(
    msg: &[u8],
    nsym: usize,
    erasures: &[usize],
    synd: Option<&[u8]>,
) -> Option<(Vec<u8>, Vec<usize>)> {
    let n = msg.len();
    if erasures.len() > nsym {
        return None;
    }
    let owned;
    let synd = match synd {
        Some(s) => s,
        None => {
            owned = rs_syndromes(msg, nsym);
            &owned
        }
    };
    if synd.iter().all(|&s| s == 0) {
        return Some((msg.to_vec(), Vec::new()));
    }
    // erasure locator Gamma(x) = prod (1 - X_i x), stored highest power first
    let mut gamma = vec![1u8];
    for &p in erasures {
        gamma = poly_mul(&gamma, &[locator(n, p), 1]);
    }
    // Forney (modified) syndromes T = Gamma * S, coefficients f .. nsym-1
    let mut t = vec![0u8; nsym];
    for (i, &gc) in gamma.iter().rev().enumerate() {
        if gc != 0 {
            for (j, &sc) in synd.iter().enumerate() {
                if i + j < nsym && sc != 0 {
                    t[i + j] ^= gf_mul(gc, sc);
                }
            }
        }
    }
    let f = erasures.len();
    let seq = &t[f..];
    // Berlekamp-Massey on the modified syndromes (lowest power first)
    let (mut c_poly, mut b_poly) = (vec![1u8], vec![1u8]);
    let (mut l, mut m, mut b) = (0usize, 1usize, 1u8);
    for k in 0..seq.len() {
        let mut d = seq[k];
        for i in 1..=l {
            if i < c_poly.len() && k >= i {
                d ^= gf_mul(c_poly[i], seq[k - i]);
            }
        }
        if d == 0 {
            m += 1;
            continue;
        }
        let coef = gf_div(d, b);
        let tmp = c_poly.clone();
        let shifted_len = m + b_poly.len();
        if shifted_len > c_poly.len() {
            c_poly.resize(shifted_len, 0);
        }
        for (i, &c) in b_poly.iter().enumerate() {
            c_poly[m + i] ^= gf_mul(coef, c);
        }
        if 2 * l <= k {
            l = k + 1 - l;
            b_poly = tmp;
            b = d;
            m = 1;
        } else {
            m += 1;
        }
    }
    while c_poly.len() > 1 && c_poly[c_poly.len() - 1] == 0 {
        c_poly.pop();
    }
    let n_err = c_poly.len() - 1;
    if 2 * n_err + f > nsym {
        return None;
    }
    // Chien search: root at X^-1 means error at position p with X = a^(n-1-p)
    let mut err_pos = Vec::new();
    if n_err > 0 {
        for p in 0..n {
            let e = (n - 1 - p) % 255;
            let mut s = 0u8;
            for (k, &c) in c_poly.iter().enumerate() {
                if c != 0 {
                    s ^= GF_EXP[(GF_LOG[c as usize] as usize + (255 - (k * e) % 255)) % 255];
                }
            }
            if s == 0 {
                err_pos.push(p);
            }
        }
        if err_pos.len() != n_err {
            return None;
        }
    }
    let mut positions: Vec<usize> = erasures.iter().chain(err_pos.iter()).copied().collect();
    positions.sort_unstable();
    positions.dedup();
    let mags = solve_magnitudes(synd, &positions, n)?;
    let mut out = msg.to_vec();
    for (&p, &e) in positions.iter().zip(mags.iter()) {
        out[p] ^= e;
    }
    if rs_syndromes(&out, nsym).iter().any(|&s| s != 0) {
        return None;
    }
    let changed = positions
        .into_iter()
        .filter(|&p| out[p] != msg[p])
        .collect();
    Some((out, changed))
}

/// Result of decoding all RS blocks of a symbol.
#[derive(Debug, Clone, PartialEq)]
pub struct RSResult {
    pub data_cw: Vec<u8>,
    pub all_cw: Vec<u8>,
    /// Codewords actually changed per block.
    pub block_errors: Vec<usize>,
    /// Erasures declared per block.
    pub block_erasures: Vec<usize>,
    pub corrected_positions: Vec<usize>,
    /// Unused error correction (0..1).
    pub uec: f64,
}

/// Default share of the check codewords that may be declared erasures.
pub const MAX_ERASURE_FRAC: f64 = 0.75;

/// [`rs_decode_symbol_frac`] with [`MAX_ERASURE_FRAC`].
pub fn rs_decode_symbol(cw: &[u8], conf: &[f64], g: &Geometry) -> Option<RSResult> {
    rs_decode_symbol_frac(cw, conf, g, MAX_ERASURE_FRAC)
}

/// Decode all blocks; on failure retry with the least confident codewords
/// declared as erasures (2 per step). Erasure decoding can "correct" garbage
/// into a valid but wrong codeword, so a reserve of unused check codewords
/// (2e + f <= n - margin) is required whenever erasures are used.
/// `conf` is indexed by stream position.
pub fn rs_decode_symbol_frac(
    cw: &[u8],
    conf: &[f64],
    g: &Geometry,
    max_erasure_frac: f64,
) -> Option<RSResult> {
    let mut fixed = cw.to_vec();
    let (mut errs, mut eras, mut changed_all) = (Vec::new(), Vec::new(), Vec::new());
    let mut uec = f64::INFINITY;
    let margin = if g.n_ecc < 20 { 2 } else { 3 };
    let max_f = (g.n_ecc as f64 * max_erasure_frac) as usize;
    for (pos, block) in deinterleave(cw, g) {
        // stable sort, ties keep stream order (Python `sorted`)
        let mut order: Vec<usize> = (0..block.len()).collect();
        order.sort_by(|&a, &b| {
            conf[pos[a]]
                .partial_cmp(&conf[pos[b]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let synd = rs_syndromes(&block, g.n_ecc); // same for every erasure set
        let mut ok = None;
        for f in (0..=max_f).step_by(2) {
            let erased = &order[..f.min(order.len())];
            let Some(r) = rs_correct(&block, g.n_ecc, erased, Some(&synd)) else {
                continue;
            };
            let n_err = r.1.iter().filter(|p| !erased.contains(p)).count();
            if f > 0 && 2 * n_err + f > g.n_ecc - margin {
                continue;
            }
            ok = Some((r, f, n_err));
            break;
        }
        let ((corr, changed), f, n_err) = ok?;
        for (k, &p) in pos.iter().enumerate() {
            fixed[p] = corr[k];
        }
        errs.push(changed.len());
        eras.push(f);
        changed_all.extend(changed.iter().map(|&k| pos[k]));
        let used = (2 * n_err + (changed.len() - n_err)) as f64;
        uec = uec.min(1.0 - used / g.n_ecc as f64);
    }
    changed_all.sort_unstable();
    Some(RSResult {
        data_cw: fixed[..g.n_data].to_vec(),
        all_cw: fixed,
        block_errors: errs,
        block_erasures: eras,
        corrected_positions: changed_all,
        uec,
    })
}
