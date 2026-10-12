// Pairwise summation derived from NumPy (loops_utils.h.src,
// @TYPE@_pairwise_sum), https://github.com/numpy/numpy, BSD 3-Clause.
// Copyright (c) 2005-2025, NumPy Developers (see NOTICE).
// Modified: ported to Rust.

//! numpy reductions with numpy's rounding. A reduction along a contiguous
//! float32 axis is pairwise (8 accumulators, blocks of 128); along a strided
//! axis (e.g. rows of an F-ordered array) numpy adds element by element.
//! Divisions are float32.

const PW_BLOCKSIZE: usize = 128;

/// `@TYPE@_pairwise_sum` of numpy's `loops_utils.h`.
pub fn sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = 0f32;
        for &x in a {
            res += x;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        sum_f32(&a[..n2]) + sum_f32(&a[n2..])
    }
}

/// Sequential sum (reduction along a strided axis).
pub fn sum_seq_f32(a: &[f32]) -> f32 {
    a.iter().fold(0f32, |s, &x| s + x)
}

/// Summation order of a reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Pairwise,
    Sequential,
}

fn sum_by(a: &[f32], o: Order) -> f32 {
    match o {
        Order::Pairwise => sum_f32(a),
        Order::Sequential => sum_seq_f32(a),
    }
}

/// `a.mean()` (NaN for an empty slice).
pub fn mean_f32(a: &[f32], o: Order) -> f32 {
    sum_by(a, o) / a.len() as f32
}

/// `a.var()` (ddof 0): mean, squared deviations, sum, divide.
pub fn var_f32(a: &[f32], o: Order) -> f32 {
    let m = mean_f32(a, o);
    let sq: Vec<f32> = a
        .iter()
        .map(|&x| {
            let d = x - m;
            d * d
        })
        .collect();
    sum_by(&sq, o) / a.len() as f32
}

/// `np.linalg.norm` of a float32 2-vector (`sqrt(dot)`, no fusion).
pub fn norm2_f32(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// `np.clip(x, lo, hi)`: `min(max(x, lo), hi)` with NaN propagated; `hi`
/// wins when `lo > hi`.
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let y = if x > lo { x } else { lo };
    if y < hi { y } else { hi }
}

/// Least squares `a x = b` for a tall `m x N` matrix (Householder QR).
/// numpy's `lstsq` uses LAPACK `gelsd` (SVD), so results agree to the
/// last few bits only. `None` if `a` is rank deficient.
#[allow(clippy::needless_range_loop)] // matrix index form
pub fn lstsq<const N: usize>(a: &[[f64; N]], b: &[f64]) -> Option<[f64; N]> {
    let m = a.len();
    let mut r: Vec<[f64; N]> = a.to_vec();
    let mut y = b.to_vec();
    for k in 0..N {
        let norm = (k..m).map(|i| r[i][k] * r[i][k]).sum::<f64>().sqrt();
        if norm == 0.0 {
            return None;
        }
        let alpha = if r[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k..m).map(|i| r[i][k]).collect();
        v[0] -= alpha;
        let vv: f64 = v.iter().map(|x| x * x).sum();
        for j in k..N {
            let f = 2.0 * (k..m).map(|i| v[i - k] * r[i][j]).sum::<f64>() / vv;
            for i in k..m {
                r[i][j] -= f * v[i - k];
            }
        }
        let f = 2.0 * (k..m).map(|i| v[i - k] * y[i]).sum::<f64>() / vv;
        for i in k..m {
            y[i] -= f * v[i - k];
        }
    }
    let mut x = [0f64; N];
    for k in (0..N).rev() {
        let s: f64 = (k + 1..N).map(|j| r[k][j] * x[j]).sum();
        x[k] = (y[k] - s) / r[k][k];
    }
    Some(x)
}

/// `np.percentile(a, q)` of float32 values with a Python-number `q`
/// (method "linear"): the weight is a weak Python float, so the
/// interpolation is float32. NaN for an empty slice (numpy raises).
pub fn percentile_f32(a: &[f32], q: f64) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::NAN;
    }
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.total_cmp(y));
    let v = (n - 1) as f64 * (q / 100.0);
    let (lo, hi, t) = if v >= (n - 1) as f64 {
        (n - 1, n - 1, v + 1.0)
    } else {
        let lo = v.floor();
        (lo as usize, lo as usize + 1, v - lo)
    };
    let (a, b) = (s[lo], s[hi]);
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t) as f32
    } else {
        a + d * t as f32
    }
}

/// Python `round(x, digits)` (correctly rounded decimal, ties to even).
pub fn round_to(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}
