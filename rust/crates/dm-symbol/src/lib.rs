//! Data Matrix symbol tables, placement map and geometry (from datamatrix_gen.py).
//!
//! Port of `SQUARE_SYMBOLS`, `RECT_SYMBOLS`, `placement_map` and `build_matrix`
//! (generator) and `DMRE_SYMBOLS`, `Geometry`, `geometry`, `deinterleave`
//! (reader).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// ECC 200 symbol attributes (ISO/IEC 16022, DMRE: ISO/IEC 21471).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Symbol {
    /// Full symbol size in modules (incl. finder / clock track).
    pub rows: usize,
    pub cols: usize,
    /// Module size of one data region (excl. finder).
    pub dregr: usize,
    pub dregc: usize,
    /// Number of data regions vertically / horizontally.
    pub regr: usize,
    pub regc: usize,
    /// Total data codeword capacity.
    pub data: usize,
    /// Error correction codewords per RS block.
    pub eccblk: usize,
    /// Number of interleaved RS blocks.
    pub blocks: usize,
}

impl Symbol {
    /// `"RxC"`, as in the report and `--expect`.
    pub fn name(&self) -> String {
        format!("{}x{}", self.rows, self.cols)
    }

    /// Data plus error correction codewords.
    pub fn n_codewords(&self) -> usize {
        self.data + self.eccblk * self.blocks
    }

    /// The Python tuple order: rows cols dregr dregc regr regc data eccblk blocks.
    pub fn as_tuple(&self) -> [usize; 9] {
        [
            self.rows,
            self.cols,
            self.dregr,
            self.dregc,
            self.regr,
            self.regc,
            self.data,
            self.eccblk,
            self.blocks,
        ]
    }
}

#[allow(clippy::too_many_arguments)]
const fn sym(
    rows: usize,
    cols: usize,
    dregr: usize,
    dregc: usize,
    regr: usize,
    regc: usize,
    data: usize,
    eccblk: usize,
    blocks: usize,
) -> Symbol {
    Symbol {
        rows,
        cols,
        dregr,
        dregc,
        regr,
        regc,
        data,
        eccblk,
        blocks,
    }
}

#[rustfmt::skip]
pub const SQUARE_SYMBOLS: [Symbol; 24] = [
    //  rows cols dregr dregc regr regc  data eccblk blocks
    sym(  10,  10,   8,    8,   1,   1,      3,     5,   1),
    sym(  12,  12,  10,   10,   1,   1,      5,     7,   1),
    sym(  14,  14,  12,   12,   1,   1,      8,    10,   1),
    sym(  16,  16,  14,   14,   1,   1,     12,    12,   1),
    sym(  18,  18,  16,   16,   1,   1,     18,    14,   1),
    sym(  20,  20,  18,   18,   1,   1,     22,    18,   1),
    sym(  22,  22,  20,   20,   1,   1,     30,    20,   1),
    sym(  24,  24,  22,   22,   1,   1,     36,    24,   1),
    sym(  26,  26,  24,   24,   1,   1,     44,    28,   1),
    sym(  32,  32,  14,   14,   2,   2,     62,    36,   1),
    sym(  36,  36,  16,   16,   2,   2,     86,    42,   1),
    sym(  40,  40,  18,   18,   2,   2,    114,    48,   1),
    sym(  44,  44,  20,   20,   2,   2,    144,    56,   1),
    sym(  48,  48,  22,   22,   2,   2,    174,    68,   1),
    sym(  52,  52,  24,   24,   2,   2,    204,    42,   2),
    sym(  64,  64,  14,   14,   4,   4,    280,    56,   2),
    sym(  72,  72,  16,   16,   4,   4,    368,    36,   4),
    sym(  80,  80,  18,   18,   4,   4,    456,    48,   4),
    sym(  88,  88,  20,   20,   4,   4,    576,    56,   4),
    sym(  96,  96,  22,   22,   4,   4,    696,    68,   4),
    sym( 104, 104,  24,   24,   4,   4,    816,    56,   6),
    sym( 120, 120,  18,   18,   6,   6,   1050,    68,   6),
    sym( 132, 132,  20,   20,   6,   6,   1304,    62,   8),
    sym( 144, 144,  22,   22,   6,   6,   1558,    62,  10),
];

#[rustfmt::skip]
pub const RECT_SYMBOLS: [Symbol; 6] = [
    sym(   8,  18,   6,   16,   1,   1,      5,     7,   1),
    sym(   8,  32,   6,   14,   1,   2,     10,    11,   1),
    sym(  12,  26,  10,   24,   1,   1,     16,    14,   1),
    sym(  12,  36,  10,   16,   1,   2,     22,    18,   1),
    sym(  16,  36,  14,   16,   1,   2,     32,    24,   1),
    sym(  16,  48,  14,   22,   1,   2,     49,    28,   1),
];

/// DMRE - Data Matrix Rectangular Extension (ISO/IEC 21471), same layout
/// rules as ECC 200.
#[rustfmt::skip]
pub const DMRE_SYMBOLS: [Symbol; 18] = [
    sym(8, 48, 6, 22, 1, 2, 18, 15, 1), sym(8, 64, 6, 14, 1, 4, 24, 18, 1),
    sym(8, 80, 6, 18, 1, 4, 32, 22, 1), sym(8, 96, 6, 22, 1, 4, 38, 28, 1),
    sym(8, 120, 6, 18, 1, 6, 49, 32, 1), sym(8, 144, 6, 22, 1, 6, 63, 36, 1),
    sym(12, 64, 10, 14, 1, 4, 43, 27, 1), sym(12, 88, 10, 20, 1, 4, 64, 36, 1),
    sym(16, 64, 14, 14, 1, 4, 62, 36, 1), sym(20, 36, 18, 16, 1, 2, 44, 28, 1),
    sym(20, 44, 18, 20, 1, 2, 56, 34, 1), sym(20, 64, 18, 14, 1, 4, 84, 42, 1),
    sym(22, 48, 20, 22, 1, 2, 72, 38, 1), sym(24, 48, 22, 22, 1, 2, 80, 41, 1),
    sym(24, 64, 22, 14, 1, 4, 108, 46, 1), sym(26, 40, 24, 18, 1, 2, 70, 38, 1),
    sym(26, 48, 24, 22, 1, 2, 90, 42, 1), sym(26, 64, 24, 14, 1, 4, 118, 50, 1),
];

/// `ALL_SYMBOLS` in the reader: square, rectangular, DMRE, in that order.
pub fn all_symbols() -> impl Iterator<Item = &'static Symbol> {
    SQUARE_SYMBOLS
        .iter()
        .chain(RECT_SYMBOLS.iter())
        .chain(DMRE_SYMBOLS.iter())
}

/// One placement cell: `(codeword number, bit)`; codeword 0 is the fixed
/// corner pattern. Bit 1 is the most significant bit.
pub type Cell = (usize, u8);

/// ECC 200 module placement (ISO/IEC 16022 Annex F) of an `nrow` x `ncol`
/// mapping matrix, row-major. `None` where no codeword bit is placed.
pub fn placement_map(nrow: usize, ncol: usize) -> Vec<Option<Cell>> {
    Placer {
        arr: vec![None; nrow * ncol],
        nrow: nrow as isize,
        ncol: ncol as isize,
    }
    .run()
}

struct Placer {
    arr: Vec<Option<Cell>>,
    nrow: isize,
    ncol: isize,
}

impl Placer {
    fn free(&self, row: isize, col: isize) -> bool {
        let idx = row * self.ncol + col;
        idx >= 0 && (idx as usize) < self.arr.len() && self.arr[idx as usize].is_none()
    }

    fn module(&mut self, mut row: isize, mut col: isize, chrn: usize, bit: u8) {
        let (nrow, ncol) = (self.nrow, self.ncol);
        if row < 0 {
            row += nrow;
            col += 4 - ((nrow + 4) % 8);
        }
        if col < 0 {
            col += ncol;
            row += 4 - ((ncol + 4) % 8);
        }
        if row >= nrow {
            // DMRE (ISO/IEC 21471) wrap
            row -= nrow;
        }
        // Python list indexing: a negative index counts from the end
        let mut idx = row * ncol + col;
        if idx < 0 {
            idx += self.arr.len() as isize;
        }
        self.arr[idx as usize] = Some((chrn, bit));
    }

    fn utah(&mut self, row: isize, col: isize, chrn: usize) {
        self.module(row - 2, col - 2, chrn, 1);
        self.module(row - 2, col - 1, chrn, 2);
        self.module(row - 1, col - 2, chrn, 3);
        self.module(row - 1, col - 1, chrn, 4);
        self.module(row - 1, col, chrn, 5);
        self.module(row, col - 2, chrn, 6);
        self.module(row, col - 1, chrn, 7);
        self.module(row, col, chrn, 8);
    }

    fn corner(&mut self, cells: [(isize, isize); 8], chrn: usize) {
        for (k, (r, c)) in cells.into_iter().enumerate() {
            self.module(r, c, chrn, k as u8 + 1);
        }
    }

    fn run(mut self) -> Vec<Option<Cell>> {
        let (nrow, ncol) = (self.nrow, self.ncol);
        let mut chrn = 1;
        let (mut row, mut col) = (4isize, 0isize);
        loop {
            if row == nrow && col == 0 {
                #[rustfmt::skip]
                self.corner([(nrow - 1, 0), (nrow - 1, 1), (nrow - 1, 2), (0, ncol - 2),
                             (0, ncol - 1), (1, ncol - 1), (2, ncol - 1), (3, ncol - 1)], chrn);
                chrn += 1;
            } else if row == nrow - 2 && col == 0 && ncol % 4 != 0 {
                #[rustfmt::skip]
                self.corner([(nrow - 3, 0), (nrow - 2, 0), (nrow - 1, 0), (0, ncol - 4),
                             (0, ncol - 3), (0, ncol - 2), (0, ncol - 1), (1, ncol - 1)], chrn);
                chrn += 1;
            } else if row == nrow - 2 && col == 0 && ncol % 8 == 4 {
                #[rustfmt::skip]
                self.corner([(nrow - 3, 0), (nrow - 2, 0), (nrow - 1, 0), (0, ncol - 2),
                             (0, ncol - 1), (1, ncol - 1), (2, ncol - 1), (3, ncol - 1)], chrn);
                chrn += 1;
            } else if row == nrow + 4 && col == 2 && ncol % 8 == 0 {
                #[rustfmt::skip]
                self.corner([(nrow - 1, 0), (nrow - 1, ncol - 1), (0, ncol - 3), (0, ncol - 2),
                             (0, ncol - 1), (1, ncol - 3), (1, ncol - 2), (1, ncol - 1)], chrn);
                chrn += 1;
            }

            loop {
                // diagonal, up and right
                if row < nrow && col >= 0 && self.free(row, col) {
                    self.utah(row, col, chrn);
                    chrn += 1;
                }
                row -= 2;
                col += 2;
                if !(row >= 0 && col < ncol) {
                    break;
                }
            }
            row += 1;
            col += 3;

            loop {
                // diagonal, down and left
                if row >= 0 && col < ncol && self.free(row, col) {
                    self.utah(row, col, chrn);
                    chrn += 1;
                }
                row += 2;
                col -= 2;
                if !(row < nrow && col >= 0) {
                    break;
                }
            }
            row += 3;
            col += 1;

            if row >= nrow && col >= ncol {
                break;
            }
        }

        let n = self.arr.len();
        if self.arr[n - 1].is_none() {
            // fixed corner pattern
            self.arr[n - 1] = Some((0, 0));
            self.arr[n - ncol as usize - 2] = Some((0, 0));
        }
        self.arr
    }
}

/// Full symbol (finder, clock track, data) for `codewords`, row-major
/// `rows * cols`, 1 = dark.
pub fn build_matrix(codewords: &[u8], sym: &Symbol) -> Vec<u8> {
    let (mrow, mcol) = (sym.dregr * sym.regr, sym.dregc * sym.regc);
    let place = placement_map(mrow, mcol);

    let mapping: Vec<u8> = place
        .iter()
        .map(|cell| match *cell {
            None => 0,
            Some((0, _)) => 1, // fixed corner pattern module
            Some((chrn, bit)) => (codewords[chrn - 1] >> (8 - bit)) & 1,
        })
        .collect();

    let (rows, cols, dregr, dregc) = (sym.rows, sym.cols, sym.dregr, sym.dregc);
    let mut matrix = vec![0u8; rows * cols];
    for rv in 0..sym.regr {
        for rh in 0..sym.regc {
            let (r0, c0) = (rv * (dregr + 2), rh * (dregc + 2));
            for i in 0..dregr + 2 {
                // left solid + right clock
                matrix[(r0 + i) * cols + c0] = 1;
                matrix[(r0 + i) * cols + c0 + dregc + 1] = u8::from((i + dregc + 1) % 2 == 0);
            }
            for j in 0..dregc + 2 {
                // bottom solid + top clock
                matrix[(r0 + dregr + 1) * cols + c0 + j] = 1;
                matrix[r0 * cols + c0 + j] = u8::from(j % 2 == 0);
            }
            for i in 0..dregr {
                // data
                for j in 0..dregc {
                    matrix[(r0 + 1 + i) * cols + c0 + 1 + j] =
                        mapping[(rv * dregr + i) * mcol + rh * dregc + j];
                }
            }
        }
    }
    matrix
}

/// Per-symbol lookup tables used by the grid engine. All grids are
/// row-major; the padded ones are `(rows + 2) x (cols + 2)`.
#[derive(Debug, Clone)]
pub struct Geometry {
    pub sym: Symbol,
    pub rows: usize,
    pub cols: usize,
    /// True on finder / timing modules.
    pub fixed_mask: Vec<bool>,
    /// Expected value there (1 = dark); the all-zero codeword template elsewhere.
    pub fixed_val: Vec<u8>,
    /// Module row / column of each codeword bit (bit 1 first).
    pub cw_rows: Vec<[usize; 8]>,
    pub cw_cols: Vec<[usize; 8]>,
    pub n_data: usize,
    pub n_ecc: usize,
    pub blocks: usize,
    /// Fixed pattern + 1 module quiet zone ring.
    pub pad_mask: Vec<bool>,
    pub pad_val: Vec<u8>,
    /// Module centres (x, y) of `pad_mask`, row-major order.
    pub fp_uv_qz: Vec<[f64; 2]>,
    /// Expected dark at `fp_uv_qz`.
    pub fp_dark_qz: Vec<bool>,
    /// Same for the fixed pattern only (no ring).
    pub fp_uv_in: Vec<[f64; 2]>,
    pub fp_dark_in: Vec<bool>,
}

impl Geometry {
    pub fn name(&self) -> String {
        format!("{}x{}", self.rows, self.cols)
    }

    fn new(sym: &Symbol) -> Self {
        let Symbol {
            rows,
            cols,
            dregr,
            dregc,
            regr,
            regc,
            data,
            eccblk,
            blocks,
        } = *sym;
        let n_cw = sym.n_codewords();
        let template = build_matrix(&vec![0u8; n_cw], sym);
        let mut mask = vec![false; rows * cols];
        for rv in 0..regr {
            let r0 = rv * (dregr + 2);
            for c in 0..cols {
                mask[r0 * cols + c] = true;
                mask[(r0 + dregr + 1) * cols + c] = true;
            }
        }
        for rh in 0..regc {
            let c0 = rh * (dregc + 2);
            for r in 0..rows {
                mask[r * cols + c0] = true;
                mask[r * cols + c0 + dregc + 1] = true;
            }
        }
        let mcol = dregc * regc;
        let place = placement_map(dregr * regr, mcol);
        let mut cw_rows = vec![[0usize; 8]; n_cw];
        let mut cw_cols = vec![[0usize; 8]; n_cw];
        for (idx, cell) in place.iter().enumerate() {
            let Some((chrn, bit)) = *cell else { continue };
            if chrn == 0 {
                continue;
            }
            let (r, c) = (idx / mcol, idx % mcol);
            let (rv, rh) = (r / dregr, c / dregc);
            cw_rows[chrn - 1][bit as usize - 1] = rv * (dregr + 2) + 1 + r % dregr;
            cw_cols[chrn - 1][bit as usize - 1] = rh * (dregc + 2) + 1 + c % dregc;
        }

        let pcols = cols + 2;
        let mut pad_mask = vec![true; (rows + 2) * pcols];
        let mut pad_val = vec![0u8; (rows + 2) * pcols];
        for r in 0..rows {
            for c in 0..cols {
                pad_mask[(r + 1) * pcols + c + 1] = mask[r * cols + c];
                pad_val[(r + 1) * pcols + c + 1] = template[r * cols + c];
            }
        }
        let (mut fp_uv_qz, mut fp_dark_qz) = (Vec::new(), Vec::new());
        let (mut fp_uv_in, mut fp_dark_in) = (Vec::new(), Vec::new());
        for pr in 0..rows + 2 {
            for pc in 0..pcols {
                if !pad_mask[pr * pcols + pc] {
                    continue;
                }
                let uv = [pc as f64 - 1.0 + 0.5, pr as f64 - 1.0 + 0.5];
                let dark = pad_val[pr * pcols + pc] == 1;
                fp_uv_qz.push(uv);
                fp_dark_qz.push(dark);
                if (1..=rows).contains(&pr) && (1..=cols).contains(&pc) {
                    fp_uv_in.push(uv);
                    fp_dark_in.push(dark);
                }
            }
        }
        Geometry {
            sym: *sym,
            rows,
            cols,
            fixed_mask: mask,
            fixed_val: template,
            cw_rows,
            cw_cols,
            n_data: data,
            n_ecc: eccblk,
            blocks,
            pad_mask,
            pad_val,
            fp_uv_qz,
            fp_dark_qz,
            fp_uv_in,
            fp_dark_in,
        }
    }
}

/// Cached geometry of `sym`; built once per symbol and kept for the process
/// lifetime (as `_GEOM_CACHE`).
pub fn geometry(sym: &Symbol) -> &'static Geometry {
    static CACHE: OnceLock<Mutex<HashMap<Symbol, &'static Geometry>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut map = cache.lock().unwrap_or_else(|e| e.into_inner());
    map.entry(*sym)
        .or_insert_with(|| Box::leak(Box::new(Geometry::new(sym))))
}

/// Per RS block: (positions in the codeword stream, codewords), data then ecc.
pub fn deinterleave(cw: &[u8], g: &Geometry) -> Vec<(Vec<usize>, Vec<u8>)> {
    (0..g.blocks)
        .map(|b| {
            let mut pos: Vec<usize> = (b..g.n_data).step_by(g.blocks).collect();
            pos.extend((0..g.n_ecc).map(|k| g.n_data + k * g.blocks + b));
            let block = pos.iter().map(|&p| cw[p]).collect();
            (pos, block)
        })
        .collect()
}
