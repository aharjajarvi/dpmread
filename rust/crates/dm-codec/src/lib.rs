//! ECC 200 codeword decoder (ASCII, C40, Text, X12, EDIFACT, Base256, ECI, GS1).
//!
//! Port of the reader's `decode_codewords`, `_decode_ctx`, `_decode_edifact`,
//! `Decoded`, `Segment`, `DataError` and `parse_gs1` (ISO/IEC 16022 5.2).
//! Python quirks are kept on purpose: list indexing past the end is
//! [`DataError::Index`] (the reader catches `IndexError` like `DataError`),
//! and a negative table index counts from the end.

use std::fmt;

const C40_BASIC: &[u8] = b" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const TEXT_BASIC: &[u8] = b" 0123456789abcdefghijklmnopqrstuvwxyz";
const SHIFT2: &[u8] = b"!\"#$%&'()*+,-./:;<=>?@[\\]^_";
const TEXT_SHIFT3: &[u8] = b"`ABCDEFGHIJKLMNOPQRSTUVWXYZ{|}~\x7f";
const X12_SET: &[u8] = b"\r*> 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Invalid codeword stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    /// Python `DataError` with its message.
    Invalid(String),
    /// Python `IndexError`: the stream ended inside a field.
    Index,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Invalid(msg) => f.write_str(msg),
            DataError::Index => f.write_str("index out of range"),
        }
    }
}

impl std::error::Error for DataError {}

fn invalid<T>(msg: impl Into<String>) -> Result<T, DataError> {
    Err(DataError::Invalid(msg.into()))
}

/// Python `seq[idx]` on a byte table, negative index from the end.
fn py_index(table: &[u8], idx: i64) -> Result<u8, DataError> {
    let i = if idx < 0 {
        idx + table.len() as i64
    } else {
        idx
    };
    if i < 0 || i >= table.len() as i64 {
        return Err(DataError::Index);
    }
    Ok(table[i as usize])
}

/// One encodation segment of the decoded stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// ASCII, C40, TEXT, X12, EDIFACT or BASE256.
    pub mode: &'static str,
    pub cw_start: usize,
    /// Exclusive.
    pub cw_end: usize,
    pub data: Vec<u8>,
    pub note: String,
}

impl Segment {
    fn new(mode: &'static str, cw_start: usize) -> Self {
        Segment {
            mode,
            cw_start,
            cw_end: 0,
            data: Vec::new(),
            note: String::new(),
        }
    }
}

/// Decoded data codewords.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Decoded {
    pub data: Vec<u8>,
    pub segments: Vec<Segment>,
    /// Codewords up to and incl. first PAD.
    pub payload_cw: usize,
    pub pad_cw: usize,
    pub gs1: bool,
    pub eci: Vec<i64>,
    /// Byte offset in `data` where each ECI of `eci` starts.
    pub eci_pos: Vec<usize>,
    /// `"05"` / `"06"` for a macro header, else empty (Python `macro`).
    pub macro_kind: String,
    pub structured_append: String,
    pub reader_programming: bool,
    pub warnings: Vec<String>,
}

fn unrand255(v: i64, pos: usize) -> i64 {
    let t = v - (((149 * pos as i64) % 255) + 1);
    if t >= 0 { t } else { t + 256 }
}

fn unrand253(v: i64, pos: usize) -> i64 {
    let t = v - (((149 * pos as i64) % 253) + 1);
    if t >= 1 { t } else { t + 254 }
}

struct Decoder<'a> {
    cw: &'a [u8],
    out: Vec<u8>,
    segs: Vec<Segment>,
    upper: bool,
}

impl Decoder<'_> {
    fn get(&self, i: usize) -> Result<i64, DataError> {
        self.cw.get(i).map(|&c| c as i64).ok_or(DataError::Index)
    }

    fn seg(&mut self) -> &mut Segment {
        let last = self.segs.len() - 1;
        &mut self.segs[last]
    }

    fn emit(&mut self, mut b: i64) {
        if self.upper {
            b += 128;
            self.upper = false;
        }
        let v = (b & 0xFF) as u8;
        self.out.push(v);
        self.seg().data.push(v);
    }

    fn new_seg(&mut self, mode: &'static str, start: usize) {
        let seg = self.seg();
        if seg.cw_end == 0 {
            seg.cw_end = start;
        }
        self.segs.push(Segment::new(mode, start));
    }

    fn add_note(&mut self, note: &str) {
        let seg = self.seg();
        seg.note = format!("{} {note}", seg.note).trim().to_string();
    }

    /// C40 / Text / X12 values from `i`; returns the index after the run.
    fn decode_ctx(&mut self, mut i: usize, mode: &'static str) -> Result<usize, DataError> {
        let n = self.cw.len();
        let mut shift = 0i64;
        while i < n {
            if self.cw[i] == 254 {
                // unlatch
                return Ok(i + 1);
            }
            if i + 1 >= n {
                // single cw left: ASCII
                return Ok(i);
            }
            let v = self.cw[i] as i64 * 256 + self.cw[i + 1] as i64 - 1;
            i += 2;
            for val in [
                v.div_euclid(1600),
                v.div_euclid(40).rem_euclid(40),
                v.rem_euclid(40),
            ] {
                if mode == "X12" {
                    if val >= X12_SET.len() as i64 {
                        return invalid("invalid X12 value");
                    }
                    self.emit(py_index(X12_SET, val)? as i64);
                    continue;
                }
                match shift {
                    0 => {
                        if val < 3 {
                            shift = val + 1;
                        } else {
                            let basic = if mode == "C40" { C40_BASIC } else { TEXT_BASIC };
                            self.emit(py_index(basic, val - 3)? as i64);
                        }
                    }
                    1 => {
                        self.emit(val);
                        shift = 0;
                    }
                    2 => {
                        if val < 27 {
                            self.emit(py_index(SHIFT2, val)? as i64);
                        } else if val == 27 {
                            self.emit(29); // FNC1 -> <GS>
                        } else if val == 30 {
                            self.upper = true;
                        } else {
                            return invalid(format!("invalid {mode} shift-2 value {val}"));
                        }
                        shift = 0;
                    }
                    _ => {
                        let b = if mode == "C40" {
                            96 + val
                        } else {
                            py_index(TEXT_SHIFT3, val)? as i64
                        };
                        self.emit(b);
                        shift = 0;
                    }
                }
            }
            if i < n && self.cw[i] == 254 {
                return Ok(i + 1);
            }
        }
        Ok(i)
    }

    /// EDIFACT from `i`; returns the index after the run.
    fn decode_edifact(&mut self, mut i: usize) -> usize {
        let n = self.cw.len();
        while i < n {
            if n - i <= 2 {
                // implicit return
                return i;
            }
            let v =
                (self.cw[i] as u32) << 16 | (self.cw[i + 1] as u32) << 8 | self.cw[i + 2] as u32;
            for (k, sh) in [18, 12, 6, 0].into_iter().enumerate() {
                let e = (v >> sh) & 0x3F;
                if e == 0x1F {
                    // unlatch: rest of its codeword is filler
                    return i + 3.min((6 * (k + 1)).div_ceil(8));
                }
                self.emit(if e < 0x20 { e | 0x40 } else { e } as i64);
            }
            i += 3;
        }
        i
    }
}

/// Decode the data codewords of a symbol (after error correction).
pub fn decode_codewords(cw: &[u8]) -> Result<Decoded, DataError> {
    let n = cw.len();
    let mut d = Decoder {
        cw,
        out: Vec::new(),
        segs: vec![Segment::new("ASCII", 0)],
        upper: false,
    };
    let mut res = Decoded {
        data: Vec::new(),
        segments: Vec::new(),
        payload_cw: n,
        pad_cw: 0,
        gs1: false,
        eci: Vec::new(),
        eci_pos: Vec::new(),
        macro_kind: String::new(),
        structured_append: String::new(),
        reader_programming: false,
        warnings: Vec::new(),
    };
    let mut i = 0usize;
    while i < n {
        let c = cw[i] as i64;
        match c {
            0 => return invalid(format!("codeword 0 at {} is invalid", i + 1)),
            1..=128 => {
                d.emit(c - 1);
                i += 1;
            }
            129 => {
                // PAD: end of data
                res.payload_cw = i;
                res.pad_cw = n - i;
                for (k, &c) in cw.iter().enumerate().skip(i + 1) {
                    if unrand253(c as i64, k + 1) != 129 {
                        res.warnings
                            .push(format!("bad padding codeword at {}", k + 1));
                        break;
                    }
                }
                break;
            }
            130..=229 => {
                let v = c - 130;
                d.emit(48 + v / 10);
                d.emit(48 + v % 10);
                i += 1;
            }
            230 | 239 | 238 => {
                // C40 / Text / X12
                let mode = match c {
                    230 => "C40",
                    239 => "TEXT",
                    _ => "X12",
                };
                d.new_seg(mode, i);
                i = d.decode_ctx(i + 1, mode)?;
                d.new_seg("ASCII", i);
            }
            231 => {
                // Base 256
                d.new_seg("BASE256", i);
                i += 1;
                if i >= n {
                    return invalid("BASE256 length missing");
                }
                let d1 = unrand255(cw[i] as i64, i + 1);
                i += 1;
                let length = if d1 == 0 {
                    (n - i) as i64
                } else if d1 < 250 {
                    d1
                } else {
                    if i >= n {
                        return invalid("BASE256 length truncated");
                    }
                    let l = 250 * (d1 - 249) + unrand255(cw[i] as i64, i + 1);
                    i += 1;
                    l
                };
                if i as i64 + length > n as i64 {
                    return invalid("BASE256 field runs past the symbol");
                }
                for _ in 0..length {
                    d.emit(unrand255(cw[i] as i64, i + 1));
                    i += 1;
                }
                d.new_seg("ASCII", i);
            }
            232 => {
                // FNC1
                if i == 0 || (i == 1 && !res.structured_append.is_empty()) {
                    res.gs1 = true;
                    d.add_note("FNC1(GS1)");
                } else {
                    d.emit(29); // <GS> separator
                }
                i += 1;
            }
            233 => {
                // Structured append
                if i + 3 > n {
                    return invalid("structured append truncated");
                }
                let seq = d.get(i + 1)?;
                let (f1, f2) = (d.get(i + 2)?, d.get(i + 3)?);
                res.structured_append = format!(
                    "symbol {} of {}, file id {f1}/{f2}",
                    (seq >> 4) + 1,
                    17 - (seq & 15)
                );
                i += 4;
            }
            234 => {
                res.reader_programming = true;
                i += 1;
            }
            235 => {
                d.upper = true;
                i += 1;
            }
            236 | 237 => {
                res.macro_kind = if c == 236 { "05" } else { "06" }.to_string();
                let header = [b"[)>\x1e".as_slice(), res.macro_kind.as_bytes(), b"\x1d"].concat();
                for b in header {
                    d.emit(b as i64);
                }
                i += 1;
            }
            240 => {
                // EDIFACT
                d.new_seg("EDIFACT", i);
                i = d.decode_edifact(i + 1);
                d.new_seg("ASCII", i);
            }
            241 => {
                // ECI
                let c1 = if i + 1 < n { cw[i + 1] as i64 } else { 0 };
                let val;
                if c1 <= 127 {
                    val = c1 - 1;
                    i += 2;
                } else if c1 <= 191 {
                    val = (c1 - 128) * 254 + (d.get(i + 2)? - 1) + 127;
                    i += 3;
                } else {
                    val = (c1 - 192) * 64516
                        + (d.get(i + 2)? - 1) * 254
                        + (d.get(i + 3)? - 1)
                        + 16383;
                    i += 4;
                }
                res.eci.push(val);
                res.eci_pos.push(d.out.len());
                d.add_note(&format!("ECI {val:06}"));
            }
            254 => i += 1, // stray unlatch, ignore
            _ => return invalid(format!("invalid codeword {c} at {}", i + 1)),
        }
    }
    let payload_cw = res.payload_cw;
    let seg = d.seg();
    if seg.cw_end == 0 {
        seg.cw_end = i.min(payload_cw);
    }
    if !res.macro_kind.is_empty() {
        d.out.extend_from_slice(b"\x1e\x04");
    }
    res.data = d.out;
    res.segments = d
        .segs
        .into_iter()
        .filter(|s| s.cw_end > s.cw_start || !s.data.is_empty())
        .collect();
    Ok(res)
}

// --------------------------------------------------------------------------
// GS1 element strings (subset of the AI table, enough for common labels)
// --------------------------------------------------------------------------

/// `(ai, fixed length or None, title)`.
#[rustfmt::skip]
pub const GS1_AI: &[(&str, Option<usize>, &str)] = &[
    ("00", Some(18), "SSCC"), ("01", Some(14), "GTIN"), ("02", Some(14), "CONTENT"),
    ("10", None, "BATCH/LOT"), ("11", Some(6), "PROD DATE"), ("12", Some(6), "DUE DATE"),
    ("13", Some(6), "PACK DATE"), ("15", Some(6), "BEST BEFORE"), ("16", Some(6), "SELL BY"),
    ("17", Some(6), "USE BY"), ("20", Some(2), "VARIANT"), ("21", None, "SERIAL"),
    ("22", None, "CPV"), ("240", None, "ADDITIONAL ID"), ("241", None, "CUST PART NO"),
    ("250", None, "SECONDARY SERIAL"), ("30", None, "VAR COUNT"), ("37", None, "COUNT"),
    ("400", None, "ORDER NUMBER"), ("410", Some(13), "SHIP TO LOC"), ("414", Some(13), "LOC NO"),
    ("420", None, "SHIP TO POST"), ("7003", Some(10), "EXPIRY TIME"), ("8004", None, "GIAI"),
    ("90", None, "INTERNAL"), ("91", None, "INTERNAL"), ("92", None, "INTERNAL"),
    ("93", None, "INTERNAL"), ("94", None, "INTERNAL"), ("95", None, "INTERNAL"),
    ("96", None, "INTERNAL"), ("97", None, "INTERNAL"), ("98", None, "INTERNAL"),
    ("99", None, "INTERNAL"),
];

const MEASURE_PREFIX: [&str; 6] = ["31", "32", "33", "34", "35", "36"];

fn gs1_lookup(ai: &str) -> Option<(Option<usize>, &'static str)> {
    GS1_AI
        .iter()
        .find(|(k, _, _)| *k == ai)
        .map(|&(_, fixed, title)| (fixed, title))
}

/// GS1 element string -> `(ai, title, value)`; `text` without the leading FNC1.
pub fn parse_gs1(text: &str) -> Vec<(String, &'static str, String)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    // Python slice text[a:b], clamped
    let slice = |a: usize, b: usize| -> String { chars[a.min(n)..b.min(n)].iter().collect() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if chars[i] == '\x1d' {
            i += 1;
            continue;
        }
        let mut ai = None;
        for ln in [2, 3, 4] {
            let cand = slice(i, i + ln);
            if gs1_lookup(&cand).is_some() {
                ai = Some(cand);
                break;
            }
            let head: String = cand.chars().take(3).collect();
            if ln == 4 && ["310", "320", "330", "390", "392", "393"].contains(&head.as_str()) {
                ai = Some(cand);
            }
        }
        let (ai, fixed, title) = match ai {
            Some(ai) => {
                let (fixed, title) = gs1_lookup(&ai).unwrap_or((None, "VALUE"));
                (ai, fixed, title)
            }
            None => {
                // measures 31nn-36nn etc.
                if MEASURE_PREFIX.contains(&slice(i, i + 2).as_str()) {
                    (slice(i, i + 4), Some(6), "MEASURE")
                } else {
                    (slice(i, i + 2), None, "?")
                }
            }
        };
        i += ai.chars().count();
        let val;
        if let Some(fixed) = fixed {
            val = slice(i, i + fixed);
            i += fixed;
        } else {
            let j = (i.min(n)..n).find(|&k| chars[k] == '\x1d').unwrap_or(n);
            val = slice(i, j);
            i = j;
        }
        out.push((ai, title, val));
    }
    out
}
