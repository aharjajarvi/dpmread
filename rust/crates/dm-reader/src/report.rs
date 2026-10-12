//! Report output: `to_dict` (JSON), `print_report` (text), `printable`,
//! `matrix_art`, `hexrows` (reader 2048–2185).

use dm_codec::{Decoded, parse_gs1};
use dm_grid::nearmiss::round2;
use dm_grid::{OUTSIDE_BUSY, Quality, STRONG_FIT};
use dm_symbol::DMRE_SYMBOLS;
use serde_json::{Map, Value, json};

use crate::charset::to_text;
use crate::read::{ReadNearMiss, ReadResult};
use crate::zxing::ZxInfo;

/// Bytes as readable text: control characters as `<GS>`, `<1F>` …; data
/// with bytes ≥ 128 through `to_text` (per ECI segment when `dec` is given).
pub fn printable(data: &[u8], dec: Option<&Decoded>) -> String {
    if !data.iter().all(|&b| b < 128) {
        return to_text(data, dec).0;
    }
    data.iter()
        .map(|&b| match b {
            29 => "<GS>".to_string(),
            30 => "<RS>".to_string(),
            4 => "<EOT>".to_string(),
            13 => "<CR>".to_string(),
            10 => "<LF>".to_string(),
            9 => "<TAB>".to_string(),
            b if b < 32 || b == 127 => format!("<{b:02X}>"),
            b => char::from(b).to_string(),
        })
        .collect()
}

/// Module matrix (`rows x cols`, 1 = dark) as half-block characters, two
/// module rows per line.
pub fn matrix_art(bits: &[u8], rows: usize, cols: usize) -> String {
    let mut lines = Vec::new();
    for r in (0..rows).step_by(2) {
        let mut line = String::from("    ");
        for c in 0..cols {
            let t = bits[r * cols + c] != 0;
            let b = r + 1 < rows && bits[(r + 1) * cols + c] != 0;
            line.push(match (t, b) {
                (true, true) => '█',
                (true, false) => '▀',
                (false, true) => '▄',
                (false, false) => ' ',
            });
        }
        lines.push(line);
    }
    lines.join("\n")
}

/// Codewords as hex, 16 per row with a 1-based index; `marks` (0-based)
/// get a `*`.
pub fn hexrows(cws: &[u8], marks: &[usize], per: usize) -> String {
    cws.chunks(per)
        .enumerate()
        .map(|(row, chunk)| {
            let i = row * per;
            let cells: Vec<String> = chunk
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    let mark = if marks.contains(&(i + k)) { '*' } else { ' ' };
                    format!("{c:02X}{mark}")
                })
                .collect();
            format!("    {:4}: {}", i + 1, cells.join(" "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `data.hex(" ").upper()`.
pub fn hex_spaced(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn zx_dict(zx: Option<&ZxInfo>) -> Value {
    match zx {
        None => json!({}),
        Some(z) => json!({
            "binarizer": z.binarizer,
            "symbology_identifier": z.symbology_identifier,
            "content_type": z.content_type,
            "orientation_deg": z.orientation_deg,
            "ec_level": z.ec_level,
        }),
    }
}

fn near_miss_dict(n: &ReadNearMiss) -> Value {
    let nm = &n.nm;
    let mut d = Map::new();
    d.insert("score".into(), json!(nm.score));
    d.insert("size".into(), json!(nm.size));
    d.insert("qz".into(), json!(nm.qz));
    d.insert("modules_px".into(), json!(nm.modules_px));
    d.insert(
        "fixed_pattern_module_errors".into(),
        json!(nm.fixed_pattern_module_errors),
    );
    d.insert("size_agree".into(), json!(nm.size_agree));
    d.insert("outside_dark".into(), json!(nm.outside_dark));
    d.insert("measured_size".into(), json!(nm.measured_size));
    // warped-grid near miss only
    if let Some(p) = nm.pitch_drift_modules {
        d.insert("pitch_drift_modules".into(), json!(p));
    }
    d.insert("modules_px_orig".into(), json!(n.modules_px_orig));
    d.insert("variant".into(), json!(n.variant));
    Value::Object(d)
}

pub fn quality_dict(q: &Quality) -> Value {
    json!({
        "symbol_contrast_pct": q.symbol_contrast_pct,
        "symbol_contrast_grade": q.symbol_contrast_grade.to_string(),
        "modulation_p5": q.modulation_p5,
        "modulation_grade": q.modulation_grade.to_string(),
        "axial_nonuniformity": q.axial_nonuniformity,
        "axial_nonuniformity_grade": q.axial_nonuniformity_grade.to_string(),
        "unused_error_correction": q.unused_error_correction,
        "unused_error_correction_grade": q.unused_error_correction_grade.to_string(),
        "fixed_pattern_damage_pct": q.fixed_pattern_damage_pct,
        "fixed_pattern_damage_grade": q.fixed_pattern_damage_grade.to_string(),
        "module_size_px": q.module_size_px,
        "symbol_size_px": [q.symbol_size_px.0, q.symbol_size_px.1],
        "rotation_deg": q.rotation_deg,
        "skew_deg": q.skew_deg,
        "pitch_drift_modules": q.pitch_drift_modules,
        "dark_reflectance_mean": q.dark_reflectance_mean,
        "light_reflectance_mean": q.light_reflectance_mean,
        "corners_px": q.corners_px,
        "overall_grade_estimate": q.overall_grade_estimate.to_string(),
    })
}

/// `to_dict`: the JSON object of one result, keys and nesting as Python's.
pub fn to_dict(res: &ReadResult) -> Value {
    let mut d = Map::new();
    d.insert("file".into(), json!(res.path.to_string_lossy()));
    d.insert("decoded".into(), json!(!res.data.is_empty()));
    d.insert("elapsed_s".into(), json!(round2(res.elapsed)));
    d.insert("attempts".into(), json!(res.attempts));
    if res.data.is_empty() {
        if let Some(n) = &res.near_miss {
            d.insert("near_miss".into(), near_miss_dict(n));
        }
        return Value::Object(d);
    }
    let (text, charset) = to_text(&res.data, res.text_dec());
    d.insert("text".into(), json!(text));
    d.insert("hex".into(), json!(hex_spaced(&res.data)));
    d.insert("length_bytes".into(), json!(res.data.len()));
    d.insert("charset".into(), json!(charset));
    d.insert("engine".into(), json!(res.engine));
    d.insert("enhancement".into(), json!(res.variant));
    d.insert("grid_sampling".into(), json!(res.sampling));
    d.insert("agreeing_reads".into(), json!(res.agreeing));
    d.insert("zxing".into(), zx_dict(res.zx.as_ref()));
    if let Some(fit) = &res.fit {
        let g = fit.geom;
        let s = &g.sym;
        let shape = if s.rows == s.cols {
            "square"
        } else if DMRE_SYMBOLS.contains(s) {
            "rectangular DMRE (ISO/IEC 21471)"
        } else {
            "rectangular"
        };
        d.insert(
            "symbol".into(),
            json!({
                "size": g.name(),
                "shape": shape,
                "data_regions": format!("{}x{} of {}x{}", s.regr, s.regc, s.dregr, s.dregc),
                "data_codewords": s.data,
                "ecc_codewords": s.eccblk * s.blocks,
                "rs_blocks": s.blocks,
                "ecc_per_block": s.eccblk,
                "max_correctable_errors_per_block": s.eccblk / 2,
                "mirrored": res.mirrored,
            }),
        );
    }
    if let Some(rs) = &res.rs {
        let fixed = res.fit.as_ref().map_or(0, |f| f.fixed_errors);
        let positions: Vec<usize> = rs.corrected_positions.iter().map(|p| p + 1).collect();
        d.insert(
            "error_correction".into(),
            json!({
                "codewords_corrected_per_block": rs.block_errors,
                "erasures_declared_per_block": rs.block_erasures,
                "corrected_positions": positions,
                "fixed_pattern_module_errors": fixed,
                "codewords": rs.all_cw,
            }),
        );
    }
    if let Some(dec) = &res.dec {
        let segments: Vec<Value> = dec
            .segments
            .iter()
            .map(|s| {
                json!({
                    "mode": s.mode,
                    "codewords": format!("{}-{}", s.cw_start + 1, s.cw_end),
                    "data": printable(&s.data, None),
                    "note": s.note,
                })
            })
            .collect();
        d.insert(
            "encodation".into(),
            json!({
                "payload_codewords": dec.payload_cw,
                "pad_codewords": dec.pad_cw,
                "gs1": dec.gs1,
                "eci": dec.eci,
                "macro": dec.macro_kind,
                "structured_append": dec.structured_append,
                "segments": segments,
                "warnings": dec.warnings,
            }),
        );
        if dec.gs1 {
            let el: Vec<Value> = parse_gs1(&text)
                .into_iter()
                .map(|(ai, title, val)| json!([ai, title, val]))
                .collect();
            d.insert("gs1_elements".into(), Value::Array(el));
        }
    }
    if let Some(q) = &res.quality {
        d.insert("quality_estimate".into(), quality_dict(q));
    }
    // --multi: the further symbols (keys absent with one symbol)
    if !res.more.is_empty() {
        d.insert("symbols_found".into(), json!(1 + res.more.len()));
        let more: Vec<Value> = res.more.iter().map(to_dict).collect();
        d.insert("more_symbols".into(), Value::Array(more));
    }
    Value::Object(d)
}

/// The `--json` output: a list of `to_dict`, indented by 2, non-ASCII kept
/// (`json.dumps(..., indent=2, ensure_ascii=False)`).
pub fn to_json(results: &[ReadResult]) -> String {
    let list: Vec<Value> = results.iter().map(to_dict).collect();
    serde_json::to_string_pretty(&list).unwrap_or_default()
}

/// Python `repr(float)`: shortest round-trip digits, `.0` on integral
/// values, exponent form below 1e-4 and from 1e16.
pub fn py_float(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let a = x.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        // Rust "1.5e-5" -> Python "1.5e-05"
        let s = format!("{x:e}");
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let (sign, digits) = e.strip_prefix('-').map_or(("+", e), |d| ("-", d));
        return format!("{m}e{sign}{digits:0>2}");
    }
    let s = format!("{x}");
    if s.contains('.') { s } else { s + ".0" }
}

/// Python `repr(list_of_ints)`.
fn py_list<T: std::fmt::Display>(v: &[T]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(", "))
}

const GENERIC_HINTS: &str = "crop closer, avoid glare, shoot straight on, add light";

/// The near-miss lines of a failed read.
fn near_miss_lines(out: &mut Vec<String>, n: &ReadNearMiss) {
    let nm = &n.nm;
    let mpx = n.modules_px_orig;
    let agree = nm.size_agree;
    out.push(format!(
        "Near-miss  : {} grid fit, score {:.1}, {} finder/timing modules wrong, ~{mpx:.1} px/module{}",
        nm.size,
        nm.score,
        nm.fixed_pattern_module_errors,
        if nm.qz { "" } else { " (no-quiet-zone)" }
    ));
    out.push(format!(
        "Size check : {}",
        match agree {
            _ if nm.pitch_drift_modules.is_some() => {
                "both timing edges have the dash count of this size".to_string()
            }
            2 => "module pitch inside the symbol confirms the size".to_string(),
            1 => format!(
                "size only partly confirmed (module pitch inside suggests ~{})",
                nm.measured_size
            ),
            _ => format!(
                "size NOT confirmed (module pitch inside suggests ~{})",
                nm.measured_size
            ),
        }
    ));
    if nm.outside_dark > OUTSIDE_BUSY {
        out.push(format!(
            "Extent     : {:.0}% of the ring around the fit is dark: it may cover only part of the symbol",
            nm.outside_dark * 100.0
        ));
    }
    let mut hints = Vec::new();
    if mpx != 0.0 && mpx < 2.0 {
        hints.push(format!(
            "modules are ~{mpx:.1} px: shoot much closer / crop tighter"
        ));
    }
    if agree == 0 {
        hints.push("symbol size uncertain: if known, pass it with --expect RxC".into());
    } else if nm.fixed_pattern_module_errors >= 6 {
        hints.push("finder pattern damaged: try another viewing angle or more light".into());
    }
    if nm.score < STRONG_FIT {
        hints.push("low finder/timing contrast: reduce ambient light / add illumination".into());
    }
    out.push(format!(
        "Hints      : {}",
        if hints.is_empty() {
            GENERIC_HINTS.to_string()
        } else {
            hints.join("; ")
        }
    ));
}

/// `print_report`: the technical report of one result, with a trailing
/// newline. `show_matrix` adds the sampled module matrix.
pub fn report_text(res: &ReadResult, show_matrix: bool) -> String {
    let n = 1 + res.more.len();
    let mut s = report_one(res, show_matrix, (1, n));
    for (k, m) in res.more.iter().enumerate() {
        s += &report_one(m, show_matrix, (k + 2, n));
    }
    s
}

/// One symbol's block; `(k, n)` with n > 1 adds "Symbol k of n" (`--multi`).
fn report_one(res: &ReadResult, show_matrix: bool, (k, n): (usize, usize)) -> String {
    let bar = "=".repeat(72);
    let mut out = vec![bar.clone(), format!("File       : {}", res.path.display())];
    if n > 1 {
        out.push(format!("Symbol     : {k} of {n}"));
    }
    if res.data.is_empty() {
        out.push(format!(
            "Result     : NOT DECODED  ({} attempts, {:.1} s)",
            res.attempts, res.elapsed
        ));
        match &res.near_miss {
            Some(n) => near_miss_lines(&mut out, n),
            None => out.push(format!("Hints      : {GENERIC_HINTS}")),
        }
        out.push(bar);
        return out.join("\n") + "\n";
    }
    let (_, charset) = to_text(&res.data, res.text_dec());
    out.push(format!(
        "Content    : {}",
        printable(&res.data, res.text_dec())
    ));
    out.push(format!("Hex        : {}", hex_spaced(&res.data)));
    out.push(format!(
        "Length     : {} bytes, charset {charset}",
        res.data.len()
    ));
    out.push(format!(
        "Decoder    : {}   ({} attempts, {:.1} s)",
        res.engine, res.attempts, res.elapsed
    ));
    out.push(format!(
        "Enhancement: {}{}",
        res.variant,
        if res.sampling.is_empty() {
            String::new()
        } else {
            format!("  | sampling: {}", res.sampling)
        }
    ));
    if res.agreeing.len() > 1 {
        out.push(format!(
            "Confirmed  : {} independent reads agree",
            res.agreeing.len()
        ));
    }
    if let Some(z) = &res.zx {
        let mut kv = Vec::new();
        for (k, v) in [
            ("binarizer", z.binarizer.clone()),
            ("symbology_identifier", z.symbology_identifier.clone()),
            ("content_type", z.content_type.clone()),
            ("orientation_deg", z.orientation_deg.to_string()),
            ("ec_level", z.ec_level.clone()),
        ] {
            if !v.is_empty() {
                kv.push(format!("{k}={v}"));
            }
        }
        out.push(format!("zxing info : {}", kv.join(", ")));
    }
    if let Some(fit) = &res.fit {
        let g = fit.geom;
        let s = &g.sym;
        let shape = if s.rows == s.cols {
            "square"
        } else if DMRE_SYMBOLS.contains(s) {
            "rectangular DMRE (ISO/IEC 21471)"
        } else {
            "rectangular"
        };
        out.push(String::new());
        out.push("Symbol".into());
        out.push(format!(
            "  size            : {} modules ({shape}), data regions {}x{} of {}x{}",
            g.name(),
            s.regr,
            s.regc,
            s.dregr,
            s.dregc
        ));
        out.push(format!(
            "  capacity        : {} data + {} ECC codewords, {} RS block(s) x {} ECC",
            s.data,
            s.eccblk * s.blocks,
            s.blocks,
            s.eccblk
        ));
        out.push(format!(
            "  correctable     : {} errors or {} erasures per block",
            s.eccblk / 2,
            s.eccblk
        ));
        if res.mirrored {
            out.push("  mirrored        : YES (read from behind / mirrored mark)".into());
        }
    }
    if let Some(rs) = &res.rs {
        let fixed = res.fit.as_ref().map_or(0, |f| f.fixed_errors);
        out.push(String::new());
        out.push("Error correction".into());
        out.push(format!(
            "  corrected cw    : {} per block  (erasures declared {})",
            py_list(&rs.block_errors),
            py_list(&rs.block_erasures)
        ));
        out.push(format!(
            "  finder damage   : {fixed} wrong finder/timing modules"
        ));
        out.push("  codewords (* = corrected):".into());
        out.push(hexrows(&rs.all_cw, &rs.corrected_positions, 16));
    }
    if let Some(dec) = &res.dec {
        out.push(String::new());
        out.push("Encodation".into());
        out.push(format!(
            "  payload         : {} codewords, padding {}",
            dec.payload_cw, dec.pad_cw
        ));
        for s in &dec.segments {
            let cws = format!("{}-{}", s.cw_start + 1, s.cw_end);
            out.push(format!(
                "  {:8} cw {cws:>8} : {:.60} {}",
                s.mode,
                printable(&s.data, None),
                s.note
            ));
        }
        let mut flags = Vec::new();
        if dec.gs1 {
            flags.push("GS1".to_string());
        }
        if !dec.eci.is_empty() {
            flags.push(format!("ECI {}", py_list(&dec.eci)));
        }
        if !dec.macro_kind.is_empty() {
            flags.push(format!("Macro {}", dec.macro_kind));
        }
        if !dec.structured_append.is_empty() {
            flags.push(dec.structured_append.clone());
        }
        if !flags.is_empty() {
            out.push(format!("  flags           : {}", flags.join(", ")));
        }
        for w in &dec.warnings {
            out.push(format!("  warning         : {w}"));
        }
        if dec.gs1 {
            let text = to_text(&res.data, Some(dec)).0;
            out.push(String::new());
            out.push("GS1 element string".into());
            for (ai, title, val) in parse_gs1(&text) {
                out.push(format!("  ({ai}) {title:16} {val}"));
            }
        }
    }
    if let Some(q) = &res.quality {
        let f = py_float;
        out.push(String::new());
        out.push("Quality estimate (ISO/IEC 15415 style, indicative only)".into());
        out.push(format!("  overall         : {}", q.overall_grade_estimate));
        out.push(format!(
            "  symbol contrast : {}%  [{}]   dark/light reflectance {}/{}",
            f(q.symbol_contrast_pct),
            q.symbol_contrast_grade,
            f(q.dark_reflectance_mean),
            f(q.light_reflectance_mean)
        ));
        out.push(format!(
            "  modulation      : {}  [{}]",
            f(q.modulation_p5),
            q.modulation_grade
        ));
        out.push(format!(
            "  unused ECC      : {}  [{}]",
            f(q.unused_error_correction),
            q.unused_error_correction_grade
        ));
        out.push(format!(
            "  fixed pattern   : {}% damaged  [{}]",
            f(q.fixed_pattern_damage_pct),
            q.fixed_pattern_damage_grade
        ));
        out.push(format!(
            "  axial nonunif.  : {}  [{}]",
            f(q.axial_nonuniformity),
            q.axial_nonuniformity_grade
        ));
        out.push(format!(
            "  geometry        : module {} px, symbol {}x{} px, rotation {} deg, skew {} deg{}",
            f(q.module_size_px),
            q.symbol_size_px.0,
            q.symbol_size_px.1,
            f(q.rotation_deg),
            f(q.skew_deg),
            if q.pitch_drift_modules != 0.0 {
                format!(", pitch drift {} module", f(q.pitch_drift_modules))
            } else {
                String::new()
            }
        ));
    }
    if show_matrix && let Some(fit) = &res.fit {
        out.push(String::new());
        out.push("Sampled module matrix".into());
        let bits = fit.bits.as_deref().unwrap_or(&[]);
        out.push(matrix_art(bits, fit.geom.rows, fit.geom.cols));
    }
    out.push(bar);
    out.join("\n") + "\n"
}
