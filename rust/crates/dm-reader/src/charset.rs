//! Decoded bytes as text: `to_text`, one charset per ECI segment (round 4;
//! Python applies one charset to the whole payload).

use dm_codec::Decoded;
use encoding_rs::Encoding;

/// IBM code page 437, bytes 0x80-0xFF (ECI 000000 / 000002).
const CP437_HIGH: &str = "ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\u{a0}";

/// How the bytes of one ECI are turned into text.
enum Charset {
    Latin1,
    Cp437,
    Ascii,
    Utf16Be,
    Enc(&'static Encoding),
}

/// ECI assignment -> (name, charset); `None` for an unknown ECI. ISO-8859-9
/// and -11 use their Windows supersets (encoding_rs has no exact table).
fn eci_charset(eci: i64) -> Option<(&'static str, Charset)> {
    use encoding_rs::*;
    let enc = |name, e| Some((name, Charset::Enc(e)));
    match eci {
        0 | 2 => Some(("CP437", Charset::Cp437)),
        1 | 3 => Some(("ISO-8859-1", Charset::Latin1)),
        4 => enc("ISO-8859-2", ISO_8859_2),
        5 => enc("ISO-8859-3", ISO_8859_3),
        6 => enc("ISO-8859-4", ISO_8859_4),
        7 => enc("ISO-8859-5", ISO_8859_5),
        8 => enc("ISO-8859-6", ISO_8859_6),
        9 => enc("ISO-8859-7", ISO_8859_7),
        10 => enc("ISO-8859-8", ISO_8859_8),
        11 => enc("ISO-8859-9", WINDOWS_1254),
        12 => enc("ISO-8859-10", ISO_8859_10),
        13 => enc("ISO-8859-11", WINDOWS_874),
        15 => enc("ISO-8859-13", ISO_8859_13),
        16 => enc("ISO-8859-14", ISO_8859_14),
        17 => enc("ISO-8859-15", ISO_8859_15),
        18 => enc("ISO-8859-16", ISO_8859_16),
        20 => enc("Shift_JIS", SHIFT_JIS),
        21 => enc("windows-1250", WINDOWS_1250),
        22 => enc("windows-1251", WINDOWS_1251),
        23 => enc("windows-1252", WINDOWS_1252),
        24 => enc("windows-1256", WINDOWS_1256),
        25 => Some(("UTF-16BE", Charset::Utf16Be)),
        26 => enc("UTF-8", UTF_8),
        27 | 170 => Some(("US-ASCII", Charset::Ascii)),
        28 => enc("Big5", BIG5),
        29 => enc("GB18030", GB18030),
        30 => enc("EUC-KR", EUC_KR),
        _ => None,
    }
}

fn latin1(b: &[u8]) -> String {
    b.iter().map(|&c| char::from(c)).collect()
}

fn decode(b: &[u8], cs: &Charset) -> String {
    match cs {
        Charset::Latin1 => latin1(b),
        Charset::Cp437 => b
            .iter()
            .map(|&c| match c {
                0..=127 => char::from(c),
                _ => CP437_HIGH
                    .chars()
                    .nth(c as usize - 128)
                    .unwrap_or('\u{fffd}'),
            })
            .collect(),
        Charset::Ascii => b
            .iter()
            .map(|&c| if c < 128 { char::from(c) } else { '\u{fffd}' })
            .collect(),
        Charset::Utf16Be => {
            let units: Vec<u16> = b
                .chunks(2)
                .map(|p| u16::from_be_bytes([p[0], *p.get(1).unwrap_or(&0)]))
                .collect();
            String::from_utf16_lossy(&units)
        }
        Charset::Enc(e) => e.decode_without_bom_handling(b).0.into_owned(),
    }
}

/// Bytes without ECI: UTF-8 if valid, else ISO-8859-1 (Python `to_text`).
fn default_text(data: &[u8]) -> (String, &'static str) {
    match std::str::from_utf8(data) {
        Ok(s) if data.iter().any(|&b| b > 127) => (s.to_string(), "UTF-8 (detected)"),
        Ok(s) => (s.to_string(), "ASCII"),
        Err(_) => (latin1(data), "ISO-8859-1 (default)"),
    }
}

/// Decoded bytes as text and the charset used. With ECIs in `dec`, each
/// segment is decoded with its own charset (bytes before the first ECI as
/// without ECI; an unknown ECI as ISO-8859-1).
pub fn to_text(data: &[u8], dec: Option<&Decoded>) -> (String, String) {
    let Some(d) = dec.filter(|d| !d.eci.is_empty() && d.eci.len() == d.eci_pos.len()) else {
        let (t, cs) = default_text(data);
        return (t, cs.to_string());
    };
    let mut text = String::new();
    let mut names: Vec<String> = Vec::new();
    let first = d.eci_pos[0].min(data.len());
    if first > 0 {
        let (t, cs) = default_text(&data[..first]);
        text += &t;
        if cs != "ASCII" {
            names.push(cs.to_string());
        }
    }
    for (k, (&eci, &pos)) in d.eci.iter().zip(&d.eci_pos).enumerate() {
        let end = d.eci_pos.get(k + 1).copied().unwrap_or(data.len());
        let (a, b) = (pos.min(data.len()), end.min(data.len()));
        let (name, cs) = eci_charset(eci).unwrap_or(("ISO-8859-1, unknown ECI", Charset::Latin1));
        text += &decode(&data[a..b.max(a)], &cs);
        let label = format!("{name} (ECI {eci:06})");
        if names.last() != Some(&label) {
            names.push(label);
        }
    }
    let label = match names.len() {
        0 => "ASCII".to_string(),
        1 => names.remove(0),
        _ => format!("mixed: {}", names.join(", ")),
    };
    (text, label)
}

#[cfg(test)]
mod tests {
    #[test]
    fn cp437_table_has_128_chars() {
        assert_eq!(super::CP437_HIGH.chars().count(), 128);
    }
}
