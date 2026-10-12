//! P4.3: `to_text` and `parse_sizes` behave like Python's.

use dm_reader::{parse_sizes, to_text};

#[test]
fn to_text_charsets() {
    assert_eq!(to_text(b"AB-1", None), ("AB-1".into(), "ASCII".into()));
    assert_eq!(
        to_text("Näyttö".as_bytes(), None),
        ("Näyttö".into(), "UTF-8 (detected)".into())
    );
    assert_eq!(
        to_text(b"\xe4x", None),
        ("äx".into(), "ISO-8859-1 (default)".into())
    );
}

#[test]
fn parse_sizes_like_python() {
    let names = |v: Option<Vec<dm_symbol::Symbol>>| {
        v.map(|s| s.iter().map(|s| s.name()).collect::<Vec<_>>())
    };
    assert_eq!(names(parse_sizes(&[])), None);
    assert_eq!(
        names(parse_sizes(&["24X24, 12×26".into(), "10x10,24x24".into()])),
        Some(vec!["10x10".into(), "12x26".into(), "24x24".into()])
    );
    assert_eq!(names(parse_sizes(&["11x11".into(), "abc".into()])), None);
    assert_eq!(
        names(parse_sizes(&["8x48".into()])),
        Some(vec!["8x48".into()])
    );
}

#[test]
fn report_helpers_like_python() {
    use dm_reader::{hexrows, printable, py_float};
    for (x, want) in [
        (58.6, "58.6"),
        (0.0, "0.0"),
        (-3.0, "-3.0"),
        (100.0, "100.0"),
        (1.5e-5, "1.5e-05"),
        (2e16, "2e+16"),
        (0.0001, "0.0001"),
    ] {
        assert_eq!(py_float(x), want);
    }
    assert_eq!(printable(b"]A\x1d01\x7f\x01", None), "]A<GS>01<7F><01>");
    assert_eq!(printable("ä".as_bytes(), None), "ä");
    let cws: Vec<u8> = (0..18).collect();
    assert_eq!(
        hexrows(&cws, &[1, 16], 16),
        "       1: 00  01* 02  03  04  05  06  07  08  09  0A  0B  0C  0D  0E  0F \n      17: 10* 11 "
    );
}

#[test]
fn to_text_per_eci_segment() {
    use dm_codec::decode_codewords;
    let text = |cw: &[u8]| {
        let d = decode_codewords(cw).unwrap();
        to_text(&d.data, Some(&d))
    };
    // zxing_cpp/datamatrix-1_eci.png: upper shift ¶, ECI 7 (ISO-8859-5), upper shift 0xB6
    assert_eq!(
        text(&[0xEB, 0x37, 0xF1, 0x08, 0xEB, 0x37, 0x81, 0x38]),
        (
            "¶Ж".into(),
            "mixed: ISO-8859-1 (default), ISO-8859-5 (ECI 000007)".into()
        )
    );
    // ECI 26 first: the whole payload is UTF-8, as before
    let mut cw = vec![0xF1, 27];
    cw.extend("ä".bytes().flat_map(|b| [0xEB, b - 127]));
    assert_eq!(text(&cw), ("ä".into(), "UTF-8 (ECI 000026)".into()));
    // ECI 2 (CP437): 0x80 'Ç', 0xFF NBSP; unknown ECI 900 as ISO-8859-1
    assert_eq!(
        text(&[0xF1, 3, 0xEB, 1, 0xEB, 128]),
        ("Ç\u{a0}".into(), "CP437 (ECI 000002)".into())
    );
    assert_eq!(
        text(&[0xF1, 131, 12, 0xEB, 0x37]).1,
        "ISO-8859-1, unknown ECI (ECI 000900)"
    );
}
