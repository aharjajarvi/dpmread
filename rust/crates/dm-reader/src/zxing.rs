//! zxing wrapper (`zxing_read`, `_binary`, `_zx_quad`; reader 1700–1741).
//! The read itself needs the `zxing` feature; the result types do not.

use dm_codec::Decoded;

use crate::variants::{Quad, Variant};

/// `res.zx` of a zxing read: Python's info dict.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZxInfo {
    pub binarizer: String,
    pub symbology_identifier: String,
    pub content_type: String,
    pub orientation_deg: i32,
    /// The crate has no getter; DataMatrix never sets it (always "").
    pub ec_level: String,
    /// ECIs of the symbol (only `data`, `eci`, `eci_pos` set), for the text
    /// when no grid decode fills `ReadResult::dec`; not in the JSON.
    pub eci: Option<Box<Decoded>>,
}

/// ECIs from zxing-cpp's ECI byte stream (ISO/IEC 15424: symbology
/// identifier, `\NNNNNN` before each ECI, `\\` for a backslash). `None`
/// when the identifier says the symbol has no ECI, or the stream does not
/// give back `data`. zxing-cpp puts the default ECI 3 before data that
/// precedes the first ECI; that one is dropped, so such bytes are read as
/// without ECI, as on the grid path.
pub fn eci_from_stream(stream: &[u8], data: &[u8]) -> Option<Decoded> {
    // "]d4".."]d6": DataMatrix with ECI (modifier 1..3 + 3)
    let rest = match stream {
        [b']', b'd', b'4'..=b'6', rest @ ..] => rest,
        _ => return None,
    };
    let mut d = Decoded::default();
    let mut i = 0;
    while i < rest.len() {
        if rest[i] != b'\\' {
            d.data.push(rest[i]);
            i += 1;
        } else if rest.get(i + 1) == Some(&b'\\') {
            d.data.push(b'\\');
            i += 2;
        } else {
            let digits = rest.get(i + 1..i + 7)?;
            if !digits.iter().all(u8::is_ascii_digit) {
                return None;
            }
            let eci = std::str::from_utf8(digits).ok()?.parse().ok()?;
            d.eci.push(eci);
            d.eci_pos.push(d.data.len());
            i += 7;
        }
    }
    if d.eci.len() > 1 && d.eci[0] == 3 && d.eci_pos[0] == 0 && d.eci_pos[1] > 0 {
        d.eci.remove(0);
        d.eci_pos.remove(0);
    }
    (d.data == data && !d.eci.is_empty()).then_some(d)
}

/// A valid zxing read: bytes, position (in variant pixels) and info.
#[derive(Clone, Debug, PartialEq)]
pub struct ZxRead {
    pub data: Vec<u8>,
    pub quad: Quad,
    pub info: ZxInfo,
}

/// `_binary`: variant is already black/white, every binarizer sees the same
/// image.
pub fn binary(v: &Variant) -> bool {
    v.name.ends_with("otsu") || v.name.ends_with("adaptive")
}

/// zxing is compiled in (Python: `zxingcpp is not None`).
pub const ZXING_AVAILABLE: bool = cfg!(feature = "zxing");

#[cfg(feature = "zxing")]
mod imp {
    use super::{Quad, Variant, ZxInfo, ZxRead, binary, eci_from_stream};
    use zxingcpp::{BarcodeFormat, Binarizer, ImageFormat, ImageView, TextMode};

    fn zx_quad(b: &zxingcpp::Barcode) -> Quad {
        let p = b.position();
        [p.top_left, p.top_right, p.bottom_right, p.bottom_left].map(|q| (q.x as f32, q.y as f32))
    }

    /// `zxing_read`: (result or None, failed positions). Binarizers in turn;
    /// FixedThreshold (threshold 127) reads some flattened / CLAHE variants
    /// the adaptive ones miss and is the only one used on binarised
    /// variants. The failed positions (seeds for the grid decoder) come from
    /// the first binarizer only. `try_invert` is false when the polarity is
    /// known.
    pub fn zxing_read(v: &Variant, try_invert: bool) -> (Option<ZxRead>, Vec<Quad>) {
        let bzs: &[Binarizer] = if binary(v) {
            &[Binarizer::FixedThreshold]
        } else {
            &[
                Binarizer::LocalAverage,
                Binarizer::GlobalHistogram,
                Binarizer::FixedThreshold,
            ]
        };
        let img = &v.img;
        let mut failed = Vec::new();
        for (i, &bz) in bzs.iter().enumerate() {
            let r = ImageView::from_slice(&img.data, img.width, img.height, ImageFormat::Lum)
                .and_then(|iv| {
                    zxingcpp::read()
                        .formats([BarcodeFormat::DataMatrix])
                        .binarizer(bz)
                        .text_mode(TextMode::Plain)
                        .try_invert(try_invert)
                        .return_errors(true)
                        .from(iv)
                })
                .unwrap_or_default();
            if let Some(b) = r.iter().find(|b| b.is_valid()) {
                let info = ZxInfo {
                    binarizer: format!("{bz:?}"),
                    symbology_identifier: b.symbology_identifier(),
                    content_type: format!("{:?}", b.content_type()),
                    orientation_deg: b.orientation(),
                    ec_level: String::new(),
                    eci: eci_from_stream(&b.bytes_eci(), &b.bytes()).map(Box::new),
                };
                let read = ZxRead {
                    data: b.bytes(),
                    quad: zx_quad(b),
                    info,
                };
                return (Some(read), Vec::new());
            }
            if i == 0 {
                failed = r.iter().map(zx_quad).collect();
            }
        }
        (None, failed)
    }
}

#[cfg(feature = "zxing")]
pub use imp::zxing_read;

#[cfg(test)]
mod tests {
    use super::eci_from_stream;

    #[test]
    fn eci_stream() {
        // zxing-cpp's stream of datamatrix-1_eci.png: default ECI 3 dropped
        let d = eci_from_stream(b"]d4\\000003\xb6\\000007\xb6", b"\xb6\xb6").unwrap();
        assert_eq!((d.eci, d.eci_pos), (vec![7], vec![1]));
        let d = eci_from_stream(b"]d4\\000003a\\\\b", b"a\\b").unwrap();
        assert_eq!((d.eci, d.eci_pos), (vec![3], vec![0]));
        let d = eci_from_stream(b"]d4\\000026\\000005x", b"x").unwrap();
        assert_eq!((d.eci, d.eci_pos), (vec![26, 5], vec![0, 0]));
        assert_eq!(eci_from_stream(b"]d1ab", b"ab"), None);
        assert_eq!(eci_from_stream(b"]d4\\0000", b""), None);
        assert_eq!(eci_from_stream(b"]d4\\000003ab", b"ac"), None);
    }
}
