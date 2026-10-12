//! `load_image`: decode like `cv2.imdecode(data, cv2.IMREAD_COLOR)`.
//!
//! OpenCV applies the EXIF orientation, expands gray and palette images to
//! three channels and reduces 16-bit samples to their high byte (libpng
//! `png_set_strip_16`). Unlike OpenCV, which drops alpha (transparent pixels
//! are often stored black, so the quiet zone turns black), alpha is
//! composited over white (round 4).

use crate::image::Bgr8;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use std::io::Cursor;
use std::path::Path;

#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoadError {}

/// Load an image file as BGR. Rust paths handle non-ASCII names natively.
pub fn load_image(path: &Path) -> Result<Bgr8, LoadError> {
    let bytes = std::fs::read(path)
        .map_err(|e| LoadError(format!("cannot read image {}: {e}", path.display())))?;
    decode_image(&bytes)
        .map_err(|e| LoadError(format!("cannot decode image {}: {}", path.display(), e.0)))
}

/// Decode an encoded image (any supported format) as BGR.
pub fn decode_image(bytes: &[u8]) -> Result<Bgr8, LoadError> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| LoadError(e.to_string()))?;
    let is_jpeg = reader.format() == Some(ImageFormat::Jpeg);
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| LoadError(e.to_string()))?;
    let orientation = decoder
        .orientation()
        .map_err(|e| LoadError(e.to_string()))?;
    let mut img = if is_jpeg && cfg!(feature = "turbojpeg") {
        decode_jpeg(bytes)?
    } else {
        DynamicImage::from_decoder(decoder).map_err(|e| LoadError(e.to_string()))?
    };
    img.apply_orientation(orientation);
    Ok(to_bgr(&img))
}

/// JPEG pixels from libjpeg-turbo with its default (OpenCV's) settings:
/// accurate integer IDCT, fancy upsampling.
#[cfg(feature = "turbojpeg")]
fn decode_jpeg(bytes: &[u8]) -> Result<DynamicImage, LoadError> {
    let img = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB)
        .map_err(|e| LoadError(e.to_string()))?;
    let (w, h) = (img.width, img.height);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        rgb.extend_from_slice(&img.pixels[y * img.pitch..y * img.pitch + w * 3]);
    }
    let buf = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| LoadError("jpeg size".into()))?;
    Ok(DynamicImage::ImageRgb8(buf))
}

#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg(_bytes: &[u8]) -> Result<DynamicImage, LoadError> {
    unreachable!()
}

fn to_bgr(img: &DynamicImage) -> Bgr8 {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = Vec::with_capacity(w * h * 3);
    let hi = |v: u16| (v >> 8) as u8;
    // over white, rounded: c * a + 255 * (255 - a)
    let over = |c: u8, a: u8| ((c as u32 * a as u32 + 255 * (255 - a as u32) + 127) / 255) as u8;
    let over16 = |c: u16, a: u16| {
        let (c, a) = (c as u32, a as u32);
        hi(((c * a + 65535 * (65535 - a) + 32767) / 65535) as u16)
    };
    match img {
        DynamicImage::ImageLuma8(b) => b.pixels().for_each(|p| out.extend_from_slice(&[p[0]; 3])),
        DynamicImage::ImageLumaA8(b) => b
            .pixels()
            .for_each(|p| out.extend_from_slice(&[over(p[0], p[1]); 3])),
        DynamicImage::ImageRgb8(b) => b
            .pixels()
            .for_each(|p| out.extend_from_slice(&[p[2], p[1], p[0]])),
        DynamicImage::ImageRgba8(b) => b.pixels().for_each(|p| {
            out.extend_from_slice(&[over(p[2], p[3]), over(p[1], p[3]), over(p[0], p[3])])
        }),
        DynamicImage::ImageLuma16(b) => b
            .pixels()
            .for_each(|p| out.extend_from_slice(&[hi(p[0]); 3])),
        DynamicImage::ImageLumaA16(b) => b
            .pixels()
            .for_each(|p| out.extend_from_slice(&[over16(p[0], p[1]); 3])),
        DynamicImage::ImageRgb16(b) => b
            .pixels()
            .for_each(|p| out.extend_from_slice(&[hi(p[2]), hi(p[1]), hi(p[0])])),
        DynamicImage::ImageRgba16(b) => b.pixels().for_each(|p| {
            out.extend_from_slice(&[over16(p[2], p[3]), over16(p[1], p[3]), over16(p[0], p[3])])
        }),
        other => other.to_rgba8().pixels().for_each(|p| {
            out.extend_from_slice(&[over(p[2], p[3]), over(p[1], p[3]), over(p[0], p[3])])
        }),
    }
    Bgr8::from_vec(w, h, out)
}
