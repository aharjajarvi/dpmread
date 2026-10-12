//! Image buffers: single-channel `Image<T>` (`Gray8`, `GrayF32`) and
//! interleaved `Bgr8`, row-major without padding.

/// Single-channel image, row-major, `data.len() == width * height`.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
}

pub type Gray8 = Image<u8>;
pub type GrayF32 = Image<f32>;

impl<T: Copy + Default> Image<T> {
    pub fn new(width: usize, height: usize) -> Self {
        Self::filled(width, height, T::default())
    }

    pub fn filled(width: usize, height: usize, v: T) -> Self {
        Image {
            width,
            height,
            data: vec![v; width * height],
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), width * height, "image buffer size");
        Image {
            width,
            height,
            data,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        self.data[y * self.width + x] = v;
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        &mut self.data[y * self.width..(y + 1) * self.width]
    }

    /// Shorter side (`min(g.shape)`).
    pub fn min_side(&self) -> usize {
        self.width.min(self.height)
    }

    /// Longer side (`max(g.shape)`).
    pub fn max_side(&self) -> usize {
        self.width.max(self.height)
    }

    /// Copy of the region `[x0, x0 + w) x [y0, y0 + h)` (numpy slice).
    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Self {
        assert!(
            x0 + w <= self.width && y0 + h <= self.height,
            "crop out of range"
        );
        let mut data = Vec::with_capacity(w * h);
        for y in y0..y0 + h {
            data.extend_from_slice(&self.row(y)[x0..x0 + w]);
        }
        Image {
            width: w,
            height: h,
            data,
        }
    }

    /// Element-wise conversion to another pixel type.
    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Image<U> {
        Image {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }
}

impl Gray8 {
    /// `g.astype(np.float32)`.
    pub fn to_f32(&self) -> GrayF32 {
        self.map(|v| v as f32)
    }

    /// `255 - g`.
    pub fn inverted(&self) -> Gray8 {
        self.map(|v| 255 - v)
    }
}

impl GrayF32 {
    /// `.astype(np.uint8)` of values known to lie in 0..=255: truncation
    /// towards zero (numpy float -> uint8 cast).
    pub fn to_u8_trunc(&self) -> Gray8 {
        self.map(|v| v as u8)
    }
}

/// 8-bit colour image, interleaved B, G, R (OpenCV order).
#[derive(Clone, Debug, PartialEq)]
pub struct Bgr8 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Bgr8 {
    pub fn from_vec(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height * 3, "image buffer size");
        Bgr8 {
            width,
            height,
            data,
        }
    }

    /// One channel (0 = blue, 1 = green, 2 = red) as a gray image.
    pub fn channel(&self, c: usize) -> Gray8 {
        Image::from_vec(
            self.width,
            self.height,
            self.data.iter().skip(c).step_by(3).copied().collect(),
        )
    }

    pub fn max_side(&self) -> usize {
        self.width.max(self.height)
    }
}
