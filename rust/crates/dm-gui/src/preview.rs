//! Detail image: the selected image with the selected symbol's fit drawn
//! on it (`annotated`, as `--debug` writes it), loaded off the UI thread;
//! a zoom / pan viewer that outlines every symbol of the image.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};

use dm_imgproc::{Bgr8, load_image};
use dm_reader::{Quad, ReadResult, annotated};
use eframe::egui;

/// Row id and symbol index.
pub type Key = (u64, usize);

struct Loaded {
    key: Key,
    image: Result<egui::ColorImage, String>,
    /// Original image size (the texture may be smaller).
    size: egui::Vec2,
}

/// BGR -> RGBA, box-shrunk by an integer factor so the longer side fits
/// `max_side` (the GPU texture limit).
fn to_color_image(bgr: &Bgr8, max_side: usize) -> egui::ColorImage {
    let f = bgr.width.max(bgr.height).div_ceil(max_side.max(1)).max(1);
    let (w, h) = (bgr.width / f, bgr.height / f);
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut s = [0u32; 3];
            for yy in y * f..(y + 1) * f {
                for xx in x * f..(x + 1) * f {
                    let i = 3 * (yy * bgr.width + xx);
                    for (acc, &v) in s.iter_mut().zip(&bgr.data[i..i + 3]) {
                        *acc += u32::from(v);
                    }
                }
            }
            let n = (f * f) as u32;
            let [b, g, r] = s.map(|v| (v / n) as u8);
            px.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    egui::ColorImage::new([w, h], px)
}

fn load(
    path: &Path,
    sym: Option<&ReadResult>,
    max_side: usize,
) -> (Result<egui::ColorImage, String>, egui::Vec2) {
    match load_image(path) {
        Ok(bgr) => {
            let size = egui::vec2(bgr.width as f32, bgr.height as f32);
            let img = sym.and_then(|r| annotated(&bgr, r)).unwrap_or(bgr);
            (Ok(to_color_image(&img, max_side)), size)
        }
        Err(e) => (Err(e.to_string()), egui::Vec2::ZERO),
    }
}

/// Map between original image pixels and the screen: `scale` screen px per
/// image px, `center` the image point at the middle of the view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub scale: f32,
    pub center: egui::Pos2,
}

impl View {
    pub fn fit(rect: egui::Rect, size: egui::Vec2) -> Self {
        let scale = (rect.width() / size.x)
            .min(rect.height() / size.y)
            .max(1e-3);
        View {
            scale,
            center: (size / 2.0).to_pos2(),
        }
    }

    pub fn to_screen(self, rect: egui::Rect, p: egui::Pos2) -> egui::Pos2 {
        rect.center() + (p - self.center) * self.scale
    }

    pub fn to_image(self, rect: egui::Rect, s: egui::Pos2) -> egui::Pos2 {
        self.center + (s - rect.center()) / self.scale
    }

    /// Zoom by `factor` keeping the image point under `at` in place.
    pub fn zoom(&mut self, rect: egui::Rect, at: egui::Pos2, factor: f32) {
        let p = self.to_image(rect, at);
        self.scale = (self.scale * factor).clamp(0.01, 64.0);
        self.center = p - (at - rect.center()) / self.scale;
    }
}

pub struct Preview {
    tx: Sender<Loaded>,
    rx: Receiver<Loaded>,
    /// Key of the newest request.
    want: Option<Key>,
    shown: Option<(Key, Result<egui::TextureHandle, String>, egui::Vec2)>,
    /// Zoom / pan set by the user; `None`: fit to the view every frame
    /// (follows pane resizes and new images).
    view: Option<View>,
}

impl Default for Preview {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Preview {
            tx,
            rx,
            want: None,
            shown: None,
            view: None,
        }
    }
}

impl Preview {
    /// Show `key`: load the image (in the background) if it is not shown.
    /// `sym`: the result whose fit is drawn (none for a failed read).
    pub fn request(
        &mut self,
        ctx: &egui::Context,
        key: Key,
        path: &Path,
        sym: Option<&ReadResult>,
    ) {
        if self.want == Some(key) {
            return;
        }
        self.want = Some(key);
        let (tx, ctx2, path, sym) = (
            self.tx.clone(),
            ctx.clone(),
            path.to_path_buf(),
            sym.cloned(),
        );
        let max_side = ctx.input(|i| i.max_texture_side);
        std::thread::spawn(move || {
            let (image, size) = load(&path, sym.as_ref(), max_side);
            let _ = tx.send(Loaded { key, image, size });
            ctx2.request_repaint();
        });
    }

    pub fn clear(&mut self) {
        self.want = None;
        self.shown = None;
        self.view = None;
    }

    fn receive(&mut self, ctx: &egui::Context) {
        for l in self.rx.try_iter().collect::<Vec<_>>() {
            if Some(l.key) != self.want {
                continue;
            }
            // another image fits again; another symbol of it keeps the zoom
            if self.shown.as_ref().map(|s| s.0.0) != Some(l.key.0) {
                self.view = None;
            }
            let tex = l
                .image
                .map(|img| ctx.load_texture("detail", img, egui::TextureOptions::LINEAR));
            self.shown = Some((l.key, tex, l.size));
        }
    }

    /// The viewer: wheel zooms at the cursor, drag pans, double-click fits.
    /// `quads`: every symbol of the image (original pixels), `selected`
    /// its index among them.
    pub fn show(&mut self, ui: &mut egui::Ui, quads: &[Quad], selected: Option<usize>) {
        self.receive(ui.ctx());
        let (rect, resp) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
        let loading = self.shown.as_ref().map(|s| s.0) != self.want;
        let Some((_, tex, size)) = &self.shown else {
            if self.want.is_some() {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "loading…",
                    egui::TextStyle::Body.resolve(ui.style()),
                    ui.visuals().weak_text_color(),
                );
            }
            return;
        };
        let tex = match tex {
            Ok(t) => t,
            Err(e) => {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    e,
                    egui::TextStyle::Body.resolve(ui.style()),
                    ui.visuals().error_fg_color,
                );
                return;
            }
        };
        let size = *size;
        if resp.double_clicked() {
            self.view = None;
        }
        let mut view = self.view.unwrap_or_else(|| View::fit(rect, size));
        if resp.dragged() && resp.drag_delta() != egui::Vec2::ZERO {
            view.center -= resp.drag_delta() / view.scale;
            self.view = Some(view);
        }
        if let Some(at) = resp.hover_pos() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (scroll * 0.0025).exp() * pinch;
            if factor != 1.0 {
                view.zoom(rect, at, factor);
                self.view = Some(view);
            }
        }
        let img_rect = egui::Rect::from_two_pos(
            view.to_screen(rect, egui::Pos2::ZERO),
            view.to_screen(rect, size.to_pos2()),
        );
        painter.image(
            tex.id(),
            img_rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        for (k, q) in quads.iter().enumerate() {
            let pts: Vec<egui::Pos2> = q
                .iter()
                .map(|&(x, y)| view.to_screen(rect, egui::pos2(x, y)))
                .collect();
            let stroke = if Some(k) == selected {
                egui::Stroke::new(2.5, egui::Color32::YELLOW)
            } else {
                egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 200, 255))
            };
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        if loading {
            painter.text(
                rect.left_top() + egui::vec2(6.0, 6.0),
                egui::Align2::LEFT_TOP,
                "loading…",
                egui::TextStyle::Small.resolve(ui.style()),
                ui.visuals().weak_text_color(),
            );
        }
        let hint = format!(
            "{:.0}%  · wheel zoom, drag pan, double-click fit",
            view.scale * 100.0
        );
        let galley = painter.layout_no_wrap(
            hint,
            egui::TextStyle::Small.resolve(ui.style()),
            ui.visuals().text_color(),
        );
        let pos = rect.left_bottom() + egui::vec2(6.0, -6.0 - galley.size().y);
        let bg = egui::Rect::from_min_size(pos, galley.size()).expand(3.0);
        painter.rect_filled(bg, 3.0, ui.visuals().extreme_bg_color.gamma_multiply(0.8));
        painter.galley(pos, galley, egui::Color32::PLACEHOLDER);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_keeps_point_under_cursor() {
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(400.0, 300.0));
        let mut v = View::fit(rect, egui::vec2(800.0, 400.0));
        assert_eq!(v.scale, 0.5);
        let at = egui::pos2(100.0, 50.0);
        let p = v.to_image(rect, at);
        v.zoom(rect, at, 3.0);
        let q = v.to_image(rect, at);
        assert!((p - q).length() < 1e-3, "{p:?} {q:?}");
        assert_eq!(v.to_screen(rect, v.center), rect.center());
    }

    #[test]
    fn big_images_shrink_to_the_texture_limit() {
        let bgr = Bgr8::from_vec(5, 3, (0..45).map(|i| i as u8).collect());
        let img = to_color_image(&bgr, 2);
        assert_eq!(img.size, [1, 1]);
        // the 3x3 box: blue channel of pixels (0..3, 0..3), red is channel 2
        let red: u32 = [0, 1, 2, 5, 6, 7, 10, 11, 12]
            .iter()
            .map(|p| 3 * p + 2)
            .sum();
        assert_eq!(img.pixels[0].r() as u32, red / 9);
        assert_eq!(to_color_image(&bgr, 4096).size, [5, 3]);
    }
}
