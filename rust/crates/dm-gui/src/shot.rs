//! Self-screenshot for checking the layout without a person at the screen:
//! with `DPMREAD_GUI_SCREENSHOT=out.png` the window saves itself as PNG once
//! every queued image is read (and `SETTLE_S` later), then closes. With
//! `DPMREAD_GUI_SELECT=N` table line N is selected first (detail pane), with
//! `DPMREAD_GUI_SETTINGS` / `DPMREAD_GUI_REPORT` set the ⚙ / report window
//! starts open (own OS windows, `popup.rs`: not in the screenshot, which
//! holds the main window only), and
//! `DPMREAD_GUI_STOP_AT=S` presses Stop S seconds after start. With
//! `DPMREAD_GUI_EXPORT=out.json` (or `.csv`) the results are exported there
//! just before the screenshot, as *Export* would.

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

pub const ENV: &str = "DPMREAD_GUI_SCREENSHOT";
const SELECT_ENV: &str = "DPMREAD_GUI_SELECT";
const STOP_ENV: &str = "DPMREAD_GUI_STOP_AT";
const EXPORT_ENV: &str = "DPMREAD_GUI_EXPORT";
const SETTLE_S: f64 = 1.5;

pub struct Shot {
    path: PathBuf,
    /// When the queue was first seen read.
    ready_at: Option<Instant>,
    requested: bool,
    /// Table line to select once read.
    pub select: Option<usize>,
    /// When to press Stop.
    stop_at: Option<Instant>,
    /// Where to export the results, and whether that is due now.
    export: Option<PathBuf>,
    export_due: bool,
}

impl Shot {
    pub fn from_env() -> Option<Self> {
        std::env::var_os(ENV).map(|p| Shot {
            path: PathBuf::from(p),
            ready_at: None,
            requested: false,
            select: std::env::var(SELECT_ENV).ok().and_then(|v| v.parse().ok()),
            stop_at: std::env::var(STOP_ENV)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .map(|s| Instant::now() + std::time::Duration::from_secs_f64(s)),
            export: std::env::var_os(EXPORT_ENV).map(PathBuf::from),
            export_due: false,
        })
    }

    /// True once, when Stop is due.
    pub fn stop_due(&mut self) -> bool {
        let due = self.stop_at.is_some_and(|t| Instant::now() >= t);
        if due {
            self.stop_at = None;
        }
        due
    }

    /// The export path, once, when the results are to be exported.
    pub fn export_due(&mut self) -> Option<PathBuf> {
        if !self.export_due {
            return None;
        }
        self.export_due = false;
        self.export.take()
    }

    /// Call every frame; `ready`: nothing left to read.
    pub fn update(&mut self, ctx: &egui::Context, ready: bool) {
        let image = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let [w, h] = img.size;
            let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            if let Err(e) = image::save_buffer(
                &self.path,
                &rgba,
                w as u32,
                h as u32,
                image::ExtendedColorType::Rgba8,
            ) {
                eprintln!("screenshot {}: {e}", self.path.display());
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if !ready {
            self.ready_at = None;
        } else if self.ready_at.is_none() {
            self.ready_at = Some(Instant::now());
        }
        let settled = self
            .ready_at
            .is_some_and(|t| t.elapsed().as_secs_f64() > SETTLE_S);
        if !self.requested && settled {
            self.requested = true;
            self.export_due = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        ctx.request_repaint();
    }
}
