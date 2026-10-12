//! dpmread-gui: egui front end over dm-reader. Drop images or folders on the
//! window (or pick them with the buttons); every image is listed once and
//! read at once by a background worker, one image at a time, so that Stop
//! leaves the rest queued.

// no console window on Windows (debug builds keep it for eprintln output)
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod detail;
#[cfg(all(unix, not(target_os = "macos")))]
mod displaylibs;
mod export;
mod inputs;
mod preview;
mod settings;
mod shot;
mod table;
mod worker;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use dm_reader::ReadOptions;
use eframe::egui;
use inputs::{Inputs, dialog_extensions};
use settings::Settings;
use table::{Row, Status, TableState};
use worker::{Event, Job, Outcome, Worker};

const APP_TITLE: &str = "dpmread";
const SAVE_DELAY_S: f64 = 1.0;

struct App {
    inputs: Inputs,
    /// One per input, ids increasing.
    rows: Vec<Row>,
    next_id: u64,
    settings: Settings,
    /// `settings` as read options, rebuilt when they change.
    opts: Arc<ReadOptions>,
    settings_open: bool,
    /// Unsaved settings change since.
    dirty: Option<Instant>,
    worker: Worker,
    /// Row being read by the worker.
    in_flight: Option<u64>,
    /// Stopped: queued rows wait for Continue (or new input).
    paused: bool,
    /// Result of the last export, shown in the status bar.
    export_msg: Option<String>,
    table: TableState,
    shot: Option<shot::Shot>,
    preview: preview::Preview,
    report_open: bool,
}

impl App {
    fn new(ctx: &egui::Context) -> Self {
        let ctx = ctx.clone();
        let settings = Settings::load();
        App {
            inputs: Inputs::default(),
            rows: Vec::new(),
            next_id: 0,
            opts: Arc::new(settings.options()),
            settings,
            // layout check of the ⚙ window with the screenshot hook
            settings_open: std::env::var_os("DPMREAD_GUI_SETTINGS").is_some(),
            dirty: None,
            worker: Worker::spawn(move || ctx.request_repaint()),
            in_flight: None,
            paused: false,
            export_msg: None,
            table: TableState::default(),
            shot: shot::Shot::from_env(),
            preview: preview::Preview::default(),
            report_open: std::env::var_os("DPMREAD_GUI_REPORT").is_some(),
        }
    }

    /// Add files and folders and queue the new images (also continues
    /// after Stop).
    fn add(&mut self, paths: &[PathBuf]) {
        let new = self.inputs.add(paths);
        if new.is_empty() {
            return;
        }
        for path in new {
            self.rows.push(Row {
                id: self.next_id,
                path,
                status: Status::Queued,
            });
            self.next_id += 1;
        }
        self.paused = false;
        self.dispatch();
    }

    /// Give the next queued row to the idle worker, with the settings of now.
    fn dispatch(&mut self) {
        if self.in_flight.is_some() || self.paused {
            return;
        }
        let Some(row) = self
            .rows
            .iter()
            .find(|r| matches!(r.status, Status::Queued))
        else {
            return;
        };
        self.worker.push(Job {
            id: row.id,
            path: row.path.clone(),
            opts: self.opts.clone(),
            workers: self.settings.workers(),
        });
        self.in_flight = Some(row.id);
    }

    /// Stop: the current image ends as stopped, the queue waits (Continue
    /// reads the stopped images again).
    fn stop(&mut self) {
        if self.in_flight.is_some() {
            self.worker.stop();
        }
        self.paused = true;
    }

    fn resume(&mut self) {
        for r in &mut self.rows {
            if matches!(r.status, Status::Stopped) {
                r.status = Status::Queued;
            }
        }
        self.paused = false;
        self.dispatch();
    }

    /// Queue a finished row again (context menu).
    fn requeue(&mut self, id: u64) {
        if let Some(r) = self.row_mut(id).filter(|r| r.finished()) {
            r.status = Status::Queued;
            self.paused = false;
            self.dispatch();
        }
    }

    /// Save the finished rows as CSV or JSON (by the chosen extension).
    fn export(&mut self) {
        let picked = rfd::FileDialog::new()
            .set_file_name("datamatrix_results.csv")
            .add_filter("CSV", &["csv"])
            .add_filter("JSON (dpmread --json)", &["json"])
            .save_file();
        if let Some(path) = picked {
            self.export_msg = Some(match export::write(&path, &self.rows) {
                Ok(()) => format!("saved {}", path.display()),
                Err(e) => format!("cannot save {}: {e}", path.display()),
            });
        }
    }

    /// Stop and forget everything.
    fn clear(&mut self) {
        if self.in_flight.is_some() {
            self.worker.stop();
        }
        self.paused = false;
        self.inputs.clear();
        self.rows.clear();
        self.table.selected = None;
        self.preview.clear();
    }

    fn row_mut(&mut self, id: u64) -> Option<&mut Row> {
        let i = self.rows.binary_search_by_key(&id, |r| r.id).ok()?;
        Some(&mut self.rows[i])
    }

    fn handle_events(&mut self) {
        let events: Vec<Event> = self.worker.poll().collect();
        for ev in events {
            match ev {
                Event::Started(id) => {
                    if let Some(r) = self.row_mut(id) {
                        r.status = Status::Running(String::new(), 0, 0);
                    }
                }
                Event::Stage(id, stage, done, total) => {
                    if let Some(r) = self.row_mut(id) {
                        r.status = Status::Running(stage, done, total);
                    }
                }
                Event::Finished(id, outcome) => {
                    if self.in_flight == Some(id) {
                        self.in_flight = None;
                    }
                    if let Some(r) = self.row_mut(id) {
                        r.status = match outcome {
                            Outcome::Read(res) => Status::Done(res),
                            Outcome::Error(e) => Status::Error(e),
                            Outcome::Stopped => Status::Stopped,
                        };
                    }
                }
            }
        }
        self.dispatch();
    }

    /// Reading, or rows queued and not stopped.
    fn busy(&self) -> bool {
        self.in_flight.is_some()
            || (!self.paused && self.rows.iter().any(|r| matches!(r.status, Status::Queued)))
    }

    fn pick_files(&mut self) {
        let picked = rfd::FileDialog::new()
            .add_filter("Images", &dialog_extensions())
            .pick_files();
        if let Some(files) = picked {
            self.add(&files);
        }
    }

    fn pick_folder(&mut self) {
        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
            self.add(&[dir]);
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Add files…").clicked() {
                self.pick_files();
            }
            if ui.button("Add folder…").clicked() {
                self.pick_folder();
            }
            if ui
                .add_enabled(!self.rows.is_empty(), egui::Button::new("Clear"))
                .on_hover_text("stop and empty the list")
                .clicked()
            {
                self.clear();
            }
            ui.separator();
            ui.checkbox(&mut self.table.failed_only, "failed only");
            ui.separator();
            self.settings.toolbar(ui);
            if ui
                .selectable_label(self.settings_open, "⚙")
                .on_hover_text("settings")
                .clicked()
            {
                self.settings_open = !self.settings_open;
            }
        });
    }

    /// The drop hint, highlighted while files are dragged over the window.
    fn drop_zone(&self, ui: &mut egui::Ui, hovering: bool) {
        let height = if self.inputs.is_empty() {
            ui.available_height()
        } else {
            48.0
        };
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let visuals = ui.visuals();
        let stroke = if hovering {
            visuals.selection.stroke
        } else {
            visuals.widgets.noninteractive.bg_stroke
        };
        if hovering {
            ui.painter()
                .rect_filled(rect, 6.0, visuals.selection.bg_fill.gamma_multiply(0.3));
        }
        ui.painter()
            .rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Inside);
        let exts = dialog_extensions().join(" ");
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("Drop images or folders here   ({exts})"),
            egui::TextStyle::Body.resolve(ui.style()),
            visuals.weak_text_color(),
        );
    }
}

impl eframe::App for App {
    /// Worker events also while the window is hidden.
    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_events();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let (hovering, dropped) = ui.input(|i| {
            let paths: Vec<PathBuf> = i
                .raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .filter(|p| !p.as_os_str().is_empty())
                .collect();
            (!i.raw.hovered_files.is_empty(), paths)
        });
        if !dropped.is_empty() {
            self.add(&dropped);
        }
        let before = self.settings.clone();
        self.settings.window(ui.ctx(), &mut self.settings_open);
        let ready = !self.busy();
        if self.shot.as_mut().is_some_and(|s| s.stop_due()) {
            self.stop();
        }
        if let Some(shot) = &mut self.shot {
            shot.update(ui.ctx(), ready);
            if let Some(n) = shot.select.filter(|_| ready) {
                let lines = self.table.lines(&self.rows);
                self.table.selected = lines.get(n).map(|&(i, s)| (self.rows[i].id, s));
                shot.select = None;
            }
        }
        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(4.0);
            self.toolbar(ui);
            ui.add_space(4.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            let n = self.rows.len();
            let count = |f: fn(&Row) -> bool| self.rows.iter().filter(|r| f(r)).count();
            let finished = count(Row::finished);
            let decoded = count(Row::decoded);
            let failed = count(Row::failed);
            let stopped = count(|r| matches!(r.status, Status::Stopped));
            ui.horizontal(|ui| {
                let frac = if n == 0 {
                    0.0
                } else {
                    finished as f32 / n as f32
                };
                ui.add(egui::ProgressBar::new(frac).desired_width(240.0));
                let mut text = format!("{finished} / {n}   decoded {decoded}   failed {failed}");
                if stopped > 0 {
                    text += &format!("   stopped {stopped}");
                }
                ui.label(text);
                let queued = self
                    .rows
                    .iter()
                    .any(|r| matches!(r.status, Status::Queued | Status::Stopped));
                if self.paused && queued && self.in_flight.is_none() {
                    if ui.button("Continue").clicked() {
                        self.resume();
                    }
                } else if ui
                    .add_enabled(self.busy() && !self.paused, egui::Button::new("Stop"))
                    .on_hover_text("stop the image being read, keep the queue")
                    .clicked()
                {
                    self.stop();
                }
                let any = self.rows.iter().any(Row::finished);
                if ui
                    .add_enabled(any, egui::Button::new("Export…"))
                    .on_hover_text("finished images as CSV or JSON")
                    .clicked()
                {
                    self.export();
                }
                if let Some(m) = &self.export_msg {
                    ui.weak(m);
                }
            });
        });
        let sel = self.table.selected.and_then(|(id, sym)| {
            let i = self.rows.binary_search_by_key(&id, |r| r.id).ok()?;
            Some((i, sym))
        });
        if let Some((i, sym)) = sel {
            egui::Panel::bottom("detail")
                .resizable(true)
                .default_size(ui.available_height() * 0.5)
                .show(ui, |ui| {
                    detail::show(
                        ui,
                        &self.rows[i],
                        sym,
                        &mut self.preview,
                        &mut self.report_open,
                    );
                });
        }
        egui::CentralPanel::default().show(ui, |ui| {
            self.drop_zone(ui, hovering);
            if !self.inputs.is_empty() {
                ui.add_space(6.0);
                if let Some(id) = self.table.show(ui, &self.rows) {
                    self.requeue(id);
                }
            }
        });
        if self.settings != before {
            self.opts = Arc::new(self.settings.options());
            self.dirty = Some(Instant::now());
        }
        // save a second after the last change (not every drag step)
        if let Some(t) = self.dirty {
            if t.elapsed().as_secs_f64() > SAVE_DELAY_S {
                self.settings.save();
                self.dirty = None;
            } else {
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(300));
            }
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if self.dirty.is_some() {
            self.settings.save();
        }
    }
}

/// winit 0.30 has no file drop on Wayland: use X11 (XWayland) when it is
/// there, unless `DPMREAD_GUI_WAYLAND` is set (native Wayland, no drop).
/// Exits with a message when the display libraries are missing (winit
/// panics on some of them).
#[cfg(all(unix, not(target_os = "macos")))]
fn event_loop_hook() -> Option<eframe::EventLoopBuilderHook> {
    use winit::platform::x11::EventLoopBuilderExtX11;
    let x11 = displaylibs::use_x11().unwrap_or_else(|msg| {
        eprintln!("{APP_TITLE}: {msg}");
        // no console when started from the desktop: also a dialog (zenity)
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title(APP_TITLE)
            .set_description(&msg)
            .show();
        std::process::exit(1);
    });
    x11.then(|| {
        Box::new(|b: &mut eframe::EventLoopBuilder<eframe::UserEvent>| {
            b.with_x11();
        }) as eframe::EventLoopBuilderHook
    })
}

#[cfg(not(all(unix, not(target_os = "macos"))))]
fn event_loop_hook() -> Option<eframe::EventLoopBuilderHook> {
    None
}

fn native_options() -> eframe::NativeOptions {
    eframe::NativeOptions {
        // glow first everywhere (wgpu would be the default on Windows, where
        // it is enabled too, as fallback)
        renderer: eframe::Renderer::Glow,
        event_loop_builder: event_loop_hook(),
        viewport: egui::ViewportBuilder::default()
            .with_title(APP_TITLE)
            .with_app_id("dpmread-gui")
            .with_inner_size([960.0, 720.0])
            .with_drag_and_drop(true),
        ..Default::default()
    }
}

fn app_creator<'a>() -> eframe::AppCreator<'a> {
    Box::new(|cc| {
        let mut app = App::new(&cc.egui_ctx);
        // files and folders on the command line ("Open with", tests)
        let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
        app.add(&args);
        Ok(Box::new(app))
    })
}

#[cfg(not(windows))]
fn main() -> eframe::Result {
    eframe::run_native(APP_TITLE, native_options(), app_creator())
}

#[cfg(windows)]
fn main() -> eframe::Result {
    eframe::run_native(APP_TITLE, native_options(), app_creator()).or_else(|glow_err| {
        // no OpenGL 2 (VM without 3D acceleration, RDP): Direct3D 12 via
        // wgpu instead; eframe keeps its event loop, so a second run works
        let options = eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            ..native_options()
        };
        eframe::run_native(APP_TITLE, options, app_creator()).inspect_err(|wgpu_err| {
            // without a console the error would be silent: message box
            rfd::MessageDialog::new()
                .set_level(rfd::MessageLevel::Error)
                .set_title(APP_TITLE)
                .set_description(format!(
                    "{APP_TITLE} could not start:\n\nOpenGL: {glow_err}\n\nDirect3D (wgpu): {wgpu_err}"
                ))
                .show();
        })
    })
}
