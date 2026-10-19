//! Reader settings of the GUI (the CLI flags that matter for one image),
//! stored as JSON in the platform config dir.

use std::path::{Path, PathBuf};

use dm_reader::{Engines, Polarity, ReadOptions, default_jobs, parse_sizes};
use eframe::egui;
use serde::{Deserialize, Serialize};

const FILE: &str = "settings.json";
/// Config dir name; settings saved before the binary rename are read once
/// from `OLD_APP_DIR` until the first save.
const APP_DIR: &str = "dpmread-gui";
const OLD_APP_DIR: &str = "dmread-gui";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub budget_s: f64,
    /// "any", "dark" or "light".
    pub polarity: String,
    /// Read every symbol of an image ("all codes", `--multi`).
    pub multi: bool,
    /// Expected sizes as typed (`--expect`), empty for all.
    pub expect: String,
    pub zxing: bool,
    pub grid: bool,
    pub exhaustive: bool,
    /// Trained symbol detector as the first try (`--no-detector` off;
    /// cargo feature `ml`, without it this does nothing).
    pub detector: bool,
    /// Module classifier on the best failed grid fits (`--no-classifier`
    /// off; cargo feature `ml`).
    pub classifier: bool,
    /// Reader threads; 0 = default (`default_jobs()`).
    pub jobs: usize,
}

impl Default for Settings {
    fn default() -> Self {
        let o = ReadOptions::default();
        Settings {
            budget_s: o.budget_s,
            polarity: "any".into(),
            multi: o.multi,
            expect: String::new(),
            zxing: o.engines.zxing,
            grid: o.engines.grid,
            exhaustive: o.exhaustive,
            detector: o.detector,
            classifier: o.classifier,
            jobs: 0,
        }
    }
}

pub const POLARITIES: [(&str, &str); 3] = [
    ("any", "any"),
    ("dark", "dark on light"),
    ("light", "light on dark"),
];

impl Settings {
    pub fn options(&self) -> ReadOptions {
        ReadOptions {
            engines: Engines {
                zxing: self.zxing,
                grid: self.grid,
            },
            exhaustive: self.exhaustive,
            budget_s: self.budget_s,
            expect: self.expect_sizes(),
            polarity: match self.polarity.as_str() {
                "dark" => Polarity::Dark,
                "light" => Polarity::Light,
                _ => Polarity::Any,
            },
            debug_dir: None,
            multi: self.multi,
            progress: None,
            analyse: true,
            fit_inputs: false,
            classifier: self.classifier,
            detector: self.detector,
            frame: false,
        }
    }

    fn expect_sizes(&self) -> Option<Vec<dm_symbol::Symbol>> {
        if self.expect.trim().is_empty() {
            return None;
        }
        parse_sizes(std::slice::from_ref(&self.expect))
    }

    /// The sizes `expect` names, for the settings window.
    fn expect_summary(&self) -> String {
        match self.expect_sizes() {
            None if self.expect.trim().is_empty() => "all sizes".to_string(),
            None => "no known size: all sizes".to_string(),
            Some(s) => s
                .iter()
                .map(|s| format!("{}x{}", s.rows, s.cols))
                .collect::<Vec<_>>()
                .join(", "),
        }
    }

    pub fn workers(&self) -> usize {
        if self.jobs == 0 {
            default_jobs()
        } else {
            self.jobs
        }
    }

    fn dir_path(app: &str) -> Option<PathBuf> {
        directories::ProjectDirs::from("", "", app).map(|d| d.config_dir().join(FILE))
    }

    fn path() -> Option<PathBuf> {
        Self::dir_path(APP_DIR)
    }

    /// The stored settings, or the defaults.
    pub fn load() -> Self {
        match (Self::path(), Self::dir_path(OLD_APP_DIR)) {
            (Some(p), Some(old)) => Self::load_first(&p, &old),
            (Some(p), None) => Self::load_from(&p),
            _ => Self::default(),
        }
    }

    /// `p` if it exists, else `old` (settings from before the rename).
    fn load_first(p: &Path, old: &Path) -> Self {
        Self::load_from(if p.exists() || !old.exists() { p } else { old })
    }

    fn load_from(p: &Path) -> Self {
        std::fs::read_to_string(p)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Some(p) = Self::path() {
            self.save_to(&p);
        }
    }

    fn save_to(&self, p: &Path) {
        let r = p
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|_| {
                std::fs::write(p, serde_json::to_string_pretty(self).unwrap_or_default())
            });
        if let Err(e) = r {
            eprintln!("cannot save settings to {}: {e}", p.display());
        }
    }

    /// Polarity, budget and "all codes" for the toolbar.
    pub fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.label("Polarity");
        polarity_combo(ui, &mut self.polarity);
        ui.label("Budget");
        ui.add(
            egui::DragValue::new(&mut self.budget_s)
                .range(1.0..=600.0)
                .speed(0.5)
                .suffix(" s"),
        )
        .on_hover_text("max seconds per image");
        ui.checkbox(&mut self.multi, "all codes")
            .on_hover_text("read every symbol in the image");
    }

    /// The ⚙ window: everything, applied to images added from now on.
    pub fn window(&mut self, ctx: &egui::Context, open: &mut bool) {
        crate::popup::popup(
            ctx,
            "settings",
            "Settings – dpmread",
            [460.0, 300.0],
            false,
            open,
            |ui| {
                egui::Grid::new("settings_grid")
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Budget");
                        ui.add(
                            egui::DragValue::new(&mut self.budget_s)
                                .range(1.0..=600.0)
                                .speed(0.5)
                                .suffix(" s per image"),
                        );
                        ui.end_row();
                        ui.label("Polarity");
                        polarity_combo(ui, &mut self.polarity);
                        ui.end_row();
                        ui.label("Expected size(s)");
                        ui.vertical(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.expect)
                                    .hint_text("all, or e.g. 24x24, 16x36"),
                            );
                            // parsed once per text (`parse_sizes` warns on stderr)
                            let id = egui::Id::new("expect_summary");
                            let cached: Option<(String, String)> = ui.data(|d| d.get_temp(id));
                            let known = match cached {
                                Some((t, k)) if t == self.expect => k,
                                _ => {
                                    let k = self.expect_summary();
                                    let v = (self.expect.clone(), k.clone());
                                    ui.data_mut(|d| d.insert_temp(id, v));
                                    k
                                }
                            };
                            ui.weak(known);
                        });
                        ui.end_row();
                        ui.label("Engines");
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.zxing, "zxing");
                            ui.checkbox(&mut self.grid, "grid");
                        });
                        ui.end_row();
                        ui.label("Read");
                        ui.vertical(|ui| {
                            ui.checkbox(&mut self.multi, "all codes in an image");
                            ui.checkbox(
                                &mut self.exhaustive,
                                "exhaustive (try every method, keep the majority)",
                            );
                        });
                        ui.end_row();
                        #[cfg(feature = "ml")]
                        {
                            ui.label("Trained models");
                            ui.vertical(|ui| {
                                ui.checkbox(&mut self.detector, "symbol detector (first try)");
                                ui.checkbox(
                                    &mut self.classifier,
                                    "module classifier (on failed grid fits)",
                                );
                            });
                            ui.end_row();
                        }
                        ui.label("Worker threads");
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut self.jobs).range(0..=64));
                            ui.weak(if self.jobs == 0 {
                                format!("default ({})", default_jobs())
                            } else {
                                String::new()
                            });
                        });
                        ui.end_row();
                    });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Defaults").clicked() {
                        *self = Settings::default();
                    }
                    ui.weak("applies to images not read yet");
                });
            },
        );
    }
}

fn polarity_combo(ui: &mut egui::Ui, value: &mut String) {
    let shown = POLARITIES
        .iter()
        .find(|p| p.0 == value)
        .map_or("any", |p| p.1);
    egui::ComboBox::from_id_salt(ui.next_auto_id())
        .selected_text(shown)
        .show_ui(ui, |ui| {
            for (k, label) in POLARITIES {
                ui.selectable_value(value, k.to_string(), label);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_cli() {
        let o = Settings::default().options();
        let d = ReadOptions::default();
        assert_eq!(o.budget_s, d.budget_s);
        assert_eq!(o.polarity, d.polarity);
        assert_eq!((o.engines.zxing, o.engines.grid), (true, true));
        assert!(!o.multi && !o.exhaustive && o.expect.is_none());
        assert_eq!((o.detector, o.classifier), (d.detector, d.classifier));
        assert!(o.detector && o.classifier);
    }

    #[test]
    fn options_and_json_round_trip() {
        let s = Settings {
            budget_s: 12.5,
            polarity: "light".into(),
            multi: true,
            expect: "24x24, 16X36".into(),
            grid: false,
            detector: false,
            jobs: 3,
            ..Default::default()
        };
        let o = s.options();
        assert_eq!(o.polarity, Polarity::Light);
        let sizes: Vec<_> = o.expect.unwrap().iter().map(|s| (s.rows, s.cols)).collect();
        assert_eq!(sizes, [(16, 36), (24, 24)]);
        assert!(o.multi && !o.engines.grid);
        assert!(!o.detector && o.classifier);
        assert_eq!(s.workers(), 3);
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
        // missing keys (older file) take the defaults
        let old: Settings = serde_json::from_str(r#"{"budget_s": 5.0}"#).unwrap();
        assert_eq!(old.budget_s, 5.0);
        assert!(old.zxing && old.detector && old.classifier);
    }

    #[test]
    fn save_and_load() {
        let d = std::env::temp_dir().join(format!("dm-gui-settings-{}", std::process::id()));
        let p = d.join("sub").join(FILE);
        assert_eq!(Settings::load_from(&p), Settings::default());
        let s = Settings {
            expect: "Näyttö".into(),
            multi: true,
            ..Default::default()
        };
        s.save_to(&p);
        assert_eq!(Settings::load_from(&p), s);
        std::fs::write(&p, "not json").unwrap();
        assert_eq!(Settings::load_from(&p), Settings::default());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn load_falls_back_to_the_old_dir() {
        let d = std::env::temp_dir().join(format!("dm-gui-old-{}", std::process::id()));
        let (new, old) = (d.join("new").join(FILE), d.join("old").join(FILE));
        let s = Settings {
            multi: true,
            ..Default::default()
        };
        s.save_to(&old);
        assert_eq!(Settings::load_first(&new, &old), s);
        let t = Settings::default();
        t.save_to(&new);
        assert_eq!(Settings::load_first(&new, &old), t);
        let _ = std::fs::remove_dir_all(&d);
    }
}
