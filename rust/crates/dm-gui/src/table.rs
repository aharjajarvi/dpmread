//! Results table: one row per image, plus one per further symbol of a
//! `multi` read ("file #2", …) right under it. Sorting by column header,
//! "failed only" filter, double-click opens the image's folder.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use dm_reader::{ReadResult, printable};
use eframe::egui;
use egui_extras::{Column, TableBuilder};

pub enum Status {
    Queued,
    /// Stage name and its consumed / total jobs (empty before the first).
    Running(String, usize, usize),
    Done(Box<ReadResult>),
    /// Load error.
    Error(String),
    /// Ended by Stop.
    Stopped,
}

pub struct Row {
    pub id: u64,
    pub path: PathBuf,
    pub status: Status,
}

impl Row {
    pub fn name(&self) -> String {
        file_name(&self.path)
    }

    pub fn finished(&self) -> bool {
        matches!(
            self.status,
            Status::Done(_) | Status::Error(_) | Status::Stopped
        )
    }

    pub fn decoded(&self) -> bool {
        matches!(&self.status, Status::Done(r) if !r.data.is_empty())
    }

    /// Finished without a read (also a load error; not stopped).
    pub fn failed(&self) -> bool {
        matches!(self.status, Status::Done(_) | Status::Error(_)) && !self.decoded()
    }

    /// The read symbols: the result itself and its `more` (empty if none).
    pub fn symbols(&self) -> Vec<&ReadResult> {
        match &self.status {
            Status::Done(r) if !r.data.is_empty() => {
                std::iter::once(&**r).chain(r.more.iter()).collect()
            }
            _ => Vec::new(),
        }
    }
}

pub fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Text of one table line.
#[derive(Debug, Default, PartialEq)]
pub struct Cells {
    pub icon: &'static str,
    pub name: String,
    pub engine: String,
    /// Decoded text, near-miss summary, progress or error.
    pub text: String,
    pub time: Option<f64>,
    pub size: String,
    /// Overall grade estimate (ISO/IEC 15415 style, from the grid fit).
    pub grade: String,
}

/// Symbol size "RxC" of a read (from the grid fit; empty if zxing alone).
fn size(r: &ReadResult) -> String {
    r.fit.as_ref().map(|f| f.geom.name()).unwrap_or_default()
}

/// Overall grade estimate of a read (empty without a grid fit).
pub fn grade(r: &ReadResult) -> String {
    r.quality
        .as_ref()
        .map(|q| q.overall_grade_estimate.to_string())
        .unwrap_or_default()
}

/// Cells of symbol `sym` of a row (0 = the row itself, k = `more[k-1]`).
pub fn cells(row: &Row, sym: usize) -> Cells {
    let name = row.name();
    if sym > 0 {
        let syms = row.symbols();
        let Some(r) = syms.get(sym) else {
            return Cells::default();
        };
        return Cells {
            icon: "✔",
            name: format!("{name} #{}", sym + 1),
            engine: r.engine.clone(),
            text: printable(&r.data, r.text_dec()),
            time: None,
            size: size(r),
            grade: grade(r),
        };
    }
    match &row.status {
        Status::Queued => Cells {
            icon: "·",
            name,
            text: "queued".into(),
            ..Default::default()
        },
        Status::Running(stage, done, total) => Cells {
            icon: "⏳",
            name,
            engine: "…".into(),
            text: if *total > 0 {
                format!("{stage} {done}/{total}")
            } else {
                "preparing".into()
            },
            ..Default::default()
        },
        Status::Done(r) if !r.data.is_empty() => Cells {
            icon: "✔",
            name,
            engine: r.engine.clone(),
            text: printable(&r.data, r.text_dec()),
            time: Some(r.elapsed),
            size: size(r),
            grade: grade(r),
        },
        Status::Done(r) => Cells {
            icon: "✖",
            name,
            engine: "—".into(),
            text: match &r.near_miss {
                Some(n) => format!("near miss {}", n.nm.size),
                None => "not decoded".into(),
            },
            time: Some(r.elapsed),
            size: String::new(),
            grade: String::new(),
        },
        Status::Error(e) => Cells {
            icon: "✖",
            name,
            engine: "—".into(),
            text: e.clone(),
            ..Default::default()
        },
        Status::Stopped => Cells {
            icon: "⏹",
            name,
            text: "stopped".into(),
            ..Default::default()
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    Status,
    Name,
    Engine,
    Text,
    Time,
    Size,
    Grade,
}

const COLUMNS: [(SortKey, &str); 7] = [
    (SortKey::Status, ""),
    (SortKey::Name, "File"),
    (SortKey::Engine, "Engine"),
    (SortKey::Text, "Text"),
    (SortKey::Time, "Time"),
    (SortKey::Size, "Size"),
    (SortKey::Grade, "Grade"),
];

/// Status order for sorting: failed, running, queued, decoded.
fn status_rank(row: &Row) -> u8 {
    match row.status {
        _ if row.failed() => 0,
        Status::Running(..) | Status::Stopped => 1,
        Status::Queued => 2,
        _ => 3,
    }
}

fn compare(a: &Row, b: &Row, key: SortKey) -> Ordering {
    let (ca, cb) = (cells(a, 0), cells(b, 0));
    match key {
        SortKey::Status => status_rank(a).cmp(&status_rank(b)),
        SortKey::Name => ca.name.to_lowercase().cmp(&cb.name.to_lowercase()),
        SortKey::Engine => ca.engine.cmp(&cb.engine),
        SortKey::Text => ca.text.cmp(&cb.text),
        SortKey::Time => ca.time.unwrap_or(-1.0).total_cmp(&cb.time.unwrap_or(-1.0)),
        // by module count, rows first
        // A first; no grade last
        SortKey::Grade => {
            let g = |c: &Cells| c.grade.chars().next().unwrap_or('Z');
            g(&ca).cmp(&g(&cb))
        }
        SortKey::Size => {
            let n = |r: &Row| {
                r.symbols()
                    .first()
                    .and_then(|s| s.fit.as_ref())
                    .map_or((0, 0), |f| (f.geom.rows * f.geom.cols, f.geom.rows))
            };
            n(a).cmp(&n(b))
        }
    }
}

/// Sort, filter and selection of the table.
#[derive(Default)]
pub struct TableState {
    /// Column and ascending; `None`: the order added.
    pub sort: Option<(SortKey, bool)>,
    pub failed_only: bool,
    /// Row id and symbol index.
    pub selected: Option<(u64, usize)>,
}

impl TableState {
    /// Lines to show as `(row index, symbol index)`: sorted and filtered
    /// rows, each followed by its further symbols.
    pub fn lines(&self, rows: &[Row]) -> Vec<(usize, usize)> {
        let mut order: Vec<usize> = (0..rows.len())
            .filter(|&i| !self.failed_only || rows[i].failed())
            .collect();
        if let Some((key, asc)) = self.sort {
            // stable: ties keep the order added
            order.sort_by(|&i, &j| {
                let o = compare(&rows[i], &rows[j], key);
                if asc { o } else { o.reverse() }
            });
        }
        order
            .into_iter()
            .flat_map(|i| {
                let n = rows[i].symbols().len().max(1);
                (0..n).map(move |s| (i, s))
            })
            .collect()
    }

    /// Header click: sort by `key` ascending, then descending, then unsorted.
    fn toggle(&mut self, key: SortKey) {
        self.sort = match self.sort {
            Some((k, true)) if k == key => Some((key, false)),
            Some((k, false)) if k == key => None,
            _ => Some((key, true)),
        };
    }

    /// Draw the table; selection and sorting change `self`. Returns the id
    /// of a row to read again (context menu).
    pub fn show(&mut self, ui: &mut egui::Ui, rows: &[Row]) -> Option<u64> {
        let mut requeue = None;
        let lines = self.lines(rows);
        // a click on the text selects the row, not the text
        ui.style_mut().interaction.selectable_labels = false;
        let row_h = ui.text_style_height(&egui::TextStyle::Body) + 6.0;
        let mono = egui::TextStyle::Monospace;
        TableBuilder::new(ui)
            .striped(true)
            .sense(egui::Sense::click())
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(Column::exact(22.0))
            .column(
                Column::initial(240.0)
                    .at_least(80.0)
                    .clip(true)
                    .resizable(true),
            )
            .column(Column::initial(60.0).at_least(40.0).resizable(true))
            .column(Column::remainder().at_least(120.0).clip(true))
            .column(Column::exact(56.0))
            .column(Column::exact(64.0))
            .column(Column::exact(48.0))
            .auto_shrink([false, false])
            .header(row_h, |mut header| {
                for (key, title) in COLUMNS {
                    header.col(|ui| {
                        let arrow = match self.sort {
                            Some((k, true)) if k == key => " ⏶",
                            Some((k, false)) if k == key => " ⏷",
                            _ => "",
                        };
                        let label = if title.is_empty() { "·" } else { title };
                        let b = egui::Button::new(
                            egui::RichText::new(format!("{label}{arrow}")).strong(),
                        )
                        .frame(false);
                        if ui.add(b).on_hover_text("sort").clicked() {
                            self.toggle(key);
                        }
                    });
                }
            })
            .body(|body| {
                body.rows(row_h, lines.len(), |mut tr| {
                    let (ri, sym) = lines[tr.index()];
                    let row = &rows[ri];
                    let c = cells(row, sym);
                    tr.set_selected(self.selected == Some((row.id, sym)));
                    tr.col(|ui| {
                        ui.label(c.icon);
                    });
                    tr.col(|ui| {
                        ui.add(egui::Label::new(&c.name).truncate())
                            .on_hover_text(row.path.display().to_string());
                    });
                    tr.col(|ui| {
                        ui.label(&c.engine);
                    });
                    tr.col(|ui| {
                        let t = egui::RichText::new(&c.text).text_style(mono.clone());
                        let t = if c.icon == "✔" { t } else { t.weak() };
                        ui.add(egui::Label::new(t).truncate());
                    });
                    tr.col(|ui| {
                        if let Some(t) = c.time {
                            ui.label(format!("{t:.1} s"));
                        }
                    });
                    tr.col(|ui| {
                        ui.label(&c.size);
                    });
                    tr.col(|ui| {
                        ui.label(&c.grade);
                    });
                    let resp = tr.response();
                    if resp.clicked() {
                        self.selected = Some((row.id, sym));
                    }
                    if resp.double_clicked() {
                        reveal(&row.path);
                    }
                    resp.context_menu(|ui| {
                        if ui
                            .add_enabled(row.finished(), egui::Button::new("Read again"))
                            .clicked()
                        {
                            requeue = Some(row.id);
                            ui.close();
                        }
                        if ui.button("Open folder").clicked() {
                            reveal(&row.path);
                            ui.close();
                        }
                        if ui.button("Copy path").clicked() {
                            ui.ctx().copy_text(row.path.display().to_string());
                            ui.close();
                        }
                    });
                });
            });
        requeue
    }
}

/// Open the folder of `path` in the file manager (the file selected where
/// the platform supports it). Errors are ignored.
pub fn reveal(path: &Path) {
    use std::process::Command;
    let r = if cfg!(windows) {
        Command::new("explorer")
            .arg(format!("/select,{}", path.display()))
            .spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg("-R").arg(path).spawn()
    } else {
        let dir = path.parent().unwrap_or(Path::new("."));
        Command::new("xdg-open").arg(dir).spawn()
    };
    if let Err(e) = r {
        eprintln!("cannot open folder of {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(id: u64, name: &str, data: &[u8], elapsed: f64, more: usize) -> Row {
        let one = |d: &[u8]| ReadResult {
            path: PathBuf::from(name),
            engine: "zxing".into(),
            data: d.to_vec(),
            elapsed,
            ..Default::default()
        };
        let mut r = one(data);
        r.more = (0..more)
            .map(|k| one(format!("more{k}").as_bytes()))
            .collect();
        Row {
            id,
            path: PathBuf::from(format!("/x/{name}")),
            status: Status::Done(Box::new(r)),
        }
    }

    fn rows() -> Vec<Row> {
        vec![
            done(0, "b.png", b"BBB", 2.0, 0),
            Row {
                id: 1,
                path: "/x/c.png".into(),
                status: Status::Error("cannot read".into()),
            },
            done(2, "A.png", b"AAA", 1.0, 2),
            done(3, "d.png", b"", 9.0, 0),
            Row {
                id: 4,
                path: "/x/e.png".into(),
                status: Status::Queued,
            },
        ]
    }

    #[test]
    fn further_symbols_get_their_own_lines() {
        let rows = rows();
        let t = TableState::default();
        assert_eq!(
            t.lines(&rows),
            [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0), (4, 0)]
        );
        let c = cells(&rows[2], 2);
        assert_eq!((c.name.as_str(), c.text.as_str()), ("A.png #3", "more1"));
        assert_eq!(cells(&rows[2], 0).time, Some(1.0));
        assert_eq!(cells(&rows[3], 0).text, "not decoded");
        assert_eq!(cells(&rows[1], 0).icon, "✖");
    }

    #[test]
    fn sort_keeps_symbols_under_their_file() {
        let rows = rows();
        let mut t = TableState::default();
        t.toggle(SortKey::Name);
        assert_eq!(
            t.lines(&rows),
            [(2, 0), (2, 1), (2, 2), (0, 0), (1, 0), (3, 0), (4, 0)]
        );
        t.toggle(SortKey::Name);
        assert_eq!(t.lines(&rows)[0], (4, 0));
        t.toggle(SortKey::Name);
        assert_eq!(t.sort, None);
        t.toggle(SortKey::Time);
        let first: Vec<usize> = t.lines(&rows).iter().map(|l| l.0).collect();
        assert_eq!(first, [1, 4, 2, 2, 2, 0, 3]);
        t.toggle(SortKey::Status);
        assert_eq!(t.lines(&rows)[..2], [(1, 0), (3, 0)]);
    }

    #[test]
    fn failed_only() {
        let rows = rows();
        let t = TableState {
            failed_only: true,
            ..Default::default()
        };
        assert_eq!(t.lines(&rows), [(1, 0), (3, 0)]);
    }
}
