//! Detail pane of the selected line: the image viewer, the key facts of
//! the symbol, Copy for the text and the CLI's full text report.

use dm_reader::{Quad, ReadResult, printable, report_text};
use eframe::egui;

use crate::preview::Preview;
use crate::table::{Row, Status};

/// `(label, value)` facts of a read symbol.
pub fn facts(r: &ReadResult) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    let engine = match &r.zx {
        Some(z) if !z.binarizer.is_empty() => format!("{} ({})", r.engine, z.binarizer),
        _ => r.engine.clone(),
    };
    out.push(("Engine", engine));
    out.push(("Variant", r.variant.clone()));
    if !r.sampling.is_empty() {
        out.push(("Sampling", r.sampling.clone()));
    }
    if let Some(f) = &r.fit {
        out.push((
            "Symbol",
            format!(
                "{}{}",
                f.geom.name(),
                if r.mirrored { ", mirrored" } else { "" }
            ),
        ));
    }
    if let Some(rs) = &r.rs {
        out.push(("ECC used", format!("{:.0}%", (1.0 - rs.uec) * 100.0)));
    }
    if let Some(q) = &r.quality {
        out.push(("Module", format!("{:.1} px", q.module_size_px)));
        out.push(("Grade (est.)", q.overall_grade_estimate.to_string()));
    }
    out.push((
        "Time",
        format!("{:.1} s, {} attempts", r.elapsed, r.attempts),
    ));
    out
}

fn fact_grid(ui: &mut egui::Ui, id: &str, rows: &[(&'static str, String)]) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            for (k, v) in rows {
                ui.weak(*k);
                ui.label(v);
                ui.end_row();
            }
        });
}

/// Draw the pane for symbol `sym` of `row`.
pub fn show(
    ui: &mut egui::Ui,
    row: &Row,
    sym: usize,
    preview: &mut Preview,
    report_open: &mut bool,
) {
    let syms = row.symbols();
    let res = syms.get(sym).copied();
    preview.request(ui.ctx(), (row.id, sym), &row.path, res);
    let quads: Vec<Quad> = syms.iter().filter_map(|s| s.quad).collect();
    let selected = res
        .and_then(|r| r.quad)
        .and_then(|q| quads.iter().position(|x| *x == q));

    egui::Panel::right("facts")
        .resizable(true)
        .default_size(ui.available_width() * 0.42)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    let title = if sym > 0 {
                        format!("{} #{}", row.name(), sym + 1)
                    } else {
                        row.name()
                    };
                    ui.label(egui::RichText::new(title).strong())
                        .on_hover_text(row.path.display().to_string());
                    ui.add_space(4.0);
                    match (&row.status, res) {
                        (_, Some(r)) => {
                            let text = printable(&r.data, r.text_dec());
                            ui.horizontal(|ui| {
                                if ui
                                    .button("Copy")
                                    .on_hover_text("copy the decoded text")
                                    .clicked()
                                {
                                    ui.ctx().copy_text(r.text.clone());
                                }
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&text).monospace().strong(),
                                    )
                                    .wrap(),
                                );
                            });
                            ui.add_space(4.0);
                            fact_grid(ui, "facts_grid", &facts(r));
                        }
                        (Status::Done(r), None) => {
                            let mut f = vec![("Result", "not decoded".to_string())];
                            if let Some(n) = &r.near_miss {
                                f.push(("Near miss", format!("{} on {}", n.nm.size, n.variant)));
                                f.push(("Module", format!("{:.1} px", n.modules_px_orig)));
                            }
                            f.push((
                                "Time",
                                format!("{:.1} s, {} attempts", r.elapsed, r.attempts),
                            ));
                            fact_grid(ui, "facts_grid", &f);
                        }
                        (Status::Error(e), None) => {
                            ui.colored_label(ui.visuals().error_fg_color, e);
                        }
                        _ => {
                            ui.weak("not read yet");
                        }
                    }
                    if let Status::Done(_) = &row.status {
                        ui.add_space(6.0);
                        if ui
                            .selectable_label(*report_open, "Technical report…")
                            .clicked()
                        {
                            *report_open = !*report_open;
                        }
                    }
                });
        });
    egui::CentralPanel::default().show(ui, |ui| {
        preview.show(ui, &quads, selected);
    });
    if let (Status::Done(r), true) = (&row.status, *report_open) {
        report_window(ui.ctx(), row, r, res, report_open);
    }
}

/// Quiet zone around the drawn module matrix, in modules.
const QUIET_MODULES: usize = 2;

/// The sampled module matrix as black / white squares with a quiet zone.
fn module_matrix(ui: &mut egui::Ui, bits: &[u8], rows: usize, cols: usize) {
    let (w, h) = (cols + 2 * QUIET_MODULES, rows + 2 * QUIET_MODULES);
    let cell = (360.0 / w.max(h) as f32).clamp(3.0, 12.0).floor();
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(w as f32 * cell, h as f32 * cell),
        egui::Sense::hover(),
    );
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, egui::Color32::WHITE);
    for r in 0..rows {
        for c in 0..cols {
            if bits.get(r * cols + c).is_some_and(|&b| b != 0) {
                let min = rect.min
                    + egui::vec2(
                        (c + QUIET_MODULES) as f32 * cell,
                        (r + QUIET_MODULES) as f32 * cell,
                    );
                p.rect_filled(
                    egui::Rect::from_min_size(min, egui::vec2(cell, cell)),
                    0.0,
                    egui::Color32::BLACK,
                );
            }
        }
    }
}

/// The CLI's text report in its own window, readable size; the module
/// matrix of the selected symbol drawn above it (the text art needs a font
/// with exact block glyphs, which **Copy report** keeps for editors).
fn report_window(
    ctx: &egui::Context,
    row: &Row,
    r: &ReadResult,
    sym: Option<&ReadResult>,
    open: &mut bool,
) {
    egui::Window::new(format!("Technical report – {}", row.name()))
        .id(egui::Id::new("technical_report"))
        .open(open)
        .resizable(true)
        .default_size([760.0, 600.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button("Copy report")
                    .on_hover_text("the full text report, with the module matrix as text art")
                    .clicked()
                {
                    ui.ctx().copy_text(report_text(r, true));
                }
            });
            ui.separator();
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if let Some(f) = sym.and_then(|s| s.fit.as_ref())
                        && let Some(bits) = &f.bits
                    {
                        ui.weak(format!("Sampled module matrix {}", f.geom.name()));
                        module_matrix(ui, bits, f.geom.rows, f.geom.cols);
                        ui.add_space(8.0);
                    }
                    ui.add(
                        egui::Label::new(egui::RichText::new(report_text(r, false)).monospace())
                            .extend(),
                    );
                });
        });
}
