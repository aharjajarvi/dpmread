//! Tool windows (settings, technical report) as their own OS windows: egui
//! immediate viewports, so they can move outside the main window and share
//! the app state directly. Where the backend has only one viewport egui
//! draws them as windows inside the main one instead (CORE-115).

use eframe::egui;

/// Show `add` in the window `id` while `*open`; the window's close button
/// (title bar or OS) clears `*open`.
pub fn popup(
    ctx: &egui::Context,
    id: &str,
    title: &str,
    size: [f32; 2],
    resizable: bool,
    open: &mut bool,
    add: impl FnOnce(&mut egui::Ui),
) {
    if !*open {
        return;
    }
    let builder = egui::ViewportBuilder::default()
        .with_title(title)
        .with_inner_size(size)
        .with_resizable(resizable)
        .with_close_button(true);
    let mut add = Some(add);
    let closed =
        ctx.show_viewport_immediate(egui::ViewportId::from_hash_of(id), builder, |ui, class| {
            if let Some(add) = add.take() {
                if class == egui::ViewportClass::EmbeddedWindow {
                    add(ui);
                } else {
                    egui::CentralPanel::default().show(ui, add);
                }
            }
            ui.ctx().input(|i| i.viewport().close_requested())
        });
    if closed {
        *open = false;
    }
}
