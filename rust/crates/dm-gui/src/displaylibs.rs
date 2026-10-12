//! Display libraries winit loads at run time (dlopen). A missing
//! `libxkbcommon-x11` makes winit panic, others give an unclear error, so
//! they are checked before the event loop starts.

use std::ffi::CString;

/// (soname, Debian / Ubuntu package) for winit's X11 backend.
const X11_LIBS: [(&str, &str); 8] = [
    ("libX11.so.6", "libx11-6"),
    ("libX11-xcb.so.1", "libx11-xcb1"),
    ("libxcb.so.1", "libxcb1"),
    ("libXcursor.so.1", "libxcursor1"),
    ("libXrandr.so.2", "libxrandr2"),
    ("libXi.so.6", "libxi6"),
    // these two are shipped next to dpmread-gui in the release zip (found
    // through the binary's RUNPATH), with libxcb-xkb.so.1
    ("libxkbcommon.so.0", "libxkbcommon0"),
    ("libxkbcommon-x11.so.0", "libxkbcommon-x11-0"),
];

/// The same for the Wayland backend.
const WAYLAND_LIBS: [(&str, &str); 2] = [
    ("libwayland-client.so.0", "libwayland-client0"),
    ("libxkbcommon.so.0", "libxkbcommon0"),
];

/// Whether dlopen finds `name` the way winit's dlopen does (called from
/// the executable, so its RUNPATH applies).
fn loads(name: &str) -> bool {
    let Ok(c) = CString::new(name) else {
        return false;
    };
    // SAFETY: plain dlopen / dlclose of a system library; the handle is
    // closed again at once and no symbol is used
    unsafe {
        let h = libc::dlopen(c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        if h.is_null() {
            return false;
        }
        libc::dlclose(h);
    }
    true
}

fn missing(libs: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    libs.iter().copied().filter(|(l, _)| !loads(l)).collect()
}

/// Message for missing libraries, with the install line.
fn message(libs: &[(&str, &str)], backend: &str) -> String {
    let names: Vec<&str> = libs.iter().map(|l| l.0).collect();
    let mut pkgs: Vec<&str> = libs.iter().map(|l| l.1).collect();
    pkgs.dedup();
    format!(
        "missing {backend} libraries: {}.\nOn Debian / Ubuntu: sudo apt install {}",
        names.join(", "),
        pkgs.join(" ")
    )
}

/// The backend to start on: `Ok(true)` X11, `Ok(false)` Wayland (or no
/// display at all: eframe reports that), `Err` the message for missing
/// libraries. X11 is preferred (file drop); without its libraries Wayland
/// is used when present, with a warning, since only drag and drop is lost.
pub fn use_x11() -> Result<bool, String> {
    let set = |v| std::env::var_os(v).is_some_and(|d| !d.is_empty());
    let wayland = set("WAYLAND_DISPLAY");
    if set("DISPLAY") && std::env::var_os("DPMREAD_GUI_WAYLAND").is_none() {
        let x = missing(&X11_LIBS);
        if x.is_empty() {
            return Ok(true);
        }
        if wayland && missing(&WAYLAND_LIBS).is_empty() {
            eprintln!(
                "warning: {}\nStarting on Wayland: drag and drop is not available, \
                 use the Add buttons.",
                message(&x, "X11")
            );
            return Ok(false);
        }
        return Err(message(&x, "X11"));
    }
    if wayland {
        let w = missing(&WAYLAND_LIBS);
        if !w.is_empty() {
            return Err(message(&w, "Wayland"));
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dlopen_check() {
        assert!(loads("libc.so.6"));
        assert!(!loads("libdpmread-does-not-exist.so.0"));
    }

    #[test]
    fn message_names_libs_and_packages() {
        let m = message(
            &[
                ("libxkbcommon-x11.so.0", "libxkbcommon-x11-0"),
                ("libX11.so.6", "libx11-6"),
            ],
            "X11",
        );
        assert_eq!(
            m,
            "missing X11 libraries: libxkbcommon-x11.so.0, libX11.so.6.\n\
             On Debian / Ubuntu: sudo apt install libxkbcommon-x11-0 libx11-6"
        );
    }
}
