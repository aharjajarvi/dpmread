// Linux: the release zip ships libxkbcommon(-x11) next to dpmread-gui (stock
// Ubuntu lacks the x11 part); winit dlopens them from the executable, which
// searches the executable's DT_RUNPATH first, so point that at the binary's
// own folder.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--enable-new-dtags,-rpath,$ORIGIN");
    }
}
