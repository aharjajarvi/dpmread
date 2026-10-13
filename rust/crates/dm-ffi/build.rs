//! Android: link the NDK's static C++ library (`c++_static`, `c++abi`) for
//! zxing-cpp. The zxing-cpp crate links `stdc++`, which on Android is the
//! system's minimal library without the `std::__ndk1` symbols, so the `.so`
//! would fail to load. The NDK linker (lld) finds both in the NDK sysroot.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        println!("cargo:rustc-link-lib=static:-bundle=c++_static");
        println!("cargo:rustc-link-lib=static:-bundle=c++abi");
    }
}
