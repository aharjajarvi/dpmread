# CMake toolchain file for the Windows MSVC build, passed by cmake-rs to the
# bundled C/C++ libraries (see rust/.cargo/config.toml). The binaries link
# the static C/C++ run-time (crt-static) so they need no VC++ redistributable;
# zxing-cpp would otherwise use the DLL run-time (CMake's default since
# policy CMP0091), and mixing both gives LNK4098 and two CRTs in one binary.
# libjpeg-turbo picks the static run-time itself (WITH_CRT_DLL off).
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded CACHE STRING "static release CRT, as Rust's crt-static")
