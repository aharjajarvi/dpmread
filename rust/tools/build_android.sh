#!/usr/bin/env bash
# Release build of the dm-ffi library for Android apps: libdm_ffi.so for
# arm64-v8a (phones) and x86_64 (emulator), minimum API 26, plus the Kotlin
# bindings generated from it.
#
#   rust/tools/build_android.sh          # -> rust/target/android/{jniLibs,kotlin}/
#
# Needs the Android NDK (ANDROID_NDK_HOME, or the newest NDK under
# $ANDROID_HOME/ndk or ~/Android/Sdk/ndk), cargo-ndk and the Rust targets
# aarch64-linux-android and x86_64-linux-android.
set -euo pipefail

RUST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$RUST_DIR/target/android"
API=26

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    SDK="${ANDROID_HOME:-$HOME/Android/Sdk}"
    ANDROID_NDK_HOME="$(ls -d "$SDK"/ndk/*/ 2>/dev/null | sort -V | tail -n 1)"
    ANDROID_NDK_HOME="${ANDROID_NDK_HOME%/}"
fi
if [[ ! -d "$ANDROID_NDK_HOME" ]]; then
    echo "no Android NDK found: set ANDROID_NDK_HOME" >&2
    exit 1
fi
# CMake's Android support (zxing-cpp, libjpeg-turbo) reads ANDROID_NDK_ROOT
export ANDROID_NDK_HOME ANDROID_NDK_ROOT="$ANDROID_NDK_HOME"

cd "$RUST_DIR"
rm -rf "$OUT/jniLibs" "$OUT/kotlin"
cargo ndk -t arm64-v8a -t x86_64 -P "$API" -o "$OUT/jniLibs" \
    build --release --locked -p dm-ffi

# the bindings come from the library's metadata; any target's build has it
cargo build --locked -p dm-ffi
cargo run -q --locked -p dm-ffi --features bindgen --bin uniffi-bindgen -- \
    generate --library target/debug/libdm_ffi.so --language kotlin \
    --no-format --out-dir "$OUT/kotlin"

find "$OUT" -type f -printf '%P  %s bytes\n' | sort
