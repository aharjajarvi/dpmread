#!/usr/bin/env bash
# Release build of dpmread and dpmread-gui on an old glibc (P6.2), in a
# manylinux_2_28 container (linux-build.Dockerfile), so the binaries run on
# most Linux distros, not only on the build machine's.
#
#   rust/tools/build_linux.sh            # -> rust/target/linux-x86_64/release/
#
# Also copies the libraries the GUI needs on stock Ubuntu next to it
# (BUNDLED_LIBS below); the release zip ships them.
#
# Needs docker. The image is tagged with the Rust version (the host's rustc
# version, or RUST_VERSION=...). The cargo registry is kept
# in rust/target/linux-x86_64/cargo-home between runs.
set -euo pipefail

RUST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_VERSION="${RUST_VERSION:-$(rustc --version 2>/dev/null | awk '{print $2}')}"
RUST_VERSION="${RUST_VERSION:-stable}"
IMAGE="dpmread-linux-build:$RUST_VERSION"
OUT="target/linux-x86_64"
MAX_GLIBC="2.28"

# cached after the first run; rebuilt when the Dockerfile changes
docker build -q --build-arg RUST_VERSION="$RUST_VERSION" -t "$IMAGE" \
    -f "$RUST_DIR/tools/linux-build.Dockerfile" "$RUST_DIR/tools" >/dev/null

mkdir -p "$RUST_DIR/$OUT/cargo-home"
docker run --rm -u "$(id -u):$(id -g)" \
    -v "$RUST_DIR:/src" -w /src \
    -e CARGO_HOME="/src/$OUT/cargo-home" \
    -e CARGO_TARGET_DIR="/src/$OUT" \
    "$IMAGE" cargo build --release --locked -p dm-cli -p dm-gui

# Libraries shipped next to dpmread-gui (found through its $ORIGIN RUNPATH,
# dm-gui/build.rs): winit dlopens libxkbcommon-x11 for X11, which stock
# Ubuntu desktops lack, as well as its dependency libxcb-xkb. libxkbcommon
# is shipped too: libxkbcommon-x11 has a copy of its internals and crashes
# with another version (0.9.1 with Ubuntu 26.04's 1.13). Both are built in
# the image (/opt/xkb, linux-build.Dockerfile), libxcb-xkb is the system's.
# The x11 copy gets its own $ORIGIN RUNPATH so it finds the other two there.
# Licences: NOTICE.
BUNDLED_LIBS="libxkbcommon.so.0 libxkbcommon-x11.so.0 libxcb-xkb.so.1"
docker run --rm -u "$(id -u):$(id -g)" -v "$RUST_DIR:/src" -w "/src/$OUT/release" \
    "$IMAGE" bash -c "for l in $BUNDLED_LIBS; do
            cp -L /opt/xkb/usr/lib64/\$l . 2>/dev/null || cp -L /usr/lib64/\$l .
            chmod 644 \$l; done
        patchelf --set-rpath '\$ORIGIN' libxkbcommon-x11.so.0"
readelf -d "$RUST_DIR/$OUT/release/dpmread-gui" | grep -q 'RUNPATH.*\[\$ORIGIN\]' || {
    echo "error: dpmread-gui has no \$ORIGIN RUNPATH" >&2
    exit 1
}

# The newest glibc symbol version each binary needs must not exceed the
# container's, or the build leaked a newer dependency.
status=0
for bin in dpmread dpmread-gui $BUNDLED_LIBS; do
    path="$RUST_DIR/$OUT/release/$bin"
    glibc=$(objdump -T "$path" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
    glibcxx=$(objdump -T "$path" | { grep -o 'GLIBCXX_[0-9.]*' || true; } | sort -uV | tail -1)
    printf '%-21s %9s bytes  %s  %s\n' "$bin" "$(stat -c %s "$path")" "$glibc" "$glibcxx"
    if [ "$(printf '%s\n' "${glibc#GLIBC_}" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]; then
        echo "error: $bin needs $glibc, more than glibc $MAX_GLIBC" >&2
        status=1
    fi
done
exit $status
