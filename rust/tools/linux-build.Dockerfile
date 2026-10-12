# Old-glibc build image for the release binaries (P6.2), used by
# build_linux.sh. manylinux_2_28 = AlmaLinux 8: glibc 2.28, GCC 14 toolset
# (zxing-cpp needs C++20), CMake. Binaries built here run on Debian 10+,
# Ubuntu 20.04+, RHEL / Alma / Rocky 8+.
FROM quay.io/pypa/manylinux_2_28_x86_64

ARG RUST_VERSION=stable
ENV RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | CARGO_HOME=/opt/cargo sh -s -- -y \
        --profile minimal --default-toolchain "$RUST_VERSION" --no-modify-path \
    && chmod -R a+rwX /opt/rustup /opt/cargo \
    && rustc --version && cmake --version | head -1 && g++ --version | head -1

# The zxing-cpp 0.5.1 build script links from <out>/lib, but GNUInstallDirs
# installs to lib64 on RHEL-like systems; the cmake crate passes this
# toolchain file to CMake.
RUN echo 'set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "")' > /opt/libdir.cmake
ENV CMAKE_TOOLCHAIN_FILE=/opt/libdir.cmake

# libxkbcommon and libxkbcommon-x11 (with libxcb-xkb) are shipped next to
# dpmread-gui: winit needs the x11 part and stock Ubuntu desktops lack it
# (build_linux.sh). Both from one current release, built here on glibc
# 2.28: AlmaLinux 8's 0.9.1 is too old for current Compose files, and the
# x11 part only works with the libxkbcommon of its own version. Installed
# to /opt/xkb with /usr paths compiled in (XKB data, X locale). Needs bison
# >= 3.6 (AlmaLinux 8: 3.0).
ARG BISON=bison-3.8.2
ARG BISON_SHA256=9bba0214ccf7f1079c5d59210045227bcf619519840ebfa80cd3849cff5a5bf2
ARG XKB=xkbcommon-1.13.1
ARG XKB_SHA256=aeb951964c2f7ecc08174cb5517962d157595e9e3f38fc4a130b91dc2f9fec18
RUN dnf -y -q install libxcb-devel m4 && dnf clean all \
    && /opt/python/cp312-cp312/bin/python -m pip install -q --root-user-action=ignore meson==1.6.1 ninja==1.11.1.3 \
    && cd /tmp \
    && curl -sSfL -o b.tar.xz "https://ftp.gnu.org/gnu/bison/$BISON.tar.xz" \
    && echo "$BISON_SHA256  b.tar.xz" | sha256sum -c - \
    && tar xf b.tar.xz && (cd "$BISON" && ./configure -q --prefix=/opt/bison \
        && make -s -j"$(nproc)" && make -s install) \
    && curl -sSfL -o x.tar.gz \
        "https://github.com/xkbcommon/libxkbcommon/archive/refs/tags/$XKB.tar.gz" \
    && echo "$XKB_SHA256  x.tar.gz" | sha256sum -c - \
    && tar xf x.tar.gz && cd "libxkbcommon-$XKB" \
    && export PATH=/opt/bison/bin:/opt/python/cp312-cp312/bin:$PATH \
    && meson setup build --prefix=/usr --libdir=lib64 \
        --buildtype=release -Denable-wayland=false -Denable-xkbregistry=false \
        -Denable-tools=false -Denable-docs=false -Denable-bash-completion=false \
    && meson compile -C build && DESTDIR=/opt/xkb meson install -C build \
    && cp LICENSE /opt/xkb/LICENSE \
    && cd / && rm -rf /tmp/* /opt/bison
