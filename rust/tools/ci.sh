#!/usr/bin/env bash
# Local CI for the Rust reader (P6.4): every check of a phase end and of an
# accepted reader change, then the release zip. Stops at the first failure.
#
#   rust/tools/ci.sh            # all steps (release check, ~27 min)
#   rust/tools/ci.sh --quick    # fmt, clippy, tests, parity (no public sets,
#                               # no container build, no zip; ~6 min)
#   rust/tools/ci.sh --fast     # default while developing (~40 s): fmt,
#                               # clippy, cargo deny (no advisories), tests
#                               # without #[ignore] and without
#                               # round_trip, parity gate (no --jobs check)
#                               # with the quick-to-build ci profile
#   rust/tools/ci.sh --gui      # GUI-only change: fmt, clippy and tests of
#                               # dm-gui, debug build of dpmread-gui; fails if
#                               # uncommitted files outside dm-gui changed
#
# Steps: cargo fmt --check; clippy -D warnings; cargo deny check (licence
# allowlist, bans, sources, advisories: deny.toml); THIRD_PARTY_LICENSES.md
# up to date with cargo about (about.toml, about.hbs); cargo test --workspace;
# parity vs the v2 baseline with the --jobs check; public sets vs
# baseline/public_rust*.json (skipped if not fetched); old-glibc container
# build (build_linux.sh, needs docker; feature ml); zip of both container-built binaries
# (plus the libraries the GUI ships, README.txt, LICENSE, NOTICE and
# THIRD_PARTY_LICENSES.md; release_zip.py) in rust/target/dist/.
#
# Private test data (not part of this repo): DM_PRIVATE_DIR points at it.
# With its fixtures/ (generated on CachyOS) the tests marked #[ignore] run
# too; with its testimages/ the parity step runs. Without them both are
# reported as skipped and the rest still has to pass.
#
# Windows (Git Bash, P6.3): same checks with the native MSVC build; instead
# of the container build, a release build of both binaries and a check that
# they need no C/C++ run-time DLL (check_dlls.py); zip ...-windows-x86_64.
# CMake is taken from Visual Studio when it is not on PATH.
set -euo pipefail

RUST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$RUST_DIR")"
PRIVATE="${DM_PRIVATE_DIR:-}"
if [ -n "$PRIVATE" ] && [ -d "$PRIVATE" ]; then
    PRIVATE="$(cd "$PRIVATE" && pwd)"
    export DM_PRIVATE_DIR="$PRIVATE"
fi
QUICK=0 GUI=0 FAST=0
case "${1:-}" in
    "") ;;
    --quick) QUICK=1 ;;
    --gui) GUI=1 ;;
    --fast) FAST=1 ;;
    *) echo "usage: ci.sh [--fast | --gui | --quick]" >&2; exit 2 ;;
esac
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
WINDOWS=0
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) WINDOWS=1 ;; esac

step() { printf '\n=== %s\n' "$*"; }

if [ $WINDOWS = 1 ] && ! command -v cmake >/dev/null; then
    vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    vs=$("$vswhere" -latest -products '*' -property installationPath | tr -d '\r')
    PATH="$(cygpath "$vs")/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin:$PATH"
    command -v cmake >/dev/null || { echo "cmake not found (Visual Studio C++ CMake tools)" >&2; exit 1; }
fi

cd "$RUST_DIR"
step "fmt"
cargo fmt --check

if [ $GUI = 1 ]; then
    # The reader checks (parity, public sets) cannot change when only dm-gui
    # does; any other uncommitted change needs --fast or more.
    step "changes outside dm-gui"
    other=$(git -C "$ROOT" status --porcelain --untracked-files=all \
        | cut -c4- | grep -v '^rust/crates/dm-gui/' || true)
    if [ -n "$other" ]; then
        printf '%s\n' "$other"
        echo "error: --gui covers changes in rust/crates/dm-gui/ only" >&2
        exit 1
    fi
    echo "none"

    step "clippy (dm-gui)"
    cargo clippy -p dm-gui --all-targets -- -D warnings
    cargo clippy -p dm-gui --features ml --all-targets -- -D warnings

    step "test (dm-gui)"
    if [ -n "$PRIVATE" ] && [ -d "$PRIVATE/testimages" ]; then
        cargo test -p dm-gui -- --include-ignored
        cargo test -p dm-gui --features ml -- --include-ignored
    else
        cargo test -p dm-gui
        echo "private test images not found (DM_PRIVATE_DIR='$PRIVATE'): tests marked #[ignore] skipped"
    fi

    step "build dpmread-gui (debug)"
    cargo build -p dm-gui
    ls -l target/debug/dpmread-gui
    step "GUI checks passed (reader checks and release build not run)"
    exit 0
fi

step "clippy"
cargo clippy --workspace --all-targets -- -D warnings
# the detector and classifier build only with the feature `ml`
cargo clippy -p dm-reader -p dm-cli -p dm-gui --features ml --all-targets -- -D warnings
cargo test -p dm-reader --features ml --lib ml::

if [ $FAST = 1 ]; then
    # advisories need the network; --quick and the full run check them
    step "cargo deny (licences, bans, sources)"
    cargo deny check licenses bans sources

    # round_trip (~55 s) and the #[ignore]d fixture tests run in --quick
    step "test (fast)"
    cargo test --workspace -- --exact --skip round_trip

    step "parity vs v2 (ci profile, no --jobs check)"
    cargo build --profile ci -p dm-cli
    cd "$ROOT"
    if [ -n "$PRIVATE" ] && [ -d "$PRIVATE/testimages" ]; then
        python rust/tools/parity.py --baseline v2 --dpmread rust/target/ci/dpmread
    else
        echo "private test images not found (DM_PRIVATE_DIR='$PRIVATE'): parity skipped"
    fi
    step "fast checks passed (ignored tests, round_trip, --jobs check and release steps not run)"
    exit 0
fi

step "cargo deny"
cargo deny check

step "THIRD_PARTY_LICENSES.md"
cargo about generate --fail about.hbs -o "$TMP/THIRD_PARTY_LICENSES.md" 2>/dev/null
if ! cmp -s "$TMP/THIRD_PARTY_LICENSES.md" "$ROOT/THIRD_PARTY_LICENSES.md"; then
    echo "error: THIRD_PARTY_LICENSES.md is out of date; regenerate it in rust/:" >&2
    echo "  cargo about generate --fail about.hbs -o ../THIRD_PARTY_LICENSES.md" >&2
    exit 1
fi
echo "up to date"

step "test"
if [ -n "$PRIVATE" ] && [ -d "$PRIVATE/fixtures" ]; then
    echo "private test data: $PRIVATE (tests marked #[ignore] included)"
    cargo test --workspace -- --include-ignored
    # the GUI's detector toggle (feature ml, private image)
    cargo test -p dm-gui --features ml -- --include-ignored
else
    cargo test --workspace
    echo "private fixtures not found (DM_PRIVATE_DIR='$PRIVATE'): tests marked #[ignore] skipped"
fi

step "parity vs v2"
cargo build --release -p dm-cli
cd "$ROOT"
if [ -n "$PRIVATE" ] && [ -d "$PRIVATE/testimages" ]; then
    python rust/tools/parity.py --baseline v2 --jobs-check
else
    echo "private test images not found (DM_PRIVATE_DIR='$PRIVATE'): parity skipped"
fi

if [ $QUICK = 1 ]; then
    step "quick run: public sets, container build and zip skipped"
    exit 0
fi

step "public sets"
if [ -d testimages_public ]; then
    python tools/score_public.py --reader rust --save "$TMP/public_rust.json"
    python tools/score_public.py --compare baseline/public_rust.json "$TMP/public_rust.json"
    python tools/score_public.py --reader rust --multi --save "$TMP/public_rust_multi.json"
    python tools/score_public.py --compare baseline/public_rust_multi.json "$TMP/public_rust_multi.json"
else
    echo "testimages_public/ not fetched (python tools/fetch_public_sets.py): skipped"
fi

if [ $WINDOWS = 1 ]; then
    step "windows build"
    cd "$RUST_DIR"
    cargo build --release --locked -p dm-cli -p dm-gui --features dm-cli/ml,dm-gui/ml
    python tools/check_dlls.py target/release/dpmread.exe target/release/dpmread-gui.exe
    os=windows-x86_64 bin_dir="$RUST_DIR/target/release" ext=.exe libs=
else
    step "old-glibc build"
    "$RUST_DIR/tools/build_linux.sh"
    os=linux-x86_64 bin_dir="$RUST_DIR/target/linux-x86_64/release" ext=
    # shipped next to dpmread-gui by build_linux.sh (stock Ubuntu lacks them)
    libs="libxkbcommon.so.0 libxkbcommon-x11.so.0 libxcb-xkb.so.1"
fi

step "release zip"
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$RUST_DIR/Cargo.toml" | head -1)
commit=$(git -C "$ROOT" describe --always --dirty)
name="dpmread-$version-$commit-$os"
mkdir -p "$RUST_DIR/target/dist"
zip="$RUST_DIR/target/dist/$name.zip"
python "$RUST_DIR/tools/release_zip.py" "$zip" "$name" "$bin_dir" "$ext" $libs
ls -l "$zip"
step "CI passed"
