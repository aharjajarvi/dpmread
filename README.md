# dpmread

**Data Matrix (ECC 200) reader for damaged and low-contrast direct part
marks: laser-burned, dot-peened, blurred.** Command-line tool and desktop
GUI for Windows and Linux, written in Rust.

General-purpose barcode libraries read clean printed codes quickly but give
up on marks burned into wood, peened into metal or photographed out of
focus. dpmread runs a fast library pass first and, when that fails, its own
grid decoder: it fits the module grid to the symbol's finder and timing
pattern (also when the module pitch drifts across the mark), samples each
module with a confidence value, and lets Reed-Solomon error correction
treat the least certain modules as erasures, which roughly doubles the
damage it can repair.

> **Status:** early (0.x). The reader works and is tested, but command-line
> flags and the JSON output may still change. See
> [CHANGELOG.md](CHANGELOG.md).

## Features

- Square, rectangular and DMRE (ISO/IEC 21471) symbol sizes; dark-on-light
  and light-on-dark; mirrored symbols.
- Enhancement ladder (contrast, background flattening, thresholds, blur,
  morphology) tried in order until a symbol decodes, within a time budget
  per image.
- Grid decoder with soft-erasure Reed-Solomon, rescue sampling and a
  warped grid for marks whose module pitch is not uniform.
- Several symbols per image (`--multi`), large photos (tiled search).
- Full technical report per symbol: size, error-correction use, encodation
  segments (ASCII, C40, Text, X12, EDIFACT, Base 256, GS1, ECI), quality
  estimate in the style of ISO/IEC 15415, corner points.
  Text or JSON output; annotated debug images.
- Desktop GUI: drag and drop files or folders, results table, annotated
  image with zoom, technical report, CSV / JSON export.
- Self-contained binaries, nothing to install: no runtime, no DLLs on
  Windows 10/11 x64; on Linux x86-64 (glibc 2.28 or newer) the GUI comes
  with the three keyboard libraries for X11 that some desktops lack.

Details, performance and benchmark results: [doc/features.md](doc/features.md).
Planned work: [doc/roadmap.md](doc/roadmap.md).

## Results on public test sets

193 images from three public sets (a research dataset of industrial marks
and the zxing / zxing-cpp test samples), one symbol per image, 60 s budget:

| Reader | Images decoded | False reads |
|---|---|---|
| dpmread 0.2 | **188 / 191** (98.4 %) | 0 |
| Python reference reader in this repo | 181 / 191 (94.8 %) | 1 |

How the sets are fetched and scored, and a comparison with zxing-cpp alone:
[doc/features.md#benchmarks](doc/features.md#benchmarks).

## Download

Release zips for Windows x64 and Linux x86-64 are on the
[releases page](https://github.com/aharjajarvi/dpmread/releases). Unpack
and run; nothing to install.

## Usage

GUI: start `dpmread-gui` and drop images or folders on the window, or pass
them on the command line: `dpmread-gui IMAGE_OR_FOLDER ...`.

![dpmread-gui: eight photos of marks on metal parts decoded; the selected one,
read by the grid decoder, outlined in the image with its text and read
details beside it](doc/gui.png)

*Images from the Karrach 2021 set (L. Karrach, E. Pivarčiová, J. Imaging 7(9), 163,
2021), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), scaled in the screenshot.*

Command line:

```sh
dpmread photo.jpg                     # text report
dpmread folder/ --json > results.json # every image in a folder, JSON
dpmread photo.jpg --multi             # every symbol in the image
dpmread photo.jpg --expect 24x24 --polarity dark   # known size and polarity: faster
dpmread photo.jpg --debug out/        # enhancement variants + annotated result
```

| Option | Meaning |
|---|---|
| `--json` | JSON output (an array with one object per image) |
| `--budget SEC` | maximum time per image (default 40) |
| `--expect RxC` | expected symbol size, repeatable; prunes the search |
| `--polarity any\|dark\|light` | mark polarity (default: try both) |
| `--multi` | read every symbol in the image |
| `--engines zxing,grid` | restrict the decoders |
| `--exhaustive` | try every method and keep the majority result |
| `--jobs N` | worker threads (results do not depend on it) |
| `--matrix` | print the sampled module matrix |
| `--debug DIR` | save the enhancement variants and an annotated image |

Exit code: 0 if every image decoded, 1 if an image did not decode or
could not be read, 2 on invalid options.

## Building from source

The reader is a Cargo workspace in `rust/`. It needs Rust (stable, via
[rustup](https://rustup.rs/)), CMake and a C++ compiler, because zxing-cpp
is built from source.

Linux:

```sh
# Arch / CachyOS: sudo pacman -S rustup cmake gcc
# Debian / Ubuntu: sudo apt install cmake g++   (plus rustup)
cd rust
cargo build --release      # rust/target/release/dpmread, dpmread-gui
```

The binaries link only against glibc and the C++ runtime; the GUI loads
X11 / Wayland and OpenGL at run time and names any missing library with
its `apt install` line. `rust/tools/build_linux.sh` builds release
binaries for older distributions (glibc 2.28) in a container (needs
Docker), with libxkbcommon, libxkbcommon-x11 and libxcb-xkb next to
`dpmread-gui` (found through its `$ORIGIN` run path).

Windows:

1. Install Visual Studio 2022 or newer, or its Build Tools, with the
   workload *Desktop development with C++* (includes MSVC and CMake).
2. Install [rustup](https://rustup.rs/) with the default MSVC toolchain.
3. Open a 64-bit developer shell: *Developer PowerShell for VS* from the
   Start menu, or from an ordinary PowerShell:

   ```powershell
   $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
   & "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
   ```

   `-Arch amd64` matters: without it the shell sets up the 32-bit x86
   compiler, while Rust's MSVC toolchain builds for x64. A message
   *'vswhere.exe' is not recognized* while the shell starts is harmless.
4. In that shell: `cd rust; cargo build --release`
   → `rust\target\release\dpmread.exe`, `dpmread-gui.exe`.

The C and C++ runtime is linked statically (`rust/.cargo/config.toml`), so
the `.exe` files run on a clean Windows without the Visual C++
Redistributable. The GUI uses OpenGL and falls back to Direct3D 12 where no
OpenGL 2 driver is present (virtual machines, remote desktop).

Tests, benchmarks and the full check script: [CONTRIBUTING.md](CONTRIBUTING.md).

## Python reference readers

`python/datamatrix_reader.py` and `python/datamatrix_reader_v2.py` are the
Python readers the Rust reader was ported from; the Rust reader is tested for
identical results against them. They are kept as references and receive
bug fixes only. `datamatrix_gen.py` is a small interactive Data Matrix
generator (Pillow only). They need Python 3, numpy, OpenCV and zxing-cpp
(`pip install -r python/requirements.txt`); pylibdmtx is optional.

## Licence

[Apache License 2.0](LICENSE). Parts of the image-processing code are
ported from OpenCV and NumPy; see [NOTICE](NOTICE). Release binaries also
contain zxing-cpp (Apache-2.0), libjpeg-turbo (IJG / BSD-3-Clause) and Rust
crates, listed with their licences in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
