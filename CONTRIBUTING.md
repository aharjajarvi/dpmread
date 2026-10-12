# Contributing to dpmread

Thanks for your interest. Bug reports, unreadable sample images and pull
requests are welcome. This is a small project, so please open an issue
before starting on a large change.

## Reporting an image that does not decode

Undecoded images are the most useful contribution. Use the *Image not
decoded* issue template and include:

- the image as an original file (camera JPEG, PNG or BMP at full
  resolution; no screenshots of screenshots, no re-compressed copies);
- the text the symbol should contain, if you know it;
- how the mark was made (laser, dot peen, ink jet, …) and on what material.

**Only post images you have the right to share**, and by posting you agree
that they may be used as test data under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Do not post marks
that encode confidential production data (serial numbers, customer or order
numbers). If that is the only kind of sample you have, say so in the issue
instead of uploading it.

## Building

The reader is a Rust workspace under `rust/`. Build steps for users are in
the [README](README.md#building-from-source); in short:

- Rust stable via [rustup](https://rustup.rs/), CMake and a C++ compiler
  (zxing-cpp is built from source).
  - Arch / CachyOS: `sudo pacman -S rustup cmake gcc`
  - Debian / Ubuntu: `sudo apt install cmake g++` plus rustup
  - Windows: Visual Studio 2022 or newer, or its Build Tools, with
    *Desktop development with C++* (MSVC and CMake), rustup with the
    default MSVC toolchain; build in a 64-bit developer shell ([README](README.md#building-from-source),
    Windows step 3).
- `cd rust && cargo build --release` builds the CLI and the GUI into
  `rust/target/release/`.
- For `rust/tools/ci.sh`: `cargo install --locked cargo-deny cargo-about`
  (Arch / CachyOS: `sudo pacman -S cargo-deny cargo-about`).

Platform notes:

- Windows: `rust/.cargo/config.toml` links the C/C++ run-time statically
  for Rust and for the CMake-built libraries (`rust/tools/msvc-crt.cmake`),
  so the `.exe` files need no Visual C++ Redistributable.
  `rust/tools/check_dlls.py` fails if a binary imports a run-time DLL; keep
  it passing when adding a C or C++ dependency.
- Linux: `rust/tools/build_linux.sh` builds the release binaries in a
  manylinux 2.28 container (Docker) so they run on glibc 2.28 and newer.
- Release builds target baseline x86-64; do not add `target-cpu` flags.

The check script and the benchmark tools need Python 3 (standard library;
Pillow too when fetching the public sets, to convert their WebP images). On Windows run `rust/tools/ci.sh` from Git Bash; it finds CMake in
the Visual Studio installation if it is not on `PATH`. The Python reference
readers additionally need numpy, OpenCV, zxing-cpp and Pillow.

## Checks before a pull request

`rust/tools/ci.sh` runs every check below, then the release build and
zip for the current OS (Linux: container build; Windows: native build and
the DLL check). It stops at the first failure:

- `cargo fmt --check`;
- `cargo clippy --all-targets -- -D warnings`;
- `cargo deny check` (licence allowlist and advisories, `rust/deny.toml`)
  and `THIRD_PARTY_LICENSES.md` up to date (`rust/about.toml`): a new
  dependency needs an allowed licence, and the file is regenerated with
  `cargo about generate --fail about.hbs -o ../THIRD_PARTY_LICENSES.md`
  in `rust/`;
- `cargo test --workspace` (tests that need the maintainer's private test
  images and fixtures are marked `#[ignore]` and skipped);
- the public benchmark sets (fetched once with
  `python tools/fetch_public_sets.py`), scored against the stored baselines
  in `baseline/`.

**Regression rule:** a change must not lose a symbol that decoded before,
change the decoded text of any symbol, or introduce a false read. Gains are
welcome; list them in the pull request. If a change improves results on
purpose, update the matching `baseline/*.json` file in the same pull
request and say why.

## Code conventions

Rust:

- `cargo fmt`; clippy clean with `-D warnings` on all targets.
- Library crates return errors on bad input instead of panicking; `expect`
  only for internal invariants, with the reason as the message.
- Results must not depend on the number of worker threads (`--jobs`) or on
  the CPU. SIMD (AVX2 / FMA) only behind runtime detection, with a scalar
  fallback that gives the same reads. Releases target baseline x86-64.
- File paths stay `Path` / `OsStr` throughout; file names with non-ASCII
  characters must work on Windows.
- Keep comments short and factual, matching the surrounding code.

Python (`python/datamatrix_reader*.py`, `python/datamatrix_gen.py`):

- The two readers are frozen references for the Rust reader; they take bug
  fixes only. New reader work goes into `rust/`.
- Single-file scripts, type hints, module-level constants in UPPER_CASE.
- Windows-safe image I/O (`np.fromfile` + `cv2.imdecode`,
  `cv2.imencode(...).tofile()`).

## Design rules

- **Grid placement is anchored on the finder and timing pattern** (the
  solid L and the alternating edges of the symbol). Changes that place
  modules from the periodicity of the data region, or decode with the fixed
  pattern ignored or missing, need a discussion in an issue before any
  code is written.
- **No third-party code without a compatible licence.** Code ported from
  another project needs an Apache-2.0-compatible licence, a header in the
  file naming the source, its licence and "modified", and an entry in
  `NOTICE`.
- **No vendor code or material from reverse engineering.** Do not
  contribute code, constants or documentation taken from decompiled,
  disassembled or otherwise reverse-engineered products or firmware.
  Published papers, standards and open-source code under a compatible
  licence are fine; cite them.

## Licence

dpmread is licensed under the [Apache License 2.0](LICENSE). By submitting
a contribution you agree that it is licensed under the same terms
(Apache-2.0 section 5), and you confirm that you have the right to submit
it.
