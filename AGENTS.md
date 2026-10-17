# AGENTS.md

Data Matrix (ECC 200) tools: a reader for poor quality marks (laser burned /
dot peened on wood, low contrast, blur) and a self-contained generator.
The main reader is the Rust one in `rust/` (CLI `dpmread`, GUI `dpmread-gui`);
the Python readers are frozen references.

## Layout

| Path | What |
|---|---|
| `python/datamatrix_reader.py` | Reader. Enhancement ladder + three engines tried in turn: zxing (zxing-cpp), libdmtx (pylibdmtx), grid (built-in ECC 200 decoder with soft-erasure Reed-Solomon). Prints a technical report or JSON. |
| `python/datamatrix_reader_v2.py` | Reader with a warped-grid tier (global timing tracking for marks whose module pitch drifts by more than ±0.45 module). The Rust reader's parity oracle. **Frozen**, like `datamatrix_reader.py` (bug fixes on request only). |
| `rust/` | **Main reader.** Cargo workspace (`crates/dm-*`): `dpmread` CLI and `dpmread-gui` (egui, drag and drop), at parity with v2; all new reader work goes here. `rust/deny.toml` (cargo-deny: licence allowlist, advisories), `rust/about.toml` + `rust/about.hbs` (cargo-about: generates `THIRD_PARTY_LICENSES.md`). `rust/tools/`: `ci.sh` (local CI), `parity.py` (check against the v2 baseline in the private test data), `build_linux.sh` + `linux-build.Dockerfile` (old-glibc release build in docker), `build_android.sh` (`dm-ffi` `.so` for Android apps and its Kotlin bindings), `check_dlls.py` + `msvc-crt.cmake` (Windows: static C/C++ run-time), `release_zip.py` + `zip_readme.txt` (release zip contents, used by `ci.sh` and the release workflow), `dump_fixtures*.py` (test fixtures in the private test data, made on CachyOS only), `*_ref.py` and `rs_bench.py` (Python counterparts of Rust tests and benchmarks). |
| `python/datamatrix_gen.py` | Interactive generator, Pillow only. Also the source of symbol tables and placement (`SQUARE_SYMBOLS`, `RECT_SYMBOLS`, `build_matrix`, `placement_map`) imported by the reader. |
| `python/requirements.txt` | Python dependencies of the readers and the generator. |
| private test data | **Not in this repo**; `DM_PRIVATE_DIR` points at it. `testimages/` (real photos and screenshots, sorted by how the reader handles them: `large/` decoded full-size photos, `easy/` decoded crops (direct sampling, ≥ 50 % unused ECC), `poor/` decoded crops that need rescue sampling or have < 50 % unused ECC, `not_recognized/` not decoded yet; move an image when its category changes), `baseline/python*_baseline.json` (frozen Python outputs, the parity oracle), `testdata/zxing/` and `fixtures/` (Rust test fixtures, not in git). Tests that need it are `#[ignore]`d. |
| `testimages_public/` | Public benchmark sets (Karrach 2021, zxing-cpp and zxing samples), **git-ignored**, fetched by `tools/fetch_public_sets.py`; each set has `expected.tsv` (ground truth) and `SOURCE.txt`. Separate from `testimages/` and from the Rust parity baseline. |
| `tools/` | `fetch_public_sets.py` (download + hash check, pins in `public_sets.lock.json`), `score_public.py` (hit / false-read / code-level scoring, `--compare`; `--reader rust` runs `dpmread`, `--multi` every symbol per image); `agreement.py` (module agreement of known texts with the symbol in an image, for checking ground-truth labels; candidate search, e.g. `--digits 4`); `export_public.sh` (`git archive HEAD` minus `public_exclude.txt` into an empty folder). |
| `baseline/` | Stored public-set scores that `score_public.py --compare` checks against: `public_v1.json`, `public_v2.json` (Python, fixed), `public_rust.json` and `public_rust_multi.json` (Rust, default / `--multi`; updated from the release run when results change on purpose). |
| `doc/` | Public docs, see below. |
| `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md` | User docs, contribution rules (checks, regression rule, design rules), how to report a vulnerability, changes per release. |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md` | Apache-2.0; ported OpenCV / NumPy code and the libraries the Linux zip ships; licences of everything linked into the binaries (generated, see `rust/about.toml`). All three go into every release zip. |
| `.github/workflows/` | `ci.yml`: the public part of `ci.sh` on Ubuntu and Windows (no private data; `--multi` public sets only in the nightly and manual runs). `release.yml`: on tag `v*`, Linux and Windows zips plus `SHA256SUMS` as a draft release. |
| `.github/ISSUE_TEMPLATE/` | Issue forms: image not decoded, bug report; security reports go to private vulnerability reporting. |

## Docs

- `doc/features.md` – what the reader does today: pipeline, features,
  performance, benchmarks on the public sets, known limits. Update it in the
  same commit as a feature change.
- `doc/roadmap.md` – what it does not do yet: planned, backlog, non-goals.
  Move an item to `features.md` when it lands.
- `doc/gui.png` – the README screenshot: eight Karrach 2021 images
  (`DMX2c_big`, `DMX2k`, `DMX2g`, `DMX2h`, `DMX2i`, `DMX2j`, `DMX3a`,
  `DMX6g`), default settings, first line selected. Remake it after a
  visible GUI change: run `dpmread-gui` on those files with
  `XDG_CONFIG_HOME` set to an empty directory (default settings),
  `DPMREAD_GUI_SCREENSHOT=doc/gui.png` and `DPMREAD_GUI_SELECT=0`.

Detailed plans may exist outside this repository; a local, untracked
`AGENTS.local.md` points to them if present.

## Running

```fish
rust/target/release/dpmread IMAGE_OR_FOLDER [...] [--json] [--debug DIR] \
    [--engines zxing,grid] [--budget SEC] [--jobs N] [--expect RxC] \
    [--polarity dark|light|any] [--exhaustive] [--matrix] [--multi] [--dump-fits DIR] \
    [--no-classifier]                # cargo build --release -p dm-cli --features ml
rust/target/release/dpmread-gui [IMAGE_OR_FOLDER ...]   # cargo build --release in rust/
rust/tools/build_linux.sh            # release binaries for other distros (glibc 2.28)
rust/tools/build_android.sh          # dm-ffi .so (arm64-v8a, x86_64) + Kotlin bindings
python python/datamatrix_reader.py IMAGE_OR_FOLDER [...] [same flags, + libdmtx engine]
python python/datamatrix_gen.py
```

Rust needs rustup (stable), cmake and a C++ compiler (zxing-cpp is built
from source): `sudo pacman -S rustup cmake gcc`. On Windows: Visual Studio
(or Build Tools) with "Desktop development with C++" (MSVC + CMake) and
rustup's MSVC toolchain; `rust/.cargo/config.toml` links the static C/C++
run-time, so the `.exe` files need no VC++ redistributable. `ci.sh` also
needs `cargo-deny` and `cargo-about` (`sudo pacman -S cargo-deny cargo-about`
or `cargo install --locked cargo-deny cargo-about`). `build_android.sh` needs the Android NDK,
`cargo-ndk` and the Rust targets `aarch64-linux-android` and
`x86_64-linux-android`.

Dependencies: numpy, opencv, zxing-cpp; pylibdmtx optional (needs system
libdmtx). Missing engines are skipped with a warning; the grid engine always
works. On CachyOS: `sudo pacman -S python-numpy python-opencv python-zxing-cpp`,
pylibdmtx from AUR (`python-pylibdmtx`) or a `--system-site-packages` venv.

## Regression check

Rust reader: `rust/tools/ci.sh` has four modes. **Agents run `--fast`
after every change**, `--gui` instead when only `dm-gui` changed, and
`--quick` only when the change needs it (table below). The full run is a
release check: run it only when the user asks for it or asks for a new
release, never on your own.

| Mode | Runs | Time (16 cores) |
|---|---|---|
| `--fast` | fmt, clippy (`-D warnings`), `cargo deny` without advisories, `cargo test --workspace` without the `#[ignore]`d tests and `round_trip`, parity gate against v2 with the `ci` build profile (no LTO) | ~40 s |
| `--gui` | fmt; fails if uncommitted files outside `rust/crates/dm-gui/` changed; clippy and tests of `dm-gui` (ignored ones included), debug build of `dpmread-gui` | ~20 s |
| `--quick` | fmt, clippy, `cargo deny check` (with advisories), `THIRD_PARTY_LICENSES.md` up to date, all tests (`--include-ignored`), release build, `parity.py --jobs-check` (output independent of `--jobs`) | ~6 min |
| (none) | `--quick`, then the public sets against `baseline/public_rust*.json`, the old-glibc build (Windows: native `.exe` build and `check_dlls.py`) and the release zip | ~27 min |

Run `--quick` instead of `--fast` when the change touches:

| Change | What only `--quick` checks |
|---|---|
| Threads, job order, `--jobs`, budgets or deadlines | `--jobs-check` |
| The numbers an image op, sampler or variant ladder computes (`dm-imgproc`, `dm-grid`, `dm-reader` variants), not a refactor | the `#[ignore]`d fixture tests, pixel-exact against Python |
| Grid sampling or decoding (`dm-grid`) or the symbol tables (`dm-symbol`) | `round_trip`: every symbol size, clean and degraded |
| `Cargo.toml`, `Cargo.lock`, build profiles, `rust/.cargo/`, build scripts | the release build that ships; advisories; `THIRD_PARTY_LICENSES.md` (regenerate in `rust/` with `cargo about generate --fail about.hbs -o ../THIRD_PARTY_LICENSES.md`) |

With `DM_PRIVATE_DIR` unset (or without its `fixtures/` / `testimages/`),
every mode reports the ignored tests and parity as skipped; a reader change
is not checked until it runs with the private data. When public-set
results change on purpose, update `baseline/public_rust*.json` from the
release run.

```fish
rust/tools/ci.sh --fast     # default after a change
rust/tools/ci.sh --gui      # change in rust/crates/dm-gui/ only
rust/tools/ci.sh --quick    # change listed in the table above
rust/tools/ci.sh            # release check: only when the user asks
```

Python readers (bug fixes only): there is no test suite: run the reader over
all test folders and compare the summary with the previous run – no image
that decoded before may be lost or change content:

```fish
python python/datamatrix_reader.py $DM_PRIVATE_DIR/testimages/{large,easy,poor,not_recognized} --budget 60
```

Public sets (round 4): fetch once, then compare against the stored score.
No hit may be lost and no false read may appear (exit code 1 otherwise):

```fish
python tools/fetch_public_sets.py
python tools/score_public.py --save /tmp/public_new.json
python tools/score_public.py --compare baseline/public_v2.json /tmp/public_new.json
# Rust reader
python tools/score_public.py --reader rust --save /tmp/public_rust.json
python tools/score_public.py --compare baseline/public_rust.json /tmp/public_rust.json
python tools/score_public.py --reader rust --multi --save /tmp/public_rust_multi.json
python tools/score_public.py --compare baseline/public_rust_multi.json /tmp/public_rust_multi.json
```

## Changes to `main`

`main` is protected by a ruleset: no direct pushes, no force pushes, no
deletion. Every change goes through a pull request:

| Step | Rule |
|---|---|
| Branch | One branch per change, from an up-to-date `main` |
| Before pushing | `rust/tools/ci.sh` as in [Regression check](#regression-check) |
| Pull request | Required checks: `test (ubuntu-24.04)`, `test (windows-latest)`, `cargo deny, third-party licences` (job names in `.github/workflows/ci.yml`; renaming a job means updating the ruleset) |
| Merge | Squash or rebase, only when all three checks pass (Windows ~30 min); linear history. Agents merge only when asked |
| Releases | Tag `v*` on `main` after the merge; `release.yml` builds a draft release |

```fish
git switch -c <branch> main
git push -u origin <branch>
gh pr create --fill
gh pr checks --watch
gh pr merge --squash --delete-branch
```

## Conventions

Rust (`rust/`):

- `cargo fmt`, clippy clean with `-D warnings` (`--all-targets`).
- Library crates return errors instead of panicking on bad input; `expect`
  only for internal invariants, with the reason as message.
- Same variant names, constants and iteration order as the Python v2 reader
  unless a change is deliberate and logged; results must not depend on
  `--jobs` or on the CPU: SIMD (AVX2 / FMA) only behind runtime detection,
  with a scalar fallback giving the same reads; releases target baseline
  x86-64.
- Paths are `Path` / `OsStr` throughout (non-ASCII names on Windows).

Python:

- Single-file scripts; keep the reader dependency-light (stdlib + numpy + cv2,
  optional decoders guarded by `try: import`).
- Must stay Windows-safe: file names contain non-ASCII (`ä`, `ö`), so
  read/write images with `np.fromfile` + `cv2.imdecode` and
  `cv2.imencode(...).tofile()`, never `cv2.imread`/`cv2.imwrite`. Worker pool
  uses the `spawn` context.
- Results are consumed in submission order, so output must not depend on
  `--jobs`.
- Match the existing style: type hints, short comments, module-level constants
  in UPPER_CASE.

Docs:

- Plain, concrete English; current fact, not history; tables for anything
  enumerable; no status adjectives without evidence ("passes `ci.sh` on
  2026-10-11", not "works well"); no duplication — link instead;
  repo-relative links; no doc over ~1 000 lines.
