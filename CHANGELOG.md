# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/). Until 1.0, minor
versions may change command-line flags and the JSON output.

## [Unreleased]

## [0.2.0] – 2026-10-13

### Added

- Grid decoder: on symbols with several data regions, every region is
  re-fitted on its own alignment lines and sampled with its own
  perspective when the whole-symbol fit does not decode.
- Grid decoder: a disc filter of the measured dot size for dot-peen and
  laser-dot marks whose finder line shows separate dots.
- README: a screenshot of the GUI.

### Changed

- GUI: *Settings* and *Technical report* open as their own windows that
  can be moved outside the main window.
- After the first grid read the remaining speculative grid jobs stop
  sooner (candidate detection checks the deadline); about 0.15 s less per
  image read by the grid decoder.
- `features.md`: the read flow as a diagram; figures for 0.2.

Reads that decoded with 0.1.0 decode identically; the public benchmark
sets give the same results.

## [0.1.0] – 2026-10-12

First public release.

- `dpmread` command-line reader and `dpmread-gui` desktop app for Data
  Matrix (ECC 200) marks, including rectangular and DMRE sizes: enhancement
  ladder, zxing-cpp pass, grid decoder with soft-erasure Reed-Solomon,
  rescue sampling and warped grid; near-miss report when nothing decodes.
- JSON and text reports, multi-symbol reads, quality estimate, debug
  images; GUI with drag and drop, detail view and CSV / JSON export.
- Release zips for Linux x86-64 and Windows x64 with licence files and
  `SHA256SUMS`, built by GitHub Actions.

[Unreleased]: https://github.com/aharjajarvi/dpmread/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/aharjajarvi/dpmread/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aharjajarvi/dpmread/releases/tag/v0.1.0
