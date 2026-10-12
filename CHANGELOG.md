# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/). Until 1.0, minor
versions may change command-line flags and the JSON output.

## [Unreleased]

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

[Unreleased]: https://github.com/aharjajarvi/dpmread/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aharjajarvi/dpmread/releases/tag/v0.1.0
