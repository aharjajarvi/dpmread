# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/). Until 1.0, minor
versions may change command-line flags and the JSON output.

## [Unreleased]

### Added

- `--multi-thorough` and the GUI setting *thorough* (builds with `ml`):
  `--multi` as before, with the whole search after each read.

### Changed

- `--multi` with the symbol detector (feature `ml`): after the first read
  the searches aim at the symbols the detector sees on the whole image
  and on 3 × 3 tiles, and stop once each is covered or tried, instead of
  searching the whole image again until a search finds nothing.
  `dm-ffi` reads with `multi` this way too.

## [0.3.0] – 2026-10-19

### Added

- `dm-ffi`: the reader as a library for Android apps (Kotlin bindings
  through uniffi), with `rust/tools/build_android.sh`; JSON identical to
  `--json`. Not in the release zips.
- `--no-analysis` (and `analyse` in `dm-ffi` options): after a zxing
  read, skip the grid fit that fills in the symbol, error-correction and
  quality fields.
- `--dump-fits DIR`: write each image's grid fits (the one that read and
  the best failed ones) with module centres and the symbol resampled
  through the fit, for training a module classifier.
- Cargo feature `ml` (`dm-cli`, `dm-gui`, `dm-reader`; off in a plain
  `cargo build`, on in the release zips): when every
  grid attempt fails, the best three failed fits are classified again by
  a trained module classifier (new crate `dm-ml`, tract, weights
  embedded) and decoded with the same error correction.
  `--no-classifier` turns it off.
- Cargo feature `ml`: a trained symbol detector as the first try (new in
  `dm-ml`, tract, weights embedded). When one zxing call on the gray image
  does not read, a finder on the whole frame and a measurer on a
  full-resolution crop give the main symbol's corners, polarity and three
  size candidates; the grid decoder tries them on a crop before zxing and
  the variant ladder, which run only when that gives no read. The main
  symbol is the large one nearest the image centre; other symbols are not
  tried first. The report names the variant `detector > …`.
  `--no-detector` turns it off. Each first-try job builds its own crop
  variant and the first job (the variant that reads most often) runs
  alone, so a phone frame read by the first try ends within a 0.3 s
  budget.
- `dm-ffi` feature `ml` (`rust/tools/build_android.sh --ml`): the
  detector first try and the module classifier in the app library; a
  `Reader` loads the detector models when it is created.
- `DPMREAD_TIMINGS`: `detector_gate`, `detector_models` and
  `detector_crops`, the parts of the detector first try.
- Camera frames: `Reader.read_frame` in `dm-ffi` (an 8-bit gray image
  from memory with its row stride, such as an Android Y plane),
  `dm_reader::read_frame` and `--frame`. One zxing pass on the gray
  image, then the detector first try (feature `ml`); no variant ladder,
  no classic grid search and no analysis after a zxing read, so a read
  ends within its budget. `dm-ffi` `Options.detector` (`--no-detector`).
- Detector first try: a later job starts only while the time left is at
  least what the first job took (building a crop variant cannot be
  stopped).
- `tools/synth_marks.py`: synthetic Data Matrix marks with exact ground
  truth (module matrix, corners, module centres) for training and testing:
  laser, ink and dot-peen styles on procedural or photo wood, scene and
  camera effects, a seeded parameter record per image to render it again,
  checks against the reader, and an HTML review sheet.
- `tools/synth_marks.py --scene WxH`: camera-sized frames with 0-3 marks
  on boards, gaps and flat surfaces, with distractors (text, labels, 1D
  bar codes, dot-matrix text, knots, strokes, stains), for training a
  symbol detector; the record holds every mark's corners, size and
  polarity. `--check` on a scene run counts reads that match no label.
- `ml/boxes.py`: symbol boxes on real images for a detector, merged from
  labelled corners, the further codes `--multi` reads, and a local page
  to draw boxes on codes no reader decodes.
- `tools/synth_marks.py --sizes`: marks of the listed symbol sizes only.
- `ml/detector.py`, `ml/detector_data.py`, `ml/train_detector.py`,
  `ml/evaluate_detector.py`: a symbol detector for a cheap first try. A
  finder on the whole frame (long side 512 px) gives up to three symbol
  quads; a measurer on a full-resolution crop of each refines the corners
  and gives polarity and the three most likely sizes. Training on scene
  frames, fine-tuning on real images with the backbone frozen, ONNX
  export, and an offline score of the chain. Not used by the reader yet.
- `tools/agreement.py`: module agreement of known texts with the symbol
  in an image, for checking ground-truth labels. Fits the grid on the
  finder and timing pattern only, then scores every candidate text
  (`--text`, `--texts-from`, `--digits N`) with the chance of a match by
  luck and the codeword errors per Reed-Solomon block.

### Changed

- Release zips: `dpmread` and `dpmread-gui` are built with the feature
  `ml`, so the symbol detector first try and the module classifier are on
  by default. The GUI settings window has a checkbox for each under
  *Trained models* (kept between sessions); the command line keeps
  `--no-detector` and `--no-classifier`.
- `features.md`: figures for 0.3, measured with the release build
  (feature `ml`): public sets 188 / 191, `--multi` code level 98.8 %.
- After a zxing read, the grid fit for the report tries the zxing outline
  alone before full detection, runs its attempts in parallel and stops
  after 2 s in total (was up to 8 s for each of 7 attempts). Public sets:
  same results, 186 → 107 s in total, slowest image 30 → 12 s.

### Fixed

- Warped grid: a square symbol fitted in the frame transposed across its
  anti-diagonal (same finder and timing pattern, mirrored data) failed
  Reed-Solomon; that frame is now decoded too. 3 more of 163 real photos
  read, no read lost or changed.
- `datamatrix_gen.py`: a C40 or Text run that ended with a shifted
  character (e.g. lowercase in C40) split across the last full triple lost
  that character. 56 of 3 000 random texts were affected; all 3 000 now
  read back with zxing-cpp.

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

[Unreleased]: https://github.com/aharjajarvi/dpmread/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/aharjajarvi/dpmread/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/aharjajarvi/dpmread/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aharjajarvi/dpmread/releases/tag/v0.1.0
