# Roadmap

What dpmread does **not** do yet. Implemented features are in
[features.md](features.md); an item moves there when it is released.
Nothing here is a promise or has a date. Each item says what and why;
discussion happens in GitHub issues.

## Planned

Next, in rough order.

### Before 0.1.0

- **Release builds from CI.** Tagged builds for Windows x64 and Linux
  x86-64 with licence files and checksums, made by GitHub Actions instead
  of a local script.
- **Third-party licence file in the release zips.** Generated from the
  dependency tree so it stays complete.

### After 0.1.0

- **zxing-cpp 3.1.** Newer library version with a detector for large and
  non-flat symbols that uses the internal alignment patterns, and an
  optional denoising pass. Why: better reads on large curved marks, more
  images read by the fast pass.
- **Alignment-pattern anchors in the warped grid.** Symbols from 32×32 up
  have several data regions separated by solid / alternating alignment
  lines. Today the grid inside the symbol is extrapolated from the outer
  edges; tracking the alignment lines as well would anchor every region.
  Why: large marks on curved or stretched material.
- **Round-dot matched filter.** Detect the dot size of dot-peen and laser
  dot marks from the finder pattern and sample each module with a filter
  shaped like one dot. Why: dot marks where the dots do not touch read
  poorly with square sampling.
- **Faster failure.** Stop earlier on images that clearly hold no symbol or
  only a destroyed one, instead of using the whole budget.

## Backlog

Ideas worth doing, not scheduled.

- **Chase-style soft decoding.** When Reed-Solomon fails with the
  erasures chosen by confidence, retry with the next least certain
  codewords flipped or erased. Why: a few more reads at the edge of the
  correction capacity.
- **Soft fusion across variants.** Combine module evidence from several
  enhancement variants, tiles and re-samplings into one decision per
  module instead of decoding each separately.
- **Learned module classifier.** A small neural network that decides dark
  / light per module from its neighbourhood, used as one more sampler in
  front of Reed-Solomon (location, grid fit and error correction stay as
  they are). Why: over-burn, wood grain and blur are easier to learn than
  to hand-tune. Needs a training set first.
- **Synthetic test marks.** A generator for realistic hard marks (burned
  and peened dots on wood and metal textures, blur, perspective, lighting)
  with exact ground truth, and a published set made with it. Why: a
  public test set of the hard cases this reader is built for, and training
  data for the classifier.
- **Data Matrix generator in Rust.** The Python generator as a `dpmread
  gen` command or a GUI tab, for test symbols and labels.
- **Python bindings.** The Rust reader as a Python package, so scripts get
  its speed and results without the Python reference code.
- **Speed.** Faster Gaussian smoothing for the large kernels of the
  background flattening, parallel candidate search, earlier abort of
  speculative grid jobs.
- **ARM64 Linux builds.**
- **Library crates on crates.io** (Reed-Solomon with erasures, symbol
  tables and encodation), once the API is stable.
- **Better localisation in cluttered scenes.** A dedicated detector for
  photos with many objects and small symbols, from the published Data
  Matrix localisation literature.

## Non-goals

- **Decoding without the finder and timing pattern.** The grid is always
  anchored on the symbol's fixed pattern; a mark without it is out of
  scope.
- **Verification grades.** The quality figures are estimates for
  diagnostics. Formal grading per ISO/IEC 15415 / TR 29158 needs controlled
  lighting and calibrated optics, which a software reader does not have.
- **Other symbologies.** QR, PDF417, linear codes and the old ECC 000–140
  Data Matrix are well served by existing libraries.
