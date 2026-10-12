"""Python timing of recorded grid jobs (`fixtures/grid` in the private data), the
reference for `cargo run --release -p dm-grid --example grid_bench FILTER`.
Single process, OpenCV with one thread (as the reader's pool workers).
`--v2` times `datamatrix_reader_v2.py` (the Rust reader's oracle since
P4b.6; its `grid_decode` adds the warp pass), default v1.

python rust/tools/grid_bench_ref.py [--v2] [FILTER]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def private_dir() -> Path:
    """Private test data (fixtures, test images, baselines), not part of
    this repo: `DM_PRIVATE_DIR`."""
    d = os.environ.get("DM_PRIVATE_DIR")
    if not d:
        sys.exit("DM_PRIVATE_DIR is not set: point it at the private test data")
    return Path(d)


PRIVATE = private_dir()
sys.path.insert(0, str(ROOT / "python"))

if "--v2" in sys.argv:
    sys.argv.remove("--v2")
    import datamatrix_reader_v2 as rd  # noqa: E402
else:
    import datamatrix_reader as rd  # noqa: E402

DIR = PRIVATE / "fixtures" / "grid"


def load(ref: dict) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(DIR / ref["file"]), np.uint8), cv2.IMREAD_GRAYSCALE)


def main() -> None:
    cv2.setNumThreads(1)
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    m = json.loads((DIR / "manifest.json").read_text(encoding="utf-8"))
    total = 0.0
    for c in m["cases"]:
        if c["op"] != "grid_decode":
            continue
        p = c["params"]
        what = f"{p['image']} [{p['variant']}] qz={str(p['qz']).lower()}"
        if flt not in what:
            continue
        g = load(c["in"]["img"])
        alts = [(n, load(a)) for n, a in c["in"]["alts"]]
        extra = [np.float32(q) for q in c["in"]["extra"]]
        t = time.perf_counter()
        r = rd.grid_decode(g, float("inf"), extra, alts, p["min_area"], (p["qz"],), None, {})
        s = time.perf_counter() - t
        total += s
        print(f"{s:7.3f} s  {'read' if r else '-'}  {what}")
    print(f"total {total:.2f} s")


if __name__ == "__main__":
    main()
