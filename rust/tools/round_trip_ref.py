"""Python reference for the P3.9 round-trip test: decode the renders that
`ROUND_TRIP_DUMP=DIR cargo test -p dm-grid --release --test round_trip`
wrote (`*.pgm` + expected text in `*.txt`) with Python's grid_decode.

python rust/tools/round_trip_ref.py DIR [clean|deg]
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def one(p: str) -> tuple[str, bool, bool]:
    import datamatrix_reader as rd
    g = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
    text = Path(p[:-4] + ".txt").read_bytes()
    r = rd.grid_decode(g, float("inf"), (), (), 0.02, (True, False))
    return os.path.basename(p), r is not None and r[2].data == text, r is not None and r[2].data != text


def main() -> None:
    kind = sys.argv[2] if len(sys.argv) > 2 else "clean"
    files = sorted(glob.glob(os.path.join(sys.argv[1], f"*_{kind}.pgm")))
    with cf.ProcessPoolExecutor() as ex:
        res = list(ex.map(one, files))
    print(f"{kind}: decoded {sum(r[1] for r in res)}/{len(res)}, wrong {sum(r[2] for r in res)}")
    print("failed:", sorted(r[0] for r in res if not r[1]))


if __name__ == "__main__":
    main()
