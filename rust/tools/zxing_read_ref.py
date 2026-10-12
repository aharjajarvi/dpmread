"""Reference for P4.1: run the reader's `zxing_read` on the gray PNGs of
`testdata/zxing` (private data) under two variant names (plain and
"... > otsu", which selects the binarised path) and both try_invert values.
Prints the TSV that `crates/dm-reader/tests/zxing_read.rs` compares with
(`testdata/zxing/zxing_read.tsv`).

python rust/tools/zxing_read_ref.py > $DM_PRIVATE_DIR/testdata/zxing/zxing_read.tsv
"""

import os
import sys
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
import datamatrix_reader as dr  # noqa: E402

DIR = PRIVATE / "testdata/zxing"


def quad(q) -> str:
    return " ".join(f"{x:g},{y:g}" for x, y in q)


def main() -> None:
    for png in sorted(DIR.glob("*.png")):
        img = cv2.imdecode(np.fromfile(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        for name in ("gray", "gray > otsu"):
            for inv in (True, False):
                r, failed = dr.zxing_read(dr.Variant(name, img), inv)
                if r:
                    info = r[2]
                    res = (f"{r[0].hex()}\t{quad(r[1])}\t{info['binarizer']}\t"
                           f"{info['symbology_identifier']}\t{info['content_type']}\t"
                           f"{info['orientation_deg']}\t{info['ec_level']}")
                else:
                    res = "-"
                fl = ";".join(quad(q) for q in failed)
                print(f"{png.name}\t{name}\t{str(inv).lower()}\t{res}\t{fl}")


if __name__ == "__main__":
    main()
