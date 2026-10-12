"""P1.5 RS micro-benchmark, Python side: time per call of rs_correct and
rs_decode_symbol on the same fixtures as `cargo run --release -p dm-ecc
--example rs_bench`.

python rust/tools/rs_bench.py
"""

import json
import os
import sys
import time
from pathlib import Path

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

import datamatrix_reader as rd  # noqa: E402

FIX = PRIVATE / "fixtures"


def main() -> None:
    correct = json.loads((FIX / "rs_correct.json").read_text())
    t = time.perf_counter()
    for c in correct:
        rd.rs_correct(c["msg"], c["nsym"], c["erasures"])
    us = (time.perf_counter() - t) * 1e6 / len(correct)
    print(f"rs_correct        {len(correct):>5} cases  {us:>9.2f} us/call")

    symbols = json.loads((FIX / "rs_symbol.json").read_text())
    cases = [(c["cw"], np.array(c["conf"]), rd.geometry(tuple(c["sym"]))) for c in symbols]
    t = time.perf_counter()
    for cw, conf, g in cases:
        rd.rs_decode_symbol(cw, conf, g)
    us = (time.perf_counter() - t) * 1e6 / len(cases)
    print(f"rs_decode_symbol  {len(cases):>5} cases  {us:>9.2f} us/call")


if __name__ == "__main__":
    main()
