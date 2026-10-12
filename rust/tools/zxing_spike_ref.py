"""Reference for the P0.3 zxing spike: write each image as a gray PNG (cv2
BGR2GRAY, as the reader does) into OUTDIR and print the same TSV lines as
`cargo run --example zxing_spike` prints for those PNGs.

python rust/tools/zxing_spike_ref.py OUTDIR IMAGE...
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

BINARIZERS = (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram,
              zxingcpp.Binarizer.FixedThreshold)


def name(e) -> str:
    return str(e).split(".")[-1]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[2:]:
        bgr = cv2.imdecode(np.fromfile(arg, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        png = out / (Path(arg).stem + ".png")
        cv2.imencode(".png", gray)[1].tofile(png)
        for bz in BINARIZERS:
            r = zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.DataMatrix,
                                       binarizer=bz, text_mode=zxingcpp.TextMode.Plain,
                                       try_invert=True, return_errors=True)
            for i, b in enumerate(r):
                p = b.position
                pos = " ".join(f"{q.x},{q.y}" for q in (p.top_left, p.top_right,
                                                         p.bottom_right, p.bottom_left))
                err = name(b.error.type) if b.error else ""
                print(f"{png.name}\t{name(bz)}\t{i}\t{str(b.valid).lower()}\t"
                      f"{bytes(b.bytes).hex()}\t{b.symbology_identifier}\t"
                      f"{name(b.content_type)}\t{b.orientation}\t{pos}\t{err}")


if __name__ == "__main__":
    main()
