"""Write Python reference results as JSON fixtures for the Rust unit tests.

Part 1 (phase 1): symbol tables, placement maps, build_matrix, geometry,
deinterleave, Reed-Solomon (rs_correct, rs_decode_symbol), codeword decoder
and parse_gs1. Fixed seeds, so the output only changes with the Python code.

python rust/tools/dump_fixtures.py [OUTDIR]        (default: fixtures/ in the private data)
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import datamatrix_gen as gen  # noqa: E402
import datamatrix_reader as rd  # noqa: E402

RS_CASES = 4000
SEED = 20261009



def private_dir() -> Path:
    """Private test data (fixtures, test images, baselines), not part of
    this repo: `DM_PRIVATE_DIR`."""
    d = os.environ.get("DM_PRIVATE_DIR")
    if not d:
        sys.exit("DM_PRIVATE_DIR is not set: point it at the private test data")
    return Path(d)

def write(out: Path, name: str, obj) -> None:
    path = out / name
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    print(f"{path}  {path.stat().st_size // 1024} KiB")


# --------------------------------------------------------------------------
# Symbols: placement, build_matrix, geometry, deinterleave
# --------------------------------------------------------------------------
def dump_symbols(rng: random.Random) -> list[dict]:
    out = []
    for sym in rd.ALL_SYMBOLS:
        rows, cols, dregr, dregc, regr, regc, data, eccblk, blocks = sym
        n_cw = data + eccblk * blocks
        place = gen.placement_map(dregr * regr, dregc * regc)
        cws = [rng.randrange(256) for _ in range(n_cw)]
        g = rd.geometry(sym)
        out.append({
            "sym": list(sym),
            "placement": [None if c is None else list(c) for line in place for c in line],
            "codewords": cws,
            "matrix": [v for line in gen.build_matrix(cws, sym) for v in line],
            "fixed_mask": g.fixed_mask.astype(int).ravel().tolist(),
            "fixed_val": g.fixed_val.astype(int).ravel().tolist(),
            "cw_rows": g.cw_rows.tolist(),
            "cw_cols": g.cw_cols.tolist(),
            "pad_mask": g.pad_mask.astype(int).ravel().tolist(),
            "pad_val": g.pad_val.astype(int).ravel().tolist(),
            "fp_uv_qz": g.fp_uv_qz.tolist(),
            "fp_dark_qz": g.fp_dark_qz.astype(int).tolist(),
            "fp_uv_in": g.fp_uv_in.tolist(),
            "fp_dark_in": g.fp_dark_in.astype(int).tolist(),
            "deinterleave": [pos for pos, _ in rd.deinterleave(cws, g)],
        })
    return out


# --------------------------------------------------------------------------
# Reed-Solomon
# --------------------------------------------------------------------------
def corrupt(rng: random.Random, msg: list[int], n_err: int) -> list[int]:
    pos = rng.sample(range(len(msg)), min(n_err, len(msg)))
    for p in pos:
        msg[p] ^= rng.randrange(1, 256)
    return pos


def dump_rs_correct(rng: random.Random) -> list[dict]:
    out = []
    for _ in range(RS_CASES):
        nsym = rng.randint(5, 68)
        k = rng.randint(1, min(255 - nsym, 200))
        data = [rng.randrange(256) for _ in range(k)]
        msg = data + gen.rs_encode(data, nsym)
        n = len(msg)
        n_err = rng.choice((0, rng.randint(0, nsym // 2 + 2), rng.randint(0, nsym + 2)))
        err = corrupt(rng, msg, n_err)
        # erasures: some on real errors, some on good codewords, sometimes too many
        f_true = rng.randint(0, len(err))
        f_false = rng.choice((0, rng.randint(0, nsym // 3 + 1), rng.randint(0, nsym + 2)))
        good = [p for p in range(n) if p not in set(err)]
        eras = rng.sample(err, f_true) + rng.sample(good, min(f_false, len(good)))
        rng.shuffle(eras)
        r = rd.rs_correct(msg, nsym, eras)
        out.append({"msg": msg, "nsym": nsym, "erasures": eras,
                    "result": None if r is None else {"out": r[0], "changed": r[1]}})
    return out


def dump_rs_symbol(rng: random.Random) -> list[dict]:
    out = []
    for sym in rd.ALL_SYMBOLS:
        g = rd.geometry(sym)
        for case in range(8):
            data = [rng.randrange(1, 256) for _ in range(g.n_data)]
            cw, _ = gen.interleave_ecc(data, sym)
            n_err = rng.randint(0, max(1, (g.n_ecc * g.blocks * (case + 1)) // 10))
            err = set(corrupt(rng, cw, n_err))
            # quantised confidences give ties (tests the stable sort); errors
            # are usually, not always, low confidence
            conf = [round(rng.random() * 0.5 if p in err and rng.random() < 0.85
                          else 0.3 + 0.7 * rng.random(), 2) for p in range(len(cw))]
            r = rd.rs_decode_symbol(cw, np.array(conf), g)
            out.append({
                "sym": list(sym), "cw": cw, "conf": conf,
                "result": None if r is None else {
                    "data_cw": r.data_cw, "all_cw": r.all_cw,
                    "block_errors": r.block_errors, "block_erasures": r.block_erasures,
                    "corrected_positions": r.corrected_positions, "uec": r.uec}})
    return out


# --------------------------------------------------------------------------
# Codeword decoder
# --------------------------------------------------------------------------
def encode(text: bytes | str) -> list[int]:
    if isinstance(text, str):
        text = text.encode("latin-1")
    return gen.encode_data(text)[0]


def padded(cw: list[int], extra: int = 0) -> list[int]:
    cw = list(cw)
    sym = gen.choose_symbol(len(cw) + extra, True, None)
    gen.add_padding(cw, sym[6])
    return cw


def codec_streams(rng: random.Random) -> list[list[int]]:
    texts: list[bytes | str] = [
        "A", "1", "12", "123456", "HELLO WORLD", "Hello, World!",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "THE QUICK BROWN FOX JUMPS 1234",
        "abcdefghijklmnopqrstuvwxyz0123456789", "the quick brown fox jumps over",
        "ABC*DEF>GHI\rJKL*MNO>PQR\r", "X12*X12*X12*X12>X12\r",
        "@ABC:DEF;GHI<JKL=MNO?@ABC:DEF", "EDI^FACT[EDI]FACT\\EDI_FACT",
        "0123456789-98-7-6-FAKE", "ÄÖÅäöå mixed latin-1 ÿ", "\x80\x81\xfe\xff",
        "C40 then text: ABCDEFGH abcdefgh ABCDEFGH", "a" * 60, "A" * 61, "9" * 101,
        bytes(range(256)), bytes(rng.randrange(256) for _ in range(300)),
        bytes(rng.randrange(256) for _ in range(40)),
    ]
    alphabets = [b"0123456789", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789",
                 b"abcdefghijklmnopqrstuvwxyz 0123456789",
                 b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*>\r ",
                 b"@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_ !\"#$%&'()*+,-./0123456789:;<=>?",
                 bytes(range(32, 127)), bytes(range(256))]
    for _ in range(150):
        alpha = rng.choice(alphabets)
        texts.append(bytes(rng.choice(alpha) for _ in range(rng.randint(1, 80))))
    streams = [padded(encode(t)) for t in texts]
    streams += [encode(t) for t in texts[:20]]                     # no padding

    # function codewords the generator does not write
    gs1 = encode("0109501101530003172501011") + [232] + encode("0ABC123")
    streams += [
        padded([232] + encode("0109501101530003") + encode("10LOT42")),
        padded([232] + gs1, 1),
        padded([241, 27] + encode("Grüße".encode("utf-8"))),          # ECI 26
        padded([241, 130, 5] + encode("ECI two")),                    # ECI 2 cw
        padded([241, 200, 7, 9] + encode("ECI three")),               # ECI 3 cw
        padded([236] + encode("06P123456")),                          # macro 05
        padded([237] + encode("1JUN123")),                            # macro 06
        padded([233, 0x12, 7, 9] + encode("PART ONE")),               # structured append
        padded([233, 0x12, 7, 9, 232] + encode("0109501101530003")),
        padded([234] + encode("PROG")),                               # reader programming
        padded([235, 66] + encode("X") + [254] + encode("Y")),        # upper shift, stray unlatch
        padded([230, 0, 1, 254, 66]),                                 # C40 shift 1 / zero triple
        padded([239, 3, 65, 254]),
        padded([230, 255, 255, 254]), padded([239, 255, 255]), padded([238, 255, 255, 254]),
        padded([238, 1, 1, 254, 66]),
    ]
    # broken / short streams
    streams += [[0], [255], [231], [231, 251], [233, 1, 2], [233, 1, 2, 3], [241],
                [241, 130], [241, 200, 5], [129], [129, 1, 2, 3], [66, 129, 129],
                [240, 0, 0], [240, 124], [240, 1, 2, 3, 124], [230, 1], [230, 0, 0],
                [239, 0, 0, 254], [238, 0, 0], [231, 0, 5, 6], [231, 7, 1]]
    # random streams, biased towards latches and function codewords
    special = list(range(229, 256)) + [129, 1, 66, 130]
    for _ in range(1500):
        n = rng.randint(1, 50)
        streams.append([rng.choice(special) if rng.random() < 0.3 else rng.randrange(256)
                        for _ in range(n)])
    # random content behind a latch: exercises C40/Text/X12/EDIFACT values
    for _ in range(600):
        n = rng.randint(2, 30)
        streams.append([rng.choice((230, 238, 239, 240, 231))]
                       + [rng.randrange(256) for _ in range(n)])
    return streams


def decoded_dict(d: rd.Decoded) -> dict:
    return {
        "data": list(d.data),
        "segments": [[s.mode, s.cw_start, s.cw_end, list(s.data), s.note] for s in d.segments],
        "payload_cw": d.payload_cw, "pad_cw": d.pad_cw, "gs1": d.gs1, "eci": d.eci,
        "macro": d.macro, "structured_append": d.structured_append,
        "reader_programming": d.reader_programming, "warnings": d.warnings,
    }


def dump_codec(rng: random.Random) -> list[dict]:
    out = []
    for cw in codec_streams(rng):
        try:
            res = {"ok": decoded_dict(rd.decode_codewords(cw))}
        except rd.DataError as exc:
            res = {"err": ["DataError", str(exc)]}
        except IndexError as exc:
            res = {"err": ["IndexError", str(exc)]}
        out.append({"cw": cw, **res})
    return out


def dump_gs1(rng: random.Random, codec: list[dict]) -> list[dict]:
    texts = ["0109501101530003", "01095011015300031725010110ABC123\x1d21XYZ",
             "3103000123", "3922123", "00123456789012345678", "7003260101123011",
             "8004GIAI\x1d99INT", "\x1d10A\x1d", "1", "", "12", "991", "4101234567890123",
             "240ID\x1d241CPN\x1d250SER", "3651234567"]
    ais = list(rd.GS1_AI) + ["310", "3103", "3922", "36", "77", "4", "x"]
    for _ in range(300):
        parts = []
        for _ in range(rng.randint(1, 4)):
            val = "".join(rng.choice("0123456789ABC") for _ in range(rng.randint(0, 20)))
            parts.append(rng.choice(ais) + val + ("\x1d" if rng.random() < 0.5 else ""))
        texts.append("".join(parts))
    texts += ["".join(map(chr, c["ok"]["data"])) for c in codec
              if "ok" in c and c["ok"]["gs1"]]
    return [{"text": t, "elements": [list(e) for e in rd.parse_gs1(t)]} for t in texts]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else private_dir() / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    write(out, "symbols.json", dump_symbols(random.Random(SEED)))
    write(out, "rs_correct.json", dump_rs_correct(random.Random(SEED + 1)))
    write(out, "rs_symbol.json", dump_rs_symbol(random.Random(SEED + 2)))
    codec = dump_codec(random.Random(SEED + 3))
    write(out, "codec.json", codec)
    write(out, "gs1.json", dump_gs1(random.Random(SEED + 4), codec))
    ok = sum("ok" in c for c in codec)
    print(f"codec: {ok} decoded, {len(codec) - ok} errors")


if __name__ == "__main__":
    main()
