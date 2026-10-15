#!/usr/bin/env python3
"""Interactive Data Matrix (ECC 200) generator.

Asks for content, writes a PNG (overwriting the previous one) and prints a full
report of the internals: encodation scheme segments, codeword usage, symbol
geometry, Reed-Solomon block layout and image metrics.

Only dependency: Pillow.  The ECC 200 encoder below is self-contained, which is
what makes the internal reporting possible.

Options can be typed at the prompt as ':name value', see ':help'.
"""

from __future__ import annotations

import math
import os
import sys

from PIL import Image

# --------------------------------------------------------------------------
# Symbol attributes (ISO/IEC 16022 ECC 200)
# rows, cols          : full symbol size in modules (incl. finder / clock track)
# dregr, dregc        : module size of one data region (excl. finder)
# regr, regc          : number of data regions vertically / horizontally
# data                : total data codeword capacity
# eccblk              : error correction codewords per RS block
# blocks              : number of interleaved RS blocks
# --------------------------------------------------------------------------
Symbol = tuple  # (rows, cols, dregr, dregc, regr, regc, data, eccblk, blocks)

SQUARE_SYMBOLS: list[Symbol] = [
    #  rows cols dregr dregc regr regc  data eccblk blocks
    (  10,  10,   8,    8,   1,   1,      3,     5,   1),
    (  12,  12,  10,   10,   1,   1,      5,     7,   1),
    (  14,  14,  12,   12,   1,   1,      8,    10,   1),
    (  16,  16,  14,   14,   1,   1,     12,    12,   1),
    (  18,  18,  16,   16,   1,   1,     18,    14,   1),
    (  20,  20,  18,   18,   1,   1,     22,    18,   1),
    (  22,  22,  20,   20,   1,   1,     30,    20,   1),
    (  24,  24,  22,   22,   1,   1,     36,    24,   1),
    (  26,  26,  24,   24,   1,   1,     44,    28,   1),
    (  32,  32,  14,   14,   2,   2,     62,    36,   1),
    (  36,  36,  16,   16,   2,   2,     86,    42,   1),
    (  40,  40,  18,   18,   2,   2,    114,    48,   1),
    (  44,  44,  20,   20,   2,   2,    144,    56,   1),
    (  48,  48,  22,   22,   2,   2,    174,    68,   1),
    (  52,  52,  24,   24,   2,   2,    204,    42,   2),
    (  64,  64,  14,   14,   4,   4,    280,    56,   2),
    (  72,  72,  16,   16,   4,   4,    368,    36,   4),
    (  80,  80,  18,   18,   4,   4,    456,    48,   4),
    (  88,  88,  20,   20,   4,   4,    576,    56,   4),
    (  96,  96,  22,   22,   4,   4,    696,    68,   4),
    ( 104, 104,  24,   24,   4,   4,    816,    56,   6),
    ( 120, 120,  18,   18,   6,   6,   1050,    68,   6),
    ( 132, 132,  20,   20,   6,   6,   1304,    62,   8),
    ( 144, 144,  22,   22,   6,   6,   1558,    62,  10),
]

RECT_SYMBOLS: list[Symbol] = [
    (   8,  18,   6,   16,   1,   1,      5,     7,   1),
    (   8,  32,   6,   14,   1,   2,     10,    11,   1),
    (  12,  26,  10,   24,   1,   1,     16,    14,   1),
    (  12,  36,  10,   16,   1,   2,     22,    18,   1),
    (  16,  36,  14,   16,   1,   2,     32,    24,   1),
    (  16,  48,  14,   22,   1,   2,     49,    28,   1),
]

ASCII_MODE, C40_MODE, TEXT_MODE, X12_MODE, EDF_MODE, B256_MODE = range(6)
MODE_NAMES = {
    ASCII_MODE: "ASCII",
    C40_MODE: "C40",
    TEXT_MODE: "TEXT",
    X12_MODE: "X12",
    EDF_MODE: "EDIFACT",
    B256_MODE: "BASE256",
}
LATCH = {C40_MODE: 230, TEXT_MODE: 239, X12_MODE: 238, EDF_MODE: 240, B256_MODE: 231}
UNLATCH_ASCII = 254
PAD = 129
UPPER_SHIFT = 235


# --------------------------------------------------------------------------
# Character set helpers
# --------------------------------------------------------------------------
def is_digit(c: int) -> bool:
    return 48 <= c <= 57


def is_extended(c: int) -> bool:
    return c > 127


def c40_value(c: int) -> list[int]:
    """C40 values for one character (may include a shift prefix)."""
    if c == 32:
        return [3]
    if is_digit(c):
        return [c - 48 + 4]
    if 65 <= c <= 90:                      # A-Z
        return [c - 65 + 14]
    if c < 32:                             # shift 1
        return [0, c]
    if 33 <= c <= 47:
        return [1, c - 33]
    if 58 <= c <= 64:
        return [1, c - 58 + 15]
    if 91 <= c <= 95:
        return [1, c - 91 + 22]
    if 96 <= c <= 127:                     # shift 3
        return [2, c - 96]
    return [1, 30] + c40_value(c - 128)    # upper shift


def text_value(c: int) -> list[int]:
    if c == 32:
        return [3]
    if is_digit(c):
        return [c - 48 + 4]
    if 97 <= c <= 122:                     # a-z
        return [c - 97 + 14]
    if c < 32:
        return [0, c]
    if 33 <= c <= 47:
        return [1, c - 33]
    if 58 <= c <= 64:
        return [1, c - 58 + 15]
    if 91 <= c <= 95:
        return [1, c - 91 + 22]
    if c == 96:
        return [2, 0]
    if 65 <= c <= 90:                      # A-Z live in set 3 for TEXT
        return [2, c - 65 + 1]
    if 123 <= c <= 127:
        return [2, c - 123 + 27]
    return [1, 30] + text_value(c - 128)


def x12_value(c: int) -> list[int] | None:
    if c == 13:
        return [0]
    if c == 42:
        return [1]
    if c == 62:
        return [2]
    if c == 32:
        return [3]
    if is_digit(c):
        return [c - 48 + 4]
    if 65 <= c <= 90:
        return [c - 65 + 14]
    return None


def is_c40_basic(c: int) -> bool:
    return c == 32 or is_digit(c) or 65 <= c <= 90


def is_text_basic(c: int) -> bool:
    return c == 32 or is_digit(c) or 97 <= c <= 122


def is_x12(c: int) -> bool:
    return x12_value(c) is not None


def is_edifact(c: int) -> bool:
    return 32 <= c <= 94


# --------------------------------------------------------------------------
# Annex P look-ahead: choose the cheapest encodation scheme from position pos
# --------------------------------------------------------------------------
def look_ahead(data: bytes, pos: int, mode: int) -> int:
    if pos >= len(data):
        return ASCII_MODE

    if mode == ASCII_MODE:
        counts = [0.0, 1.0, 1.0, 1.0, 1.0, 1.25]
    else:
        counts = [1.0, 2.0, 2.0, 2.0, 2.0, 2.25]
        counts[mode] = 0.0

    n = pos
    while True:
        if n == len(data):
            ints = [math.ceil(x) for x in counts]
            lo = min(ints)
            if ints[ASCII_MODE] == lo:
                return ASCII_MODE
            if ints.count(lo) == 1:
                for m in (B256_MODE, EDF_MODE, TEXT_MODE, X12_MODE):
                    if ints[m] == lo:
                        return m
            return C40_MODE

        c = data[n]
        n += 1

        # ASCII
        if is_digit(c):
            counts[ASCII_MODE] += 0.5
        elif is_extended(c):
            counts[ASCII_MODE] = math.ceil(counts[ASCII_MODE]) + 2.0
        else:
            counts[ASCII_MODE] = math.ceil(counts[ASCII_MODE]) + 1.0
        # C40
        if is_c40_basic(c):
            counts[C40_MODE] += 2.0 / 3.0
        elif is_extended(c):
            counts[C40_MODE] += 8.0 / 3.0
        else:
            counts[C40_MODE] += 4.0 / 3.0
        # TEXT
        if is_text_basic(c):
            counts[TEXT_MODE] += 2.0 / 3.0
        elif is_extended(c):
            counts[TEXT_MODE] += 8.0 / 3.0
        else:
            counts[TEXT_MODE] += 4.0 / 3.0
        # X12
        if is_x12(c):
            counts[X12_MODE] += 2.0 / 3.0
        elif is_extended(c):
            counts[X12_MODE] += 13.0 / 3.0
        else:
            counts[X12_MODE] += 10.0 / 3.0
        # EDIFACT
        if is_edifact(c):
            counts[EDF_MODE] += 3.0 / 4.0
        elif is_extended(c):
            counts[EDF_MODE] += 17.0 / 4.0
        else:
            counts[EDF_MODE] += 13.0 / 4.0
        # BASE256
        counts[B256_MODE] += 1.0

        if n - pos < 4:
            continue

        ints = [math.ceil(x) for x in counts]
        others = lambda *ms: min(ints[m] for m in ms)  # noqa: E731

        if ints[ASCII_MODE] < others(C40_MODE, TEXT_MODE, X12_MODE, EDF_MODE, B256_MODE):
            return ASCII_MODE
        if (ints[B256_MODE] < ints[ASCII_MODE]
                or ints[B256_MODE] + 1 < others(C40_MODE, TEXT_MODE, X12_MODE, EDF_MODE)):
            return B256_MODE
        if ints[EDF_MODE] + 1 < others(ASCII_MODE, C40_MODE, TEXT_MODE, X12_MODE, B256_MODE):
            return EDF_MODE
        if ints[TEXT_MODE] + 1 < others(ASCII_MODE, C40_MODE, X12_MODE, EDF_MODE, B256_MODE):
            return TEXT_MODE
        if ints[X12_MODE] + 1 < others(ASCII_MODE, C40_MODE, TEXT_MODE, EDF_MODE, B256_MODE):
            return X12_MODE
        if ints[C40_MODE] + 1 < others(ASCII_MODE, TEXT_MODE, EDF_MODE, B256_MODE):
            if ints[C40_MODE] < ints[X12_MODE]:
                return C40_MODE
            if ints[C40_MODE] == ints[X12_MODE]:
                p = n + 1
                while p < len(data):
                    tc = data[p]
                    if tc in (13, 42, 62):       # X12 terminator / separator
                        return X12_MODE
                    if not is_x12(tc):
                        break
                    p += 1
                return C40_MODE


# --------------------------------------------------------------------------
# High level encoder -> data codewords + per-segment log
# --------------------------------------------------------------------------
class Segment:
    def __init__(self, mode, char_start, cw_start):
        self.mode = mode
        self.char_start = char_start
        self.char_end = char_start
        self.cw_start = cw_start
        self.cw_end = cw_start

    @property
    def chars(self):
        return self.char_end - self.char_start

    @property
    def cws(self):
        return self.cw_end - self.cw_start


def encode_data(data: bytes) -> tuple[list[int], list[Segment], list[int]]:
    """Return (data codewords, segment log, positions of EDIFACT unlatch codewords)."""
    cw: list[int] = []
    segs: list[Segment] = []
    edf_unlatch: list[int] = []
    mode = ASCII_MODE
    pos = 0

    seg = Segment(ASCII_MODE, 0, 0)
    segs.append(seg)

    def close_seg(char_pos):
        seg.char_end = char_pos
        seg.cw_end = len(cw)

    def open_seg(new_mode, char_pos):
        nonlocal seg
        close_seg(char_pos)
        seg = Segment(new_mode, char_pos, len(cw))
        segs.append(seg)

    while pos < len(data):
        if mode == ASCII_MODE:
            new_mode = look_ahead(data, pos, ASCII_MODE)
            if new_mode == EDF_MODE and not is_edifact(data[pos]):
                new_mode = ASCII_MODE          # never latch into a dead end
            if new_mode == X12_MODE and not is_x12(data[pos]):
                new_mode = ASCII_MODE
            if new_mode != ASCII_MODE:
                cw.append(LATCH[new_mode])
                open_seg(new_mode, pos)
                mode = new_mode
                continue
            c = data[pos]
            if is_digit(c) and pos + 1 < len(data) and is_digit(data[pos + 1]):
                cw.append(130 + int(chr(c) + chr(data[pos + 1])))
                pos += 2
            elif is_extended(c):
                cw.append(UPPER_SHIFT)
                cw.append(c - 128 + 1)
                pos += 1
            else:
                cw.append(c + 1)
                pos += 1

        elif mode in (C40_MODE, TEXT_MODE, X12_MODE):
            conv = {C40_MODE: c40_value, TEXT_MODE: text_value, X12_MODE: x12_value}[mode]
            values: list[int] = []
            # last point where a triple ended on a character boundary
            clean_pos, clean_cw = pos, len(cw)
            while pos < len(data):
                v = conv(data[pos])
                if v is None:                       # not representable (X12)
                    break
                values.extend(v)
                pos += 1
                while len(values) >= 3:
                    a, b, d = values[0], values[1], values[2]
                    del values[0:3]
                    x = 1600 * a + 40 * b + d + 1
                    cw.append(x >> 8)
                    cw.append(x & 0xFF)
                if not values:                      # on a clean triple boundary
                    clean_pos, clean_cw = pos, len(cw)
                    nm = look_ahead(data, pos, mode)
                    if nm != mode:
                        break
            # end of run: return to ASCII, re-encoding in ASCII everything after
            # the last clean boundary (a shifted character can straddle the
            # last full triple, so an unfinished triple alone is not enough)
            del cw[clean_cw:]
            cw.append(UNLATCH_ASCII)
            pos = clean_pos
            open_seg(ASCII_MODE, pos)
            mode = ASCII_MODE

        elif mode == EDF_MODE:
            values: list[int] = []
            quad_start = pos
            while pos < len(data) and is_edifact(data[pos]):
                values.append(data[pos] & 0x3F)
                pos += 1
                if len(values) == 4:
                    _write_edifact(cw, values)
                    values = []
                    quad_start = pos
                    nm = look_ahead(data, pos, EDF_MODE)
                    if nm != EDF_MODE:
                        break
            # UNLATCH (011111) zero-filled to the codeword boundary; an
            # unfinished quadruple is cheaper re-encoded in ASCII
            edf_unlatch.append(len(cw))
            cw.append(0x1F << 2)
            pos = quad_start
            open_seg(ASCII_MODE, pos)
            mode = ASCII_MODE

        elif mode == B256_MODE:
            chunk = bytearray()
            while pos < len(data):
                chunk.append(data[pos])
                pos += 1
                nm = look_ahead(data, pos, B256_MODE)
                if nm != B256_MODE:
                    break
            _write_base256(cw, chunk)
            open_seg(ASCII_MODE, pos)
            mode = ASCII_MODE

    close_seg(pos)
    segs = [s for s in segs if s.chars or s.cws]
    return cw, segs, edf_unlatch


def _write_edifact(cw: list[int], values: list[int]) -> None:
    v = (values[0] << 18) | (values[1] << 12) | (values[2] << 6) | values[3]
    cw.append((v >> 16) & 0xFF)
    cw.append((v >> 8) & 0xFF)
    cw.append(v & 0xFF)


def _randomize_255(value: int, position: int) -> int:
    """253/255-state randomizing for BASE256 (position is 1-based)."""
    pseudo = ((149 * position) % 255) + 1
    tmp = value + pseudo
    return tmp if tmp <= 255 else tmp - 256


def _write_base256(cw: list[int], chunk: bytes) -> None:
    n = len(chunk)
    if n <= 249:                                     # single length codeword
        cw.append(_randomize_255(n, len(cw) + 1))
    else:                                            # two length codewords
        cw.append(_randomize_255(n // 250 + 249, len(cw) + 1))
        cw.append(_randomize_255(n % 250, len(cw) + 1))
    for b in chunk:
        cw.append(_randomize_255(b, len(cw) + 1))


def add_padding(cw: list[int], capacity: int) -> int:
    """Append PAD codewords (first plain 129, rest 253-state randomized)."""
    added = 0
    if len(cw) < capacity:
        cw.append(PAD)
        added += 1
    while len(cw) < capacity:
        pseudo = ((149 * (len(cw) + 1)) % 253) + 1
        tmp = PAD + pseudo
        cw.append(tmp if tmp <= 254 else tmp - 254)
        added += 1
    return added


# --------------------------------------------------------------------------
# Reed-Solomon over GF(256), primitive polynomial 0x12D (x^8+x^5+x^3+x^2+1)
# --------------------------------------------------------------------------
_ALOG = [1] * 255
_LOG = [0] * 256
for _i in range(1, 255):
    _v = _ALOG[_i - 1] * 2
    if _v >= 256:
        _v ^= 0x12D
    _ALOG[_i] = _v
    _LOG[_v] = _i


def rs_generator(nsym: int) -> list[int]:
    g = [1]
    for i in range(1, nsym + 1):
        ng = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            ng[j] ^= c
            if c:
                ng[j + 1] ^= _ALOG[(_LOG[c] + i) % 255]
        g = ng
    return g                                    # highest power first


def rs_encode(block: list[int], nsym: int) -> list[int]:
    g = rs_generator(nsym)[::-1]                # ascending powers
    ecc = [0] * nsym
    for b in block:
        factor = b ^ ecc[nsym - 1]
        for i in range(nsym - 1, 0, -1):
            ecc[i] = ecc[i - 1]
            if factor and g[i]:
                ecc[i] ^= _ALOG[(_LOG[g[i]] + _LOG[factor]) % 255]
        ecc[0] = 0
        if factor and g[0]:
            ecc[0] = _ALOG[(_LOG[g[0]] + _LOG[factor]) % 255]
    return ecc[::-1]


def interleave_ecc(data_cw: list[int], sym: Symbol):
    """Split into RS blocks, compute ECC, return (all_codewords, block_info)."""
    _, _, _, _, _, _, _, eccblk, blocks = sym
    block_data = [data_cw[i::blocks] for i in range(blocks)]
    block_ecc = [rs_encode(b, eccblk) for b in block_data]
    ecc_stream: list[int] = []
    for i in range(eccblk):
        for b in block_ecc:
            ecc_stream.append(b[i])
    return data_cw + ecc_stream, list(zip(block_data, block_ecc))


# --------------------------------------------------------------------------
# ECC 200 module placement (ISO/IEC 16022 Annex F)
# --------------------------------------------------------------------------
def placement_map(nrow: int, ncol: int) -> list[list[tuple[int, int]]]:
    arr: list[tuple[int, int] | None] = [None] * (nrow * ncol)

    def free(row, col):
        idx = row * ncol + col
        return 0 <= idx < len(arr) and arr[idx] is None

    def module(row, col, chrn, bit):
        if row < 0:
            row += nrow
            col += 4 - ((nrow + 4) % 8)
        if col < 0:
            col += ncol
            row += 4 - ((ncol + 4) % 8)
        if row >= nrow:                              # DMRE (ISO/IEC 21471) wrap
            row -= nrow
        arr[row * ncol + col] = (chrn, bit)

    def utah(row, col, chrn):
        module(row - 2, col - 2, chrn, 1)
        module(row - 2, col - 1, chrn, 2)
        module(row - 1, col - 2, chrn, 3)
        module(row - 1, col - 1, chrn, 4)
        module(row - 1, col, chrn, 5)
        module(row, col - 2, chrn, 6)
        module(row, col - 1, chrn, 7)
        module(row, col, chrn, 8)

    def corner1(chrn):
        module(nrow - 1, 0, chrn, 1)
        module(nrow - 1, 1, chrn, 2)
        module(nrow - 1, 2, chrn, 3)
        module(0, ncol - 2, chrn, 4)
        module(0, ncol - 1, chrn, 5)
        module(1, ncol - 1, chrn, 6)
        module(2, ncol - 1, chrn, 7)
        module(3, ncol - 1, chrn, 8)

    def corner2(chrn):
        module(nrow - 3, 0, chrn, 1)
        module(nrow - 2, 0, chrn, 2)
        module(nrow - 1, 0, chrn, 3)
        module(0, ncol - 4, chrn, 4)
        module(0, ncol - 3, chrn, 5)
        module(0, ncol - 2, chrn, 6)
        module(0, ncol - 1, chrn, 7)
        module(1, ncol - 1, chrn, 8)

    def corner3(chrn):
        module(nrow - 3, 0, chrn, 1)
        module(nrow - 2, 0, chrn, 2)
        module(nrow - 1, 0, chrn, 3)
        module(0, ncol - 2, chrn, 4)
        module(0, ncol - 1, chrn, 5)
        module(1, ncol - 1, chrn, 6)
        module(2, ncol - 1, chrn, 7)
        module(3, ncol - 1, chrn, 8)

    def corner4(chrn):
        module(nrow - 1, 0, chrn, 1)
        module(nrow - 1, ncol - 1, chrn, 2)
        module(0, ncol - 3, chrn, 3)
        module(0, ncol - 2, chrn, 4)
        module(0, ncol - 1, chrn, 5)
        module(1, ncol - 3, chrn, 6)
        module(1, ncol - 2, chrn, 7)
        module(1, ncol - 1, chrn, 8)

    chrn = 1
    row, col = 4, 0
    while True:
        if row == nrow and col == 0:
            corner1(chrn); chrn += 1
        elif row == nrow - 2 and col == 0 and ncol % 4:
            corner2(chrn); chrn += 1
        elif row == nrow - 2 and col == 0 and ncol % 8 == 4:
            corner3(chrn); chrn += 1
        elif row == nrow + 4 and col == 2 and ncol % 8 == 0:
            corner4(chrn); chrn += 1

        while True:                                  # diagonal, up and right
            if row < nrow and col >= 0 and free(row, col):
                utah(row, col, chrn); chrn += 1
            row -= 2
            col += 2
            if not (row >= 0 and col < ncol):
                break
        row += 1
        col += 3

        while True:                                  # diagonal, down and left
            if row >= 0 and col < ncol and free(row, col):
                utah(row, col, chrn); chrn += 1
            row += 2
            col -= 2
            if not (row < nrow and col >= 0):
                break
        row += 3
        col += 1

        if row >= nrow and col >= ncol:
            break

    if arr[nrow * ncol - 1] is None:                 # fixed corner pattern
        arr[nrow * ncol - 1] = (0, 0)
        arr[nrow * ncol - ncol - 2] = (0, 0)
    return [arr[r * ncol:(r + 1) * ncol] for r in range(nrow)]


def build_matrix(codewords: list[int], sym: Symbol) -> list[list[int]]:
    rows, cols, dregr, dregc, regr, regc, *_ = sym
    mrow, mcol = dregr * regr, dregc * regc
    place = placement_map(mrow, mcol)

    mapping = [[0] * mcol for _ in range(mrow)]
    for r in range(mrow):
        for c in range(mcol):
            cell = place[r][c]
            if cell is None:
                continue
            chrn, bit = cell
            if chrn == 0:                            # fixed corner pattern module
                mapping[r][c] = 1
            else:
                mapping[r][c] = (codewords[chrn - 1] >> (8 - bit)) & 1

    matrix = [[0] * cols for _ in range(rows)]
    for rv in range(regr):
        for rh in range(regc):
            r0, c0 = rv * (dregr + 2), rh * (dregc + 2)
            for i in range(dregr + 2):               # left solid + right clock
                matrix[r0 + i][c0] = 1
                matrix[r0 + i][c0 + dregc + 1] = 1 if (i + dregc + 1) % 2 == 0 else 0
            for j in range(dregc + 2):               # bottom solid + top clock
                matrix[r0 + dregr + 1][c0 + j] = 1
                matrix[r0][c0 + j] = 1 if j % 2 == 0 else 0
            for i in range(dregr):                   # data
                for j in range(dregc):
                    matrix[r0 + 1 + i][c0 + 1 + j] = mapping[rv * dregr + i][rh * dregc + j]
    return matrix


# --------------------------------------------------------------------------
# Symbol selection / full encode
# --------------------------------------------------------------------------
def choose_symbol(n_data: int, allow_rect: bool, forced: str | None) -> Symbol:
    candidates = SQUARE_SYMBOLS + (RECT_SYMBOLS if allow_rect else [])
    if forced:
        for s in SQUARE_SYMBOLS + RECT_SYMBOLS:
            if f"{s[0]}x{s[1]}" == forced:
                if s[6] < n_data:
                    raise ValueError(
                        f"{forced} holds {s[6]} data codewords, {n_data} needed")
                return s
        raise ValueError(f"unknown symbol size {forced}")
    candidates.sort(key=lambda s: (s[6], s[0] * s[1]))
    for s in candidates:
        if s[6] >= n_data:
            return s
    raise ValueError(f"content too long: {n_data} data codewords, max 1558")


class Result:
    pass


def generate(text: str, allow_rect: bool = False, forced_size: str | None = None) -> Result:
    try:
        raw = text.encode("latin-1")
        charset = "ISO-8859-1 (latin-1), 1 byte/char"
    except UnicodeEncodeError:
        raw = text.encode("utf-8")
        charset = "UTF-8 (no ECI emitted - reader must assume UTF-8)"

    data_cw, segments, edf_unlatch = encode_data(raw)

    # A decoder leaves EDIFACT on its own once 2 or fewer codewords are left in
    # the symbol, so an unlatch landing there would be read as data: drop it.
    while True:
        sym = choose_symbol(len(data_cw), allow_rect, forced_size)
        stale = [p for p in edf_unlatch if sym[6] - p <= 2]
        if not stale:
            break
        p = min(stale)
        del data_cw[p]
        edf_unlatch = [q for q in edf_unlatch if q != p]
        for s in segments:
            s.cw_start -= 1 if s.cw_start > p else 0
            s.cw_end -= 1 if s.cw_end > p else 0

    payload_cw = len(data_cw)
    pad_cw = add_padding(data_cw, sym[6])
    all_cw, blocks = interleave_ecc(data_cw, sym)
    matrix = build_matrix(all_cw, sym)

    res = Result()
    res.text = text
    res.raw = raw
    res.charset = charset
    res.segments = segments
    res.data_cw = data_cw
    res.payload_cw = payload_cw
    res.pad_cw = pad_cw
    res.all_cw = all_cw
    res.blocks = blocks
    res.sym = sym
    res.matrix = matrix
    return res


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_png(matrix, path: str, scale: int, quiet: int) -> tuple[int, int]:
    rows, cols = len(matrix), len(matrix[0])
    w, h = (cols + 2 * quiet) * scale, (rows + 2 * quiet) * scale
    img = Image.new("1", (w, h), 1)                  # 1 = white
    px = img.load()
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c]:
                x0, y0 = (c + quiet) * scale, (r + quiet) * scale
                for y in range(y0, y0 + scale):
                    for x in range(x0, x0 + scale):
                        px[x, y] = 0
    img.save(path, "PNG")
    return w, h


# preview styles: dark module -> glyph, light module -> glyph
ART_STYLES = {
    "half":   None,                       # 1 char per module, 2 rows per line
    "full":   ("██", "  "),     # solid blocks, 2 chars per module
    "dots":   ("██", "··"),
    "ascii":  ("##", "  "),
    "hash":   ("##", ".."),
}
_HALF_GLYPHS = " ▄▀█"      # index = (top << 1) | bottom


def _encodable(text: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def ascii_art(matrix, style: str = "half", quiet: int = 2, invert: bool = False) -> str:
    cols = len(matrix[0])
    blank = [0] * (cols + 2 * quiet)
    grid = [list(blank) for _ in range(quiet)]
    grid += [[0] * quiet + list(row) + [0] * quiet for row in matrix]
    grid += [list(blank) for _ in range(quiet)]
    if invert:                            # for dark terminals: light modules glow
        grid = [[1 - v for v in row] for row in grid]

    if style == "half":
        if len(grid) % 2:
            grid.append(list(blank))
        lines = ["".join(_HALF_GLYPHS[(t << 1) | b] for t, b in zip(grid[i], grid[i + 1]))
                 for i in range(0, len(grid), 2)]
    else:
        dark, light = ART_STYLES.get(style) or ART_STYLES["ascii"]
        lines = ["".join(dark if m else light for m in row) for row in grid]

    art = "\n".join(lines)
    if not _encodable(art):               # console cannot print block glyphs
        return ascii_art(matrix, "ascii", quiet, invert)
    return art


def hexdump(cws: list[int], per_line: int = 16, indent: str = "    ") -> str:
    out = []
    for i in range(0, len(cws), per_line):
        chunk = cws[i:i + per_line]
        out.append(f"{indent}[{i:4d}] " + " ".join(f"{v:02X}" for v in chunk))
    return "\n".join(out)


def report(res: Result, path: str, scale: int, quiet: int, img_w: int, img_h: int,
           art_style: str, art_invert: bool) -> None:
    rows, cols, dregr, dregc, regr, regc, cap, eccblk, nblocks = res.sym
    ecc_total = eccblk * nblocks
    total_cw = cap + ecc_total
    mrow, mcol = dregr * regr, dregc * regc
    dark = sum(sum(r) for r in res.matrix)
    total_modules = rows * cols

    p = print
    p("")
    p("=" * 72)
    p(f"  CONTENT   {res.text!r}")
    p("=" * 72)

    p("\n-- INPUT ------------------------------------------------------------")
    p(f"  Characters ............ {len(res.text)}")
    p(f"  Encoded bytes ......... {len(res.raw)}")
    p(f"  Character set ......... {res.charset}")
    p(f"  Bytes ................. {' '.join(f'{b:02X}' for b in res.raw[:32])}"
      + (" ..." if len(res.raw) > 32 else ""))

    p("\n-- ENCODATION (ISO/IEC 16022 ECC 200) -------------------------------")
    p("  seg  scheme    chars  codewords  bytes/cw  content")
    for i, s in enumerate(res.segments, 1):
        ratio = f"{s.chars / s.cws:.2f}" if s.cws else "-"
        frag = res.raw[s.char_start:s.char_end].decode("latin-1", "replace")
        if len(frag) > 22:
            frag = frag[:19] + "..."
        p(f"  {i:>3}  {MODE_NAMES[s.mode]:<8}  {s.chars:>5}  {s.cws:>9}  {ratio:>8}"
          f"  {frag!r}")
    p("  (latch / unlatch codewords are counted in the segment that emits them)")
    schemes = " -> ".join(MODE_NAMES[s.mode] for s in res.segments)
    p(f"  Scheme sequence ....... {schemes}")
    p(f"  Payload codewords ..... {res.payload_cw}")
    if res.pad_cw == 0:
        p("  Pad codewords ......... 0 (symbol exactly filled)")
    else:
        p(f"  Pad codewords ......... {res.pad_cw}   (PAD 0x81"
          + (f" + {res.pad_cw - 1} 253-state randomized)" if res.pad_cw > 1 else ")"))
    p(f"  Encoding density ...... {len(res.raw) / res.payload_cw:.2f} bytes per data codeword")

    p("\n-- SYMBOL -----------------------------------------------------------")
    kind = "square" if rows == cols else "rectangular"
    p(f"  Symbol size ........... {rows} x {cols} modules ({kind})")
    p(f"  Data regions .......... {regr} x {regc} = {regr * regc} region(s)"
      f" of {dregr} x {dregc} modules")
    p(f"  Mapping matrix ........ {mrow} x {mcol} = {mrow * mcol} modules"
      f" ({total_cw} codewords x 8 bits = {total_cw * 8} bits)")
    p(f"  Finder / clock track .. {total_modules - mrow * mcol} modules"
      f" (solid L + clock track around each region)")
    p(f"  Total modules ......... {total_modules}"
      f"  ({dark} dark / {total_modules - dark} light, {100 * dark / total_modules:.1f}% dark)")

    p("\n-- CAPACITY ---------------------------------------------------------")
    p(f"  Data codeword capacity  {cap}")
    p(f"  Used by payload ....... {res.payload_cw} ({100 * res.payload_cw / cap:.1f}%)")
    p(f"  Free (padded) ......... {res.pad_cw} ({100 * res.pad_cw / cap:.1f}%)")
    spare = cap - res.payload_cw
    p(f"  Room left in symbol ... ~{spare} ASCII chars / ~{spare * 2} digits"
      f" without growing the symbol")
    nxt = next((s for s in SQUARE_SYMBOLS if s[6] > cap), None)
    if nxt:
        p(f"  Next square symbol .... {nxt[0]}x{nxt[1]} ({nxt[6]} data codewords)")

    p("\n-- ERROR CORRECTION (Reed-Solomon, GF(256), poly 0x12D) -------------")
    p(f"  ECC codewords ......... {ecc_total}"
      f"  ({100 * ecc_total / total_cw:.1f}% of the {total_cw} codewords)")
    p(f"  Interleaved RS blocks . {nblocks}")
    p(f"  Per block ............. {eccblk} ECC codewords ->"
      f" corrects {eccblk // 2} errors or {eccblk} erasures")
    for i, (bd, be) in enumerate(res.blocks, 1):
        p(f"    block {i:>2}: {len(bd):>4} data + {len(be):>3} ecc codewords")
    p(f"  Symbol-wide ........... up to {sum(len(b[1]) for b in res.blocks) // 2}"
      f" corrupted codewords recoverable")

    p("\n-- CODEWORDS --------------------------------------------------------")
    p(f"  Data ({len(res.data_cw)}):")
    p(hexdump(res.data_cw))
    p(f"  ECC ({ecc_total}, interleaved):")
    p(hexdump(res.all_cw[cap:]))

    p("\n-- IMAGE ------------------------------------------------------------")
    p(f"  File .................. {path}")
    p(f"  Module size ........... {scale} px")
    p(f"  Quiet zone ............ {quiet} modules ({quiet * scale} px per side)")
    p(f"  Image size ............ {img_w} x {img_h} px")
    p(f"  Print size @ 300 dpi .. {img_w / 300 * 25.4:.2f} x {img_h / 300 * 25.4:.2f} mm"
      f"  (module {scale / 300 * 25.4:.3f} mm)")
    p(f"  Print size @ 600 dpi .. {img_w / 600 * 25.4:.2f} x {img_h / 600 * 25.4:.2f} mm"
      f"  (module {scale / 600 * 25.4:.3f} mm)")

    if art_style != "off":
        p("\n-- PREVIEW ----------------------------------------------------------")
        p(ascii_art(res.matrix, art_style, quiet=2, invert=art_invert))
    p("")


HELP = """
Commands
  <text>          encode <text>, write the PNG and print the report
  :file NAME      output file name            (current: {file})
  :scale N        module size in pixels       (current: {scale})
  :quiet N        quiet zone in modules       (current: {quiet})
  :rect on|off    allow rectangular symbols   (current: {rect})
  :size auto|RxC  force a symbol size, e.g. 20x20 or auto  (current: {size})
  :art STYLE      preview style               (current: {art})
                  half   compact blocks, 1 char per module  (default)
                  full   solid blocks, 2 chars per module
                  dots   solid blocks on a dotted background
                  ascii  '##' on blanks, plain ASCII
                  hash   '##' on dots, plain ASCII
                  off    no preview
  :invert on|off  preview light modules instead of dark, for
                  dark terminals                (current: {invert})
  :help           this text
  :quit           exit  (Ctrl-C / Ctrl-D also work)
"""


def main() -> None:
    opts = {
        "file": "datamatrix.png",
        "scale": 8,
        "quiet": 4,
        "rect": False,
        "size": "auto",
        "art": "half",
        "invert": False,
    }

    print("Data Matrix (ECC 200) generator - ':help' for options, ':quit' to exit.")
    while True:
        try:
            line = input("\ncontent> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not line.strip():
            continue

        if line.startswith(":"):
            parts = line[1:].split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            try:
                if cmd in ("quit", "exit", "q"):
                    print("bye")
                    return
                elif cmd in ("help", "h", "?"):
                    print(HELP.format(rect="on" if opts["rect"] else "off",
                                      invert="on" if opts["invert"] else "off", **{
                                          k: v for k, v in opts.items()
                                          if k not in ("rect", "invert")}))
                elif cmd == "file":
                    opts["file"] = arg or opts["file"]
                    print(f"  output file = {opts['file']}")
                elif cmd == "scale":
                    opts["scale"] = max(1, int(arg))
                    print(f"  module size = {opts['scale']} px")
                elif cmd == "quiet":
                    opts["quiet"] = max(0, int(arg))
                    print(f"  quiet zone = {opts['quiet']} modules")
                elif cmd == "rect":
                    opts["rect"] = arg.lower() in ("on", "1", "true", "yes")
                    print(f"  rectangular symbols = {'on' if opts['rect'] else 'off'}")
                elif cmd == "art":
                    style = arg.lower()
                    if style in ("on", "1", "true", "yes"):
                        style = "half"
                    if style in ("0", "false", "no"):
                        style = "off"
                    if style not in ART_STYLES and style != "off":
                        print(f"  unknown style '{style}' - try:"
                              f" {', '.join(list(ART_STYLES) + ['off'])}")
                    else:
                        opts["art"] = style
                        print(f"  preview = {opts['art']}")
                elif cmd == "invert":
                    opts["invert"] = arg.lower() in ("on", "1", "true", "yes")
                    print(f"  preview invert = {'on' if opts['invert'] else 'off'}")
                elif cmd == "size":
                    opts["size"] = arg.lower() if arg else "auto"
                    print(f"  symbol size = {opts['size']}")
                else:
                    print(f"  unknown command ':{cmd}' - try ':help'")
            except ValueError:
                print(f"  bad argument for ':{cmd}'")
            continue

        try:
            forced = None if opts["size"] == "auto" else opts["size"]
            res = generate(line, allow_rect=opts["rect"], forced_size=forced)
            path = os.path.abspath(opts["file"])
            w, h = write_png(res.matrix, path, opts["scale"], opts["quiet"])
            report(res, path, opts["scale"], opts["quiet"], w, h,
                   opts["art"], opts["invert"])
        except ValueError as exc:
            print(f"  error: {exc}")
        except OSError as exc:
            print(f"  could not write PNG: {exc}")


if __name__ == "__main__":
    main()
