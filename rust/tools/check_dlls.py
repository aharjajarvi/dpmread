#!/usr/bin/env python3
"""Check that Windows binaries need no C/C++ run-time DLL (P6.3).

dpmread.exe and dpmread-gui.exe link the static run-time (crt-static, see
rust/.cargo/config.toml), so they run on a clean Windows without the VC++
redistributable. Lists the DLLs a PE file imports and exits 1 if one of them
is a run-time DLL (vcruntime, msvcp, ucrtbase, api-ms-win-crt-*).

    python rust/tools/check_dlls.py rust/target/release/dpmread.exe ...
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

CRT_DLL = re.compile(r"^(vcruntime|msvcp|msvcr|ucrtbase|api-ms-win-crt-)", re.IGNORECASE)
IMPORT_DIR = 1


def imported_dlls(path: Path) -> list[str]:
    """Names in the import table of a PE32+ file (no delay-load imports)."""
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{path}: not a PE file")
    n_sections, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    if struct.unpack_from("<H", data, opt)[0] != 0x20B:
        raise ValueError(f"{path}: not PE32+ (64-bit)")
    import_rva, = struct.unpack_from("<I", data, opt + 112 + 8 * IMPORT_DIR)
    sections = [struct.unpack_from("<IIII", data, opt + opt_size + 40 * i + 8)
                for i in range(n_sections)]  # virtual size, rva, raw size, raw offset

    def offset(rva: int) -> int:
        for vsize, va, rsize, raw in sections:
            if va <= rva < va + max(vsize, rsize):
                return raw + rva - va
        raise ValueError(f"{path}: rva {rva:#x} outside the sections")

    names, desc = [], offset(import_rva)
    while True:
        name_rva, = struct.unpack_from("<I", data, desc + 12)
        if name_rva == 0:
            return names
        start = offset(name_rva)
        names.append(data[start:data.index(b"\0", start)].decode("ascii"))
        desc += 20


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bad = False
    for arg in sys.argv[1:]:
        dlls = imported_dlls(Path(arg))
        crt = [d for d in dlls if CRT_DLL.match(d)]
        print(f"{arg}: {', '.join(dlls)}")
        if crt:
            print(f"  needs the C/C++ run-time DLLs {', '.join(crt)}: build with crt-static")
            bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
