"""Pack a release zip: dpmread, dpmread-gui, the libraries shipped next to
the GUI, README.txt, LICENSE, NOTICE and THIRD_PARTY_LICENSES.md, all in
one top-level folder NAME/. Used by ci.sh and .github/workflows/release.yml.

python rust/tools/release_zip.py ZIP NAME BIN_DIR EXT [LIB ...]
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md")


def main() -> None:
    zip_path, name, bin_dir, ext = sys.argv[1:5]
    bin_dir, libs = Path(bin_dir), sys.argv[5:]
    files = [(bin_dir / b, 0o755) for b in ("dpmread" + ext, "dpmread-gui" + ext)]
    # bundled libraries: NOTICE carries their licences
    files += [(bin_dir / lib, 0o644) for lib in libs]
    files.append((ROOT / "rust" / "tools" / "zip_readme.txt", 0o644, "README.txt"))
    files += [(ROOT / f, 0o644) for f in DOCS]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path, mode, *arcname in files:
            info = zipfile.ZipInfo.from_file(path, f"{name}/{arcname[0] if arcname else path.name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100000 | mode) << 16  # keep the executable bit
            z.writestr(info, path.read_bytes())


if __name__ == "__main__":
    main()
