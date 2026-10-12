"""Fetch public Data Matrix test sets into testimages_public/<set>/ (round 4, A1/A2).

Each set folder gets the images (webp converted to png), `expected.tsv` and
`SOURCE.txt`. Downloads are pinned by commit / URL and SHA-256 in
`tools/public_sets.lock.json` (tracked); a hash mismatch stops the script.
Raw downloads are cached in testimages_public/_cache/, so reruns are offline
and give the same folders.

python tools/fetch_public_sets.py                      # all sets
python tools/fetch_public_sets.py --sets karrach2021
python tools/fetch_public_sets.py --karrach-zip FILE   # use a manual download
python tools/fetch_public_sets.py --update-lock        # re-list sources, rewrite lock

expected.tsv: `file<TAB>text[<TAB>text...]`, one row per image, one text per
code (Karrach images hold several codes; repeats count). A row with no text
is a negative: any read on it is a false read. A row whose only text is `*`
has no ground truth: reads are listed for review, not scored. Texts are escaped: `\\\\`,
`\\t`, `\\n`, `\\r` and `\\xNN` for other control characters; everything
else is literal UTF-8. Binary payloads (zxing TextHex) are stored as
Latin-1 text, matching the reader's `hex` bytes.

Sets: karrach2021 (CC BY 4.0, J. Imaging 7(9) 163), zxing_cpp
(Apache-2.0, test/samples/datamatrix-*), zxing_java (Apache-2.0,
core/src/test/resources/blackbox/datamatrix-*, minus images identical to
zxing_cpp). roboflow_dpm needs an account and is not fetched.
Needs Pillow only when a set contains webp.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "tools" / "public_sets.lock.json"
DEFAULT_OUT = ROOT / "testimages_public"
USER_AGENT = "datamatrix-reader-fetch/1.0"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

KARRACH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8471265/supplementaryFiles"
KARRACH_MEMBER = "jimaging-07-00163-s001.zip"

GITHUB_SETS = {
    "zxing_cpp": {
        "repo": "zxing-cpp/zxing-cpp",
        "commit": "2c3dcfeffa7878950a58c1703c86774a428b0c91",
        "prefix": "test/samples/",
        "dirs": ["datamatrix-1", "datamatrix-2", "datamatrix-3", "datamatrix-4", "datamatrix-5"],
        "licence": "Apache-2.0",
    },
    "zxing_java": {
        "repo": "zxing/zxing",
        "commit": "33dfdefcb35576612841e614d44ba9edc9aee2b5",
        "prefix": "core/src/test/resources/blackbox/",
        "dirs": ["datamatrix-1", "datamatrix-2", "datamatrix-3"],
        "licence": "Apache-2.0",
    },
}
ALL_SETS = ["karrach2021", "zxing_cpp", "zxing_java"]
DATA_EXT = {".txt", ".toml", ".bin"}

# zxing-cpp EscapeNonGraphical names, used by TextEscaped ("<GS>")
ASCII_NAMES = ["NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL", "BS", "HT", "LF", "VT",
               "FF", "CR", "SO", "SI", "DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB",
               "CAN", "EM", "SUB", "ESC", "FS", "GS", "RS", "US"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http_get(url: str, accept: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


UNANNOTATED = "*"


def escape_text(s: str) -> str:
    if s == UNANNOTATED:
        return "\\*"                       # a real text "*", not the marker
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return "".join(out)


def unescape_text(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch != "\\" or i + 1 >= len(s):
            out.append(ch)
            i += 1
            continue
        nxt = s[i + 1]
        if nxt == "x":
            out.append(chr(int(s[i + 2:i + 4], 16)))
            i += 4
            continue
        out.append({"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}.get(nxt, nxt))
        i += 2
    return "".join(out)


def read_expected(path: Path) -> dict[str, list[str] | None]:
    """expected.tsv -> {file: [text, ...] or None if unannotated} (used by score_public.py)."""
    out = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        cols = line.split("\t")
        if cols[1:] == [UNANNOTATED]:
            out[cols[0]] = None
        else:
            out[cols[0]] = [unescape_text(c) for c in cols[1:]]
    return out


def load_lock() -> dict:
    if LOCK.exists():
        return json.loads(LOCK.read_text(encoding="utf-8"))
    return {}


def save_lock(lock: dict) -> None:
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_set(out: Path, name: str, images: dict[str, bytes],
              expected: dict[str, list[str] | None], source: str) -> None:
    """Replace testimages_public/<name>/ with the given images + expected.tsv."""
    d = out / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for fn, data in images.items():
        (d / fn).write_bytes(data)
    rows = ["\t".join([fn, UNANNOTATED] if expected[fn] is None
                      else [fn] + [escape_text(t) for t in expected[fn]]) for fn in sorted(images)]
    (d / "expected.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    (d / "SOURCE.txt").write_text(source, encoding="utf-8", newline="\n")
    n_codes = sum(len(expected[fn] or []) for fn in images)
    n_neg = sum(1 for fn in images if expected[fn] == [])
    n_un = sum(1 for fn in images if expected[fn] is None)
    print(f"{name:12} {len(images):4} images  {n_codes:4} codes  {n_neg:3} negatives  "
          f"{n_un} unannotated  -> {d}")


def webp_to_png(data: bytes) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is needed to convert webp samples (pip install pillow)")
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# karrach2021
# --------------------------------------------------------------------------
def fetch_karrach(out: Path, lock: dict, update: bool, local_zip: Path | None) -> None:
    cache = out / "_cache" / "karrach2021" / KARRACH_MEMBER
    entry = lock.get("karrach2021", {})
    if local_zip:
        data = local_zip.read_bytes()
    elif cache.exists():
        data = cache.read_bytes()
    else:
        print(f"karrach2021: downloading {KARRACH_URL}")
        data = http_get(KARRACH_URL)
    # Europe PMC wraps all supplementary files in a freshly built zip; the
    # pinned file is the inner dataset zip
    zf = zipfile.ZipFile(io.BytesIO(data))
    if KARRACH_MEMBER in zf.namelist():
        data = zf.read(KARRACH_MEMBER)
    digest = sha256(data)
    if update or "sha256" not in entry:
        lock["karrach2021"] = {"url": KARRACH_URL, "member": KARRACH_MEMBER, "sha256": digest,
                               "licence": "CC BY 4.0"}
    elif digest != entry["sha256"]:
        sys.exit(f"karrach2021: SHA-256 mismatch {digest} != {entry['sha256']}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.write_bytes(data)

    zf = zipfile.ZipFile(io.BytesIO(data))
    ann = {}
    # split on LF only: splitlines() would also split at the GS / RS inside GS1 texts
    for line in zf.read("annotations.txt").decode("utf-8").split("\n"):
        line = line.rstrip("\r")
        if "=" in line:
            stem, text = line.split("=", 1)
            ann[stem.strip()] = text.split("|")
    images, expected = {}, {}
    for name in sorted(zf.namelist()):
        stem, ext = os.path.splitext(os.path.basename(name))
        if ext.lower() not in IMAGE_EXT:
            continue
        images[name] = zf.read(name)
        expected[name] = ann.get(stem)          # DMX5o.jpg has no annotation
    if set(ann) - {os.path.splitext(n)[0] for n in images}:
        sys.exit("karrach2021: annotations without an image")
    source = (
        "Karrach L., Pivarciova E., Comparative Study of Data Matrix Codes Localization and\n"
        "Recognition Methods, J. Imaging 2021, 7(9) 163, doi:10.3390/jimaging7090163.\n"
        "Supplementary dataset (jimaging-07-00163-s001.zip), licence CC BY 4.0.\n"
        f"Fetched via {KARRACH_URL}\n"
        f"SHA-256 {digest}\n"
        "Ground truth: the dataset's annotations.txt, kept verbatim (some texts carry\n"
        "quotes or an 'FNC1' prefix; score_public.py also scores a normalised match).\n"
        "Unannotated in the dataset (marked '*', not scored): "
        + ", ".join(n for n in sorted(images) if expected[n] is None) + "\n")
    write_set(out, "karrach2021", images, expected, source)


# --------------------------------------------------------------------------
# zxing samples (GitHub, pinned commit)
# --------------------------------------------------------------------------
def list_github(cfg: dict) -> list[str]:
    url = f"https://api.github.com/repos/{cfg['repo']}/git/trees/{cfg['commit']}?recursive=1"
    tree = json.loads(http_get(url, "application/vnd.github+json"))
    if tree.get("truncated"):
        sys.exit(f"{cfg['repo']}: tree listing truncated")
    paths = []
    for e in tree["tree"]:
        p = e["path"]
        if e["type"] != "blob" or not p.startswith(cfg["prefix"]):
            continue
        rel = p[len(cfg["prefix"]):]
        d, _, fn = rel.partition("/")
        ext = os.path.splitext(fn)[1].lower()
        if d in cfg["dirs"] and "/" not in fn and (ext in IMAGE_EXT or ext in DATA_EXT) \
                and not fn.endswith(".metadata.txt"):
            paths.append(rel)
    return sorted(paths)


def fetch_github(name: str, out: Path, lock: dict, update: bool) -> dict[str, bytes]:
    """Download (or read from cache) every locked file; returns {rel path: bytes}."""
    cfg = GITHUB_SETS[name]
    entry = lock.get(name)
    if update or not entry or entry.get("commit") != cfg["commit"]:
        print(f"{name}: listing {cfg['repo']}@{cfg['commit'][:10]}")
        entry = {"repo": cfg["repo"], "commit": cfg["commit"], "licence": cfg["licence"],
                 "files": {p: None for p in list_github(cfg)}}
        lock[name] = entry
    cache = out / "_cache" / name
    files = {}
    for rel, want in entry["files"].items():
        dst = cache / rel
        if dst.exists():
            data = dst.read_bytes()
        else:
            url = (f"https://raw.githubusercontent.com/{cfg['repo']}/{cfg['commit']}/"
                   + urllib.parse.quote(cfg["prefix"] + rel))
            data = http_get(url)
        digest = sha256(data)
        if want is None:
            entry["files"][rel] = digest
        elif digest != want:
            sys.exit(f"{name}: SHA-256 mismatch for {rel}")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        files[rel] = data
    return files


WS = " \t\r\n"     # str.strip() would also eat GS/RS (0x1c-0x1f) inside values


def parse_toml_value(v: str) -> str:
    v = v.strip(WS)
    if v.startswith('"') and v.endswith('"'):
        body, out, i = v[1:-1], [], 0
        while i < len(body):
            if body[i] == "\\" and i + 1 < len(body):
                out.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
                           .get(body[i + 1], body[i + 1]))
                i += 2
            else:
                out.append(body[i])
                i += 1
        return "".join(out)
    return v


def unescape_zxing(s: str) -> str:
    """zxing-cpp TextEscaped: <GS> etc. back to control characters."""
    for code, nm in enumerate(ASCII_NAMES):
        s = s.replace(f"<{nm}>", chr(code))
    return s.replace("<DEL>", "\x7f")


def toml_texts(text: str) -> list[str]:
    """Expected texts of a zxing-cpp .toml ([[symbol]] records allowed)."""
    records, cur, defaults, top = [], {}, {}, True
    for line in text.split("\n"):
        line = line.strip(WS)
        if not line or line.startswith("#"):
            continue
        if line == "[[symbol]]":
            if top:
                defaults, top = dict(cur), False    # top-level keys apply to every record
            else:
                records.append(cur)
            cur = dict(defaults)
            continue
        k, _, v = line.partition("=")
        cur[k.strip(WS)] = parse_toml_value(v)
    records.append(cur)
    out = []
    for r in records:
        if "TextHex" in r:
            out.append(bytes.fromhex(r["TextHex"]).decode("latin-1"))
        elif "TextEscaped" in r:
            out.append(unescape_zxing(r["TextEscaped"]))
        elif "TextPlain" in r:
            out.append(r["TextPlain"])
    return out


def build_zxing_cpp(out: Path, lock: dict, update: bool) -> None:
    files = fetch_github("zxing_cpp", out, lock, update)
    images, expected, hard = {}, {}, []
    for rel in files:
        d, fn = rel.split("/", 1)
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in IMAGE_EXT:
            continue
        # BlackboxTestRunner lookup: <stem>.toml / .txt, else cut the stem
        # at its last '-', '_' or '!' and retry
        key, texts = stem, None
        while texts is None:
            for dext in (".toml", ".txt"):
                cand = f"{d}/{key}{dext}"
                if cand in files:
                    raw = files[cand].decode("utf-8")
                    texts = toml_texts(raw) if dext == ".toml" else [raw]
                    break
            else:
                cut = max(key.rfind(c) for c in "-_!")
                if cut < 0:
                    texts = []
                key = key[:cut]
        if texts in (["<unkown>"], ["<unknown>"]):   # zxing-cpp placeholder: text unknown
            texts = None
        if stem.endswith("!"):
            hard.append(f"{d}_{stem}")
        data = files[rel]
        if ext.lower() == ".webp":
            data, ext = webp_to_png(data), ".png"
        out_name = f"{d}_{stem}{ext}"
        images[out_name] = data
        expected[out_name] = texts
    cfg = GITHUB_SETS["zxing_cpp"]
    source = (
        f"zxing-cpp test samples, https://github.com/{cfg['repo']} @ {cfg['commit']}\n"
        f"{cfg['prefix']}{{{','.join(cfg['dirs'])}}}, licence Apache-2.0.\n"
        "File names: <sample dir>_<original stem>; webp converted to png.\n"
        "Expected text from <stem>.txt / .toml (TextPlain, TextEscaped, TextHex), looked up\n"
        "like zxing-cpp's BlackboxTestRunner. TextPlain drops control characters, so\n"
        "score on the normalised match. zxing-cpp marks these as expected failures ('!'):\n"
        + "".join(f"  {h}\n" for h in sorted(hard)))
    write_set(out, "zxing_cpp", images, expected, source)


def build_zxing_java(out: Path, lock: dict, update: bool) -> None:
    files = fetch_github("zxing_java", out, lock, update)
    cpp_hashes = {h for h in lock.get("zxing_cpp", {}).get("files", {}).values() if h}
    images, expected, dropped = {}, {}, []
    for rel, data in files.items():
        d, fn = rel.split("/", 1)
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in IMAGE_EXT:
            continue
        if sha256(data) in cpp_hashes:
            dropped.append(rel)
            continue
        # AbstractBlackBoxTestCase: <stem>.txt (UTF-8) or <stem>.bin (Latin-1)
        if f"{d}/{stem}.txt" in files:
            texts = [files[f"{d}/{stem}.txt"].decode("utf-8")]
        elif f"{d}/{stem}.bin" in files:
            texts = [files[f"{d}/{stem}.bin"].decode("latin-1")]
        else:
            texts = []
        out_name = f"{d}_{stem}{ext}"
        images[out_name] = data
        expected[out_name] = texts
    cfg = GITHUB_SETS["zxing_java"]
    source = (
        f"ZXing (Java) blackbox samples, https://github.com/{cfg['repo']} @ {cfg['commit']}\n"
        f"{cfg['prefix']}{{{','.join(cfg['dirs'])}}}, licence Apache-2.0.\n"
        "File names: <sample dir>_<original stem>. Expected text from <stem>.txt.\n"
        f"Dropped as byte-identical to zxing_cpp samples ({len(dropped)}):\n"
        + "".join(f"  {p}\n" for p in sorted(dropped)))
    write_set(out, "zxing_java", images, expected, source)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", default=",".join(ALL_SETS),
                    help=f"comma list of {', '.join(ALL_SETS)}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--update-lock", action="store_true",
                    help="re-list sources and record new hashes instead of checking them")
    ap.add_argument("--karrach-zip", type=Path,
                    help="manually downloaded Karrach supplementary zip (inner or Europe PMC zip)")
    args = ap.parse_args()
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    for s in sets:
        if s not in ALL_SETS:
            sys.exit(f"unknown set {s!r}")
    lock = load_lock()
    if "zxing_java" in sets and "zxing_cpp" not in lock and "zxing_cpp" not in sets:
        sets.insert(0, "zxing_cpp")       # java dedupe needs the cpp hashes
    for s in ALL_SETS:
        if s not in sets:
            continue
        if s == "karrach2021":
            fetch_karrach(args.out, lock, args.update_lock, args.karrach_zip)
        elif s == "zxing_cpp":
            build_zxing_cpp(args.out, lock, args.update_lock)
        else:
            build_zxing_java(args.out, lock, args.update_lock)
        save_lock(lock)


if __name__ == "__main__":
    main()
