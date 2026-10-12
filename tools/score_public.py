"""Score a reader against the public test sets in testimages_public/ (round 4, A3).

python tools/score_public.py                                # run v2 on all sets, score
python tools/score_public.py --reader python/datamatrix_reader.py --save baseline/public_v1.json
python tools/score_public.py --reader rust --save /tmp/rust.json  # Rust dpmread (release build)
python tools/score_public.py --reader rust --multi                # every symbol per image
python tools/score_public.py --engines zxing                # zxing pass only
python tools/score_public.py --json-in out.json             # score a saved run (Python or
                                                            # Rust dpmread --json output)
python tools/score_public.py --compare baseline/public_v2.json new.json

Per image (one symbol per image is read):
  hit      decoded text matches one expected text after normalising (line
           breaks -> space, other control characters dropped, surrounding
           quotes and an 'FNC1' prefix stripped); `exact` counts byte-exact
           matches (text or Latin-1 hex)
  false    decoded, but matches no expected text (must stay 0); with
           `--multi` output also when a further symbol matches no expected
           text (a further copy of an expected text is not false)
  code %   codes hit / all expected codes; each expected code counts once.
           Without `--multi` one code per image at most, so Karrach (several
           codes per image) is capped below 100 %; comparable with the
           Dynamsoft table
Unannotated images (`*` in expected.tsv) are listed when read, never scored.

--compare exits 1 when a hit is lost or a false read appears.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_public_sets import DEFAULT_OUT, ROOT, read_expected  # noqa: E402

DEFAULT_READER = ROOT / "python" / "datamatrix_reader_v2.py"
RUST_READER = ROOT / "rust" / "target" / "release" / ("dpmread.exe" if sys.platform == "win32" else "dpmread")


# --------------------------------------------------------------------------
def key_of(path: str) -> str:
    """'.../testimages_public/<set>/<file>' -> '<set>/<file>'."""
    return "/".join(path.replace("\\", "/").split("/")[-2:])


def normalise(s: str) -> str:
    # line breaks / tabs -> one space (Karrach flattens multi-line texts),
    # other control characters (GS, RS, EOT) dropped
    s = "".join(" " if c in "\r\n\t" else c for c in s
                if c in "\r\n\t" or (ord(c) >= 0x20 and ord(c) != 0x7F))
    s = " ".join(s.split())
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    if s.startswith("FNC1"):
        s = s[4:]
    return s


def load_expected(root: Path, sets: list[str]) -> dict[str, list[str] | None]:
    exp = {}
    for s in sets:
        for fn, texts in read_expected(root / s / "expected.tsv").items():
            exp[f"{s}/{fn}"] = texts
    return exp


def slim(d: dict) -> dict:
    """Keep the fields needed for scoring and comparison."""
    out = {"file": key_of(d["file"]), "decoded": bool(d.get("decoded")),
           "elapsed_s": d.get("elapsed_s")}
    if out["decoded"]:
        out.update(text=d.get("text"), hex=d.get("hex"), engine=d.get("engine"))
        if isinstance(d.get("symbol"), dict):
            out["symbol"] = d["symbol"].get("size")
        more = d.get("more_symbols") or d.get("more")  # reader JSON / saved slim
        if more:
            out["more"] = [{"text": m.get("text"), "hex": m.get("hex")} for m in more]
    return out


def load_results(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("meta", {}), [slim(r) for r in data["results"]]
    return {}, [slim(r) for r in data]


def run_reader(args: argparse.Namespace, sets: list[str]) -> tuple[dict, list[dict]]:
    reader = RUST_READER if str(args.reader) == "rust" else args.reader
    # a .py reader runs under this Python, anything else (dpmread) directly
    prefix = [sys.executable] if reader.suffix == ".py" else []
    cmd = [*prefix, str(reader), *[str(args.root / s) for s in sets],
           "--json", "--budget", str(args.budget)]
    if args.engines:
        cmd += ["--engines", args.engines]
    if args.jobs:
        cmd += ["--jobs", str(args.jobs)]
    if args.multi:
        cmd += ["--multi"]
    print("running:", " ".join(cmd), file=sys.stderr)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    # dpmread exits 1 when an image is not decoded; only worse is an error
    if proc.returncode not in (0, 1):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    wall = time.monotonic() - t0
    meta = {"reader": reader.name, "budget": args.budget,
            "engines": args.engines or "default", "jobs": args.jobs or "default",
            "multi": args.multi,
            "wall_s": round(wall, 1), "date": datetime.date.today().isoformat(),
            "sets": sets}
    return meta, [slim(r) for r in json.loads(proc.stdout)]


# --------------------------------------------------------------------------
def classify(r: dict, expected: list[str] | None) -> str:
    """hit / exact / false / miss / negative-ok / unannotated(-read)."""
    if expected is None:
        return "unannotated-read" if r["decoded"] else "unannotated"
    if not r["decoded"]:
        return "miss" if expected else "negative-ok"
    text = r.get("text") or ""
    raw = bytes.fromhex(r["hex"]).decode("latin-1") if r.get("hex") else ""
    if text in expected or raw in expected:
        return "exact"
    if normalise(text) in {normalise(e) for e in expected}:
        return "hit"
    return "false"


def code_hits(r: dict, expected: list[str] | None) -> tuple[int, int]:
    """(expected codes matched, further symbols matching none) of a read;
    each expected code is used once (first symbol first). A further symbol
    with the text of an expected code that is used up is neither (the
    annotation may list fewer copies than the image has, e.g. Karrach
    DMX5k: 3 identical codes in a row, 2 annotated)."""
    if not expected or not r["decoded"]:
        return 0, 0
    left = list(expected)
    hits = unmatched = 0
    for k, s in enumerate([r, *r.get("more", [])]):
        text = s.get("text") or ""
        raw = bytes.fromhex(s["hex"]).decode("latin-1") if s.get("hex") else ""
        m = next((e for e in left if e in (text, raw)), None)
        if m is None:
            m = next((e for e in left if normalise(e) == normalise(text)), None)
        if m is None:
            known = text in expected or raw in expected or normalise(text) in {
                normalise(e) for e in expected}
            unmatched += k > 0 and not known
            continue
        left.remove(m)
        hits += 1
    return hits, unmatched


def score(results: list[dict], exp: dict, verbose: bool) -> None:
    by_set: dict[str, dict] = {}
    seen = set()
    for r in results:
        k = r["file"]
        if k not in exp:
            print(f"warning: {k} not in any expected.tsv", file=sys.stderr)
            continue
        seen.add(k)
        s = by_set.setdefault(k.split("/")[0], {
            "images": 0, "pos": 0, "neg": 0, "unann": 0, "hit": 0, "exact": 0, "false": 0,
            "codes": 0, "code_hits": 0, "time": 0.0, "notes": []})
        e = exp[k]
        c = classify(r, e)
        ch, extra_false = code_hits(r, e)
        s["code_hits"] += ch
        if extra_false and c != "false":
            c = "false"
            s["notes"].append(f"FALSE  {k}: {extra_false} further symbol(s) match no expected code")
        s["images"] += 1
        s["time"] += r.get("elapsed_s") or 0.0
        if e is None:
            s["unann"] += 1
        elif e:
            s["pos"] += 1
            s["codes"] += len(e)
        else:
            s["neg"] += 1
        if c in ("hit", "exact"):
            s["hit"] += 1
            s["exact"] += c == "exact"
        elif c == "false":
            s["false"] += 1
            if not extra_false:
                s["notes"].append(f"FALSE  {k}: {r.get('text')!r} (expected {e!r})")
        elif c == "unannotated-read":
            s["notes"].append(f"REVIEW {k}: unannotated, read {r.get('text')!r}")
        elif c == "miss" and verbose:
            s["notes"].append(f"miss   {k}")
    missing = sorted(k for k in exp if k.split("/")[0] in by_set and k not in seen)
    for k in missing:
        print(f"warning: no result for {k}", file=sys.stderr)

    hdr = (f"{'set':12} {'images':>6} {'pos':>4} {'neg':>4} {'hit':>4} {'img %':>6} "
           f"{'exact':>5} {'false':>5} {'codes':>5} {'code %':>6} {'time s':>7}")
    print(hdr)
    print("-" * len(hdr))
    tot = {k: 0 for k in ("images", "pos", "neg", "hit", "exact", "false", "codes", "code_hits")}
    tot["time"] = 0.0
    for name in sorted(by_set):
        s = by_set[name]
        for k in tot:
            tot[k] += s[k]
        print(fmt_row(name, s))
    if len(by_set) > 1:
        print("-" * len(hdr))
        print(fmt_row("total", tot))
    for name in sorted(by_set):
        for n in by_set[name]["notes"]:
            print(n)


def fmt_row(name: str, s: dict) -> str:
    img = 100.0 * s["hit"] / s["pos"] if s["pos"] else 0.0
    code = 100.0 * s["code_hits"] / s["codes"] if s["codes"] else 0.0
    return (f"{name:12} {s['images']:6} {s['pos']:4} {s['neg']:4} {s['hit']:4} {img:6.1f} "
            f"{s['exact']:5} {s['false']:5} {s['codes']:5} {code:6.1f} {s['time']:7.1f}")


def compare(old_p: Path, new_p: Path, exp: dict) -> int:
    _, old = load_results(old_p)
    _, new = load_results(new_p)
    o = {r["file"]: r for r in old}
    n = {r["file"]: r for r in new}
    lost, gained, changed, new_false = [], [], [], []
    for k in sorted(set(o) | set(n)):
        if k not in exp:
            continue
        ro, rn = o.get(k), n.get(k)
        if ro is None or rn is None:
            print(f"only in {'new' if ro is None else 'old'}: {k}")
            continue
        co, cn = classify(ro, exp[k]), classify(rn, exp[k])
        (ho, fo), (hn, fn) = code_hits(ro, exp[k]), code_hits(rn, exp[k])
        co, cn = ("false" if fo else co), ("false" if fn else cn)
        ok_o, ok_n = co in ("hit", "exact"), cn in ("hit", "exact")
        if (ok_o and not ok_n) or hn < ho:
            lost.append(k)
        elif (ok_n and not ok_o) or hn > ho:
            gained.append(k)
        if cn == "false" and co != "false":
            new_false.append(k)
        if ro["decoded"] and rn["decoded"] and ro.get("hex") != rn.get("hex"):
            changed.append(k)
    t_old = sum(r.get("elapsed_s") or 0 for r in old)
    t_new = sum(r.get("elapsed_s") or 0 for r in new)
    print(f"compare {old_p} -> {new_p}")
    print(f"  lost {len(lost)}, gained {len(gained)}, changed {len(changed)}, "
          f"new false {len(new_false)}; time {t_old:.1f} s -> {t_new:.1f} s")
    for tag, lst in (("LOST", lost), ("gained", gained), ("CHANGED", changed),
                     ("NEW FALSE", new_false)):
        for k in lst:
            print(f"  {tag:9} {k}  {o[k].get('text')!r} -> {n[k].get('text')!r}")
    return 1 if lost or new_false else 0


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sets", help="comma list (default: every folder with expected.tsv)")
    ap.add_argument("--reader", type=Path, default=DEFAULT_READER,
                    help="reader script or executable; 'rust' = rust/target/release/dpmread")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--engines", help="passed to the reader, e.g. zxing")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--multi", action="store_true",
                    help="pass --multi (every symbol per image; Rust reader only)")
    ap.add_argument("--json-in", type=Path, help="score this saved / reader JSON instead of running")
    ap.add_argument("--save", type=Path, help="write the slim results + run info as JSON")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("OLD", "NEW"))
    ap.add_argument("-v", "--verbose", action="store_true", help="also list misses")
    args = ap.parse_args()

    if args.sets:
        sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    else:
        sets = sorted(p.parent.name for p in args.root.glob("*/expected.tsv"))
    if not sets:
        sys.exit(f"no sets under {args.root}; run tools/fetch_public_sets.py first")
    exp = load_expected(args.root, sets)

    if args.compare:
        sys.exit(compare(*args.compare, exp))
    if args.json_in:
        meta, results = load_results(args.json_in)
    else:
        meta, results = run_reader(args, sets)
    results = [r for r in results if r["file"].split("/")[0] in sets]
    if meta:
        print("run:", ", ".join(f"{k}={v}" for k, v in meta.items() if k != "sets"))
    score(results, exp, args.verbose)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({"meta": meta, "results": results}, indent=1,
                                        ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
