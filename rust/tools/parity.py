"""Parity check of the Rust reader against the Python reader (plan P4.8,
Gate 4; oracle datamatrix_reader_v2.py since P4b.6). Compares the `--json` outputs by file name.

Gate: every image Python decodes is decoded by Rust with the same text and
hex. Reported, not gated: engine, enhancement variant, symbol size, RS
errors per block, quality grades, near-miss size, time; extra Rust reads
are listed separately (check them by hand: a wrong read is worse than none).

python rust/tools/parity.py                    # baseline JSON vs a fresh Rust run
python rust/tools/parity.py --run-python       # fresh runs of both
python rust/tools/parity.py --rust-json r.json # an existing Rust output
python rust/tools/parity.py --jobs-check       # also: Rust --jobs 1 == default jobs
python rust/tools/parity.py --baseline v1      # the frozen v1 oracle (phases 0-4)
python rust/tools/parity.py --dpmread rust/target/ci/dpmread  # another build (ci.sh --fast)

The jobs check runs dpmread twice more (default jobs and --jobs 1) with
--jobs-budget (default 600 s): with the gate's 60 s the sequential run
stops early on the large failing photos (~130 s of CPU each), which changes
attempts and near misses through the budget, not through the job order.

The test images and the baselines are private data, not part of this repo:
`DM_PRIVATE_DIR` points at them. Both readers run with that
directory as working directory, so file names match the baselines
(`testimages/...`).

Exit code 0 if the gate (and the jobs check, if asked) passes, else 1.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path(os.environ.get("DM_PRIVATE_DIR") or "/nonexistent")
BASELINES = {  # --baseline -> (frozen --json output, Python reader)
    "v1": (PRIVATE / "baseline/python_baseline.json", ROOT / "python/datamatrix_reader.py"),
    "v2": (PRIVATE / "baseline/python_v2_baseline.json", ROOT / "python/datamatrix_reader_v2.py"),
}
DPMREAD = ROOT / "rust/target/release/dpmread"
FOLDERS = ["testimages/large", "testimages/easy", "testimages/poor", "testimages/not_recognized"]
GRADES = ("symbol_contrast_grade", "modulation_grade", "axial_nonuniformity_grade",
          "unused_error_correction_grade", "fixed_pattern_damage_grade",
          "overall_grade_estimate")


def run(cmd: list[str]) -> tuple[list[dict], float]:
    t0 = time.time()
    out = subprocess.run(cmd, cwd=PRIVATE, capture_output=True, text=True, encoding="utf-8")
    if out.returncode not in (0, 1) or not out.stdout.strip():
        sys.exit(f"{cmd[0]} failed ({out.returncode}):\n{out.stderr}")
    return json.loads(out.stdout), time.time() - t0


def run_rust(args: argparse.Namespace, jobs: int | None = None,
             budget: float | None = None) -> tuple[list[dict], float]:
    exe = (args.dpmread or DPMREAD).resolve()  # dpmread runs in DM_PRIVATE_DIR
    exe = exe.with_suffix(".exe") if os.name == "nt" else exe
    if not exe.exists():
        sys.exit(f"{exe} missing: cargo build --release -p dm-cli (in rust/)")
    cmd = [str(exe), "--json", "--budget", str(budget or args.budget), *args.folders]
    if jobs is not None:
        cmd += ["--jobs", str(jobs)]
    return run(cmd)


def summary(e: dict | None) -> dict:
    """The report columns of one result."""
    if e is None:
        return {}
    q = e.get("quality_estimate") or {}
    return {
        "engine": e.get("engine", ""),
        "variant": e.get("enhancement", ""),
        "size": (e.get("symbol") or {}).get("size", ""),
        "rs": (e.get("error_correction") or {}).get("codewords_corrected_per_block", []),
        "grades": "".join(q.get(k, "-") for k in GRADES) if q else "",
        "near_miss": (e.get("near_miss") or {}).get("size", ""),
        "attempts": e.get("attempts"),
    }


def strip_times(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "elapsed_s"} for e in results]


def speed(results: list[dict]) -> tuple[float, float, float]:
    ok = sum(e["elapsed_s"] for e in results if e["decoded"])
    bad = sum(e["elapsed_s"] for e in results if not e["decoded"])
    return ok + bad, ok, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", choices=sorted(BASELINES), default="v2",
                    help="Python reader the oracle comes from (default v2, since P4b.6)")
    ap.add_argument("--python-json", type=Path,
                    help="Python --json output (default: the frozen baseline of --baseline)")
    ap.add_argument("--run-python", action="store_true", help="run the Python reader instead")
    ap.add_argument("--rust-json", type=Path, help="Rust --json output (default: run dpmread)")
    ap.add_argument("--dpmread", type=Path, help=f"dpmread to run (default: {DPMREAD})")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--jobs-check", action="store_true",
                    help="also run dpmread with --jobs 1 and default jobs and compare")
    ap.add_argument("--jobs-budget", type=float, default=600.0,
                    help="budget of the two jobs-check runs (large enough for --jobs 1)")
    ap.add_argument("folders", nargs="*", default=FOLDERS)
    args = ap.parse_args()

    if not (PRIVATE / "testimages").is_dir():
        sys.exit("no testimages/ in DM_PRIVATE_DIR: point it at the private test data")
    base_json, script = BASELINES[args.baseline]
    if args.run_python:
        py, py_wall = run([sys.executable, str(script), "--json",
                           "--budget", str(args.budget), *args.folders])
    else:
        py_json = args.python_json or base_json
        py, py_wall = json.loads(py_json.read_text(encoding="utf-8")), None
    if args.rust_json:
        rs, rs_wall = json.loads(args.rust_json.read_text(encoding="utf-8")), None
    else:
        rs, rs_wall = run_rust(args)

    by_py = {Path(e["file"]).as_posix(): e for e in py}
    by_rs = {Path(e["file"]).as_posix(): e for e in rs}
    lost, changed, extra, missing = [], [], [], []
    print(f"{'file':44s} {'py':4s} {'rs':4s} {'engine':12s} {'variant':34s} "
          f"{'size':11s} {'rs errors':17s} {'grades':13s} {'near miss':11s} time py/rs")
    for f in sorted(by_py.keys() | by_rs.keys()):
        a, b = by_py.get(f), by_rs.get(f)
        if b is None:
            missing.append(f)
        da, db = bool(a and a["decoded"]), bool(b and b["decoded"])
        if da and not db:
            lost.append(f)
        elif da and (a["text"], a["hex"]) != (b["text"], b["hex"]):
            changed.append(f)
        elif db and not da:
            extra.append(f)
        sa, sb = summary(a), summary(b)

        def col(k, w):
            x, y = sa.get(k, ""), sb.get(k, "")
            s = str(x) if x == y else f"{x}|{y}"
            return f"{s:{w}s}"
        ta = f"{a['elapsed_s']:.1f}" if a else "-"
        tb = f"{b['elapsed_s']:.1f}" if b else "-"
        print(f"{Path(f).name[:44]:44s} {'OK' if da else '--':4s} {'OK' if db else '--':4s} "
              f"{col('engine', 12)} {col('variant', 34)} {col('size', 11)} {col('rs', 17)} "
              f"{col('grades', 13)} {col('near_miss', 11)} {ta}/{tb}")

    ok_py = sum(1 for e in py if e["decoded"])
    ok_rs = sum(1 for e in rs if e["decoded"])
    print(f"\ndecoded: Python {ok_py}/{len(py)}, Rust {ok_rs}/{len(rs)}")
    tp, tr = speed(py), speed(rs)
    print("time (sum of elapsed_s)   total   successes  failures")
    print(f"  Python               {tp[0]:8.1f}  {tp[1]:9.1f}  {tp[2]:8.1f}")
    print(f"  Rust                 {tr[0]:8.1f}  {tr[1]:9.1f}  {tr[2]:8.1f}")
    walls = [f"{n} {w:.1f} s" for n, w in (("Python", py_wall), ("Rust", rs_wall)) if w is not None]
    if walls:
        print("wall time: " + ", ".join(walls))
    for name, lst in (("LOST", lost), ("CHANGED", changed), ("MISSING in Rust output", missing),
                      ("EXTRA Rust reads (check by hand)", extra)):
        for f in lst:
            print(f"{name}: {f}")

    ok = not lost and not changed and not missing
    if args.jobs_check:
        par, par_wall = run_rust(args, budget=args.jobs_budget)
        seq, seq_wall = run_rust(args, jobs=1, budget=args.jobs_budget)
        same = strip_times(seq) == strip_times(par)
        print(f"jobs check (budget {args.jobs_budget:g} s): --jobs 1 ({seq_wall:.0f} s) "
              f"{'identical to' if same else 'DIFFERS from'} default jobs ({par_wall:.0f} s)")
        for a, b in zip(strip_times(par), strip_times(seq)):
            if a != b:
                keys = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
                print(f"  {a['file']}: {', '.join(sorted(keys))}")
        ok &= same
    print(f"\nGATE: {'PASS' if ok else 'FAIL'} ({len(lost)} lost, {len(changed)} changed)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
