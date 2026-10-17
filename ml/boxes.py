#!/usr/bin/env python3
"""Symbol boxes on real images, for training and scoring a detector.

Every code in an image needs a box: a detector trained on images where
some codes are unlabelled learns to miss codes. Three sources, one CSV:

  manifest  the labelled target code of each image (`corners` column of
            the image list, CSV with file, split, content, text, size,
            polarity, corners as "x,y x,y x,y x,y")
  reader    every further code `dpmread --multi --json` reads (its
            `corners_px`); a read whose centre lies inside the manifest
            quad is the target itself and is skipped; an image whose
            target has no corners gets the reader's box of its text
  review    codes nobody can read, drawn on the review page; they have no
            text, size or polarity and count as "don't care"

Output CSV: file, split, source, corners, text, size, polarity. Corners
are in pixels of the image as the reader and a browser show it (EXIF
rotation applied), in the order the source gives them.

Usage:
  python ml/boxes.py merge --list LIST.csv --json DIR [--review review.csv] --out boxes.csv
  python ml/boxes.py review --boxes boxes.csv --out page.html [--root DIR]
      # page.html: draw the missing boxes, then "Save CSV" -> review.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path

import numpy as np

FIELDS = ("file", "split", "source", "corners", "text", "size", "polarity")
CODE = ("code", "code-cut")


def parse_quad(s: str) -> np.ndarray | None:
    pts = [p.split(",") for p in s.split()]
    return np.array(pts, float) if len(pts) == 4 else None


def fmt_quad(q) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in q)


def inside(pt: np.ndarray, quad: np.ndarray) -> bool:
    """Point in a convex quad (either winding)."""
    signs = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        signs.append(np.sign((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])))
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def reader_symbols(doc: dict) -> list[dict]:
    """Decoded symbols of one `dpmread --json` result, with corners."""
    out = []
    for r in [doc, *doc.get("more_symbols", [])]:
        q = (r.get("quality_estimate") or {}).get("corners_px")
        if r.get("decoded") and q:
            out.append({"text": r.get("text", ""), "corners": np.array(q, float),
                        "size": (r.get("symbol") or {}).get("size", ""),
                        "polarity": ""})          # the report has no polarity
    return out


def merge(args) -> int:
    rows = [r for r in csv.DictReader(open(args.list, encoding="utf-8"))
            if r["split"] in args.splits.split(",")]
    reads: dict[str, list[dict]] = {}
    for d in args.json:
        for f in sorted(Path(d).glob("*.json")):
            for doc in json.loads(f.read_text(encoding="utf-8")):
                reads[os.path.normpath(doc["file"])] = reader_symbols(doc)
    out, missing = [], 0
    for r in rows:
        file = os.path.normpath(r["file"])
        syms = reads.get(file)
        if syms is None:
            missing += 1
            syms = []
        target = parse_quad(r["corners"]) if r["content"] in CODE else None
        if r["content"] in CODE and target is None:
            # no drawn corners: the reader's box of the labelled text
            hit = [s for s in syms if s["text"] == r["text"]]
            if hit:
                target = hit[0]["corners"]
                out.append({"file": file, "split": r["split"], "source": "reader",
                            "corners": fmt_quad(target), "text": r["text"],
                            "size": r["size"] or hit[0]["size"],
                            "polarity": r["polarity"] or hit[0]["polarity"]})
                syms = [s for s in syms if s is not hit[0]]
        elif target is not None:
            out.append({"file": file, "split": r["split"], "source": "manifest",
                        "corners": fmt_quad(target), "text": r["text"], "size": r["size"],
                        "polarity": r["polarity"]})
        for s in syms:
            if target is not None and inside(s["corners"].mean(0), target):
                continue
            out.append({"file": file, "split": r["split"], "source": "reader",
                        "corners": fmt_quad(s["corners"]), "text": s["text"],
                        "size": s["size"], "polarity": s["polarity"]})
    split_of = {os.path.normpath(r["file"]): r["split"] for r in rows}
    if args.review:
        for r in csv.DictReader(open(args.review, encoding="utf-8")):
            file = os.path.normpath(r["file"])
            if file in split_of and r.get("corners"):
                out.append({"file": file, "split": split_of[file], "source": "review",
                            "corners": r["corners"], "text": "", "size": "",
                            "polarity": ""})
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    by = {s: sum(o["source"] == s for o in out) for s in ("manifest", "reader", "review")}
    print(f"{len(rows)} images, {len(out)} boxes {by}; {missing} images without reader "
          "output")
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Box review</title>
<style>
:root { --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --line: #d8d8d2; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1b1b1a; --fg: #ececea; --muted: #a0a09a; --line: #3a3a37; }
}
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.4 system-ui, sans-serif; }
header { position: sticky; top: 0; background: var(--bg); padding: 8px 16px;
  border-bottom: 1px solid var(--line); display: flex; gap: 12px;
  align-items: center; flex-wrap: wrap; z-index: 1; }
button { font: inherit; padding: 4px 10px; }
#name { color: var(--muted); }
#wrap { padding: 8px 16px; }
canvas { max-width: 100%; cursor: crosshair; display: block; }
.k { color: var(--muted); }
</style></head><body>
<header>
  <button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
  <span id="pos"></span>
  <label><input type="checkbox" id="done"> every code has a box</label>
  <button id="undo">Undo box</button>
  <button id="save">Save CSV</button>
  <span id="name"></span>
  <span class="k">Click the 4 corners of each code without a box.
  Green: target, blue: read, orange: drawn here. Keys: n / p, u = undo, d = done.</span>
</header>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const IMAGES = __IMAGES__;
const KEY = "box-review:" + __KEY__;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = 0, img = new Image(), pts = [];
const c = document.getElementById("c"), g = c.getContext("2d");
function st(f) { return state[f] || (state[f] = {boxes: [], done: false}); }
function store() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function poly(q, col) {
  g.strokeStyle = col; g.lineWidth = Math.max(3, img.width / 500);
  g.beginPath(); q.forEach((p, k) => k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
  g.closePath(); g.stroke();
}
function draw() {
  const it = IMAGES[i], s = st(it.file);
  c.width = img.width; c.height = img.height; g.drawImage(img, 0, 0);
  it.boxes.forEach(b => poly(b.q, b.src === "manifest" ? "#1db954" : "#2f7bff"));
  s.boxes.forEach(q => poly(q, "#ff8a00"));
  g.fillStyle = "#ff8a00";
  pts.forEach(p => { g.beginPath(); g.arc(p[0], p[1], img.width / 300, 0, 7); g.fill(); });
  document.getElementById("pos").textContent = (i + 1) + " / " + IMAGES.length +
    "  (" + IMAGES.filter(x => st(x.file).done).length + " done)";
  document.getElementById("name").textContent = it.file;
  document.getElementById("done").checked = s.done;
}
function show(k) {
  i = (k + IMAGES.length) % IMAGES.length; pts = [];
  img = new Image(); img.onload = draw; img.src = IMAGES[i].src;
}
c.addEventListener("click", e => {
  const r = c.getBoundingClientRect();
  pts.push([(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]);
  if (pts.length === 4) { st(IMAGES[i].file).boxes.push(pts); pts = []; store(); }
  draw();
});
function undo() {
  if (pts.length) pts.pop(); else st(IMAGES[i].file).boxes.pop();
  store(); draw();
}
function done() { const s = st(IMAGES[i].file); s.done = !s.done; store(); draw(); }
document.getElementById("prev").onclick = () => show(i - 1);
document.getElementById("next").onclick = () => show(i + 1);
document.getElementById("undo").onclick = undo;
document.getElementById("done").onchange = done;
document.addEventListener("keydown", e => {
  if (e.key === "n") show(i + 1); else if (e.key === "p") show(i - 1);
  else if (e.key === "u") undo(); else if (e.key === "d") done();
});
document.getElementById("save").onclick = () => {
  let s = "file,corners,done\\n";
  IMAGES.forEach(it => {
    const t = st(it.file), q = x => '"' + x.replace(/"/g, '""') + '"';
    t.boxes.forEach(b => s += q(it.file) + "," +
      q(b.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ")) + "," + (t.done ? 1 : 0) + "\\n");
    if (!t.boxes.length) s += q(it.file) + ",," + (t.done ? 1 : 0) + "\\n";
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type: "text/csv"}));
  a.download = "review.csv"; a.click();
};
show(0);
</script></body></html>
"""


def review(args) -> int:
    rows = list(csv.DictReader(open(args.boxes, encoding="utf-8")))
    files = sorted({r["file"] for r in rows} | set(args.extra or []))
    if args.list:
        files = sorted(set(files) | {os.path.normpath(r["file"]) for r in csv.DictReader(
            open(args.list, encoding="utf-8")) if r["split"] in args.splits.split(",")
            and r["content"] in CODE})
    root = Path(args.root).resolve()
    base = args.out.resolve().parent
    images = []
    for f in files:
        src = os.path.relpath(root / f, base)
        boxes = [{"q": parse_quad(r["corners"]).tolist(), "src": r["source"]}
                 for r in rows if r["file"] == f and r["source"] != "review"]
        images.append({"file": f, "src": src, "boxes": boxes})
    page = PAGE.replace("__IMAGES__", json.dumps(images)).replace(
        "__KEY__", json.dumps(html.escape(args.out.name)))
    args.out.write_text(page, encoding="utf-8")
    print(f"{len(images)} images in {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--list", type=Path, required=True)
    m.add_argument("--json", type=Path, nargs="+", required=True,
                   help="folders of `dpmread --multi --json` output")
    m.add_argument("--review", type=Path)
    m.add_argument("--splits", default="val,test")
    m.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("review")
    r.add_argument("--boxes", type=Path, required=True)
    r.add_argument("--list", type=Path, help="also every code image of this list")
    r.add_argument("--splits", default="val,test")
    r.add_argument("--extra", nargs="*")
    r.add_argument("--root", default=".", help="folder the file paths start from")
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return merge(args) if args.cmd == "merge" else review(args)


if __name__ == "__main__":
    sys.exit(main())
