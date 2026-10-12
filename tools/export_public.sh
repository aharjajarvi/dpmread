#!/usr/bin/env bash
# Export the public tree: `git archive HEAD` into an empty directory, minus
# the paths listed in tools/public_exclude.txt (one path or glob per line,
# relative to the repo root; # starts a comment). Untracked and ignored
# files never reach the export.
#
#   tools/export_public.sh OUTDIR     # OUTDIR must not exist or be empty
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: export_public.sh OUTDIR}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "error: $OUT is not empty" >&2
    exit 1
fi
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "error: uncommitted changes; the export takes HEAD only" >&2
    exit 1
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$OUT"

cd "$OUT"
removed=0
while IFS= read -r line; do
    line="${line%%#*}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    for path in $line; do
        if [ -e "$path" ]; then
            rm -rf -- "$path"
            echo "excluded: $path"
            removed=$((removed + 1))
        fi
    done
done < "$ROOT/tools/public_exclude.txt"
echo "exported $(git -C "$ROOT" rev-parse --short HEAD) to $OUT ($removed paths excluded, $(find . -type f | wc -l) files)"
