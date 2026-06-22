#!/usr/bin/env bash
# Compute SHA256 checksums for every best.pt in a local checkpoint tree.
# Usage:  CKPT_ROOT=/path/to/kan-guppylm/checkpoints bash gen_checksums.sh
# Writes ./SHA256SUMS (paths relative to CKPT_ROOT). Upload it with the checkpoints.
set -euo pipefail
CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to your local checkpoints directory}"
OUT="$(cd "$(dirname "$0")" && pwd)/SHA256SUMS"

if command -v sha256sum >/dev/null 2>&1; then
  HASH() { sha256sum "$1" | awk '{print $1}'; }
else
  HASH() { shasum -a 256 "$1" | awk '{print $1}'; }   # macOS
fi

: > "$OUT"
count=0
# shellcheck disable=SC2044
for f in $(cd "$CKPT_ROOT" && find . -name 'best.pt' | sort); do
  h="$(HASH "$CKPT_ROOT/$f")"
  printf '%s  %s\n' "$h" "${f#./}" >> "$OUT"
  count=$((count+1))
  echo "  hashed $f"
done
echo "Wrote $OUT ($count files)"
