#!/usr/bin/env bash
# API Lambda'sının zip paketini üretir.
#
# İçinde torch, transformers veya pdfplumber yoktur: API yalnızca iş kaydı
# açar ve durum okur. boto3 Lambda çalışma ortamında hazır gelir.
#
#   scripts/build_api_zip.sh dist/api.zip
set -euo pipefail

out="$(realpath -m "${1:-dist/api.zip}")"
root="$(cd "$(dirname "$0")/.." && pwd)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT

python -m pip install --quiet --target "$build" \
  --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 \
  --only-binary=:all: \
  "fastapi>=0.110" "mangum>=0.17" "pydantic-settings>=2.2" "python-multipart>=0.0.9"

cp -r "$root/src/ozetle" "$build/"
find "$build" -name "__pycache__" -prune -exec rm -rf {} +

mkdir -p "$(dirname "$out")"
rm -f "$out"
(cd "$build" && zip -qr -X "$out" .)
echo "$out ($(du -h "$out" | cut -f1))"
