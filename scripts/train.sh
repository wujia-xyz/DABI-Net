#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "Python not found. Please set PYTHON_BIN to a valid interpreter." >&2
  exit 1
fi

"$PY" scripts/train.py "$@"
