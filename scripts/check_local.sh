#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/validate_release.py
"$PYTHON_BIN" -m compileall -q src scripts

while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(find scripts -maxdepth 1 -type f -name '*.sh' -print0)

test_dirs=(tests)
for dir in tests_*; do
  [[ -d "$dir" ]] && test_dirs+=("$dir")
done

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  "$PYTHON_BIN" -m pytest -q "${test_dirs[@]}"

PYTHONPATH=src "$PYTHON_BIN" -m aperture.cli --help >/dev/null
PYTHONPATH=src "$PYTHON_BIN" -m aperture.cli plan --suite core

echo "APERTURE local validation: OK"
