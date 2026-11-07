#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec .venv-table/bin/python scripts/run_aperture_comprehensive.py run --models q25 q34 g34 internvl3 phi --gpus 0,1
