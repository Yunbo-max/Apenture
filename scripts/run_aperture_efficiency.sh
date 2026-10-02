#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:-plan}"
if (( $# > 0 )); then shift; fi
gpus="0"
if [[ "$action" == "run" && $# -gt 0 && "$1" != --* ]]; then gpus="$1"; shift; fi
case "$action" in
  plan|prepare|check|summarize)
    exec bash scripts/run_aperture_final.sh "$action" --domains hawaii-wildfire hospital_2 --seeds 0 "$@"
    ;;
  run)
    exec bash scripts/run_aperture_final.sh run "$gpus" --domains hawaii-wildfire hospital_2 --seeds 0 "$@"
    ;;
  *) echo "usage: $0 {plan|prepare|check|run|summarize} [GPU_IDS]" >&2; exit 2 ;;
esac
