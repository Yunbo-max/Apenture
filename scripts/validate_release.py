#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "AGENTS.md", "pyproject.toml", "src/aperture/cli.py",
    "configs/final/medical_multihospital.json", "configs/final/efficiency_rank16.json",
    "docs/CLAIM_FREEZE.md", "docs/EXPERIMENT_MATRIX.md", "docs/MIGRATION_PROVENANCE.md",
    "provenance/SOURCE_LOCK.json", "paper/visual_lens/main.tex"
]
missing = [path for path in required if not (ROOT / path).is_file()]
if missing:
    print("missing required files:", *missing, sep="\n- ", file=sys.stderr)
    raise SystemExit(1)
for path in ("configs/final/medical_multihospital.json", "configs/final/efficiency_rank16.json", "provenance/SOURCE_LOCK.json"):
    json.loads((ROOT / path).read_text())
text = (ROOT / "README.md").read_text()
for phrase in ("Learning to Refocus Visual Evidence", "five-hospital", "rank-16"):
    if phrase not in text:
        raise SystemExit(f"README missing required phrase: {phrase}")
print("APERTURE release structure: OK")
