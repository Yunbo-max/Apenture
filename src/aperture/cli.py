from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "core": ROOT / "scripts" / "run_visual_lens_all.sh",
    "final-calibration": ROOT / "scripts" / "run_aperture_final.sh",
    "medical-multihospital": ROOT / "scripts" / "run_aperture_medical_multimodel.sh",
    "efficiency-rank16": ROOT / "scripts" / "run_aperture_efficiency.sh",
}

def build_command(action: str, suite: str, gpus: str, extra: list[str]) -> list[str]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
    if action == "reproduce":
        action = "run"
    command = ["bash", str(SUITES[suite]), action]
    if action == "run":
        command.append(gpus)
    command.extend(extra)
    return command

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified APERTURE experiment entry point")
    parser.add_argument("action", choices=["prepare", "plan", "check", "run", "reproduce", "summarize"])
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--gpus", default="0")
    args, extra = parser.parse_known_args(argv)
    return subprocess.call(build_command(args.action, args.suite, args.gpus, extra), cwd=ROOT)

if __name__ == "__main__":
    raise SystemExit(main())
