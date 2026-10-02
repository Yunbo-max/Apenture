from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aperture.cli import build_command

def test_core_run_forwards_gpu_argument():
    command = build_command("run", "core", "2", ["--example"])
    assert command[-3:] == ["run", "2", "--example"]
    assert command[0] == "bash"

def test_reproduce_aliases_run():
    command = build_command("reproduce", "medical-multihospital", "0,1", [])
    assert command[-2:] == ["run", "0,1"]

def test_efficiency_suite_uses_dedicated_runner():
    command = build_command("plan", "efficiency-rank16", "0", [])
    assert command[-1] == "plan"
    assert command[1].endswith("run_aperture_efficiency.sh")
