#!/usr/bin/env python3
"""CG-v1 only. Run list/plan/check/audit before training. --execute is explicit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_conditional_gated.cli import main
if __name__=='__main__':raise SystemExit(main())
