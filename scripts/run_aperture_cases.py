#!/usr/bin/env python3
"""Independent APERTURE reviewer cases. Run `list` first; GPU execution is explicit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_rebuttal_cases.cli import main
if __name__=='__main__':raise SystemExit(main())
