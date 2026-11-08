#!/usr/bin/env python3
"""Independent E10 entry point; existing E01-E09 source hashes stay unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_retention.cli import main
if __name__=='__main__':raise SystemExit(main())
