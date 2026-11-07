#!/usr/bin/env python3
"""Run the separately versioned six-backbone comprehensive experiment suite."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_comprehensive.cli import main
if __name__=='__main__':main()
