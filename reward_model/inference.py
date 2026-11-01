"""Compatibility entry point for the restored RichHF inference command."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RichHF.inference import main

if __name__ == '__main__':
    main()
