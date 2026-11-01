"""Compatibility entry point. Use --infer/--eval and --limit for a short run."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RichHF.inference import main

__test__ = False
if __name__ == '__main__':
    main()
