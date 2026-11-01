"""Scratch directories with inherited ACLs for Windows sandbox test execution."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid


@contextmanager
def scratch_directory():
    path = Path(__file__).resolve().parents[1] / 'tmp/restoration_tests' / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path)
