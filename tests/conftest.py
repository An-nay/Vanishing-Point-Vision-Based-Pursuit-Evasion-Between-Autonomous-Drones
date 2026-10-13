"""Make the flat-layout modules importable from tests/ and run every test from the repo root
(several tests load models/ and configs by relative path)."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)
