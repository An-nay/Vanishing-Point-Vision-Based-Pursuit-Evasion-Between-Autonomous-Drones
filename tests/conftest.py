import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)
