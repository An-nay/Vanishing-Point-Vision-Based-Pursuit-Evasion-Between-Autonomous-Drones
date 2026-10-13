"""Project-root-anchored locations, so default paths work from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
