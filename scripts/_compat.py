"""Temporary legacy-path bootstrap; listed in configs/path_exceptions.json."""
import sys
from pathlib import Path


def ensure_src() -> None:
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
