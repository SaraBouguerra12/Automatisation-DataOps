import os
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path="configs/params.yaml"):
    """Charge la configuration unique du pipeline (aucun chemin ni seuil en dur dans src/)."""
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg):
    """Chemins du projet. DATA_BASE / REPORTS_DIR permettent de rediriger (tests, CI sur échantillon)."""
    base = Path(os.environ.get("DATA_BASE", ROOT / cfg["paths"]["base"]))
    reports = Path(os.environ.get("REPORTS_DIR", ROOT / cfg["paths"]["reports"]))
    return SimpleNamespace(base=base, raw=base / "raw", interim=base / "interim", processed=base / "processed",
                           logs=base / "logs", reports=reports)
