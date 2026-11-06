from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path="configs/params.yaml"):
    """Charge la configuration unique du pipeline."""
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)
