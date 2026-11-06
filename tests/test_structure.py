import importlib

from src.common.config import ROOT, load_config

MODULES = ["src.ingestion.kaggle_source", "src.ingestion.reddit_source", "src.validation.schemas",
           "src.validation.quality_gate", "src.preprocessing.run", "src.alignment.dedup_groups",
           "src.alignment.split", "src.export.manifest", "src.reporting.data_report", "src.common.runlog"]


def test_modules_importables():
    for m in MODULES:
        importlib.import_module(m)


def test_config_complete():
    cfg = load_config()
    for key in ("seed", "paths", "ingestion", "preprocessing", "alignment", "split", "quality"):
        assert key in cfg


def test_dvc_yaml_present():
    assert (ROOT / "dvc.yaml").exists()
