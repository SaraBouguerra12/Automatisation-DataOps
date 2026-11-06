import pytest

from src.common.config import get_paths, load_config
from src.common.sample_data import make_sample_raw


@pytest.fixture()
def sample_env(tmp_path, monkeypatch):
    """Environnement isolé : données synthétiques dans un dossier temporaire."""
    monkeypatch.setenv("DATA_BASE", str(tmp_path / "data"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    P = get_paths(load_config())
    make_sample_raw(P)
    return P
