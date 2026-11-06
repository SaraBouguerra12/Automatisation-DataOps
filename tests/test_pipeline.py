"""Test de bout en bout sur données synthétiques + échec contrôlé du contrôle qualité."""
import pandas as pd
import pytest

from src.run_pipeline import main as run_all
from src.validation import quality_gate


def test_pipeline_end_to_end(sample_env):
    P = sample_env
    run_all()
    m = pd.read_parquet(P.processed / "manifest.parquet")
    assert set(m.source) == {"hateful_memes", "sentiment140"}
    assert len(m[m.source == "hateful_memes"]) == 297  # 300 mèmes - 3 doublons exacts
    assert m.sample_id.is_unique and (P.reports / "data_report.html").exists()
    assert pd.read_csv(P.logs / "rejects.csv").reason.str.contains("doublon exact").sum() == 3


def test_quality_gate_fails_on_data_leak(sample_env):
    """Échec contrôlé : un même groupe est réparti sur train et val -> le pipeline doit s'arrêter."""
    P = sample_env
    run_all()
    m = pd.read_parquet(P.processed / "manifest.parquet")
    hm = m[(m.source == "hateful_memes") & m.hate_binary.notna()]
    tr, va = hm[hm.split == "train"].index[0], hm[hm.split == "val"].index[0]
    m.loc[va, "group_id"] = m.loc[tr, "group_id"]  # même groupe dans train ET val = fuite
    m.to_parquet(P.processed / "manifest.parquet", index=False)
    with pytest.raises(SystemExit) as e:
        quality_gate.main()
    assert e.value.code == 1
