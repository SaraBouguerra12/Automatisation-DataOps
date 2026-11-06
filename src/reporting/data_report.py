"""Statistiques et rapport de qualité HTML (observabilité). Sorties dans reports/."""
import html

import pandas as pd

from src.common.config import get_paths, load_config
from src.validation.labels import LABEL_MAP


def main():
    cfg = load_config()
    P = get_paths(cfg)
    P.reports.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(P.processed / "manifest.parquet")
    counters = pd.read_csv(P.logs / "counters.csv").fillna("")
    tests = pd.read_csv(P.reports / "quality_tests.csv") if (P.reports / "quality_tests.csv").exists() else pd.DataFrame()

    stats = m.groupby(["source", "split", "label"]).size().rename("n").reset_index()
    stats.to_csv(P.reports / "stats_summary.csv", index=False)
    LABEL_MAP.to_csv(P.reports / "label_mapping.csv", index=False)
    langs = (m.groupby("source").lang.value_counts(normalize=True).rename("part").reset_index()
             .groupby("source").head(8).round(3))
    blocks = [("Volumes par source, ensemble et label", stats), ("Compteurs du pipeline", counters),
              ("Langues principales par source", langs), ("Tests de qualité", tests)]
    body = "".join(f"<h2>{html.escape(t)}</h2>{d.to_html(index=False, border=0)}" for t, d in blocks if len(d))
    (P.reports / "data_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Rapport de qualité des données</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}"
        "td,th{padding:4px 10px;border-bottom:1px solid #ddd}</style>"
        f"<h1>Rapport de qualité des données</h1><p>{len(m)} échantillons dans le manifeste.</p>{body}", encoding="utf-8")
    print("Rapport :", P.reports / "data_report.html")


if __name__ == "__main__":
    main()
