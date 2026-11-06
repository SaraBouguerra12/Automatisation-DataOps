"""Manifeste unifié (Parquet) + consolidation des journaux (rejets, compteurs) + fiche de run.

Entrée : data/interim/split.parquet et data/logs/*. Sorties : data/processed/manifest.parquet,
data/logs/rejects.csv, data/logs/counters.csv, data/logs/run_summary.json.
"""
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

from src.common.config import get_paths, load_config
from src.common.io_utils import sha256_file
from src.validation.schemas import MANIFEST_SCHEMA

COLUMNS = ["sample_id", "source", "text", "text_clean", "img_path", "img_hash", "img_sha256", "width", "height",
           "lang", "lang_conf", "label", "label_raw", "hate_binary", "label_task", "label_source",
           "orig_split", "split", "group_id", "collected_at"]
LOG_STAGES = ("preprocess", "align")


def _pkg(n):
    try:
        return version(n)
    except PackageNotFoundError:
        return "n/a"


def main():
    cfg = load_config()
    P = get_paths(cfg)
    manifest = pd.read_parquet(P.interim / "split.parquet")[COLUMNS]
    for c in ("width", "height", "label_raw", "hate_binary"):
        manifest[c] = manifest[c].astype("Int64")
    manifest = MANIFEST_SCHEMA.validate(manifest, lazy=True)
    P.processed.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(P.processed / "manifest.parquet", index=False)

    rejects = [pd.read_csv(P.logs / f"{s}_rejects.csv") for s in LOG_STAGES if (P.logs / f"{s}_rejects.csv").exists()]
    pd.concat(rejects, ignore_index=True).to_csv(P.logs / "rejects.csv", index=False)
    counters = [r for s in LOG_STAGES if (P.logs / f"{s}.json").exists()
                for r in json.loads((P.logs / f"{s}.json").read_text(encoding="utf-8"))["counters"]]
    counters += [dict(source=src, kind="in_manifest", reason="", n=len(g)) for src, g in manifest.groupby("source")]
    pd.DataFrame(counters).to_csv(P.logs / "counters.csv", index=False)

    summary = dict(date=datetime.now(timezone.utc).isoformat(), config=cfg, n_rows_manifest=len(manifest),
                   manifest_sha256=sha256_file(P.processed / "manifest.parquet"), counters=counters,
                   packages={p: _pkg(p) for p in ["pandas", "pandera", "imagehash", "langdetect", "pyarrow", "scikit-learn"]})
    (P.logs / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"manifest.parquet : {len(manifest)} lignes | sha256 {summary['manifest_sha256'][:16]}…")


if __name__ == "__main__":
    main()
