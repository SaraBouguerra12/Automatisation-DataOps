"""Collecte des sources statiques (algorithme A) : téléchargement Kaggle, SHA-256, zone raw datée.

Usage : python -m src.ingestion.kaggle_source <hateful_memes|sentiment140>
"""
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

from src.common.config import get_paths, load_config
from src.common.io_utils import sha256_file
from src.common.runlog import RunLog


def ingest_kaggle(source, slug, P, rl):
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest = P.raw / source / run_date
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)], check=True)
    archive = next(dest.glob("*.zip"))
    sha = sha256_file(archive)
    extracted = dest / "extracted"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    entry = dict(slug=slug, date=run_date, archive=archive.relative_to(P.base).as_posix(), sha256=sha,
                 extracted=extracted.relative_to(P.base).as_posix(), downloaded_at=datetime.now(timezone.utc).isoformat())
    (P.raw / source / "registry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    rl.event(step="kaggle", source=source, action="download", slug=slug, sha256=sha, date=run_date)
    return entry


def main(source=None):
    source = source or sys.argv[1]
    cfg = load_config()
    P = get_paths(cfg)
    rl = RunLog(f"ingest_{source}", P.logs)
    ingest_kaggle(source, cfg["ingestion"]["kaggle"][source], P, rl)
    rl.save()


if __name__ == "__main__":
    main()
