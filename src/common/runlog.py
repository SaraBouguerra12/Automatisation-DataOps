"""Journal d'exécution : événements structurés, compteurs collectés / rejetés (avec raison), table des rejets."""
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REJECT_COLS = ["source", "sample_id", "reason", "detail", "ts"]


def _now():
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, stage, logs_dir):
        self.stage, self.logs_dir = stage, Path(logs_dir)
        self.rejects, self.counters, self.events = [], Counter(), []
        self.log = logging.getLogger(f"pipeline.{stage}.{id(self)}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        if not self.log.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [" + stage + "] %(message)s"))
            self.log.addHandler(h)

    def raw(self, source, n):
        self.counters[f"{source}|raw|"] += int(n)

    def ok(self, source, n=1, kind="collected"):
        self.counters[f"{source}|{kind}|"] += int(n)

    def reject(self, source, sample_id, reason, detail=""):
        self.rejects.append(dict(source=source, sample_id=str(sample_id), reason=reason, detail=str(detail)[:200], ts=_now()))
        self.counters[f"{source}|rejected|{reason}"] += 1
        self.log.warning("REJECT %s %s %s", source, sample_id, reason)

    def reject_bulk(self, source, ids, reason, max_examples=500):
        ids = list(ids)
        self.counters[f"{source}|rejected|{reason}"] += len(ids)
        for i in ids[:max_examples]:
            self.rejects.append(dict(source=source, sample_id=str(i), reason=reason, detail="(exemple)", ts=_now()))
        self.log.warning("REJECT_BULK %s %s n=%d", source, reason, len(ids))

    def event(self, **kw):
        kw["ts"] = _now()
        self.events.append(kw)
        self.log.info(json.dumps(kw, default=str, ensure_ascii=False))

    def rejects_df(self):
        return pd.DataFrame(self.rejects, columns=REJECT_COLS)

    def counters_records(self):
        rows = []
        for k, v in self.counters.items():
            s, kind, reason = k.split("|")
            rows.append(dict(source=s, kind=kind, reason=reason, n=v))
        return rows

    def save(self):
        """Écrit logs/<stage>.json et logs/<stage>_rejects.csv (relus par l'étape build_manifest)."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / f"{self.stage}.json").write_text(
            json.dumps(dict(stage=self.stage, counters=self.counters_records(), events=self.events),
                       indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        self.rejects_df().to_csv(self.logs_dir / f"{self.stage}_rejects.csv", index=False)
