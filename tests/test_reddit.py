"""Logique de rejet du connecteur Reddit sur des posts SIMULÉS (aucun accès réseau)."""
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.common.config import load_config
from src.common.runlog import RunLog
from src.ingestion.reddit_source import process_reddit_post


def _png(side=200, seed=0):
    arr = np.random.default_rng(seed).integers(0, 255, (side, side, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def test_reddit_rejection_reasons(tmp_path):
    cfg = load_config()
    blobs = {"https://i.redd.it/a.png": _png(seed=1), "https://i.redd.it/dup.png": _png(seed=1),
             "https://i.redd.it/bad.png": b"pas une image", "https://i.redd.it/tiny.png": _png(10, 5)}

    def fake_fetch(url, dest):
        if url not in blobs:
            raise ConnectionError("404 simulé")
        Path(dest).write_bytes(blobs[url])

    def mk(i, url, title="un titre", **kw):
        d = dict(id=i, url=url, title=title, selftext="", over_18=False, stickied=False, is_video=False,
                 subreddit="test", score=1, created_utc=0)
        d.update(kw)
        return SimpleNamespace(**d)

    posts = [mk("p1", "https://i.redd.it/a.png"), mk("p2", "https://i.redd.it/dup.png"),
             mk("p3", "https://v.redd.it/x", is_video=True), mk("p4", "https://reddit.com/r/x/comments/1"),
             mk("p5", "https://i.redd.it/a.png", title=""), mk("p6", "https://i.redd.it/missing.png"),
             mk("p7", "https://i.redd.it/bad.png"), mk("p8", "https://i.redd.it/tiny.png"),
             mk("p9", "https://i.redd.it/a.png", over_18=True)]
    rl, seen_ids, seen_ph, accepted = RunLog("t", tmp_path / "logs"), set(), set(), []
    for p in posts:
        rec = process_reddit_post(p, seen_ids, seen_ph, tmp_path, tmp_path, cfg, rl, fetch=fake_fetch)
        if rec:
            accepted.append(rec["sample_id"])
            seen_ids.add(p.id)
            seen_ph.add(rec["phash"])
    reasons = rl.rejects_df().set_index("sample_id").reason.to_dict()
    assert accepted == ["rd_p1"]
    assert reasons["rd_p2"].startswith("image déjà collectée") and reasons["rd_p3"].startswith("url non image")
    assert reasons["rd_p5"] == "texte vide" and reasons["rd_p6"] == "échec téléchargement"
    assert reasons["rd_p7"] == "image illisible" and reasons["rd_p8"] == "image trop petite" and "NSFW" in reasons["rd_p9"]
