"""Collecte Reddit (algorithme B) : PRAW (OAuth2), rejets journalisés, retry + backoff, pHash anti-repost.

Optionnel : nécessite REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USERNAME et ingestion.reddit.enabled: true.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.common.config import get_paths, load_config
from src.common.io_utils import rel
from src.common.runlog import RunLog
from src.preprocessing.images import inspect_image

UA_BASE = "script:moderation-multimodale:v1.0"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def is_image_url(url):
    return str(url).lower().split("?")[0].endswith(IMG_EXT)


def is_video(post):
    return bool(getattr(post, "is_video", False)) or "v.redd.it" in str(post.url) or getattr(post, "post_hint", "") == "hosted:video"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)), reraise=True)
def download_image(url, dest, max_bytes=15_000_000, timeout=15):
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": UA_BASE}) as r:
        r.raise_for_status()
        if not r.headers.get("Content-Type", "").startswith("image/"):
            raise ValueError(f"content-type inattendu : {r.headers.get('Content-Type')}")
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("image trop volumineuse")
                f.write(chunk)


def process_reddit_post(post, seen_ids, seen_phash, out_dir, base, cfg, rl, fetch=download_image):
    """Traite UN post. Retourne un enregistrement, ou None (rejet journalisé avec sa raison)."""
    sid = f"rd_{post.id}"
    rcfg, pcfg = cfg["ingestion"]["reddit"], cfg["preprocessing"]
    if rcfg["exclude_nsfw"] and getattr(post, "over_18", False):
        rl.reject("reddit", sid, "contenu NSFW (over_18)")
        return None
    if getattr(post, "stickied", False):
        rl.reject("reddit", sid, "post épinglé")
        return None
    if is_video(post) or getattr(post, "is_gallery", False) or not is_image_url(post.url):
        rl.reject("reddit", sid, "url non image ou vidéo", post.url)
        return None
    if post.id in seen_ids:
        return None
    text = (post.title or "").strip() + (("\n" + post.selftext.strip()) if getattr(post, "selftext", "") else "")
    if not text.strip():
        rl.reject("reddit", sid, "texte vide")
        return None
    dest = out_dir / f"{post.id}{Path(post.url.split('?')[0]).suffix.lower()}"
    try:
        fetch(post.url, dest)
    except Exception as e:
        Path(dest).unlink(missing_ok=True)
        rl.reject("reddit", sid, "échec téléchargement", repr(e))
        return None
    info = inspect_image(dest, pcfg["min_img_side"])
    if not info["ok"]:
        dest.unlink(missing_ok=True)
        rl.reject("reddit", sid, info["reason"], info["detail"])
        return None
    if info["phash"] in seen_phash:
        dest.unlink(missing_ok=True)
        rl.reject("reddit", sid, "image déjà collectée (repost)", info["phash"])
        return None
    return dict(sample_id=sid, text_raw=text, title=post.title, url=post.url, permalink=getattr(post, "permalink", ""),
                subreddit=str(getattr(post, "subreddit", "")), score=int(getattr(post, "score", 0)),
                created_utc=float(getattr(post, "created_utc", 0)), img_rel=rel(dest, base), sha256=info["sha256"],
                phash=info["phash"], w=info["w"], h=info["h"], collected_at=datetime.now(timezone.utc).isoformat())


def collect_reddit(reddit, cfg, P, rl):
    rcfg = cfg["ingestion"]["reddit"]
    out = P.raw / "reddit" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (out / "img").mkdir(parents=True, exist_ok=True)
    seen_ids, seen_ph = set(), set()
    with open(out / "reddit_posts.jsonl", "a", encoding="utf-8") as f:
        for sub in rcfg["subreddits"]:
            for post in reddit.subreddit(sub).hot(limit=rcfg["limit"]):
                rec = process_reddit_post(post, seen_ids, seen_ph, out / "img", P.base, cfg, rl)
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen_ids.add(post.id)
                    seen_ph.add(rec["phash"])
                    rl.ok("reddit")
                time.sleep(0.3)  # politesse envers l'API (quota)


def main():
    cfg = load_config()
    P = get_paths(cfg)
    rl = RunLog("ingest_reddit", P.logs)
    if not cfg["ingestion"]["reddit"]["enabled"]:
        rl.event(step="reddit", action="ignoré (ingestion.reddit.enabled = false)")
    else:
        import praw
        reddit = praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"], client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                             user_agent=f"{UA_BASE} (by /u/{os.environ.get('REDDIT_USERNAME', 'anonymous')})",
                             check_for_async=False)
        reddit.read_only = True
        collect_reddit(reddit, cfg, P, rl)
    rl.save()


if __name__ == "__main__":
    main()
