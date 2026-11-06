"""Génère un petit jeu de données SYNTHÉTIQUE au format des sources réelles (CI, tests, démo sans Kaggle).

Usage : DATA_BASE=data_sample python -m src.common.sample_data
"""
import json

import numpy as np
from PIL import Image

from src.common.config import get_paths, load_config

WORDS = ("people city music green future market sunday coffee never happy world story small night "
         "bridge garden silver moment simple travel window summer answer reason letter pocket").split()


def _sentence(rng, n=7):
    return " ".join(rng.choice(WORDS, n))


def make_sample_raw(P, n_hm=300, n_tweets=600, seed=0):
    rng = np.random.default_rng(seed)
    root = P.raw / "hateful_memes" / "sample" / "extracted" / "data"
    (root / "img").mkdir(parents=True, exist_ok=True)
    splits = {"train": range(0, 220), "dev": range(220, 260), "test": range(260, n_hm)}
    texts = {i: _sentence(rng) for i in range(n_hm)}
    for i in range(n_hm):
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / "img" / f"{i:05d}.png")
    # 3 doublons exacts (même image, même texte) et 1 image corrompue
    for k, (src, dst) in enumerate([(0, 10), (1, 11), (2, 12)]):
        (root / "img" / f"{dst:05d}.png").write_bytes((root / "img" / f"{src:05d}.png").read_bytes())
        texts[dst] = texts[src]
    for name, ids in splits.items():
        with open(root / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                rec = dict(id=i, img=f"img/{i:05d}.png", text=texts[i])
                if name != "test":
                    rec["label"] = int(rng.random() < 0.37)
                f.write(json.dumps(rec) + "\n")
    s = P.raw / "sentiment140" / "sample" / "extracted"
    s.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_tweets):
        rows.append((int(rng.choice([0, 4])), 1000 + i, "Sat May 16 23:58:44 UTC 2009", "NO_QUERY", f"user{i}",
                     _sentence(rng, 9) + f" {i}"))
    rows.append(rows[5])  # un doublon (retweet)
    import csv
    with open(s / "training.1600000.processed.noemoticon.csv", "w", encoding="latin-1", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    P = get_paths(load_config())
    make_sample_raw(P)
    print("Échantillon synthétique écrit dans", P.raw)


if __name__ == "__main__":
    main()
