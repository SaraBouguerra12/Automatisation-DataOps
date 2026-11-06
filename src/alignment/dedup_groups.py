"""Alignement : doublons exacts (image + texte) et groupes anti-fuite (images quasi identiques ou textes identiques).

Entrée : data/interim/clean.parquet. Sortie : data/interim/aligned.parquet (colonne group_id).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import get_paths, load_config
from src.common.runlog import RunLog


class UF:
    """Union-Find (composantes connectées)."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def hamming_roots(hashes_hex, max_dist):
    """Racine de groupe pour chaque image : pHash 64 bits, distance de Hamming <= max_dist."""
    H = np.array([int(h, 16) for h in hashes_hex], dtype=np.uint64)
    uf = UF(len(H))
    for i in range(len(H) - 1):
        x = np.ascontiguousarray(H[i] ^ H[i + 1:])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.flatnonzero(d <= max_dist):
            uf.union(i, i + 1 + int(j))
    return [uf.find(i) for i in range(len(H))]


def make_groups(df, phash_max_dist, min_text_chars):
    """df indexé 0..n-1 : groupes = images quasi identiques OU textes identiques (assez longs)."""
    n = len(df)
    uf = UF(n)
    first = {}
    for i, root in enumerate(hamming_roots(df.img_hash.tolist(), phash_max_dist)):
        if root in first:
            uf.union(first[root], i)
        else:
            first[root] = i
    long_txt = df.text_norm[df.text_norm.str.len() >= min_text_chars]
    for _, idx in long_txt.groupby(long_txt).groups.items():
        idx = list(idx)
        for j in idx[1:]:
            uf.union(idx[0], j)
    codes = pd.factorize(pd.Series([uf.find(i) for i in range(n)]))[0]
    return pd.Series(["g" + str(c).zfill(5) for c in codes], index=df.index)


def main():
    cfg = load_config()
    P = get_paths(cfg)
    acfg = cfg["alignment"]
    rl = RunLog("align", P.logs)
    df = pd.read_parquet(P.interim / "clean.parquet")
    mm = df[df.source != "sentiment140"].reset_index(drop=True)
    s_f = df[df.source == "sentiment140"].copy()

    if len(mm):
        dup = mm.duplicated(subset=["img_sha256", "text_norm"], keep="first")
        for r in mm[dup].itertuples():
            rl.reject(r.source, r.sample_id, "doublon exact (image + texte)", Path(r.img_path).name)
        mm = mm[~dup].reset_index(drop=True)
        mm["group_id"] = make_groups(mm, acfg["phash_max_dist"], acfg["min_text_group_chars"])
        sizes = mm.group_id.value_counts()
        rl.event(step="groups", n_groups=len(sizes), multi_sample_groups=int((sizes > 1).sum()), largest=int(sizes.max()))
        if sizes.max() > 0.01 * len(mm):
            rl.event(step="groups", warning="un groupe dépasse 1 % des données : vérifier alignment.phash_max_dist")
    s_f["group_id"] = s_f.sample_id  # texte seul : un tweet = un groupe
    out = pd.concat([mm, s_f], ignore_index=True)
    out.to_parquet(P.interim / "aligned.parquet", index=False)
    rl.save()


if __name__ == "__main__":
    main()
