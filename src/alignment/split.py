"""Découpage train / val / test stratifié PAR GROUPE (aucune fuite entre ensembles).

Entrée : data/interim/aligned.parquet. Sortie : data/interim/split.parquet (colonne split).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.common.config import get_paths, load_config


def assign_splits(df, cfg):
    seed, scfg = cfg["seed"], cfg["split"]
    df = df.copy()
    df["split"] = "unlabeled"
    lab = df[df.hate_binary.notna()]
    if len(lab):
        n_splits = max(3, int(round(1 / min(scfg["val_size"], scfg["test_size"]))))
        n_test, n_val = round(scfg["test_size"] * n_splits), round(scfg["val_size"] * n_splits)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [te for _, te in sgkf.split(lab, lab.hate_binary.astype(int), groups=lab[scfg["group_col"]])]
        sp = pd.Series("train", index=lab.index)
        for k in range(n_test):
            sp.iloc[folds[k]] = "test"
        for k in range(n_test, n_test + n_val):
            sp.iloc[folds[k]] = "val"
        df.loc[lab.index, "split"] = sp
    s = df[df.source == "sentiment140"]
    if len(s):
        _, va = train_test_split(np.arange(len(s)), test_size=scfg["s140_val_size"], stratify=s.label, random_state=seed)
        sp = pd.Series("train", index=s.index)
        sp.iloc[va] = "val"
        df.loc[s.index, "split"] = sp
    return df


def main():
    cfg = load_config()
    P = get_paths(cfg)
    df = assign_splits(pd.read_parquet(P.interim / "aligned.parquet"), cfg)
    df.to_parquet(P.interim / "split.parquet", index=False)
    print(df[df.hate_binary.notna()].groupby("split").agg(n=("sample_id", "size"), pct_hate=("hate_binary", "mean")).round(3))


if __name__ == "__main__":
    main()
