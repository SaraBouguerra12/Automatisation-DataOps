"""Prétraitement : validation de schéma, synchronisation image-texte, nettoyage du texte, langue.

Entrée : data/raw/. Sortie : data/interim/clean.parquet (schéma unifié, avant doublons / groupes / découpage).
"""
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import get_paths, load_config
from src.common.io_utils import rel
from src.common.runlog import RunLog
from src.preprocessing.images import inspect_image
from src.preprocessing.text import clean_text, detect_lang, norm_key, set_seed
from src.validation.schemas import HM_SCHEMA, S140_SCHEMA, validate_schema

warnings.filterwarnings("ignore", category=FutureWarning)


def collected_at(source, P):
    reg = P.raw / source / "registry.json"
    if reg.exists():
        return json.loads(reg.read_text(encoding="utf-8")).get("downloaded_at")
    return datetime.now(timezone.utc).isoformat()


def add_lang(df, col="text_clean"):
    langs = [detect_lang(t) for t in df[col]]
    df["lang"], df["lang_conf"] = [lang for lang, _ in langs], [c for _, c in langs]
    return df


def process_hateful_memes(cfg, P, rl):
    pc = cfg["preprocessing"]
    root = next((P.raw / "hateful_memes").rglob("train.jsonl")).parent
    parts = []
    for split in ("train", "dev", "test"):
        d = pd.read_json(root / f"{split}.jsonl", lines=True)
        if "label" not in d.columns:
            d["label"] = np.nan  # test.jsonl public : sans labels
        d["orig_split"] = split
        parts.append(d)
    hm_raw = pd.concat(parts, ignore_index=True)
    rl.raw("hateful_memes", len(hm_raw))
    hm = validate_schema(hm_raw, HM_SCHEMA, "hateful_memes", "id", rl).copy()
    hm["sample_id"] = "hm_" + hm["id"].astype(int).astype(str).str.zfill(5)  # clé unique image-texte
    hm["img_rel"] = hm["img"].map(lambda p: rel(root / p, P.base))
    hm["text_raw"] = hm["text"].fillna("").astype(str)

    paths = [P.base / p for p in hm.img_rel]
    with ThreadPoolExecutor(pc["n_workers"]) as ex:
        infos = list(ex.map(lambda p: inspect_image(p, pc["min_img_side"]), paths))
    hm = pd.concat([hm.reset_index(drop=True), pd.DataFrame(infos).add_prefix("i_")], axis=1)
    bad = hm[~hm.i_ok]
    for r in bad.itertuples():
        rl.reject("hateful_memes", r.sample_id, r.i_reason, Path(r.img).name)
    hm = hm[hm.i_ok].copy()

    hm["text_clean"] = hm.text_raw.map(clean_text)
    empty = hm.text_clean.eq("")
    for r in hm[empty].itertuples():
        rl.reject("hateful_memes", r.sample_id, "texte vide", Path(r.img).name)
    hm = hm[~empty].copy()
    hm = add_lang(hm)

    # Synchronisation dans les DEUX sens : fichiers du dossier sans ligne -> rejet journalisé (orphelins)
    img_dir = (root / hm_raw.img.iloc[0]).parent
    on_disk = {p.name for p in img_dir.iterdir() if p.is_file()}
    orphans = sorted(on_disk - {Path(p).name for p in hm_raw.img})
    for f in orphans:
        rl.reject("hateful_memes", f"file:{f}", "fichier image orphelin (non référencé)", f)
    rl.event(step="hateful_memes", raw=len(hm_raw), images_rejected=len(bad), empty_texts=int(empty.sum()),
             orphans=len(orphans), kept=len(hm))

    f = pd.DataFrame(dict(
        sample_id=hm.sample_id, source="hateful_memes", text=hm.text_raw, text_clean=hm.text_clean, img_path=hm.img_rel,
        img_hash=hm.i_phash, img_sha256=hm.i_sha256, width=hm.i_w, height=hm.i_h, lang=hm.lang, lang_conf=hm.lang_conf,
        label_raw=hm.label, orig_split=hm.orig_split, collected_at=collected_at("hateful_memes", P)))
    f["label"] = f.label_raw.map({0.0: "non_hate", 1.0: "hate"}).fillna("unlabeled")
    f["hate_binary"] = f.label_raw.map({0.0: 0, 1.0: 1}).astype("Int64")
    f["label_raw"] = f.label_raw.astype("Int64")
    f["label_task"] = "hate_detection"
    f["label_source"] = np.where(f.hate_binary.notna(), "annotateur", "non_labellise")
    f["text_norm"] = f.text_clean.map(norm_key)
    return f


def process_reddit(cfg, P, rl):
    files = sorted((P.raw / "reddit").glob("*/reddit_posts.jsonl"))
    rows = [json.loads(line) for fp in files for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    rd = pd.DataFrame(rows)
    rd["text_clean"] = rd.text_raw.map(clean_text)
    empty = rd.text_clean.eq("")
    for r in rd[empty].itertuples():
        rl.reject("reddit", r.sample_id, "texte vide")
    rd = add_lang(rd[~empty].copy())
    f = pd.DataFrame(dict(
        sample_id=rd.sample_id, source="reddit", text=rd.text_raw, text_clean=rd.text_clean, img_path=rd.img_rel,
        img_hash=rd.phash, img_sha256=rd.sha256, width=rd.w, height=rd.h, lang=rd.lang, lang_conf=rd.lang_conf,
        orig_split="reddit", collected_at=rd.collected_at))
    f["label_raw"] = pd.array([pd.NA] * len(f), dtype="Int64")
    f["label"], f["hate_binary"] = "unlabeled", pd.array([pd.NA] * len(f), dtype="Int64")
    f["label_task"], f["label_source"] = "none", "non_labellise"
    f["text_norm"] = f.text_clean.map(norm_key)
    rl.raw("reddit", len(rows))
    return f


def process_sentiment140(cfg, P, rl):
    seed = cfg["seed"]
    csv = next((P.raw / "sentiment140").rglob("*.csv"))
    s_raw = pd.read_csv(csv, encoding="latin-1", header=None, names=["target", "ids", "date", "flag", "user", "text"])
    rl.raw("sentiment140", len(s_raw))
    s = validate_schema(s_raw[["target", "ids", "text"]], S140_SCHEMA, "sentiment140", "ids", rl).copy()  # 'user' abandonné

    d = s.duplicated("ids", keep="first")
    rl.reject_bulk("sentiment140", s.loc[d, "ids"], "id dupliqué")
    s = s[~d]
    d = s.duplicated("text", keep="first")
    rl.reject_bulk("sentiment140", s.loc[d, "ids"], "texte brut dupliqué (retweets, spam)")
    s = s[~d]

    n_per = min(cfg["preprocessing"]["n_s140"] // 2, int(s.target.value_counts().min()))
    n_before = len(s)
    s = s.groupby("target", group_keys=False).sample(n=n_per, random_state=seed)  # échantillon équilibré
    rl.ok("sentiment140", n_before - len(s), kind="not_sampled")

    s["text_clean"] = s.text.map(clean_text)
    s["text_norm"] = s.text_clean.map(norm_key)
    e = s.text_norm.eq("")
    rl.reject_bulk("sentiment140", s.loc[e, "ids"], "texte vide après nettoyage")
    s = s[~e]
    d = s.duplicated("text_norm", keep="first")
    rl.reject_bulk("sentiment140", s.loc[d, "ids"], "doublon après nettoyage")
    s = add_lang(s[~d].copy())

    f = pd.DataFrame(dict(
        sample_id="s140_" + s.ids.astype(str), source="sentiment140", text=s.text, text_clean=s.text_clean,
        img_path=None, img_hash=None, img_sha256=None, width=pd.array([pd.NA] * len(s), dtype="Int64"),
        height=pd.array([pd.NA] * len(s), dtype="Int64"), lang=s.lang, lang_conf=s.lang_conf,
        label_raw=s.target.astype("Int64"), orig_split="train", collected_at=collected_at("sentiment140", P)))
    f["label"] = s.target.map({0: "negative", 4: "positive"})
    f["hate_binary"] = pd.array([pd.NA] * len(f), dtype="Int64")
    f["label_task"], f["label_source"] = "sentiment", "auto (emoticônes)"
    f["text_norm"] = s.text_norm.values
    return f


def main():
    cfg = load_config()
    P = get_paths(cfg)
    set_seed(cfg["seed"])
    rl = RunLog("preprocess", P.logs)
    frames = [process_hateful_memes(cfg, P, rl)]
    rd = process_reddit(cfg, P, rl)
    if rd is not None:
        frames.append(rd)
    frames.append(process_sentiment140(cfg, P, rl))
    out = pd.concat(frames, ignore_index=True)
    P.interim.mkdir(parents=True, exist_ok=True)
    out.to_parquet(P.interim / "clean.parquet", index=False)
    rl.event(step="preprocess", rows=len(out), by_source=out.source.value_counts().to_dict())
    rl.save()


if __name__ == "__main__":
    main()
