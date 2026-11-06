"""Contrôle qualité automatique : le pipeline s'arrête (code de sortie 1) si un contrôle CRITIQUE échoue.

Contrôles : schéma et types, valeurs manquantes, doublons, taux de rejet, dimensions des images,
cohérence inter-modalités (synchronisation image-texte dans les deux sens), absence de fuite
train / val / test, cohérence des labels.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.common.config import get_paths, load_config
from src.preprocessing.text import norm_key
from src.validation.labels import LABEL_MAP
from src.validation.schemas import MANIFEST_SCHEMA, SchemaErrors

REQUIRED = ["sample_id", "source", "text", "text_clean", "lang", "label", "split", "collected_at"]


def run_checks(manifest, cfg, base, rejects, counters, hm_img_names=None):
    q = cfg["quality"]
    tests = []

    def check(name, ok, detail="", critical=True):
        tests.append(dict(test=name, résultat="PASS" if ok else ("FAIL" if critical else "WARN"), détail=str(detail)))

    try:
        MANIFEST_SCHEMA.validate(manifest, lazy=True)
        check("schéma et types du manifeste (Pandera)", True)
    except SchemaErrors as e:
        check("schéma et types du manifeste (Pandera)", False, f"{len(e.failure_cases)} violations")

    check("sample_id unique", manifest.sample_id.is_unique)
    miss = manifest[REQUIRED].isna().mean().max()
    check("taux de valeurs manquantes (colonnes obligatoires)", miss <= q["max_missing_rate"], f"max = {miss:.4f}")

    multi = manifest[manifest.source.isin(["hateful_memes", "reddit"])]
    check("sources multimodales : img_path, img_hash et texte non vides",
          multi.img_path.notna().all() and multi.img_hash.notna().all() and multi.text_clean.str.strip().ne("").all())
    check("Sentiment140 : texte seul (pas d'image)", manifest[manifest.source == "sentiment140"].img_path.isna().all())
    small = int(((multi.width < q["expected_image_min_side"]) | (multi.height < q["expected_image_min_side"])).sum())
    check("dimensions des images >= seuil", small == 0, f"{small} images trop petites")

    img = manifest[manifest.img_sha256.notna()].assign(k=lambda d: d.text_clean.map(norm_key))
    dup_rate = float(img.duplicated(["img_sha256", "k"]).mean()) if len(img) else 0.0
    check("taux de doublons exacts (image + texte)", dup_rate <= q["max_duplicate_rate"], f"{dup_rate:.4f}")

    for src, g in counters[counters.kind.isin(["raw", "rejected"])].groupby("source"):
        raw, rej = g[g.kind == "raw"].n.sum(), g[g.kind == "rejected"].n.sum()
        if raw:
            check(f"taux de rejet {src} <= seuil", rej / raw <= q["max_reject_rate"], f"{rej}/{raw} = {rej / raw:.4f}")

    missing = [p for p in multi.img_path if not (base / p).exists()]
    check("sens 1 : chaque img_path existe sur disque", not missing, f"{len(missing)} manquants")
    if hm_img_names is not None:
        in_manifest = {Path(p).name for p in multi[multi.source == "hateful_memes"].img_path}
        logged = set(rejects[rejects.source == "hateful_memes"].detail.astype(str))
        unexplained = (hm_img_names - in_manifest) - logged
        check("sens 2 : chaque image du dossier HM figure dans le manifeste OU dans le journal des rejets",
              not unexplained, f"{len(unexplained)} non expliquées")

    sup = manifest[manifest.hate_binary.notna()]
    check("aucun groupe réparti sur plusieurs ensembles (pas de fuite)", sup.groupby("group_id").split.nunique().max() == 1)
    prev = sup.groupby("split").hate_binary.mean()
    check(f"prévalence 'hate' homogène entre ensembles (écart < {q['max_prevalence_gap']})",
          prev.max() - prev.min() < q["max_prevalence_gap"], prev.round(3).to_dict(), critical=False)

    ref = LABEL_MAP.assign(label_raw=lambda d: pd.to_numeric(d.label_raw, errors="coerce").astype("Int64"))
    chk = manifest.merge(ref, on=["source", "label_raw"], how="left", suffixes=("", "_ref"))
    lab_rows = chk[chk.label != "unlabeled"]
    check("labels conformes à la table de correspondance", (lab_rows.label == lab_rows.label_ref).all())
    check("hate_binary renseigné uniquement pour la détection de haine labellisée",
          manifest[manifest.label_task != "hate_detection"].hate_binary.isna().all())
    return pd.DataFrame(tests)


def main():
    cfg = load_config()
    P = get_paths(cfg)
    manifest = pd.read_parquet(P.processed / "manifest.parquet")
    rejects, counters = pd.read_csv(P.logs / "rejects.csv"), pd.read_csv(P.logs / "counters.csv")
    hm_names = None
    train = next((P.raw / "hateful_memes").rglob("train.jsonl"), None)
    if train is not None and (train.parent / "img").exists():
        hm_names = {p.name for p in (train.parent / "img").iterdir() if p.is_file()}
    tests = run_checks(manifest, cfg, P.base, rejects, counters, hm_names)
    P.reports.mkdir(parents=True, exist_ok=True)
    tests.to_csv(P.reports / "quality_tests.csv", index=False)
    n_fail = int((tests.résultat == "FAIL").sum())
    (P.reports / "quality.json").write_text(json.dumps(dict(
        n_tests=len(tests), n_pass=int((tests.résultat == "PASS").sum()), n_fail=n_fail,
        n_warn=int((tests.résultat == "WARN").sum()), manifest_rows=len(manifest)), indent=2), encoding="utf-8")
    print(tests.to_string(index=False))
    if n_fail:
        print(f"\nÉCHEC DU CONTRÔLE QUALITÉ : {n_fail} test(s) critique(s) en échec -> pipeline arrêté.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
