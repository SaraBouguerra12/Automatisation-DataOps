# Modération de contenu multimodale (image + texte) — Pipeline DataOps

Pipeline reproductible de collecte et de préparation des données (Hateful Memes, Sentiment140, Reddit optionnel). Projet du module Deep Learning, Tâche 3.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows ; Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

## Récupérer les données (DVC)
Les données ne sont pas dans Git : elles sont versionnées avec DVC (remote `storage`).
1. Cloner le dépôt et installer les dépendances.
2. Indiquer le dossier de stockage (propre à chaque machine, non versionné) :
   `dvc remote modify --local storage url CHEMIN_DU_DOSSIER_PARTAGE`
3. Récupérer les données : `dvc pull`
4. Vérifier : `dvc status` doit répondre « Data and pipelines are up to date ».

## Exécuter le pipeline
```bash
dvc repro      # pipeline réel (la première fois : token Kaggle requis)
make sample    # démo complète sur données synthétiques, sans Kaggle
make test      # tests unitaires, bout en bout et échec contrôlé du contrôle qualité
```
Token Kaggle (Settings → API → Generate New Token) : `$env:KAGGLE_API_TOKEN="..."` sous Windows, `export KAGGLE_API_TOKEN=...` sous Linux/macOS.

## Configuration
Tous les chemins et seuils sont dans `configs/params.yaml`. Les variables `DATA_BASE` et `REPORTS_DIR` permettent de rediriger les sorties.

## Étapes du pipeline (dvc.yaml)
`ingest_*` → `preprocess` → `align` → `split` → `build_manifest` → `quality_gate` → `report`.
Le `quality_gate` termine avec le code 1 si un contrôle critique échoue : le pipeline s'arrête.

## Partie B : module de fusion (prototype)
```bash
pip install -r requirements-fusion.txt
pytest -q tests/test_fusion.py
python -m src.fusion.demo_forward    # passe avant sur un batch réel (télécharge CLIP, environ 600 Mo)
```
Fiche de données : `docs/DATA_CARD.md`. Suivi d'expériences : MLflow (`src/fusion/tracking.py`).
