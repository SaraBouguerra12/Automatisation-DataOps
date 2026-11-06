# Modération de contenu multimodale (image + texte) — Pipeline DataOps

Pipeline reproductible de collecte et de préparation des données (Hateful Memes, Sentiment140, Reddit optionnel).

## Installation
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécuter le pipeline
```bash
make sample      # démo complète sur données SYNTHÉTIQUES (sans Kaggle) -> data_sample/, reports_sample/
make test        # tests unitaires + tests de bout en bout + échec contrôlé du contrôle qualité
make pipeline    # = dvc repro : pipeline réel (nécessite le token Kaggle)
```
Token Kaggle (Settings -> API -> Generate New Token) : `export KAGGLE_API_TOKEN=...` (Windows : `$env:KAGGLE_API_TOKEN="..."`).

## Récupérer les données
*(à compléter à l'étape 3 : remote DVC, `dvc pull`)*

## Configuration
Tous les chemins et seuils sont dans `configs/params.yaml`. `DATA_BASE` et `REPORTS_DIR` permettent de rediriger les sorties.

## Étapes (dvc.yaml)
`ingest_*` -> `preprocess` -> `align` -> `split` -> `build_manifest` -> `quality_gate` -> `report`.
Le `quality_gate` termine avec le code 1 si un contrôle critique échoue : le pipeline s'arrête.
Les étapes de fusion / entraînement / évaluation (Partie B) sont prévues en commentaire dans `dvc.yaml`.
