# Modération de contenu multimodale (image + texte) ” Pipeline DataOps

Pipeline reproductible de collecte et de préparation des données (Hateful Memes, Sentiment140, Reddit optionnel).

## Installation
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécuter le pipeline
```bash
make sample      # démo complete sur données SYNTHéTIQUES (sans Kaggle) -> data_sample/, reports_sample/
make test        # tests unitaires + tests de bout en bout + échec du controle contre de qualité
make pipeline    # = dvc repro : pipeline réel (nécessite le token Kaggle)
```
Token Kaggle (Settings -> API -> Generate New Token) : `export KAGGLE_API_TOKEN=...` (Windows : `$env:KAGGLE_API_TOKEN="..."`).

## Récupérer les données
*(Ã  compléter Ã  l'étape 3 : remote DVC, `dvc pull`)*

## Configuration
Tous les chemins et seuils sont dans `configs/params.yaml`. `DATA_BASE` et `REPORTS_DIR` permettent de rediriger les sorties.

## Étapes (dvc.yaml)
`ingest_*` -> `preprocess` -> `align` -> `split` -> `build_manifest` -> `quality_gate` -> `report`.
Le `quality_gate` termine avec le code 1 si un controle critique échoue : le pipeline s'arréte.
Les Étapes de fusion / entrainement / évaluation (Partie B) sont prévues en commentaire dans `dvc.yaml`.

## Récupérer les données (DVC)
Les données ne sont pas dans Git : elles sont versionnées avec DVC (remote `storage`).
1. Cloner le dépôt et installer les dépendances (voir Installation).
2. Pointer vers le dossier de stockage partagé (non versionné, propre à chaque machine) :
   dvc remote modify --local storage url CHEMIN_DU_DOSSIER_PARTAGE
3. Récupérer les données : dvc pull
4. Vérifier : dvc status   (doit répondre : Data and pipelines are up to date)
