# Modération de contenu multimodale (image + texte) — Pipeline DataOps

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
## Récupérer les données
*(à compléter à l'étape 3 : remote DVC, `dvc pull`)*
## Exécuter le pipeline
```bash
make pipeline      # = dvc repro
make test
```
## Structure
Voir l'arborescence : configs/, src/{ingestion,validation,preprocessing,alignment,fusion}, tests/, dvc.yaml.
