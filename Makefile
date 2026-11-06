.PHONY: pipeline test lint sample
pipeline:      ## reconstruit toutes les données préparées (nécessite le token Kaggle)
	dvc repro
test:
	pytest -q
lint:
	flake8 src tests --max-line-length 150
sample:        ## pipeline complet sur échantillon synthétique (CI, démo sans Kaggle)
	DATA_BASE=data_sample REPORTS_DIR=reports_sample python -m src.common.sample_data
	DATA_BASE=data_sample REPORTS_DIR=reports_sample python -m src.run_pipeline
