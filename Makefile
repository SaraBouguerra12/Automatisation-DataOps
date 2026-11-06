.PHONY: pipeline test lint sample
pipeline:      ## reconstruit toutes les données préparées
	dvc repro
test:
	pytest -q
lint:
	flake8 src tests --max-line-length 130
sample:        ## pipeline sur échantillon réduit (CI)
	CI_SAMPLE=1 dvc repro
