"""Exécute les étapes dans l'ordre (utilisé par la CI sur échantillon). En production : `dvc repro`."""
from src.alignment import dedup_groups, split
from src.export import manifest
from src.preprocessing import run as preprocess
from src.reporting import data_report
from src.validation import quality_gate

STAGES = [("preprocess", preprocess.main), ("align", dedup_groups.main), ("split", split.main),
          ("build_manifest", manifest.main), ("quality_gate", quality_gate.main), ("report", data_report.main)]


def main():
    for name, fn in STAGES:
        print(f"\n=== étape : {name} ===")
        fn()


if __name__ == "__main__":
    main()
