"""Suivi d'expériences MLflow (optionnel : sans MLflow installé, le code s'exécute quand même)."""
from contextlib import contextmanager


@contextmanager
def tracked_run(cfg, run_name, params=None):
    try:
        import mlflow
    except ImportError:
        yield None
        return
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield mlflow
