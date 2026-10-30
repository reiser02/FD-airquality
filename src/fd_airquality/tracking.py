"""Explicit MLflow tracking without autologging or uploading station data."""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from fd_airquality.artifacts import file_digest
from fd_airquality.models import model_metadata
from fd_airquality.settings import RunSettings


def tracking_uri(settings: RunSettings) -> str:
    if settings.mlflow_tracking_uri:
        uri = settings.mlflow_tracking_uri
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            return f"sqlite:///{Path(uri.removeprefix('sqlite:///')).expanduser().resolve()}"
        return uri
    database = Path(settings.output_dir).resolve().parent / "mlflow.db"
    return f"sqlite:///{database}"


def write_run(output: Path, settings: RunSettings, manifest: dict, mode: str) -> None:
    versions = {}
    for name in (
        "fd-airquality", "flwr", "ray", "torch", "darts", "numpy", "pandas", "numba",
        "scipy", "scikit-learn", "granite-tsfm", "transformers", "datasets", "mlflow",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    (output / "run.json").write_text(json.dumps({
        "mode": mode, "settings": asdict(settings),
        "manifest_sha256": file_digest(Path(settings.manifest)), "manifest": manifest,
        "model": model_metadata(manifest["geometry"], settings),
        "python": sys.version, "versions": versions,
        "granite_source": json.loads(distribution("granite-tsfm").read_text("direct_url.json") or "null"),
        "application_sources": {path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "training_budget": {
            "stages": 0 if mode == "zero-shot" else settings.num_server_rounds,
            "epochs_per_stage": settings.local_epochs,
            "optimizer_reset_each_stage": True,
        },
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class RunTracker:
    """One run per method/model; only the server or baseline coordinator logs."""

    def __init__(self, settings: RunSettings, output: Path, mode: str, manifest: dict):
        self.settings, self.output, self.mode, self.manifest = settings, output, mode, manifest
        self.client = None
        self.run_id = None

    def __enter__(self):
        self.started = time.perf_counter()
        if self.settings.mlflow_tracking_uri == "disabled":
            return self
        from mlflow import MlflowClient

        uri = tracking_uri(self.settings)
        self.client = MlflowClient(tracking_uri=uri)
        experiment = self.client.get_experiment_by_name(self.settings.mlflow_experiment)
        if experiment is None:
            artifacts = self.output.parent / "mlflow-artifacts"
            experiment_id = self.client.create_experiment(
                self.settings.mlflow_experiment, artifact_location=(
                    artifacts.resolve().as_uri() if uri.startswith(("sqlite:", "file:")) else None
                ),
            )
        else:
            experiment_id = experiment.experiment_id
        run = self.client.create_run(experiment_id, tags={
            "mlflow.runName": f"{self.settings.model}-{self.mode}-{self.output.name}",
            "model": self.settings.model, "mode": self.mode,
            "pollutant": self.manifest["pollutant"], "protocol": self.manifest["protocol"],
            "manifest_sha256": file_digest(Path(self.settings.manifest)),
            "adaptation": model_metadata(self.manifest["geometry"], self.settings).get("adaptation", "full"),
        })
        self.run_id = run.info.run_id
        try:
            self._log_params()
        except Exception:
            self.client.set_terminated(self.run_id, "FAILED")
            raise
        (self.output / "mlflow.json").write_text(json.dumps({
            "tracking_uri": uri, "experiment_id": experiment_id, "run_id": self.run_id,
        }, indent=2) + "\n", encoding="utf-8")
        return self

    def _log_params(self):
        for key, value in asdict(self.settings).items():
            self.client.log_param(self.run_id, key, value)
        for key, value in self.manifest["geometry"].items():
            self.client.log_param(self.run_id, f"data.{key}", value)
        for key, value in model_metadata(self.manifest["geometry"], self.settings).items():
            serialized = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            if len(serialized) > 5000:
                self.client.log_param(self.run_id, f"model.{key}.sha256", sha256(serialized.encode()).hexdigest())
            else:
                self.client.log_param(self.run_id, f"model.{key}", serialized)
        self.client.log_param(self.run_id, "num_clients", len(self.manifest["clients"]))

    def log_metrics(self, split: str, metrics: dict, step: int, *, partition: int | None = None) -> None:
        if self.client is None:
            return
        from mlflow.entities import Metric

        prefix = split if partition is None else f"station_{partition:03d}/{split}"
        timestamp = int(time.time() * 1000)
        records = [Metric(f"{prefix}/{key}", float(value), timestamp, step)
                   for key, value in metrics.items() if isinstance(value, (int, float)) and math.isfinite(value)]
        self.client.log_batch(self.run_id, metrics=records)

    def __exit__(self, exc_type, exc_value, traceback):
        resources = {
            "wall-seconds": time.perf_counter() - self.started,
            "coordinator-peak-process-rss-mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "status": "FAILED" if exc_type else "FINISHED",
            "note": "RSS es el máximo de vida del proceso, no memoria exclusiva de una ronda. Bytes comunicados son payload tensorial, no bytes de red.",
        }
        (self.output / "resource_metrics.json").write_text(json.dumps(resources, indent=2) + "\n", encoding="utf-8")
        if self.client is not None:
            status = "FAILED"
            try:
                self.log_metrics("runtime", resources, 0)
                # Allowlist: never recursively log the project, data, NPZ or scalers.
                for name in ("run.json", "round_metrics.csv", "station_metrics.csv", "test_metrics.json", "resource_metrics.json", "excluded_stations.csv"):
                    if (self.output / name).is_file():
                        self.client.log_artifact(self.run_id, str(self.output / name))
                if self.settings.mlflow_log_models:
                    for path in sorted(self.output.glob("*.pt")):
                        self.client.log_artifact(self.run_id, str(path), artifact_path="checkpoints")
                if exc_type is None:
                    status = "FINISHED"
            finally:
                self.client.set_terminated(self.run_id, status)
        return False
