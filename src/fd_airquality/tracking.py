"""Explicit MLflow tracking without autologging or uploading station data."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import resource
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import torch

from fd_airquality.artifacts import file_digest
from fd_airquality.missing import resolve_missing_policy
from fd_airquality.models import model_metadata
from fd_airquality.run_logging import flush_run_logs
from fd_airquality.settings import RunSettings

MODEL_LABELS = {
    "mlp": "MLP", "autoencoder": "Autoencoder", "nlinear": "NLinear", "nhits": "NHITS",
    "ttm-r3": "TTM-R3", "patchtst-fm-r2": "PatchTST-FM-R2", "flowstate-r1": "FlowState-R1",
}
MODE_LABELS = {"local": "local", "centralized": "centralizado", "federated": "federado", "zero-shot": "zero-shot"}
ADAPTATION_LABELS = {
    "full": "completo", "none": "sin adaptación", "lora": "LoRA", "ffa-lora": "FFA-LoRA",
    "flora": "FLoRA", "lora-fair": "LoRA-FAIR",
}

_EVALUATION_METRICS = (
    "macro-mase", "macro-rmsse", "crps", "wis", "target-coverage",
    "robust-num-clients", "robust-macro-mase", "robust-macro-rmsse",
)
_TRACKED_METRICS = {
    "train": (
        "train-seconds", "peak-cuda-allocated-mib", "trainable-parameters",
        "uplink-tensor-bytes", "downlink-tensor-bytes", "global-state-tensor-bytes",
        "server-aggregation-seconds", "server-aggregation-cuda",
    ),
    "validation": _EVALUATION_METRICS,
    "test": _EVALUATION_METRICS,
    "selection": ("round", "validation_score"),
    "runtime": ("wall-seconds",),
}


def tracking_uri(settings: RunSettings) -> str:
    if settings.mlflow_tracking_uri:
        uri = settings.mlflow_tracking_uri
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            return f"sqlite:///{Path(uri.removeprefix('sqlite:///')).expanduser().resolve()}"
        return uri
    database = Path(settings.output_dir).resolve().parent / "mlflow.db"
    return f"sqlite:///{database}"


def backtest_tags(output_dir: str | Path, pollutant: str) -> dict[str, str]:
    """Group all folds of a backtest without changing the child process settings."""
    case = Path(output_dir)
    fold_dir = case.parent.parent
    fold = re.fullmatch(r"fold_([0-9]{3,})", fold_dir.name)
    stamp = re.fullmatch(r"backtest_([0-9]{8}_[0-9]{6})(?:_[0-9]+)?", fold_dir.parent.name)
    if (case.parent.name != "runs" or not re.fullmatch(r"case_[0-9]{4,}", case.name)
            or fold is None or stamp is None):
        return {}
    try:
        started = datetime.strptime(stamp[1], "%Y%m%d_%H%M%S")
    except ValueError:
        return {}
    return {"backtest": f"{pollutant} | {started:%Y-%m-%d %H:%M:%S} UTC", "fold": str(int(fold[1]))}


def run_labels(settings: RunSettings, mode: str, pollutant: str, adaptation: str,
               *, started_at: datetime | None = None) -> dict[str, str]:
    """Use the effective method in the display name; seed and rank stay in params."""
    parts = [MODEL_LABELS.get(settings.model, settings.model), MODE_LABELS.get(mode, mode),
             ADAPTATION_LABELS.get(adaptation, adaptation)]
    if mode == "federated":
        parts.append("FedProx" if settings.strategy == "fedprox" else "FedAvg")
    candidate = " | ".join(parts)
    tags = backtest_tags(settings.output_dir, pollutant)
    started = (started_at or datetime.now(UTC)).astimezone(UTC)
    origin = f"{tags['backtest']} | fold {tags['fold']}" if tags else f"{pollutant} | {started:%Y-%m-%d %H:%M:%S} UTC"
    return tags | {"candidate": candidate, "mlflow.runName": f"{candidate} | {origin}"}


def initialize_tracking(settings: RunSettings, artifacts: Path) -> None:
    if settings.mlflow_tracking_uri == "disabled":
        return
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri(settings))
    if client.get_experiment_by_name(settings.mlflow_experiment) is None:
        client.create_experiment(settings.mlflow_experiment, artifact_location=(
            artifacts.resolve().as_uri() if tracking_uri(settings).startswith(("sqlite:", "file:")) else None
        ))


def write_run(output: Path, settings: RunSettings, manifest: dict, mode: str) -> None:
    versions = {}
    for name in (
        "fd-airquality", "flwr", "ray", "torch", "neuralforecast", "numpy", "pandas", "numba",
        "scipy", "scikit-learn", "granite-tsfm", "transformers", "datasets", "mlflow",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    hardware = {"platform": platform.platform(), "machine": platform.machine(),
                "torch_cuda": torch.version.cuda, "cudnn": None,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark}
    if settings.device == "cuda":
        try:
            hardware["cudnn"] = torch.backends.cudnn.version()
        except RuntimeError as exc:
            hardware["cudnn_metadata_error"] = str(exc)
        hardware["gpus"] = [{"name": torch.cuda.get_device_properties(index).name,
                             "total_memory": torch.cuda.get_device_properties(index).total_memory}
                            for index in range(torch.cuda.device_count())]
    progressive = bool(mode == "federated" and settings.arrival_period_hours
                       and manifest["geometry"].get("split_mode") == "global")
    (output / "run.json").write_text(json.dumps({
        "mode": mode, "settings": asdict(settings),
        "manifest_sha256": file_digest(Path(settings.manifest)), "manifest": manifest,
        "model": model_metadata(manifest["geometry"], settings),
        "python": sys.version, "versions": versions,
        "granite_source": json.loads(distribution("granite-tsfm").read_text("direct_url.json") or "null"),
        "neuralforecast_source": json.loads(distribution("neuralforecast").read_text("direct_url.json") or "null"),
        "application_sources": {path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "execution": {"pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "hardware": hardware,
        "development_lock_sha256": file_digest(Path("uv.lock")) if Path("uv.lock").is_file() else None,
        "initialization": {"fold": "fresh", "trained_checkpoint_reused": False,
                           "base": "pinned_pretrained" if settings.model in {"ttm-r3", "patchtst-fm-r2", "flowstate-r1"} else "seeded_random"},
        "missing_data": {"policy": resolve_missing_policy(settings.model, settings.missing_policy, settings.missing_policies),
                         "target_policy": "observed_only", "window_policy": manifest["geometry"].get("window_policy", "strict"),
                         "training_threshold": settings.training_data_availability_threshold,
                         "model_thresholds": json.loads(settings.training_availability),
                          "normalization": settings.normalization, "scaler_type": settings.scaler_type},
        "window_protocol": {
            "training_padding": manifest["geometry"].get("window_padding", "none"),
            "evaluation_padding": "none",
            "training_window_sampling": "resample_without_replacement_each_epoch",
            "max_train_windows_per_station_per_epoch": manifest["geometry"].get("max_train_windows"),
            "short_training_batch": "fill_with_replacement_from_current_epoch_selection",
            "federation_weight": ("mean_observed_targets_per_epoch_before_replacement"
                                  if manifest.get("schema_version", 4) >= 5 else "windows_per_epoch"),
        },
        "probabilistic_protocol": {"crps": "quantile_integral_trapezoid_constant_tails_approximation",
                                   "truth": "canonical_observed_original_units"},
        "arrival_protocol": {"period_hours": settings.arrival_period_hours if progressive else 0,
                              "history": "new-only" if progressive else None,
                              "training_targets": "current_block_only" if progressive else "full_fold_training",
                              "block_schedule": "initial_prefix_then_calendar_increments" if progressive else None,
                              "validation": "after_final_arrival" if progressive else "each_stage",
                              "weight_continuity": "within_fold_rounds_only"},
        "metric_protocol": {"implementation": "masked_seasonal_errors_v2",
                             "reference": manifest["geometry"].get("error_scale_reference", "pre-evaluation"),
                             "policy": manifest["geometry"].get("error_scale_policy", "unfloored")},
        "training_budget": {
            "stages": 0 if mode == "zero-shot" else settings.num_server_rounds,
            "epochs_per_stage": settings.local_epochs,
            "optimizer_reset_each_stage": True,
        },
        "training_controls": {
            "scope": "local_epochs_per_stage_or_round",
            "monitor": "pooled_observed_mse_original_units",
            "early_stopping": settings.early_stopping,
            "patience": settings.early_stopping_patience,
            "min_delta": settings.early_stopping_min_delta,
            "scheduler": settings.lr_scheduler,
            "scheduler_patience": settings.lr_scheduler_patience,
            "scheduler_factor": settings.lr_scheduler_factor,
            "min_lr": settings.min_learning_rate,
            "federated_monitor": "fixed_reserve_before_round_one" if progressive else None,
            "round_validation_hours": settings.round_validation_hours if progressive else None,
            "single_epoch_budget": settings.local_epochs == 1,
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
        metadata = model_metadata(self.manifest["geometry"], self.settings)
        adaptation = metadata.get("adaptation", "full")
        run = self.client.create_run(experiment_id, tags={
            "model": self.settings.model, "mode": self.mode,
            "pollutant": self.manifest["pollutant"], "protocol": self.manifest["protocol"],
            "manifest_sha256": file_digest(Path(self.settings.manifest)),
            "adaptation": adaptation,
            **run_labels(self.settings, self.mode, self.manifest["pollutant"], adaptation),
        })
        self.run_id = run.info.run_id
        try:
            self._log_params(metadata)
        except Exception:
            self.client.set_terminated(self.run_id, "FAILED")
            raise
        (self.output / "mlflow.json").write_text(json.dumps({
            "tracking_uri": uri, "experiment_id": experiment_id, "run_id": self.run_id,
        }, indent=2) + "\n", encoding="utf-8")
        return self

    def _log_params(self, metadata):
        settings = self.settings
        geometry = self.manifest["geometry"]
        adaptation = metadata.get("adaptation", "full")
        keys = ["seed", "device", "output_dir", "selection_metric", "probabilistic", "normalization"]
        if settings.normalization != "none":
            keys.append("scaler_type")
        if self.mode != "zero-shot":
            keys.extend(("num_server_rounds", "local_epochs", "batch_size", "learning_rate", "weight_decay"))
        if self.mode == "federated":
            keys.append("strategy")
            if settings.strategy == "fedprox":
                keys.append("proximal_mu")
        if adaptation in {"lora", "ffa-lora", "flora", "lora-fair"}:
            keys.extend(("lora_rank", "lora_alpha", "lora_dropout", "lora_targets"))
        if adaptation == "lora-fair":
            keys.extend(("fair_iterations", "fair_learning_rate", "fair_lambda"))
        if settings.model == "flowstate-r1":
            keys.append("flowstate_scale_factor")
        params = {key: getattr(settings, key) for key in keys} | {
            "data.context": geometry["context"], "data.horizon": geometry["horizon"],
            "num_clients": len(self.manifest["clients"]),
            "missing_policy": resolve_missing_policy(settings.model, settings.missing_policy, settings.missing_policies),
        }
        if "revision" in metadata:
            params["model.revision"] = metadata["revision"]
        for key, value in params.items():
            self.client.log_param(self.run_id, key, value)

    def log_metrics(self, split: str, metrics: dict, step: int) -> None:
        if self.client is None:
            return
        from mlflow.entities import Metric

        timestamp = int(time.time() * 1000)
        records = [Metric(f"{split}/{key}", float(value), timestamp, step)
                   for key in _TRACKED_METRICS.get(split, ())
                   if isinstance(value := metrics.get(key), (int, float)) and math.isfinite(value)]
        if records:
            self.client.log_batch(self.run_id, metrics=records)

    def __exit__(self, exc_type, exc_value, traceback):
        logging.getLogger(__name__).info("Tracking finalize status=%s", "FAILED" if exc_type else "FINISHED")
        if exc_type is not None:
            logging.getLogger(__name__).error("Run failed: %s", exc_value, exc_info=(exc_type, exc_value, traceback))
        flush_run_logs()
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
                for name in ("run.json", "round_metrics.csv", "test_metrics.json", "resource_metrics.json", "excluded_stations.csv", "run.log"):
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
