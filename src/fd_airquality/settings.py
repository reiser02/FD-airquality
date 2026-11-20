"""Shared configuration for the CLI and Flower App entrypoints."""

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class RunSettings:
    manifest: str = "data/federated/manifest.json"
    output_dir: str = "reports/federated"
    num_server_rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 128
    pad_last_batch: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    hidden_size: int = 64
    strategy: str = "fedavg"
    proximal_mu: float = 0.01
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 1
    model: str = "mlp"
    latent_size: int = 16
    reconstruction_weight: float = 0.1
    hf_revision: str = ""
    flowstate_scale_factor: float = 1.0
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "fd-airquality"
    mlflow_log_models: int = 1
    adaptation: str = "auto"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_targets: str = "auto"
    fair_iterations: int = 1000
    fair_learning_rate: float = 0.01
    fair_lambda: float = 1.0
    selection_metric: str = "macro-mase"
    heartbeat_seconds: float = 60.0
    log_level: str = "INFO"
    missing_policy: str = "auto"
    missing_policies: str = "{}"
    deterministic: int = 0
    probabilistic: int = 0
    calibrate_point_forecasts: int = 0
    calibration_min_pairs: int = 20
    training_data_availability_threshold: str = "0.0"
    training_availability: str = "{}"
    normalization: str = "window"
    scaler_type: str = "standard"
    normalization_reference_end: str = ""
    arrival_period_hours: int = 24
    federated_update_epochs: int = 0
    early_stopping: int = 1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    lr_scheduler: str = "plateau"
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    min_learning_rate: float = 0.000001
    round_validation_hours: int = 24
    round_validation_max_hours: int = 0

    def __post_init__(self) -> None:
        for name in ("num_server_rounds", "local_epochs", "batch_size", "hidden_size", "torch_threads", "latent_size", "lora_rank", "fair_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} debe ser un entero positivo.")
        if (
            not all(isfinite(value) for value in (self.learning_rate, self.weight_decay, self.proximal_mu))
            or self.learning_rate <= 0 or self.weight_decay < 0 or self.proximal_mu < 0
        ):
            raise ValueError("learning_rate debe ser positivo; weight_decay y proximal_mu, no negativos.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed debe ser un entero entre 0 y 2**32 - 1.")
        if self.strategy not in {"fedavg", "fedprox"}:
            raise ValueError("strategy debe ser fedavg o fedprox.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device debe ser cpu o cuda.")
        from fd_airquality.models import MODEL_NAMES

        if self.model not in MODEL_NAMES:
            raise ValueError(f"Modelo desconocido: {self.model}. Opciones: {', '.join(MODEL_NAMES)}")
        if not isfinite(self.reconstruction_weight) or self.reconstruction_weight < 0:
            raise ValueError("reconstruction_weight debe ser finito y no negativo.")
        if not isfinite(self.flowstate_scale_factor) or self.flowstate_scale_factor <= 0:
            raise ValueError("flowstate_scale_factor debe ser finito y positivo.")
        if self.hf_revision and not re.fullmatch(r"[0-9a-f]{40}", self.hf_revision):
            raise ValueError("hf_revision debe ser un SHA inmutable de 40 caracteres hexadecimales.")
        if self.mlflow_log_models not in {0, 1} or not self.mlflow_experiment.strip():
            raise ValueError("mlflow_log_models debe ser 0/1 y mlflow_experiment no puede estar vacío.")
        if self.adaptation not in {"auto", "none", "lora", "ffa-lora", "flora", "lora-fair"}:
            raise ValueError("Adaptación desconocida.")
        if (not all(isfinite(value) for value in (self.lora_alpha, self.lora_dropout, self.fair_learning_rate, self.fair_lambda))
                or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1
                or self.fair_learning_rate <= 0 or self.fair_lambda < 0 or not self.lora_targets.strip()):
            raise ValueError("Hiperparámetros LoRA/LoRA-FAIR inválidos.")
        if self.selection_metric not in {"macro-mase", "macro-rmsse"}:
            raise ValueError("selection_metric debe ser macro-mase o macro-rmsse.")
        from fd_airquality.missing import resolve_missing_policy

        resolve_missing_policy(self.model, self.missing_policy, self.missing_policies)
        from fd_airquality.availability import availability_threshold

        availability_threshold(self)
        if self.normalization not in {"window", "fixed", "none"}:
            raise ValueError("normalization debe ser window, fixed o none.")
        if self.scaler_type not in {"identity", "standard", "robust", "minmax"}:
            raise ValueError("scaler_type debe ser identity, standard, robust o minmax.")
        if self.normalization_reference_end:
            import pandas as pd

            date = pd.Timestamp(self.normalization_reference_end)
            if pd.isna(date) or date.tz is not None or date != date.floor("h"):
                raise ValueError("normalization_reference_end debe ser una fecha horaria sin zona.")
        if type(self.arrival_period_hours) is not int or self.arrival_period_hours < 0:
            raise ValueError("arrival_period_hours debe ser un entero no negativo (0 desactiva llegada).")
        if type(self.federated_update_epochs) is not int or self.federated_update_epochs < 0:
            raise ValueError("federated_update_epochs debe ser un entero no negativo (0 hereda local_epochs).")
        if type(self.pad_last_batch) is not int or self.pad_last_batch not in {0, 1}:
            raise ValueError("pad_last_batch debe ser 0/1.")
        if type(self.early_stopping) is not int or self.early_stopping not in {0, 1}:
            raise ValueError("early_stopping debe ser 0/1.")
        for name in ("early_stopping_patience", "round_validation_hours"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} debe ser un entero positivo.")
        max_hours = self.round_validation_max_hours
        if type(max_hours) is not int or (max_hours != 0 and max_hours < self.round_validation_hours):
            raise ValueError("round_validation_max_hours debe ser 0 o al menos round_validation_hours.")
        if type(self.lr_scheduler_patience) is not int or self.lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience debe ser un entero no negativo.")
        if self.lr_scheduler not in {"plateau", "none"}:
            raise ValueError("lr_scheduler debe ser plateau o none.")
        if not isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta debe ser finito y no negativo.")
        if not isfinite(self.lr_scheduler_factor) or not 0 < self.lr_scheduler_factor < 1:
            raise ValueError("lr_scheduler_factor debe estar entre 0 y 1.")
        if not isfinite(self.min_learning_rate) or not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate debe ser positivo y no superar learning_rate.")
        if type(self.deterministic) is not int or self.deterministic not in {0, 1}:
            raise ValueError("deterministic debe ser 0/1.")
        if type(self.probabilistic) is not int or self.probabilistic not in {0, 1}:
            raise ValueError("probabilistic debe ser 0/1.")
        if type(self.calibrate_point_forecasts) is not int or self.calibrate_point_forecasts not in {0, 1}:
            raise ValueError("calibrate_point_forecasts debe ser 0/1.")
        if type(self.calibration_min_pairs) is not int or self.calibration_min_pairs < 2:
            raise ValueError("calibration_min_pairs debe ser un entero >= 2.")
        if not isfinite(self.heartbeat_seconds) or self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds debe ser finito y positivo.")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level debe ser DEBUG, INFO, WARNING o ERROR.")

    @classmethod
    def from_mapping(cls, config: Mapping) -> "RunSettings":
        known = {field.name for field in fields(cls)}
        values = {key.replace("-", "_"): value for key, value in config.items()}
        unknown = values.keys() - known
        if unknown:
            raise ValueError(f"Opciones de ejecución desconocidas: {sorted(unknown)}")
        return cls(**values)

    def flower_config(self) -> dict:
        return {key.replace("_", "-"): value for key, value in asdict(self).items()}


def resolve_manifest(path: str | None, *, backtest: bool = False) -> Path:
    if path:
        result = Path(path).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(f"No existe el manifest: {result}")
        return result
    from fd_airquality.loaders import POLLUTANT

    root, filename, command = (("data/backtests", "backtest_manifest.json", "prepare-backtest")
                               if backtest else ("data/federated", "manifest.json", "prepare"))
    candidates = [entry for entry in Path(root).glob(f"*/{filename}")
                  if json.loads(entry.read_text(encoding="utf-8")).get("pollutant") == POLLUTANT]
    if not candidates:
        raise FileNotFoundError(f"No hay datos preparados de {POLLUTANT}; ejecuta uv run fd-airquality {command}.")
    return max(candidates, key=lambda entry: entry.stat().st_mtime_ns).resolve()
