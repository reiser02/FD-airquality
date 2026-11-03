"""Shared configuration for the CLI and Flower App entrypoints."""

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
        if self.selection_metric not in {"macro-mase", "macro-rmsse", "mae"}:
            raise ValueError("selection_metric debe ser macro-mase, macro-rmsse o mae.")
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


def resolve_manifest(path: str | None) -> Path:
    if path:
        result = Path(path).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(f"No existe el manifest: {result}")
        return result
    candidates = list(Path("data/federated").glob("*/manifest.json"))
    if not candidates:
        raise FileNotFoundError("No hay datos preparados; ejecuta uv run fd-airquality prepare.")
    return max(candidates, key=lambda entry: entry.stat().st_mtime_ns).resolve()
