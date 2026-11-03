"""Plain PyTorch training and station-local metrics, independent of Flower."""

from __future__ import annotations

import logging
import random
import resource
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fd_airquality.data import WindowDataset
from fd_airquality.metrics import compute_mase, compute_rmsse
from fd_airquality.models import ForecastAutoencoder, ForecastMLP, make_model, shared_state  # noqa: F401
from fd_airquality.run_logging import report_progress
from fd_airquality.settings import RunSettings


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        try:
            if not torch.cuda.is_available():
                logging.getLogger(__name__).warning("CUDA no disponible; se usará CPU.")
                warnings.warn("CUDA no disponible; se usará CPU.", stacklevel=2)
                return torch.device("cpu")
            # Availability alone does not guarantee kernels for this GPU architecture.
            probe = torch.ones(1, device="cuda")
            probe.add_(1)
            torch.cuda.synchronize()
            return torch.device("cuda")
        except (RuntimeError, AssertionError) as exc:
            reason = str(exc).splitlines()[0]
            logging.getLogger(__name__).warning("CUDA no utilizable (%s); se usará CPU.", reason)
            warnings.warn(f"CUDA no utilizable ({reason}); se usará CPU.", stacklevel=2)
    return torch.device("cpu")


def configure_torch(settings: RunSettings, seed: int) -> torch.device:
    device = resolve_device(settings.device)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(settings.torch_threads)
    return device


def _microbatch_size(settings: RunSettings, device: torch.device) -> int:
    # Execution limit, not an optimizer batch-size change or an OOM retry policy.
    if settings.model == "patchtst-fm-r2" and device.type == "cuda":
        return min(settings.batch_size, 8)
    return settings.batch_size


def train_local(
    model: nn.Module,
    dataset: Dataset,
    settings: RunSettings,
    device: torch.device,
    seed: int,
    proximal_mu: float = 0.0,
) -> dict[str, float | int]:
    started = time.perf_counter()
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No hay parámetros entrenables; zero-shot no debe entrar en train_local.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reference = [parameter.detach().clone() for parameter in parameters] if proximal_mu else []
    loader = DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    # Local optimizer state is deliberately reset at each communication round.
    optimizer = torch.optim.AdamW(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    microbatch_size = _microbatch_size(settings, device)
    optimizer_steps = 0
    total_squared_error, targets = 0.0, 0
    for epoch in range(1, settings.local_epochs + 1):
        report_progress("train", f"epoch={epoch}/{settings.local_epochs} batches=0/{len(loader)} device={device} "
                        f"effective_batch_size={settings.batch_size} microbatch_size={microbatch_size}")
        last_report = time.monotonic()
        for batch, (inputs, actual) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(inputs), microbatch_size):
                micro_inputs = inputs[start:start + microbatch_size].to(device)
                micro_actual = actual[start:start + microbatch_size].to(device)
                mse = nn.functional.mse_loss(model(micro_inputs), micro_actual)
                loss = mse
                if isinstance(model, ForecastAutoencoder) and settings.reconstruction_weight:
                    loss = loss + settings.reconstruction_weight * model.reconstruction_loss(micro_inputs)
                if not torch.isfinite(loss):
                    raise ValueError("Pérdida local no finita; se aborta para no contaminar los pesos globales.")
                if not loss.requires_grad:
                    raise RuntimeError("El modelo cortó autograd: los adaptadores no reciben gradientes.")
                # Weight by targets, including uneven microbatches and the final
                # partial logical batch. Backward releases each activation graph.
                (loss * (micro_actual.numel() / actual.numel())).backward()
                total_squared_error += float(mse.detach()) * micro_actual.numel()
                targets += micro_actual.numel()
                del loss, mse
            if proximal_mu:
                penalty = sum(
                    (parameter - initial).square().sum()
                    for parameter, initial in zip(parameters, reference, strict=True)
                )
                loss = proximal_mu * penalty / 2
                if not torch.isfinite(loss):
                    raise ValueError("Penalización proximal no finita; se aborta el entrenamiento.")
                # FedProx is added once per logical batch, not per microbatch.
                loss.backward()
                del loss, penalty
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            if time.monotonic() - last_report >= settings.heartbeat_seconds:
                report_progress("train", f"epoch={epoch}/{settings.local_epochs} batches={batch}/{len(loader)}")
                last_report = time.monotonic()
        report_progress("train", f"epoch={epoch}/{settings.local_epochs} batches={len(loader)}/{len(loader)} "
                        f"mse_normalized={total_squared_error / targets:.6g}")
    if any(not torch.isfinite(value).all() for value in shared_state(model).values()):
        raise ValueError("El entrenamiento produjo pesos no finitos.")
    metrics = {
        "num-examples": len(dataset), "train-mse-normalized": total_squared_error / targets,
        "train-seconds": time.perf_counter() - started,
        "trainable-parameters": sum(parameter.numel() for parameter in parameters),
        "peak-process-rss-mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "effective-batch-size": settings.batch_size, "microbatch-size": microbatch_size,
        "optimizer-steps": optimizer_steps,
    }
    if device.type == "cuda":
        metrics["peak-cuda-allocated-mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return metrics


@torch.no_grad()
def evaluate_local(
    model: nn.Module,
    dataset: WindowDataset,
    settings: RunSettings,
    device: torch.device,
    artifact: dict[str, np.ndarray],
    seasonality: int,
) -> dict[str, float | int]:
    model.to(device).eval()
    predictions, actuals, baselines = [], [], []
    loader = DataLoader(dataset, batch_size=_microbatch_size(settings, device), num_workers=0)
    last_report = time.monotonic()
    for batch, (inputs, actual) in enumerate(loader, start=1):
        forecast = model(inputs.to(device)).cpu().numpy().astype(np.float64)
        predictions.append(forecast * dataset.scale + dataset.mean)
        actuals.append(actual.numpy().astype(np.float64) * dataset.scale + dataset.mean)
        baselines.append(np.repeat(
            inputs[:, -1:].numpy().astype(np.float64) * dataset.scale + dataset.mean,
            dataset.horizon, axis=1,
        ))
        if time.monotonic() - last_report >= settings.heartbeat_seconds:
            report_progress(dataset.split, f"batches={batch}/{len(loader)} device={device}")
            last_report = time.monotonic()
    prediction, actual, baseline = map(np.concatenate, (predictions, actuals, baselines))
    if not np.isfinite(prediction).all():
        raise ValueError("Predicciones no finitas.")
    error, baseline_error = prediction - actual, baseline - actual
    stats: dict[str, float | int] = {
        "num-examples": len(dataset),
        "num-targets": actual.size,
        "absolute-error-sum": float(np.abs(error).sum()),
        "squared-error-sum": float(np.square(error).sum()),
        "baseline-absolute-error-sum": float(np.abs(baseline_error).sum()),
        "baseline-squared-error-sum": float(np.square(baseline_error).sum()),
        "mae": float(np.abs(error).mean()),
        "mse": float(np.square(error).mean()),
        "baseline-mae": float(np.abs(baseline_error).mean()),
        "mase-sum": 0.0,
        "mase-count": 0,
        "rmsse-sum": 0.0,
        "rmsse-count": 0,
    }
    prefix = f"{dataset.split}_" if f"{dataset.split}_mase_scale" in artifact else ""
    if f"{prefix}mase_scale" in artifact and f"{prefix}rmsse_scale" in artifact:
        if prefix:
            stats["error-scale-pairs"] = int(artifact[f"{prefix}scale_pairs"])
        for name in ("mase", "rmsse"):
            denominator = float(artifact[f"{prefix}{name}_scale"])
            if np.isfinite(denominator) and denominator > 0:
                per_origin = np.abs(error).mean(axis=1) / denominator if name == "mase" else np.sqrt(np.square(error).mean(axis=1) / denominator)
                stats[f"{name}-sum"] = float(per_origin.sum())
                stats[f"{name}-count"] = len(per_origin)
                stats[name] = stats["mae"] / denominator if name == "mase" else float(np.sqrt(stats["mse"] / denominator))
        return stats
    # Historical schema-2 artifacts only: their original origin-dependent scales
    # are not silently relabelled as a fixed-reference protocol.
    series = pd.Series(artifact["values"], index=pd.to_datetime(artifact["timestamps"]))
    for row, start in enumerate(dataset.starts):
        origin = int(start) + dataset.context
        timestamps = series.index[origin : origin + dataset.horizon]
        actual_series = pd.Series(actual[row], index=timestamps)
        predicted_series = pd.Series(prediction[row], index=timestamps)
        # Preserve NaN and time geometry in the entire available raw prefix.
        prefix = series.iloc[:origin]
        for name, metric in (("mase", compute_mase), ("rmsse", compute_rmsse)):
            value = metric(actual_series, predicted_series, prefix, seasonality_m=seasonality)
            if np.isfinite(value):
                stats[f"{name}-sum"] += value
                stats[f"{name}-count"] += 1
    return stats


def selection_value(metrics: dict, settings: RunSettings, *, local: bool = False) -> float:
    key = settings.selection_metric.removeprefix("macro-") if local else settings.selection_metric
    if key not in metrics or not np.isfinite(metrics[key]):
        raise ValueError(f"Selección {key} indefinida: revisa escalas/cobertura del periodo de referencia o elige otra métrica explícitamente.")
    return float(metrics[key])


def validate_selection_coverage(manifest: dict, settings: RunSettings, mode: str) -> None:
    if manifest["schema_version"] < 4 or settings.selection_metric == "mae":
        return
    metric = settings.selection_metric.removeprefix("macro-")
    clients = manifest["clients"]
    undefined = sum(not client[f"validation_{metric}_defined"] for client in clients)
    if undefined:
        raise ValueError(
            f"Selección {settings.selection_metric} indefinida en {undefined}/{len(clients)} "
            f"estaciones con esta referencia de validación (modo {mode}). "
            "Ejecuta fd-airquality simulate/baseline/benchmark para descartar estas estaciones "
            "antes de entrenar, con el mismo panel para todos los candidatos."
        )
