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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from fd_airquality.batching import CompleteBatchSampler, training_padding_counts
from fd_airquality.data import WindowDataset
from fd_airquality.metrics import compute_mase, compute_rmsse
from fd_airquality.missing import (
    forecast,
    forecast_distribution_with_raw,
    resolve_missing_policy,
)
from fd_airquality.models import ForecastAutoencoder, make_model, shared_state  # noqa: F401
from fd_airquality.run_logging import report_progress
from fd_airquality.settings import RunSettings
from fd_airquality.training_control import (
    ContextSupervisionDataset,
    TrainingControl,
    controls_enabled,
)


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
    if settings.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
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
    *,
    max_train_windows: int | None = None,
    validation_data: Dataset | None = None,
    group_metrics: list[dict[str, float | int]] | None = None,
) -> dict[str, float | int]:
    started = time.perf_counter()
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No hay parámetros entrenables; zero-shot no debe entrar en train_local.")
    if not len(dataset):
        raise ValueError("No hay ventanas admisibles de entrenamiento tras aplicar disponibilidad/llegada.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reference = [parameter.detach().clone() for parameter in parameters] if proximal_mu else []
    grouped = isinstance(dataset, ConcatDataset)
    children = dataset.datasets if grouped else [dataset]
    restrict_reconstruction = any(getattr(child, "arrival_lower", None) is not None
                                  or getattr(child, "monitoring_interval", None) is not None
                                  for child in children)
    loader_dataset = dataset
    if isinstance(model, ForecastAutoencoder) and settings.reconstruction_weight and restrict_reconstruction:
        supervised = [ContextSupervisionDataset(child) for child in children]
        loader_dataset = ConcatDataset(supervised) if grouped else supervised[0]
    group_sizes = [len(child) for child in children] if grouped else None
    sampler = CompleteBatchSampler(
        len(dataset), settings.batch_size, torch.Generator().manual_seed(seed),
        max_windows=max_train_windows, group_sizes=group_sizes, pad_last_batch=bool(settings.pad_last_batch),
    )
    loader = DataLoader(loader_dataset, batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    visited = np.zeros(len(dataset), dtype=bool) if sampler.epoch_size < len(dataset) else None
    if group_metrics is not None:
        if not grouped:
            raise ValueError("La auditoría por grupo requiere un ConcatDataset.")
        observed_by_draw = np.empty(sampler.epoch_size, dtype=np.int64)
        group_targets = np.zeros(len(children), dtype=np.int64)
    # Local optimizer state is deliberately reset at each communication round.
    optimizer = torch.optim.AdamW(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    control = None
    if controls_enabled(settings):
        if validation_data is None or not len(validation_data):
            raise ValueError("Early stopping/scheduler requieren una validación interna observada, nunca el test.")
        control = TrainingControl(model, optimizer, settings)
        control.observe(0, training_validation_mse(model, validation_data, settings, device))
    microbatch_size = _microbatch_size(settings, device)
    optimizer_steps = 0
    window_occurrences = selected_targets = 0
    total_squared_error, total_objective, targets = 0.0, 0.0, 0
    for epoch in range(1, settings.local_epochs + 1):
        report_progress("train", f"epoch={epoch}/{settings.local_epochs} batches=0/{len(loader)} device={device} "
                        f"effective_batch_size={settings.batch_size} microbatch_size={microbatch_size}")
        last_report = time.monotonic()
        for batch, values in enumerate(loader, start=1):
            inputs, actual = values[:2]
            window_occurrences += len(inputs)
            context_allowed = values[2] if len(values) == 3 else torch.ones_like(inputs, dtype=torch.bool)
            observed = torch.isfinite(actual)
            observed_targets = int(observed.sum())
            # Only the suffix of the final batch can contain replacement draws.
            first_draw = (batch - 1) * settings.batch_size
            remaining = sampler.epoch_size - first_draw
            original = observed[:remaining]
            selected_targets += observed_targets if remaining >= len(actual) else int(original.sum())
            if group_metrics is not None:
                observed_by_draw[first_draw:first_draw + len(original)] = original.sum(dim=1).numpy()
            if not observed_targets:
                continue
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(inputs), microbatch_size):
                micro_inputs = inputs[start:start + microbatch_size].to(device)
                micro_actual = actual[start:start + microbatch_size].to(device)
                mask = torch.isfinite(micro_actual)
                if not mask.any():
                    continue
                policy = resolve_missing_policy(settings.model, settings.missing_policy, settings.missing_policies)
                if hasattr(model, "training_objective"):
                    objective, squared = model.training_objective(
                        micro_inputs, micro_actual, missing_policy=policy,
                    )
                    loss = objective * int(mask.sum()) / observed_targets
                else:
                    prediction = forecast(model, micro_inputs, settings)
                    if prediction.shape != micro_actual.shape or not torch.isfinite(prediction).all():
                        raise ValueError("Forma de predicción incompatible o predicciones no finitas.")
                    squared = (prediction[mask] - micro_actual[mask]).square().sum()
                    loss = squared / observed_targets
                total_objective += float(loss.detach()) * observed_targets
                if isinstance(model, ForecastAutoencoder) and settings.reconstruction_weight:
                    filled = model.prepare_context(micro_inputs, policy)
                    centered = filled - filled[:, -1:]
                    reconstruction = model.reconstruction(model.encoder(centered))
                    context_mask = torch.isfinite(micro_inputs) & context_allowed[start:start + microbatch_size].to(device)
                    context_count = int((torch.isfinite(inputs) & context_allowed).sum())
                    loss = loss + settings.reconstruction_weight * (reconstruction[context_mask] - centered[context_mask]).square().sum() / max(1, context_count)
                if not torch.isfinite(loss):
                    raise ValueError("Pérdida local no finita; se aborta para no contaminar los pesos globales.")
                if not loss.requires_grad:
                    raise RuntimeError("El modelo cortó autograd: los adaptadores no reciben gradientes.")
                # Weight by observed targets, including uneven microbatches.
                # Backward releases each activation graph.
                loss.backward()
                total_squared_error += float(squared.detach())
                targets += int(mask.sum())
                del loss, squared
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
        if visited is not None:
            visited[sampler.selected_indices.numpy()] = True
        if group_metrics is not None:
            selected_groups = np.searchsorted(
                dataset.cumulative_sizes, sampler.selected_indices.numpy(), side="right",
            )
            np.add.at(group_targets, selected_groups, observed_by_draw)
        if not targets:
            raise ValueError("No hay targets de entrenamiento observados; no se puede optimizar.")
        report_progress("train", f"epoch={epoch}/{settings.local_epochs} batches={len(loader)}/{len(loader)} "
                        f"mse_normalized={total_squared_error / targets:.6g}")
        if control is not None:
            score = training_validation_mse(model, validation_data, settings, device)
            stop = control.observe(epoch, score)
            report_progress("training-validation", f"epoch={epoch} mse={score:.6g} "
                            f"lr={optimizer.param_groups[0]['lr']:.6g} patience={control.bad_epochs}/"
                            f"{settings.early_stopping_patience} stop={stop}")
            if stop:
                break
    epochs_completed = epoch
    if control is not None:
        control.restore()
    if any(not torch.isfinite(value).all() for value in shared_state(model).values()):
        raise ValueError("El entrenamiento produjo pesos no finitos.")
    padded_context, padded_targets = training_padding_counts(dataset, visited)
    metrics = {
        "num-examples": sampler.epoch_size, "train-mse-normalized": total_squared_error / targets,
        "admissible-windows": len(dataset),
        "unique-training-windows": len(dataset) if visited is None else int(visited.sum()),
        "train-seconds": time.perf_counter() - started,
        "trainable-parameters": sum(parameter.numel() for parameter in parameters),
        "peak-process-rss-mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "effective-batch-size": settings.batch_size, "microbatch-size": microbatch_size,
        "optimizer-steps": optimizer_steps,
        "num-targets": selected_targets / epochs_completed,
        "selected-target-occurrences": selected_targets,
        "sampled-window-occurrences": window_occurrences,
        "sampled-target-occurrences": targets,
        "replacement-window-occurrences": window_occurrences - sampler.epoch_size * epochs_completed,
        "padded-context-slots": padded_context,
        "padded-target-slots": padded_targets,
        "epochs-completed": epochs_completed,
        "early-stopped": int(epochs_completed < settings.local_epochs),
        "learning-rate-initial": settings.learning_rate,
        "learning-rate-final": float(optimizer.param_groups[0]["lr"]),
        "learning-rate-reductions": control.reductions if control is not None else 0,
        "best-epoch": control.best_epoch if control is not None else epochs_completed,
    }
    if control is not None:
        metrics["training-validation-mse"] = control.best_score
        metrics["training-validation-windows"] = len(validation_data)
    metrics["train-objective-normalized"] = total_objective / targets
    if hasattr(dataset, "candidate_windows"):
        metrics.update({"candidate-windows": dataset.candidate_windows,
                        "availability-excluded-windows": dataset.availability_excluded_windows})
    if isinstance(dataset, WindowDataset) and getattr(dataset, "available_through", None) is not None:
        first, last = dataset.target_time_bounds(dataset.starts[visited] if visited is not None else None)
        metrics.update({"train-first-target-ns": first, "train-last-target-ns": last})
    if device.type == "cuda":
        metrics["peak-cuda-allocated-mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    if group_metrics is not None:
        offset = 0
        for child, count, support in zip(children, sampler.group_samples, group_targets, strict=True):
            end = offset + len(child)
            used = visited[offset:end] if visited is not None else None
            row = {
                "num-examples": count, "num-targets": int(support) / epochs_completed,
                "admissible-windows": len(child), "epochs-completed": epochs_completed if count else 0,
                "selected-target-occurrences": int(support),
                "unique-training-windows": len(child) if used is None else int(used.sum()),
            }
            if count and getattr(child, "available_through", None) is not None:
                first, last = child.target_time_bounds(child.starts[used] if used is not None else None)
                row.update({"train-first-target-ns": first, "train-last-target-ns": last})
            group_metrics.append(row)
            offset = end
    return metrics


@torch.no_grad()
def training_validation_mse(model, dataset, settings, device) -> float:
    """Pool observed squared residuals in original units before dividing by support."""
    was_training = model.training
    model.eval()
    try:
        error, count = _training_validation_totals(model, dataset, settings, device)
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("La validación de entrenamiento no tiene targets observados.")
    return error / count


def _training_validation_totals(model, dataset, settings, device):
    if isinstance(dataset, ConcatDataset):
        totals = [_training_validation_totals(model, child, settings, device) for child in dataset.datasets]
        return sum(row[0] for row in totals), sum(row[1] for row in totals)
    error, count = 0., 0
    loader = DataLoader(dataset, batch_size=_microbatch_size(settings, device), num_workers=0,
                        generator=torch.Generator().manual_seed(settings.seed))
    offset = 0
    for inputs, actual in loader:
        prediction = forecast(model, inputs.to(device), settings).cpu()
        if prediction.shape != actual.shape or not torch.isfinite(prediction).all():
            raise ValueError("Predicciones de monitorización incompatibles o no finitas.")
        if isinstance(dataset, WindowDataset):
            starts = dataset.starts[offset:offset + len(actual)]
            actual = torch.from_numpy(dataset.target_values(starts).astype(np.float64))
            prediction = prediction.double() * dataset.scale + dataset.mean
            scale = 1.
        else:
            scale = getattr(dataset, "scale", 1.)
        mask = torch.isfinite(actual)
        error += float((prediction[mask].double() - actual[mask]).square().sum()) * scale**2
        count += int(mask.sum())
        offset += len(actual)
    return error, count


@torch.no_grad()
def evaluate_local(
    model: nn.Module,
    dataset: WindowDataset,
    settings: RunSettings,
    device: torch.device,
    artifact: dict[str, np.ndarray],
    seasonality: int,
    *,
    calibration=None,
) -> dict[str, float | int]:
    model.to(device).eval()
    predictions, actuals = [], []
    missing_context = 0
    quantile_batches, raw_quantile_batches, quantile_levels = [], [], None
    row_offset = 0
    loader = DataLoader(dataset, batch_size=_microbatch_size(settings, device), num_workers=0)
    last_report = time.monotonic()
    for batch, (inputs, actual) in enumerate(loader, start=1):
        if settings.probabilistic:
            predicted, quantiles, levels, raw_quantiles = forecast_distribution_with_raw(
                model, inputs.to(device), settings,
            )
            if quantiles is not None:
                if quantile_levels is not None and list(levels) != list(quantile_levels):
                    raise ValueError("Los niveles de cuantiles cambiaron entre lotes.")
                quantile_levels = levels
                quantile_batches.append(quantiles.cpu().numpy().astype(np.float64) * dataset.scale + dataset.mean)
                if raw_quantiles is not None:
                    raw_quantile_batches.append(raw_quantiles.cpu().numpy().astype(np.float64) * dataset.scale + dataset.mean)
        else:
            predicted = forecast(model, inputs.to(device), settings)
        if predicted.shape != actual.shape or not torch.isfinite(predicted).all():
            raise ValueError("Forma de predicción incompatible o predicciones no finitas.")
        predictions.append(predicted.cpu().numpy().astype(np.float64) * dataset.scale + dataset.mean)
        starts = dataset.starts[row_offset:row_offset + len(actual)]
        row_offset += len(actual)
        positions = starts[:, None] + dataset.context + np.arange(dataset.horizon)
        # Score the canonical labels, not a normalization round trip whose
        # rounding could depend on the candidate's training scaler.
        actuals.append(artifact["values"][positions].astype(np.float64))
        missing_context += int(torch.isnan(inputs).sum())
        if time.monotonic() - last_report >= settings.heartbeat_seconds:
            report_progress(dataset.split, f"batches={batch}/{len(loader)} device={device}")
            last_report = time.monotonic()
    prediction, actual = map(np.concatenate, (predictions, actuals))
    if not np.isfinite(prediction).all():
        raise ValueError("Predicciones no finitas.")
    observed = np.isfinite(actual)
    count = int(observed.sum())
    if not count:
        raise ValueError("No hay targets observados para evaluar.")
    error = np.where(observed, prediction - actual, 0.0)
    days = (pd.to_datetime(artifact["timestamps"][dataset.starts + dataset.context]).strftime("%Y%m%d").to_numpy()
            if dataset.split == "test" and "timestamps" in artifact else None)
    signed_error = float(error.sum())
    negative = int((prediction < 0).sum())
    stats: dict[str, float | int] = {
        "num-examples": len(dataset),
        "num-targets": count,
        "num-target-slots": actual.size,
        "missing-targets": actual.size - count,
        "missing-context-values": missing_context,
        "signed-error-sum": signed_error,
        "bias": signed_error / count,
        "negative-predictions": negative,
        "negative-prediction-fraction": negative / actual.size,
        "mase-sum": 0.0,
        "mase-count": 0,
        "rmsse-sum": 0.0,
        "rmsse-count": 0,
    }
    scale = float(artifact.get("validation_mase_scale", 0.))
    stats["scaled-crps-eligible"] = int(np.isfinite(scale) and scale > 0)
    if calibration is not None:
        stats["calibration-min-lead-support"] = int(calibration.counts.min())
        if calibration.effective_counts is not None:
            stats["calibration-min-effective-support"] = int(calibration.effective_counts.min())
            stats["calibration-neighbor-leads"] = int((calibration.effective_counts > calibration.counts).sum())
        stats["calibration-available"] = int(calibration.offsets is not None)
    provided = bool(quantile_batches or (calibration is not None and calibration.offsets is not None))
    stats["probabilistic-available"] = int(provided)
    if provided:
        from fd_airquality.calibration import ESTIMATED_LEVELS
        from fd_airquality.probabilistic import quantile_statistics

        raw = np.concatenate(raw_quantile_batches) if raw_quantile_batches else None
        if quantile_batches:
            values = np.concatenate(quantile_batches)
        else:
            values = prediction[..., None] + calibration.offsets
            quantile_levels = ESTIMATED_LEVELS
        stats.update(quantile_statistics(actual, values, quantile_levels, raw_quantiles=raw,
                                         seasonal_scale=scale, origin_days=days))
    if days is not None:
        for day in np.unique(days):
            stats[f"block-{day}-observed-target-count"] = int(observed[days == day].sum())
    lead_counts = observed.sum(axis=0)
    for lead, support in enumerate(lead_counts, start=1):
        stats[f"lead-{lead:02d}-count"] = int(support)
    prefix = f"{dataset.split}_" if f"{dataset.split}_mase_scale" in artifact else ""
    if f"{prefix}mase_scale" in artifact and f"{prefix}rmsse_scale" in artifact:
        if prefix:
            stats["error-scale-pairs"] = int(artifact[f"{prefix}scale_pairs"])
        origin_counts = observed.sum(axis=1)
        valid = origin_counts > 0
        for name, transform in (("mase", np.abs), ("rmsse", np.square)):
            denominator = float(artifact[f"{prefix}{name}_scale"])
            if not np.isfinite(denominator) or denominator <= 0:
                continue
            scaled_error = transform(error) / denominator
            by_origin = scaled_error.sum(axis=1)
            total = float(by_origin.sum())
            per_origin = by_origin[valid] / origin_counts[valid]
            stats[f"{name}-sum"] = float((per_origin if name == "mase" else np.sqrt(per_origin)).sum())
            stats[f"{name}-count"] = len(per_origin)
            stats[name] = total / count if name == "mase" else float(np.sqrt(total / count))
            stats[f"{name}-scaled-error-sum"] = total
            stats[f"{name}-target-count"] = count
            for lead, numerator in enumerate(scaled_error.sum(axis=0), start=1):
                stats[f"lead-{lead:02d}-{name}-scaled-error-sum"] = float(numerator)
                stats[f"lead-{lead:02d}-{name}-target-count"] = int(lead_counts[lead - 1])
            if days is not None:
                for day in np.unique(days):
                    rows = days == day
                    stats[f"block-{day}-{name}-scaled-error-sum"] = float(by_origin[rows].sum())
                    stats[f"block-{day}-{name}-target-count"] = int(origin_counts[rows].sum())
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
    if manifest["schema_version"] < 4:
        return
    metric = settings.selection_metric.removeprefix("macro-")
    clients = manifest["clients"]
    undefined = sum(not client[f"validation_{metric}_defined"] for client in clients)
    if undefined:
        raise ValueError(
            f"Selección {settings.selection_metric} indefinida en {undefined}/{len(clients)} "
            f"estaciones con esta referencia de validación (modo {mode}). "
            "Ejecuta fd-airquality simulate/baseline/backtest para descartar estas estaciones "
            "antes de entrenar, con el mismo panel para todos los candidatos."
        )
