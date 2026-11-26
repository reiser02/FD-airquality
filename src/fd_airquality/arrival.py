"""Round-local data blocks and fold-local normalization; no inter-fold state."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from fd_airquality.availability import filter_training_windows
from fd_airquality.calibration import validation_view
from fd_airquality.data import WindowDataset
from fd_airquality.splitting import window_observation_counts
from fd_airquality.training_control import controls_enabled


def round_cutoff(geometry: dict, settings, round_number: int) -> pd.Timestamp | None:
    if not settings.arrival_period_hours or geometry.get("split_mode") != "global":
        return None
    if not 1 <= round_number <= settings.num_server_rounds:
        raise ValueError("Ronda fuera del calendario de llegada.")
    return pd.Timestamp(geometry["validation_start"]) - pd.Timedelta(
        hours=(settings.num_server_rounds - round_number) * settings.arrival_period_hours,
    )


def round_bounds(geometry: dict, settings, round_number: int):
    """Half-open target interval; old observations may still provide context."""
    cutoff = round_cutoff(geometry, settings, round_number)
    lower = None
    if cutoff is not None and round_number > 1:
        lower = cutoff - pd.Timedelta(hours=settings.arrival_period_hours)
    return lower, cutoff


def round_epoch_limit(geometry: dict, settings, round_number: int) -> int:
    """Only updates on new arrivals get the additional epoch cap."""
    if (round_number > 1 and settings.federated_update_epochs
            and round_cutoff(geometry, settings, round_number) is not None):
        return min(settings.local_epochs, settings.federated_update_epochs)
    return settings.local_epochs


def _round_validation_starts(artifact, geometry, lower, cutoff):
    times = artifact["timestamps"]
    starts = artifact["train_starts"]
    first, last = starts + geometry["context"], starts + geometry["context"] + geometry["horizon"] - 1
    keep = (starts >= 0) & (last < len(times))
    starts, first, last = starts[keep], first[keep], last[keep]
    return starts[(times[first] >= lower.value) & (times[last] < cutoff.value)]


def round_validation_bounds(geometry: dict, settings, round_number: int, *, allow_extension: bool = False):
    """The trailing validation interval for this round, clipped to its training block."""
    lower, cutoff = round_bounds(geometry, settings, round_number)
    if cutoff is None or not controls_enabled(settings) or round_epoch_limit(geometry, settings, round_number) == 1:
        return None
    horizon = pd.Timedelta(hours=geometry["horizon"])
    hours = settings.round_validation_max_hours if allow_extension else 0
    start = cutoff - max(horizon, pd.Timedelta(hours=max(hours, settings.round_validation_hours)))
    if lower is not None:
        start = max(start, lower)
    if geometry.get("train_start"):
        start = max(start, pd.Timestamp(geometry["train_start"]))
    return (start, cutoff) if cutoff - start >= horizon else None


def _round_validation_plan(artifact, geometry, settings, round_number):
    bounds = round_validation_bounds(geometry, settings, round_number)
    if bounds is None:
        return None
    start, cutoff = bounds
    earliest, _ = round_validation_bounds(geometry, settings, round_number, allow_extension=True)
    starts = _round_validation_starts(artifact, geometry, earliest, cutoff)
    if not len(starts):
        return None
    origins = artifact["timestamps"][starts + geometry["context"]]
    # Extend only when the nominal tail has no validation windows.
    start = min(start, pd.Timestamp(origins[-1]))
    return (start, cutoff), starts[origins >= start.value]


def round_validation_view(artifact: dict, geometry: dict, settings, training):
    if training.available_through is None:
        return validation_view(artifact, geometry, settings)
    if training.round_validation_interval is None:
        return None
    times = artifact["timestamps"]
    size = int(np.searchsorted(times, training.available_through.value))
    local = artifact | {"validation_starts": training.round_validation_starts,
                        "values": artifact["values"][:size], "timestamps": times[:size]}
    return WindowDataset(local, "validation", geometry["context"], geometry["horizon"])


def normalization_artifact(artifact: dict, geometry: dict, settings) -> dict:
    """Runtime copy only. Never fit a fixed scaler with an unreleased observation."""
    result = artifact.copy()
    reference_end = ""
    if settings.normalization != "fixed":
        result.update(mean=np.array(0.), scale=np.array(1.))
    elif geometry.get("split_mode") == "global":
        first_arrival = round_cutoff(geometry, settings, 1)
        bounds = training_view(artifact, geometry, settings, 1).round_validation_interval
        allowed = np.isfinite(artifact["values"])
        if geometry.get("train_start"):
            allowed &= artifact["timestamps"] >= pd.Timestamp(geometry["train_start"]).value
        validation_start = bounds[0] if bounds is not None else None
        reference_end = settings.normalization_reference_end or str(
            validation_start or first_arrival or pd.Timestamp(geometry["validation_start"]),
        )
        cutoff = pd.Timestamp(reference_end)
        if cutoff > (first_arrival or pd.Timestamp(geometry["validation_start"])):
            raise ValueError("La normalización fija utiliza datos aún no disponibles en la primera ronda.")
        if validation_start is not None and cutoff > validation_start:
            raise ValueError("La normalización fija invade la reserva de validación de la primera ronda.")
        mask = allowed & (artifact["timestamps"] < cutoff.value)
        values = artifact["values"][mask].astype(np.float64)
        if not len(values):
            raise ValueError("No hay observaciones autorizadas para ajustar la normalización fija.")
        scale = float(values.std())
        result.update(mean=np.array(values.mean()), scale=np.array(scale if scale >= 1e-8 else 1.))
        result["normalization_reference_mask"] = mask
    digest = sha256()
    digest.update(settings.normalization.encode())
    digest.update(reference_end.encode())
    digest.update(np.asarray([result["mean"], result["scale"]], dtype="<f8").tobytes())
    if settings.normalization == "fixed":
        mask = (result["normalization_reference_mask"] if "normalization_reference_mask" in result
                else artifact["train_mask"])
        digest.update(artifact["timestamps"][mask].astype("<i8").tobytes())
        digest.update(artifact["values"][mask].astype("<f4").tobytes())
    result["normalization_sha256"] = digest.hexdigest()
    return result


def audit_normalization(artifact: dict, settings, partition: int, output: Path | None) -> None:
    """Record only a hash of this fold's scaler, never reuse another fold's fit."""
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / f"normalization_{partition:03d}.json").write_text(
            json.dumps({"sha256": artifact["normalization_sha256"],
                        "normalization": settings.normalization}, indent=2) + "\n",
        )


def training_view(artifact: dict, geometry: dict, settings, round_number: int | None = None):
    """Return the full admissible pool for this round, before epoch sampling."""
    local = artifact.copy()
    lower, cutoff = round_bounds(geometry, settings, round_number) if round_number is not None else (None, None)
    starts = artifact["train_starts"]
    padded = str(artifact.get("window_padding", "none")) != "none"
    target_end = int(artifact.get("train_target_end", len(artifact["values"])))
    if cutoff is not None:
        times = artifact["timestamps"]
        # Padding may extend the nominal horizon, never its supervised labels.
        final = starts + geometry["context"] + (0 if padded else geometry["horizon"] - 1)
        starts = starts[times[final] < cutoff.value]
        if lower is not None:
            # Only forecast targets must be new. Their context may precede the
            # block; a padded horizon can extend right, never back into old labels.
            starts = starts[times[starts + geometry["context"]] >= lower.value]
        size = int(np.searchsorted(times, cutoff.value))
        local["values"], local["timestamps"] = artifact["values"][:size], times[:size]
        target_end = min(target_end, size)
    plan = _round_validation_plan(artifact, geometry, settings, round_number) if cutoff is not None else None
    validation_excluded = 0
    if plan is not None:
        bounds, _ = plan
        training_end = min(target_end, int(np.searchsorted(times, bounds[0].value)))
        last = starts + geometry["context"] + (0 if padded else geometry["horizon"] - 1)
        training_starts = starts[last < training_end]
        dataset, arrival_unobserved = _admissible_training(
            local, geometry, settings, training_starts, training_end, padded,
        )
        if len(dataset):
            validation_excluded = len(starts) - len(training_starts)
        else:
            plan = None
    if plan is None:
        dataset, arrival_unobserved = _admissible_training(local, geometry, settings, starts, target_end, padded)
    dataset.available_through = cutoff
    dataset.arrival_lower = lower
    dataset.round_validation_interval, dataset.round_validation_starts = plan if plan is not None else (None, ())
    dataset.round_validation_excluded_windows = validation_excluded
    dataset.arrival_unobserved_windows = arrival_unobserved
    return dataset


def _admissible_training(local, geometry, settings, starts, target_end, padded):
    arrival_unobserved = 0
    if padded:
        context_counts, target_counts = window_observation_counts(
            local["values"], starts, geometry["context"], geometry["horizon"], target_end,
        )
        keep = ((context_counts >= geometry.get("min_context_observed", 1))
                & (target_counts >= geometry.get("min_target_observed", 1)))
        arrival_unobserved = int((~keep).sum())
        starts = starts[keep]
    local = local | {"train_target_end": np.array(target_end), "train_starts": starts}
    dataset = filter_training_windows(WindowDataset(
        local, "train", geometry["context"], geometry["horizon"], allow_empty=True,
    ), settings)
    return dataset, arrival_unobserved


def arrival_metrics(dataset) -> dict:
    metrics = {
        "candidate-windows": dataset.candidate_windows,
        "availability-excluded-windows": dataset.availability_excluded_windows,
        "minimum-context-observed": dataset.availability_minima[0],
        "minimum-target-observed": dataset.availability_minima[1],
        "round-validation-windows": len(dataset.round_validation_starts),
        "round-validation-excluded-windows": dataset.round_validation_excluded_windows,
        "arrival-unobserved-windows": dataset.arrival_unobserved_windows,
    }
    if dataset.available_through is not None:
        metrics["available-through-ns"] = dataset.available_through.value
        if dataset.round_validation_interval is not None:
            start, end = dataset.round_validation_interval
            metrics.update({"round-validation-from-ns": start.value, "round-validation-through-ns": end.value})
        if dataset.arrival_lower is not None:
            metrics["available-from-ns"] = dataset.arrival_lower.value
    return metrics
