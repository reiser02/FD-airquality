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


def round_validation_bounds(geometry: dict, settings):
    """Fixed monitoring reserve, wholly available BEFORE round one."""
    if not controls_enabled(settings):
        return None
    end = round_cutoff(geometry, settings, 1)
    if end is None:
        return None
    start = end - pd.Timedelta(hours=max(settings.round_validation_hours, geometry["horizon"]))
    if geometry.get("train_start") and start < pd.Timestamp(geometry["train_start"]):
        raise ValueError("El histórico del fold no alcanza la reserva de monitorización de la primera ronda.")
    return start, end


def round_validation_view(artifact: dict, geometry: dict, settings):
    bounds = round_validation_bounds(geometry, settings)
    if bounds is None:
        return validation_view(artifact, geometry, settings)
    lower, cutoff = bounds
    times = artifact["timestamps"]
    starts = artifact["train_starts"]
    first, last = starts + geometry["context"], starts + geometry["context"] + geometry["horizon"] - 1
    keep = (starts >= 0) & (last < len(times))
    starts, first, last = starts[keep], first[keep], last[keep]
    starts = starts[(times[first] >= lower.value) & (times[last] < cutoff.value)]
    size = int(np.searchsorted(times, cutoff.value))
    local = artifact | {"validation_starts": starts, "values": artifact["values"][:size],
                        "timestamps": times[:size]}
    if not len(starts):
        raise ValueError("Sin ventanas observadas en la reserva de monitorización anterior a la primera ronda.")
    return WindowDataset(local, "validation", geometry["context"], geometry["horizon"])


def normalization_artifact(artifact: dict, geometry: dict, settings) -> dict:
    """Runtime copy only. Never fit a fixed scaler with an unreleased observation."""
    result = artifact.copy()
    reference_end = ""
    if settings.normalization != "fixed":
        result.update(mean=np.array(0.), scale=np.array(1.))
    elif geometry.get("split_mode") == "global":
        reference_end = settings.normalization_reference_end
        if not reference_end:
            end = pd.Timestamp(geometry["validation_start"])
            reference_end = str(end - pd.Timedelta(
                hours=max(0, settings.num_server_rounds - 1) * settings.arrival_period_hours,
            ))
            bounds = round_validation_bounds(geometry, settings)
            if bounds is not None:
                reference_end = str(bounds[0])
        cutoff = pd.Timestamp(reference_end)
        first_arrival = round_cutoff(geometry, settings, 1)
        if cutoff > (first_arrival or pd.Timestamp(geometry["validation_start"])):
            raise ValueError("La normalización fija utiliza datos aún no disponibles en la primera ronda.")
        bounds = round_validation_bounds(geometry, settings)
        if bounds is not None and cutoff > bounds[0]:
            raise ValueError("La normalización fija invade la reserva de monitorización.")
        mask = np.isfinite(artifact["values"]) & (artifact["timestamps"] < cutoff.value)
        if geometry.get("train_start"):
            mask &= artifact["timestamps"] >= pd.Timestamp(geometry["train_start"]).value
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
    monitoring_excluded = 0
    arrival_unobserved = 0
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
        bounds = round_validation_bounds(geometry, settings)
        if bounds is not None:
            first = times[starts + geometry["context"]]
            if padded:
                keep = (first < bounds[0].value) | (first >= bounds[1].value)
            else:
                last = times[starts + geometry["context"] + geometry["horizon"] - 1]
                keep = (last < bounds[0].value) | (first >= bounds[1].value)
            monitoring_excluded = int((~keep).sum())
            starts = starts[keep]
            if round_number == 1:
                target_end = min(target_end, int(np.searchsorted(times, bounds[0].value)))
        size = int(np.searchsorted(times, cutoff.value))
        local["values"], local["timestamps"] = artifact["values"][:size], times[:size]
        target_end = min(target_end, size)
    if padded:
        context_counts, target_counts = window_observation_counts(
            local["values"], starts, geometry["context"], geometry["horizon"], target_end,
        )
        keep = ((context_counts >= geometry.get("min_context_observed", 1))
                & (target_counts >= geometry.get("min_target_observed", 1)))
        arrival_unobserved = int((~keep).sum())
        starts = starts[keep]
    local["train_target_end"] = np.array(target_end)
    local["train_starts"] = starts
    dataset = filter_training_windows(WindowDataset(
        local, "train", geometry["context"], geometry["horizon"], allow_empty=True,
    ), settings)
    dataset.available_through = cutoff
    dataset.arrival_lower = lower
    dataset.timestamps = local["timestamps"]
    dataset.monitoring_excluded_windows = monitoring_excluded
    dataset.monitoring_interval = round_validation_bounds(geometry, settings) if cutoff is not None else None
    dataset.arrival_unobserved_windows = arrival_unobserved
    return dataset


def arrival_metrics(dataset) -> dict:
    metrics = {
        "candidate-windows": dataset.candidate_windows,
        "availability-excluded-windows": dataset.availability_excluded_windows,
        "minimum-context-observed": dataset.availability_minima[0],
        "minimum-target-observed": dataset.availability_minima[1],
        "monitoring-excluded-windows": dataset.monitoring_excluded_windows,
        "arrival-unobserved-windows": dataset.arrival_unobserved_windows,
    }
    if dataset.available_through is not None:
        metrics["available-through-ns"] = dataset.available_through.value
        if dataset.arrival_lower is not None:
            metrics["available-from-ns"] = dataset.arrival_lower.value
    return metrics
