"""Station-local uncertainty; checkpoint selection and calibration never share targets."""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fd_airquality.data import WindowDataset
from fd_airquality.missing import forecast
from fd_airquality.models import AUTOENCODER_MODELS

ESTIMATED_LEVELS = tuple(i / 10 for i in range(1, 10))
POINT_MODELS = (*AUTOENCODER_MODELS, "linear-6lags", "holt", "nlinear-point")
SCALED_CRPS_PROTOCOL = "station_crps_divided_by_training_mase_scale_macro_v1"


def calibration_enabled(settings) -> bool:
    return bool(settings.probabilistic and settings.calibrate_point_forecasts)


def _validation_bounds(geometry, artifact) -> tuple[int, int]:
    if geometry.get("split_mode") == "global":
        return pd.Timestamp(geometry["validation_start"]).value, pd.Timestamp(geometry["test_start"]).value
    context = geometry["context"]
    times = artifact["timestamps"]
    return (int(times[artifact["validation_starts"][0] + context]),
            int(times[artifact["test_starts"][0] + context]))


def validation_view(artifact, geometry, settings, *, calibration=False):
    context, horizon = geometry["context"], geometry["horizon"]
    if not calibration_enabled(settings):
        if calibration:
            raise ValueError("La calibración no está habilitada.")
        return WindowDataset(artifact, "validation", context, horizon)
    start, end = _validation_bounds(geometry, artifact)
    hour = pd.Timedelta(hours=1).value
    cut = start + ((end - start) // hour // 2) * hour
    times = artifact["timestamps"]
    starts = artifact["validation_starts"]
    if calibration:
        keep = (times[starts + context] >= cut) & (times[starts + context + horizon - 1] < end)
        upper = end
    else:
        keep = times[starts + context + horizon - 1] < cut
        upper = cut
    # Discard every window that would use labels on both sides of the split.
    size = int(np.searchsorted(times, upper))
    local = artifact | {"validation_starts": starts[keep], "values": artifact["values"][:size],
                        "timestamps": times[:size]}
    if not calibration and not keep.any():
        raise ValueError("Sin ventanas de selección antes del corte de calibración; "
                         "amplía la validación o usa --calibrate-point-forecasts 0.")
    return WindowDataset(local, "validation", context, horizon, allow_empty=calibration)


@dataclass(frozen=True)
class Calibration:
    offsets: np.ndarray | None
    counts: np.ndarray
    method: str
    effective_counts: np.ndarray | None = None

    def metadata(self) -> dict:
        return {"method": self.method, "levels": list(ESTIMATED_LEVELS),
                "counts_by_lead": self.counts.tolist(),
                "effective_counts_by_lead": (self.effective_counts if self.effective_counts is not None
                                             else self.counts).tolist(),
                "offsets_original_units": self.offsets.tolist() if self.offsets is not None else None}


def residual_calibration(actual, prediction, min_pairs: int, *, target_positions=None) -> Calibration:
    if actual.ndim != 2 or actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Predicciones de calibración incompatibles o no finitas.")
    if target_positions is not None and target_positions.shape != actual.shape:
        raise ValueError("Identidades de targets de calibración incompatibles.")
    observed = np.isfinite(actual)
    counts = observed.sum(axis=0)
    effective, offsets = counts.copy(), []
    residuals = actual - prediction
    # If a lead is sparse, add the nearest leads from the SAME station. Keep
    # at most one error per physical target time (nearest lead wins).
    for lead, mask in enumerate(observed.T):
        errors = residuals[mask, lead]
        if len(errors) < min_pairs and target_positions is not None:
            by_time = {}
            for neighbor in sorted(range(len(counts)), key=lambda index: abs(index - lead)):
                keep = observed[:, neighbor]
                for time, error in zip(target_positions[keep, neighbor], residuals[keep, neighbor], strict=True):
                    by_time.setdefault(int(time), float(error))
                if len(by_time) >= min_pairs:
                    break
            errors = np.fromiter(by_time.values(), dtype=float)
            effective[lead] = len(errors)
        if len(errors) >= min_pairs:
            offsets.append(np.quantile(errors, ESTIMATED_LEVELS))
    method = "empirical_residual_quantiles_by_lead"
    if (effective > counts).any():
        method += "_with_neighbor_leads"
    return Calibration(np.stack(offsets) if len(offsets) == len(counts) else None, counts, method, effective)


@torch.no_grad()
def point_calibration(model, artifact, geometry, settings, device) -> Calibration:
    data = validation_view(artifact, geometry, settings, calibration=True)
    model.eval()
    forecasts = [forecast(model, x.to(device), settings).to("cpu", torch.float64).numpy()
                 for x, _ in DataLoader(data, batch_size=settings.batch_size)]
    points = np.concatenate(forecasts) if forecasts else np.empty((0, data.horizon))
    positions = data.starts[:, None] + data.context + np.arange(data.horizon)
    return residual_calibration(data.target_values(), points * data.scale + data.mean,
                                settings.calibration_min_pairs, target_positions=positions)


def seasonal_calibration(artifact, geometry, client, min_pairs: int) -> Calibration:
    """StatsForecast-style Gaussian intervals using only observed calendar pairs."""
    season, horizon = geometry["seasonality"], geometry["horizon"]
    times = pd.to_datetime(artifact["timestamps"])
    # The validation reference ends at the training cutoff under either scale policy.
    start, end = (pd.Timestamp(client[k]) for k in ("validation_scale_start", "validation_scale_end"))
    if geometry.get("train_start"):
        start = max(start, pd.Timestamp(geometry["train_start"]))
    series = pd.Series(artifact["values"], index=times).loc[(times >= start) & (times < end)]
    errors = series.to_numpy(dtype=float) - series.reindex(
        series.index - pd.Timedelta(hours=season)
    ).to_numpy(dtype=float)
    observed = errors[np.isfinite(errors)]
    offsets = None
    if len(observed) >= min_pairs:
        sigma = np.sqrt(np.square(observed).mean())
        step_sigma = sigma * np.sqrt(1 + np.arange(horizon) // season)
        standard_normal = NormalDist()
        offsets = step_sigma[:, None] * [standard_normal.inv_cdf(q) for q in ESTIMATED_LEVELS]
    return Calibration(offsets, np.full(horizon, len(observed)), "gaussian_seasonal_residuals_from_train")


def forecast_calibration(model, artifact, geometry, client, settings, device) -> Calibration | None:
    if not calibration_enabled(settings):
        return None
    if settings.model == "seasonal-naive" and "validation_scale_start" in client:
        return seasonal_calibration(artifact, geometry, client, settings.calibration_min_pairs)
    if settings.model in POINT_MODELS:
        return point_calibration(model, artifact, geometry, settings, device)
    return None


def calibration_options(model, artifact, geometry, client, settings, device,
                        output: Path | None, partition: int) -> dict:
    calibration = forecast_calibration(model, artifact, geometry, client, settings, device)
    if calibration is None:
        return {}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / f"calibration_{partition:03d}.json").write_text(json.dumps(
            calibration.metadata(), indent=2, allow_nan=False,
        ) + "\n", encoding="utf-8")
    return {"calibration": calibration}
