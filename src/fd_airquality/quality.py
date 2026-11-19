"""Coverage accounting and path-independent identities of evaluation targets."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pandas as pd


def target_signature(series: pd.Series, starts: np.ndarray, context: int, horizon: int,
                      reference_start: str, reference_end: str) -> str:
    digest = sha256()
    digest.update(series.index.asi8[starts + context].astype("<i8").tobytes())
    positions = starts[:, None] + context + np.arange(horizon)
    values = series.to_numpy(dtype="<f4")[positions]
    observed = np.isfinite(values)
    digest.update(observed.tobytes())
    digest.update(np.where(observed, values, 0).astype("<f4").tobytes())
    reference = series.loc[(series.index >= pd.Timestamp(reference_start)) & (series.index < pd.Timestamp(reference_end))]
    ref_values = reference.to_numpy(dtype="<f8")
    valid = np.isfinite(ref_values)
    digest.update(reference.index.asi8.astype("<i8").tobytes())
    digest.update(valid.tobytes())
    digest.update(np.where(valid, ref_values, 0).astype("<f8").tobytes())
    return digest.hexdigest()


def evaluation_identity(manifest: dict) -> str | None:
    if not all("test_signature" in client for client in manifest["clients"]):
        return None
    geometry = manifest["geometry"]
    definition = {"pollutant": manifest["pollutant"],
                  "geometry": {key: geometry[key] for key in ("horizon", "seasonality", "error_scale_hours",
                                                               "error_scale_policy", "error_scale_epsilon", "min_scale_pairs")},
                  "error_scale_reference": geometry.get("error_scale_reference", "pre-evaluation"),
                  "clients": [{"station": client["station"], "validation": client["validation_signature"],
                               "test": client["test_signature"]} for client in manifest["clients"]]}
    return sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def coverage_rows(series: pd.Series, config, candidates: dict, station: str) -> list[dict]:
    rows = []
    for name, lower, upper in (("train", config.train_start, config.validation_start),
                                ("validation", config.validation_start, config.test_start),
                                ("test", config.test_start, config.test_end)):
        lo = pd.Timestamp(lower) if lower else series.index[0]
        hi = pd.Timestamp(upper)
        calendar = pd.date_range(lo, hi, freq="h", inclusive="left")
        available = series.reindex(calendar)
        starts = candidates[f"{name}_starts"]
        hour_coverage = np.zeros(len(series) + 1, dtype=np.int64)
        target_end = min(len(series), int(series.index.searchsorted(hi)))
        np.add.at(hour_coverage, np.clip(starts + config.context, 0, target_end), 1)
        np.add.at(hour_coverage, np.clip(starts + config.context + config.horizon, 0, target_end), -1)
        targets = (np.cumsum(hour_coverage[:-1]) > 0) & np.isfinite(series.to_numpy())
        scheduled = max(0, (len(calendar) - config.horizon) // config.stride + 1) if name != "train" else None
        rows.append({"station": station, "split": name, "calendar_hours": len(calendar),
                     "observed_hours": int(available.notna().sum()),
                     "observed_target_hours_used": int(targets.sum()), "windows": len(starts),
                     "window_count_stage": "before_runtime_training_sampling" if name == "train" else "evaluation",
                     "epoch_window_budget_upper_bound": ((min(len(starts), config.max_train_windows)
                                                          if config.max_train_windows is not None else len(starts))
                                                         if name == "train" else None),
                     "scheduled_origins": scheduled,
                     "origin_coverage": len(starts) / scheduled if scheduled else None})
    return rows
