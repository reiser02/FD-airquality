"""Seasonal error scales on real timestamps, plus legacy Darts metric wrappers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mase as darts_mase
from darts.metrics import rmsse as darts_rmsse

logger = logging.getLogger(__name__)


def seasonal_error_scales(series: pd.Series, seasonality: int) -> tuple[float, float, int]:
    """Return absolute/squared seasonal difference means and valid pair count.

    Only the endpoints must be observed. Datetime lags are elapsed hours, never
    positions in a compacted array. The caller supplies the reference interval.
    """
    values = series.to_numpy(dtype=float)
    if not isinstance(seasonality, int) or isinstance(seasonality, bool) or seasonality < 1:
        raise ValueError("La estacionalidad debe ser un entero positivo.")
    if isinstance(series.index, pd.DatetimeIndex):
        previous = series.reindex(series.index - pd.Timedelta(hours=seasonality)).to_numpy(dtype=float)
    else:
        if len(values) <= seasonality:
            return 0.0, 0.0, 0
        previous, values = values[:-seasonality], values[seasonality:]
    usable = np.isfinite(values) & np.isfinite(previous)
    differences = values[usable] - previous[usable]
    if not len(differences):
        return 0.0, 0.0, 0
    return float(np.abs(differences).mean()), float(np.square(differences).mean()), len(differences)


def compute_mase(
    actual: pd.Series | TimeSeries,
    pred: pd.Series | TimeSeries,
    insample: pd.Series | TimeSeries,
    *,
    seasonality_m: int,
) -> float:
    actual_s = actual.to_series() if isinstance(actual, TimeSeries) else actual
    pred_s = pred.to_series() if isinstance(pred, TimeSeries) else pred
    insample_s = insample.to_series() if isinstance(insample, TimeSeries) else insample

    aligned_idx = actual_s.index.intersection(pred_s.index)
    actual_s = actual_s.reindex(aligned_idx).astype(float)
    pred_s = pred_s.reindex(aligned_idx).astype(float)
    valid = np.isfinite(actual_s.to_numpy()) & np.isfinite(pred_s.to_numpy())
    actual_values = actual_s.to_numpy()[valid]
    pred_values = pred_s.to_numpy()[valid]
    if len(actual_values) == 0:
        return float("nan")

    insample_values = insample_s.astype(float).to_numpy()
    if len(insample_values) <= seasonality_m or np.isinf(insample_values).any():
        return float("nan")

    # Synthetic indices meet Darts' insample boundary contract without compacting history.
    split = len(insample_values)
    insample_ts = TimeSeries.from_times_and_values(pd.RangeIndex(split), insample_values)
    eval_index = pd.RangeIndex(split, split + len(actual_values))
    actual_ts = TimeSeries.from_times_and_values(eval_index, actual_values)
    pred_ts = TimeSeries.from_times_and_values(eval_index, pred_values)
    try:
        value = float(darts_mase(
            actual_series=actual_ts, pred_series=pred_ts, insample=insample_ts, m=int(seasonality_m),
        ))
        return value if np.isfinite(value) else float("nan")
    except Exception as exc:  # noqa: BLE001 -- undefined Darts scales remain missing.
        logger.warning("MASE no computable: %s", exc)
        return float("nan")


def compute_rmsse(
    actual: pd.Series | TimeSeries,
    pred: pd.Series | TimeSeries,
    insample: pd.Series | TimeSeries,
    *,
    seasonality_m: int,
) -> float:
    actual_s = actual.to_series() if isinstance(actual, TimeSeries) else actual
    pred_s = pred.to_series() if isinstance(pred, TimeSeries) else pred
    insample_s = insample.to_series() if isinstance(insample, TimeSeries) else insample

    aligned_idx = actual_s.index.intersection(pred_s.index)
    actual_s = actual_s.reindex(aligned_idx).astype(float)
    pred_s = pred_s.reindex(aligned_idx).astype(float)
    valid = np.isfinite(actual_s.to_numpy()) & np.isfinite(pred_s.to_numpy())
    actual_values = actual_s.to_numpy()[valid]
    pred_values = pred_s.to_numpy()[valid]
    if len(actual_values) == 0:
        return float("nan")

    insample_values = insample_s.astype(float).to_numpy()
    if len(insample_values) <= seasonality_m or np.isinf(insample_values).any():
        return float("nan")

    split = len(insample_values)
    insample_ts = TimeSeries.from_times_and_values(pd.RangeIndex(split), insample_values)
    eval_index = pd.RangeIndex(split, split + len(actual_values))
    actual_ts = TimeSeries.from_times_and_values(eval_index, actual_values)
    pred_ts = TimeSeries.from_times_and_values(eval_index, pred_values)
    try:
        value = float(darts_rmsse(
            actual_series=actual_ts, pred_series=pred_ts, insample=insample_ts, m=int(seasonality_m),
        ))
        return value if np.isfinite(value) else float("nan")
    except Exception as exc:  # noqa: BLE001 -- undefined Darts scales remain missing.
        logger.warning("RMSSE no computable: %s", exc)
        return float("nan")
