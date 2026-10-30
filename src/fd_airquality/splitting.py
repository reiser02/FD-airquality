"""Retrospective station holdout and strictly earlier train/validation windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import extract_subseries

from fd_airquality.preprocessing import ensure_datetime_series


def global_window_starts(
    series: pd.Series, *, context: int, horizon: int, validation_start: str,
    test_start: str, test_end: str, train_stride: int = 1, stride: int = 6,
) -> dict[str, np.ndarray]:
    """Enumerate complete windows within global, half-open target intervals.

    Histories may overlap earlier splits; targets may not straddle a boundary.
    Evaluation origins use a global grid, not the beginning of each observed run.
    The train mask counts unique observations before any optional window cap.
    """
    index = series.index
    width = context + horizon
    cuts = [int(index.searchsorted(pd.Timestamp(date))) for date in (validation_start, test_start, test_end)]
    starts = {name: [] for name in ("train", "validation", "test")}
    train_mask = np.zeros(len(series), dtype=bool)
    hour = pd.Timedelta(hours=1)
    for first, end in _observed_runs(series):
        train_end = min(end, cuts[0])
        if train_end - first >= width:
            starts["train"].extend(range(first, train_end - width + 1, train_stride))
            train_mask[first:train_end] = True
        for name, lower, upper in (("validation", validation_start, test_start), ("test", test_start, test_end)):
            lo, hi = pd.Timestamp(lower), pd.Timestamp(upper)
            earliest = max(lo, index[first] + context * hour)
            latest = min(hi - horizon * hour, index[end - 1] - (horizon - 1) * hour)
            offset = int((earliest - lo) / hour)
            origin = lo + ((offset + stride - 1) // stride) * stride * hour
            if origin <= latest:
                for target in pd.date_range(origin, latest, freq=f"{stride}h"):
                    starts[name].append(int(index.get_loc(target)) - context)
    return {f"{name}_starts": np.asarray(values, dtype=np.int64) for name, values in starts.items()} | {"train_mask": train_mask}


def _observed_runs(series: pd.Series) -> list[tuple[int, int]]:
    observed = series.notna().to_numpy()
    breaks = np.r_[False, np.diff(series.index.asi8) != pd.Timedelta(hours=1).value] if isinstance(series.index, pd.DatetimeIndex) else np.zeros(len(series), dtype=bool)
    runs: list[tuple[int, int]] = []
    cur_start: int | None = None
    for i, is_obs in enumerate(observed):
        if cur_start is not None and breaks[i]:
            runs.append((cur_start, i))
            cur_start = None
        if is_obs and cur_start is None:
            cur_start = i
        elif not is_obs and cur_start is not None:
            runs.append((cur_start, i))
            cur_start = None
    if cur_start is not None:
        runs.append((cur_start, len(observed)))
    return runs


def select_holdout_window(
    series: pd.Series,
    *,
    holdout: int,
    context_len: int,
    train_min_len: int,
    validation_len: int = 48,
    freq: str = "h",
    host_min_len: int | None = None,
) -> dict | None:
    if min(holdout, context_len, train_min_len, validation_len) <= 0:
        raise ValueError("Las longitudes de train, validacion y test deben ser positivas")
    if host_min_len is not None and host_min_len <= 0:
        raise ValueError("host_min_len debe ser positivo")

    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    runs = _observed_runs(s)
    host_len = host_min_len or train_min_len + validation_len
    index = pd.DatetimeIndex(s.index)
    for start, end in reversed(runs):
        if end - start < context_len + holdout:
            continue
        test_target_pos = end - holdout
        test_start_pos = test_target_pos - context_len
        train_runs = _observed_runs(s.iloc[:test_target_pos])
        if not any(host_end - host_start >= host_len for host_start, host_end in train_runs):
            continue
        return {
            "train_index": index[:test_target_pos],
            "train_end": index[test_target_pos - 1],
            "test_context_start": index[test_start_pos],
            "source_run_start": index[start],
            "test_index": index[test_start_pos:end],
            "context_index": index[test_start_pos:test_target_pos],
            "test_target_index": index[test_target_pos:end],
            "test_target_start": index[test_target_pos],
            "test_target_end": index[end - 1],
            "test_target_hours": holdout,
        }
    return None


def split_train_val_subseries(
    train_ts: TimeSeries,
    *,
    input_chunk: int,
    size_k: int,
    validation_len: int = 48,
    validation_stride: int | None = None,
) -> tuple[list[TimeSeries], list[TimeSeries]] | None:
    validation_stride = size_k if validation_stride is None else validation_stride
    if min(input_chunk, size_k, validation_len, validation_stride) <= 0:
        raise ValueError("Las longitudes y el stride de validacion deben ser positivos")
    if validation_len < size_k:
        raise ValueError("validation_len debe ser >= size_k")
    width = input_chunk + size_k
    subseries = [ss for ss in extract_subseries(train_ts, min_gap_size=1) if len(ss) >= width]
    if not subseries:
        return None

    # Only the latest viable host and earlier blocks can enter training.
    for i in range(len(subseries) - 1, -1, -1):
        if len(subseries[i]) < width + validation_len:
            continue
        val_host = subseries[i]
        train_subs = [
            ss for ss in (*subseries[:i], val_host[:-validation_len]) if len(ss) >= width
        ]
        if not train_subs:
            return None
        first_target = len(val_host) - validation_len
        val_subs = []
        for target in range(first_target, len(val_host) - size_k + 1, validation_stride):
            window_start = target - input_chunk
            val_subs.append(val_host[window_start : window_start + width])
        return train_subs, val_subs
    return None
