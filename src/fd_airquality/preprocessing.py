"""Five-minute quality filtering, hourly aggregation, and gap-preserving blocks."""

from __future__ import annotations

import pandas as pd

DETECTION_LIMITS = {"CO": 0.0, "NO2": 0.0, "O3": 0.0}
MIN_RUN = 6
MIN_USEFUL = 1
ZERO_RUN_HOURS = 24


def ensure_datetime_series(series: pd.Series, freq: str, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"La serie '{name}' debe ser pd.Series; recibido: {type(series)}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"La serie '{name}' debe tener DatetimeIndex")
    out = series.sort_index().copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.asfreq(freq)
    out.name = str(series.name) if series.name is not None else name
    return out.astype(float)


def frozen_mask(series: pd.Series, min_run: int = MIN_RUN, *, quality_mode: str = "retrospective") -> pd.Series:
    valid = series.notna()
    same_as_previous = valid & series.shift().notna() & series.eq(series.shift())
    block = (~same_as_previous).cumsum()
    if quality_mode not in {"causal", "retrospective"}:
        raise ValueError("quality_mode debe ser causal o retrospective.")
    # Online detection cannot retroactively erase the first min_run-1 readings
    # using observations which had not arrived when that hour was completed.
    run_len = valid.groupby(block).cumcount() + 1 if quality_mode == "causal" else valid.groupby(block).transform("sum")
    # Zero plateaus use a separate hourly threshold after aggregation.
    return valid & series.ne(0) & run_len.ge(min_run)


def hourly_mean(
    df: pd.DataFrame,
    pollutant: str,
    *,
    min_run: int = MIN_RUN,
    min_useful: int = MIN_USEFUL,
    quality_mode: str = "retrospective",
) -> pd.DataFrame:
    col = df.columns[0]
    series = ensure_datetime_series(df[col], freq="5min", name=col)
    hour = series.index.floor("h")
    is_useful = (series >= DETECTION_LIMITS[pollutant]) & ~frozen_mask(series, min_run=min_run, quality_mode=quality_mode)
    n_useful = is_useful.groupby(hour).sum()
    mean_useful = series.where(is_useful).groupby(hour).mean()
    value = mean_useful.where(n_useful >= min_useful)
    value.index = pd.DatetimeIndex(value.index)
    return value.asfreq("h").rename(col).to_frame()


def preprocess(
    dfs: list[pd.DataFrame],
    pollutant: str,
    *,
    min_run: int = MIN_RUN,
    min_useful: int = MIN_USEFUL,
    zero_run_hours: int = ZERO_RUN_HOURS,
    quality_mode: str = "retrospective",
) -> tuple[list[pd.DataFrame], list[int]]:
    if not isinstance(zero_run_hours, int) or isinstance(zero_run_hours, bool) or zero_run_hours < 1:
        raise ValueError("zero_run_hours debe ser un entero positivo.")
    processed, frozen_counts = [], []
    for df in dfs:
        hourly = hourly_mean(df, pollutant, min_run=min_run, min_useful=min_useful, quality_mode=quality_mode)
        col = hourly[hourly.columns[0]]
        # Missing hours pause the counter without contributing zero observations.
        # Only an observed, nonzero hourly mean resets evidence of a zero plateau.
        zero = col.eq(0)
        groups = (col.notna() & col.ne(0)).cumsum()
        zero_length = (zero.groupby(groups).cumsum() if quality_mode == "causal"
                       else zero.groupby(groups).transform("sum"))
        # Causal detection preserves the first zero_run_hours-1 observed zero hours.
        is_frozen = (col.ne(0) & col.eq(col.shift())) | (zero & zero_length.ge(zero_run_hours))
        frozen_counts.append(int(is_frozen.sum()))
        hourly.loc[is_frozen, hourly.columns[0]] = pd.NA
        processed.append(hourly)
    return processed, frozen_counts


def contiguous_observed_segments(series: pd.Series, min_len: int = 1) -> list[pd.Series]:
    observed = series.notna()
    boundaries = observed != observed.shift()
    if isinstance(series.index, pd.DatetimeIndex):
        boundaries |= series.index.to_series().diff().ne(pd.Timedelta(hours=1))
    block_ids = boundaries.cumsum()
    return [
        series.loc[block.index]
        for _, block in observed.groupby(block_ids)
        if bool(block.iloc[0]) and len(block) >= int(min_len)
    ]


def observed_blocks(series: pd.Series) -> pd.DataFrame:
    segments = contiguous_observed_segments(series)
    return pd.DataFrame(
        {
            "start": [segment.index[0] for segment in segments],
            "end": [segment.index[-1] for segment in segments],
            "hours": [len(segment) for segment in segments],
        },
        columns=["start", "end", "hours"],
    )
