"""Calendar-preserving seasonal scales and observed-label external scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_error_scales(series: pd.Series, seasonality: int) -> tuple[float, float, int]:
    """Average valid seasonal pairs, preserving real hourly lags across gaps.

    The caller defines the historical reference. No interpolation, compaction,
    or extension of that reference is performed.
    """
    values = series.to_numpy(dtype=float)
    if not isinstance(seasonality, int) or isinstance(seasonality, bool) or seasonality < 1:
        raise ValueError("La estacionalidad debe ser un entero positivo.")
    if isinstance(series.index, pd.DatetimeIndex):
        if not series.index.is_unique:
            raise ValueError("La referencia estacional contiene timestamps duplicados.")
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


def effective_error_scales(absolute: float, squared: float, pairs: int, *,
                           policy: str = "unfloored", epsilon: float = 0.01,
                           min_pairs: int = 1) -> tuple[float, float]:
    # Read old immutable manifests without silently changing their definition.
    policy = {"gift": "unfloored", "impermanent": "epsilon"}.get(policy, policy)
    if policy not in {"unfloored", "epsilon"} or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Política de escalas o epsilon inválidos.")
    if not np.isfinite([absolute, squared]).all() or absolute < 0 or squared < 0:
        raise ValueError("Las escalas deben ser finitas y no negativas.")
    if type(pairs) is not int or type(min_pairs) is not int or pairs < 0 or min_pairs < 1:
        raise ValueError("Los recuentos de pares deben ser enteros válidos.")
    if pairs < min_pairs:
        return 0.0, 0.0
    if policy == "epsilon":
        return max(absolute, epsilon), max(squared, epsilon**2)
    return absolute, squared


def _scaled_error(actual, pred, insample, seasonality_m: int, squared: bool,
                  policy: str, epsilon: float) -> float:
    actual = actual if isinstance(actual, pd.Series) else actual.to_series()
    pred = pred if isinstance(pred, pd.Series) else pred.to_series()
    insample = insample if isinstance(insample, pd.Series) else insample.to_series()
    if not actual.index.equals(pred.index):
        raise ValueError("Predicción y targets deben tener exactamente el mismo índice.")
    forecasts = pred.to_numpy(dtype=float)
    if not np.isfinite(forecasts).all():
        raise ValueError("Predicciones no finitas; no se pueden omitir para mejorar la métrica.")
    truth = actual.to_numpy(dtype=float)
    mask = np.isfinite(truth)
    absolute, square, pairs = seasonal_error_scales(insample, seasonality_m)
    absolute, square = effective_error_scales(absolute, square, pairs, policy=policy, epsilon=epsilon)
    denominator = square if squared else absolute
    if not mask.any() or denominator <= 0:
        return float("nan")
    errors = forecasts[mask] - truth[mask]
    return float(np.sqrt(np.square(errors).mean() / denominator) if squared else np.abs(errors).mean() / denominator)


def compute_mase(actual, pred, insample, *, seasonality_m: int,
                 policy: str = "unfloored", epsilon: float = 0.01) -> float:
    return _scaled_error(actual, pred, insample, seasonality_m, False, policy, epsilon)


def compute_rmsse(actual, pred, insample, *, seasonality_m: int,
                   policy: str = "unfloored", epsilon: float = 0.01) -> float:
    return _scaled_error(actual, pred, insample, seasonality_m, True, policy, epsilon)
