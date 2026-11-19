"""Masked quantile scores; no fabricated intervals for point-only models."""

from __future__ import annotations

import numpy as np


def quantile_statistics(actual: np.ndarray, quantiles: np.ndarray, levels) -> dict:
    levels = np.asarray(levels, dtype=float)
    if (quantiles.shape != actual.shape + (len(levels),) or len(levels) < 2
            or not np.isfinite(quantiles).all() or not np.isfinite(levels).all()
            or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1):
        raise ValueError("Cuantiles no finitos, niveles inválidos o forma incompatible.")
    observed = np.isfinite(actual)
    truth, forecasts = actual[observed], quantiles[observed]
    if not len(truth):
        raise ValueError("No hay targets observados para evaluar cuantiles.")
    delta = truth[:, None] - forecasts
    pinball = np.maximum(levels * delta, (levels - 1) * delta)
    # Trapezoidal quadrature over [0,1], with constant quantile tails. Retain
    # crossing diagnostics; do not silently sort or claim exact continuous CRPS.
    grid = np.r_[0., levels, 1.]
    errors = np.column_stack([delta[:, 0], delta, delta[:, -1]])
    integrand = np.maximum(grid * errors, (grid - 1) * errors)
    crps = 2 * np.trapezoid(integrand, grid, axis=1)
    stats = {"quantile-loss-sum": float((2 * pinball).sum()),
             "quantile-count": int(pinball.size),
             "quantile-crossings": int((np.diff(forecasts, axis=1) < 0).any(axis=1).sum()),
             "probabilistic-targets": len(truth),
             # Finite-grid approximation of 2 ∫ pinball dτ, not the
             # unnormalized training MQLoss nor exact continuous CRPS.
             "crps-sum": float(crps.sum()),
             "crps-count": len(truth),
             "crps": float(crps.mean())}
    median = np.flatnonzero(np.isclose(levels, .5))
    wis = .5 * np.abs(truth - forecasts[:, median[0]]) if len(median) else None
    intervals = 0
    for index, lower_level in enumerate(levels):
        if lower_level >= .5:
            continue
        upper = np.flatnonzero(np.isclose(levels, 1 - lower_level))
        if not len(upper):
            continue
        lo, hi = forecasts[:, index], forecasts[:, upper[0]]
        alpha = 2 * lower_level
        width = hi - lo
        interval_score = width + 2 / alpha * (np.maximum(lo - truth, 0) + np.maximum(truth - hi, 0))
        if wis is not None:
            wis += alpha / 2 * interval_score
        intervals += 1
        key = f"interval-{round((1 - alpha) * 100):02d}"
        stats.update({f"{key}-covered": int(((truth >= lo) & (truth <= hi)).sum()),
                      f"{key}-width-sum": float(width.sum()), f"{key}-count": len(truth)})
    if wis is not None and intervals:
        stats.update({"wis-sum": float((wis / (intervals + .5)).sum()), "wis-count": len(truth)})
    return stats


def sample_crps(actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Exact CRPS of an empirical distribution, on observed labels only.

    E|X-y| - E|X-X'|/2, including diagonal terms (empirical, not fair CRPS).
    """
    if (samples.shape[:-1] != actual.shape or samples.shape[-1] < 1
            or not np.isfinite(samples).all()):
        raise ValueError("Muestras CRPS no finitas o de forma incompatible.")
    observed = np.isfinite(actual)
    if not observed.any():
        raise ValueError("No hay targets observados para CRPS.")
    ordered = np.sort(samples[observed], axis=-1)
    size = ordered.shape[-1]
    weights = 2 * np.arange(1, size + 1) - size - 1
    return np.abs(ordered - actual[observed, None]).mean(axis=-1) - (
        ordered * weights
    ).sum(axis=-1) / size**2
