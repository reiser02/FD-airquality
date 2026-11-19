"""Paired calendar-block uncertainty, resampling all stations jointly."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def paired_block_interval(a: np.ndarray, b: np.ndarray, counts: np.ndarray, *,
                          block_days: int = 7, replicates: int = 1000,
                          seed: int = 42) -> tuple[float, float]:
    """Bootstrap a pooled-MAE difference, not independent overlapping targets."""
    if a.shape != b.shape or a.shape != counts.shape or a.ndim != 1:
        raise ValueError("Las sumas y recuentos pareados deben tener la misma forma unidimensional.")
    if (not len(a) or counts.sum() <= 0 or block_days < 1 or replicates < 1
            or not np.isfinite(np.r_[a, b, counts]).all() or (counts < 0).any()):
        raise ValueError("Bloques de evaluación inválidos.")
    rng = np.random.default_rng(seed)
    if len(a) < 2 * block_days:
        raise ValueError("Se requieren al menos dos bloques completos para estimar incertidumbre.")
    estimates = []
    for _ in range(replicates):
        starts = rng.integers(0, len(a) - block_days + 1, size=int(np.ceil(len(a) / block_days)))
        indices = (starts[:, None] + np.arange(block_days)).ravel()[:len(a)]
        n = counts[indices].sum()
        if n:
            estimates.append((a[indices].sum() - b[indices].sum()) / n)
    if not estimates:
        raise ValueError("Ninguna réplica contiene targets observados.")
    return tuple(float(value) for value in np.quantile(estimates, [.025, .975]))


def compare_calendar_blocks(rows: list[dict], output) -> None:
    results = []
    for a, b in combinations(rows, 2):
        days_a = {key[6:14] for key in a if key.startswith("block-") and key.endswith("-count")}
        days_b = {key[6:14] for key in b if key.startswith("block-") and key.endswith("-count")}
        if days_a != days_b:
            raise ValueError("Los candidatos no tienen los mismos bloques de test.")
        if not days_a:
            continue
        calendar = pd.date_range(pd.to_datetime(min(days_a)), pd.to_datetime(max(days_a)), freq="D")
        counts = np.array([a.get(f"block-{day:%Y%m%d}-count", 0) for day in calendar], dtype=float)
        counts_b = np.array([b.get(f"block-{day:%Y%m%d}-count", 0) for day in calendar], dtype=float)
        if not np.array_equal(counts, counts_b):
            raise ValueError("Máscaras de targets distintas en los bloques pareados.")
        errors = [np.array([row.get(f"block-{day:%Y%m%d}-absolute-error-sum", 0.) for day in calendar]) for row in (a, b)]
        if len(days_a) < 14:
            results.append({"run_a": a["run"], "run_b": b["run"], "metric": "mae",
                            "delta_a_minus_b": (errors[0].sum() - errors[1].sum()) / counts.sum(),
                            "observed_origin_days": len(days_a), "block_days": 7, "replicates": 0,
                            "interpretation": "No estimado: menos de 14 días de origen observados."})
            continue
        lower, upper = paired_block_interval(*errors, counts)
        results.append({"run_a": a["run"], "run_b": b["run"], "metric": "mae",
                        "delta_a_minus_b": (errors[0].sum() - errors[1].sum()) / counts.sum(),
                        "ci95_lower": lower, "ci95_upper": upper, "observed_origin_days": len(days_a),
                        "block_days": min(7, len(calendar)), "replicates": 1000,
                        "interpretation": "Exploratorio; bloques temporales conjuntos, sin corrección de multiplicidad."})
    pd.DataFrame(results, columns=["run_a", "run_b", "metric", "delta_a_minus_b", "ci95_lower", "ci95_upper",
                                  "observed_origin_days", "block_days", "replicates", "interpretation"]).to_csv(
                                      output / "block_bootstrap_comparison.csv", index=False)
