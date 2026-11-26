"""External CV reductions: pool numerators/counts, then explicitly macro stations."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

OPERATIONAL_SETTINGS = {
    "seed", "manifest", "output_dir", "mlflow_tracking_uri",
    "mlflow_experiment", "mlflow_log_models", "heartbeat_seconds", "log_level",
    "gateway_topology", "gateway_num_clients", "gateway_seed",
}


def candidate_identity(run: dict) -> str:
    definition = {
        "settings": {key: value for key, value in run["settings"].items()
                     if key not in OPERATIONAL_SETTINGS},
        "mode": run["mode"], "model": run["model"],
        "gateway_topology_sha256": run.get("gateway_topology_sha256"),
        "sources": run["application_sources"], "versions": run["versions"],
    }
    return sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def temporal_run_rows(paths: list[Path], fold: int) -> tuple[list[dict], list[dict]]:
    rows, stations = [], []
    for path in paths:
        run = json.loads((path / "run.json").read_text())
        result = json.loads((path / "test_metrics.json").read_text())
        identity = {
            "temporal_configuration": candidate_identity(run), "seed": run["settings"]["seed"],
            "model": run["settings"]["model"], "mode": run["mode"],
            "adaptation": run["model"].get("adaptation", "full"),
            "strategy": run["settings"]["strategy"], "fold": fold, "run_path": str(path),
        }
        rows.append(identity | result["test"])
        frame = pd.read_csv(path / "station_metrics.csv")
        stations.extend(dict(record) | identity for record in frame.loc[frame.split == "test"].to_dict("records"))
    return rows, stations


def _pooled(frame: pd.DataFrame) -> dict:
    count = int(frame["num-targets"].sum())
    if not count:
        raise ValueError("La CV no tiene targets observados; no se puede asignar error cero.")
    result = {
        "num-targets": count, "num-folds": frame["fold"].nunique(),
    }
    if "num-target-slots" in frame:
        result["target-coverage"] = count / frame["num-target-slots"].sum()
    for name in ("mase", "rmsse", "crps", "scaled-crps"):
        count_key = f"{name}-count" if name.endswith("crps") else f"{name}-target-count"
        numerator = f"{name}-sum" if name.endswith("crps") else f"{name}-scaled-error-sum"
        if count_key not in frame or numerator not in frame:
            continue
        support = frame[count_key].sum()
        if support:
            score = frame[numerator].sum() / support
            result[name] = float(np.sqrt(score) if name == "rmsse" else score)
            result[count_key] = int(support)
    return result


def summarize_cv(rows: list[dict], stations: list[dict], output: Path) -> pd.DataFrame:
    frame, local = pd.DataFrame(rows), pd.DataFrame(stations)
    summaries, fold_means = [], []
    for (configuration, seed), group in frame.groupby(["temporal_configuration", "seed"], sort=False):
        identity = {key: group.iloc[0][key] for key in (
            "temporal_configuration", "seed", "model", "mode", "adaptation", "strategy",
        )}
        means = identity | {"num-folds": group["fold"].nunique(),
                            "aggregation": "unweighted_fold_mean"}
        fold_scores = []
        for _, fold_group in group.groupby("fold", sort=False):
            fold_score = _pooled(fold_group)
            fold_score.update({key: fold_group.iloc[0][key] for key in (
                "macro-mase", "macro-rmsse", "macro-crps", "macro-scaled-crps",
            ) if key in fold_group})
            fold_scores.append(fold_score)
        fold_frame = pd.DataFrame(fold_scores)
        for metric in ("mase", "rmsse", "crps", "macro-mase", "macro-rmsse", "macro-crps", "macro-scaled-crps"):
            if metric in fold_frame:
                values = fold_frame[metric]
                defined = np.isfinite(values)
                means[f"{metric}-defined-folds"] = int(defined.sum())
                means[metric] = float(values.mean()) if defined.all() else np.nan
                means[f"{metric}-std"] = float(values.std(ddof=1)) if defined.all() and len(values) > 1 else (
                    0. if defined.all() else np.nan)
        fold_means.append(means)
        score = _pooled(group)
        panel = local.loc[(local.temporal_configuration == configuration) & (local.seed == seed)]
        per_station = [_pooled(group) for _, group in panel.groupby("station", sort=False)]
        for name in ("mase", "rmsse", "crps"):
            valid = [record[name] for record in per_station if name in record]
            score[f"macro-{name}-stations"] = len(valid)
            if valid:
                score[f"macro-{name}"] = float(np.mean(valid))
        if "macro-scaled-crps" in group and group["macro-scaled-crps"].notna().all():
            valid = [record["scaled-crps"] for record in per_station if "scaled-crps" in record]
            if valid:
                score.update({"macro-scaled-crps": float(np.mean(valid)),
                              "macro-scaled-crps-stations": len(valid)})
        summaries.append(identity | score)
    result = pd.DataFrame(summaries)
    result.to_csv(output / "cv_summary.csv", index=False)
    pd.DataFrame(fold_means).to_csv(output / "cv_fold_mean.csv", index=False)
    return result


def select_configuration(summary: pd.DataFrame, metric: str) -> str:
    if metric not in summary or not np.isfinite(summary[metric]).all():
        raise ValueError(f"La métrica externa de selección {metric} es indefinida; no hay fallback implícito.")
    # Seeds are replications, never candidate hyperparameters to cherry-pick.
    scores = summary.groupby("temporal_configuration", sort=True)[metric].mean()
    return str(scores.idxmin())
