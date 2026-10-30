"""Compare completed runs only when their exact prepared panel is identical."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


def _same_inference(run: dict, reference: dict) -> bool:
    model = run["settings"].get("model", "mlp")
    return (
        model == reference["settings"].get("model", "mlp")
        and all(run.get("model", {}).get(key) == reference.get("model", {}).get(key)
                for key in ("revision", "point_forecast"))
        and (model != "flowstate-r1" or run["settings"].get("flowstate_scale_factor", 1.0)
             == reference["settings"].get("flowstate_scale_factor", 1.0))
    )


def _metric_difference(metric: str, value_a: float, value_b: float) -> dict:
    valid = np.isfinite(value_a) and np.isfinite(value_b)
    return {
        "metric": metric, "value_a": value_a, "value_b": value_b,
        "delta_a_minus_b": value_a - value_b if valid else np.nan,
        "improvement_a_vs_b_pct": 100 * (1 - value_a / value_b) if valid and value_b else np.nan,
    }


def _write_pairwise(rows: list[dict], stations: list[pd.DataFrame], output: Path) -> None:
    global_rows, station_rows = [], []
    indexed = [frame.set_index("partition-id") for frame in stations]
    for a, b in permutations(range(len(rows)), 2):
        identity = {"run_a": rows[a]["run"], "run_b": rows[b]["run"]}
        for metric in ("mae", "rmse", "macro-mase", "macro-rmsse"):
            global_rows.append(identity | _metric_difference(
                metric, rows[a].get(metric, np.nan), rows[b].get(metric, np.nan),
            ))
        for partition, station_a in indexed[a].iterrows():
            station_b = indexed[b].loc[partition]
            for metric in ("mae", "rmse", "mase", "rmsse"):
                station_rows.append(identity | {"partition-id": partition, "station": station_a["station"]}
                                    | _metric_difference(metric, station_a.get(metric, np.nan), station_b.get(metric, np.nan)))
    pd.DataFrame(global_rows).to_csv(output / "pairwise_comparison.csv", index=False)
    pd.DataFrame(station_rows).to_csv(output / "station_pairwise_comparison.csv", index=False)


def compare_runs(paths: list[Path], output: Path) -> Path:
    if len(paths) < 2 or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Indica al menos dos ejecuciones diferentes y completas.")
    runs, rows, station_frames, warnings = [], [], [], []
    for path in paths:
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
        result = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
        if runs and run["manifest_sha256"] != runs[0]["manifest_sha256"]:
            raise ValueError("Comparación rechazada: los manifests/paneles/ventanas no son idénticos.")
        if runs and run["versions"] != runs[0]["versions"]:
            warnings.append(f"Entorno distinto: {path}. Consulta versions en run.json.")
        if runs and run.get("granite_source") != runs[0].get("granite_source"):
            warnings.append(f"Código Granite distinto: {path}.")
        if runs and run.get("application_sources") != runs[0].get("application_sources"):
            warnings.append(f"Código de la aplicación distinto: {path}.")
        if runs and any(run["settings"].get(key) != runs[0]["settings"].get(key) for key in (
            "seed", "num_server_rounds", "local_epochs", "batch_size", "learning_rate", "weight_decay",
        )):
            warnings.append(f"Presupuesto/semilla/optimizador distinto: {path}.")
        mode = run.get("mode", "federated")
        model = run["settings"].get("model", "mlp")
        adaptation = run.get("model", {}).get("adaptation", "full")
        label = f"{model}/{adaptation}/{run['settings']['strategy'] if mode == 'federated' else mode}/{path.name}"
        metrics = result["test"]
        expected = len(run["manifest"]["clients"])
        if metrics["num-clients"] != expected or not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Test incompleto o no finito: {path}")
        stations = pd.read_csv(path / "station_metrics.csv")
        stations = stations.loc[stations["split"] == "test"].copy()
        if len(stations) != expected or set(stations["partition-id"]) != set(range(expected)):
            raise ValueError(f"Panel de test incompleto o duplicado: {path}")
        for client in run["manifest"]["clients"]:
            station = stations.loc[stations["partition-id"] == client["partition_id"]].iloc[0]
            if (station["station"] != client["station"] or station["num-examples"] != client["num_test_windows"]
                    or station["num-targets"] != client["num_test_windows"] * run["manifest"]["geometry"]["horizon"]):
                raise ValueError(f"Identidad o recuentos de test incompatibles: {path}")
        if metrics["num-targets"] != sum(client["num_test_windows"] for client in run["manifest"]["clients"]) * run["manifest"]["geometry"]["horizon"]:
            raise ValueError(f"Número de targets incompatible: {path}")
        stations["run"], stations["model"], stations["mode"], stations["adaptation"] = label, model, mode, adaptation
        station_frames.append(stations)
        rows.append({
            "run": label, "model": model, "mode": mode, "adaptation": adaptation,
            "seed": run["settings"].get("seed"), "lora_rank": run["settings"].get("lora_rank"),
            "flowstate_scale_factor": run["settings"].get("flowstate_scale_factor") if model == "flowstate-r1" else None,
            "revision": run.get("model", {}).get("revision"),
            "selection_metric": result.get("selection_metric"),
            "strategy": run["settings"]["strategy"] if mode == "federated" else "none",
            "selected_round": result["selected_round"], **metrics,
            "mae_improvement_vs_persistence_pct": (
                100 * (1 - metrics["mae"] / metrics["baseline-mae"]) if metrics["baseline-mae"] else None
            ),
        })
        runs.append(run)
    for row, run, stations in zip(rows, runs, station_frames, strict=True):
        references = [(ref, frame, reference_run) for ref, frame, reference_run in zip(rows, station_frames, runs, strict=True)
                       if ref["model"] == row["model"] and ref["mode"] == "zero-shot"
                       and _same_inference(run, reference_run)]
        if len(references) == 1:
            reference, reference_stations, _ = references[0]
            for metric in ("mae", "macro-mase", "macro-rmsse"):
                denominator = reference.get(metric)
                row[f"{metric.replace('-', '_')}_improvement_vs_zero_shot_pct"] = (
                    100 * (1 - row[metric] / denominator) if denominator and metric in row else None
                )
            baseline = reference_stations.set_index("partition-id")["mae"]
            denominators = stations["partition-id"].map(baseline).replace(0, np.nan)
            stations["mae_improvement_vs_zero_shot_pct"] = 100 * (1 - stations["mae"] / denominators)
        elif len(references) > 1:
            warnings.append(f"Varias referencias zero-shot para {row['model']}; no se elige una silenciosamente.")
        elif any(ref["model"] == row["model"] and ref["mode"] == "zero-shot" for ref in rows):
            warnings.append(f"Sin zero-shot con la misma configuración de inferencia para {row['run']}; "
                            "se conserva en todas las comparaciones por pares.")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.concat(station_frames, ignore_index=True).to_csv(output / "station_comparison.csv", index=False)
    _write_pairwise(rows, station_frames, output)
    (output / "comparison.json").write_text(json.dumps({
        "manifest_sha256": runs[0]["manifest_sha256"], "runs": [str(path.resolve()) for path in paths],
        "warnings": warnings,
        "interpretation": "Comparación descriptiva; targets rolling solapados, sin test de significación.",
        "pairwise_convention": "Todos los pares ordenados A/B. Delta = A - B; negativo favorece A. "
                               "Mejora (%) = 100 * (1 - A/B); positiva favorece A; indefinida si B=0 o falta una métrica.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["model", "adaptation", "mode", "strategy", "mae", "rmse", "baseline-mae"]].to_string(index=False))
    for warning in warnings:
        print(f"AVISO: {warning}")
    print(f"Comparación: {output}")
    return output
