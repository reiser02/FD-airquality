"""Bounded real-CLI integration check; never the full scientific experiment.

Run from the repository root:
uv run --frozen python experiments/validate_neuralforecast.py --foundations
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fd_airquality.arrival import normalization_artifact, training_view
from fd_airquality.artifacts import create_run_dir, file_digest
from fd_airquality.data import WindowDataset, load_client_data
from fd_airquality.missing import forecast
from fd_airquality.models import restore_checkpoint
from fd_airquality.settings import RunSettings


def run_cli(root: Path, label: str, *arguments: str) -> None:
    print(f"CLI: {label}", flush=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", FLWR_TELEMETRY_ENABLED="0", FLWR_DISABLE_UPDATE_CHECK="1")
    command = ["uv", "run", "--frozen", "fd-airquality", *arguments]
    with (root / f"{label}.log").open("w") as log:
        result = subprocess.run(command, check=False, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=3600)
    if result.returncode:
        raise RuntimeError(f"Falló {label}; consulta {root / f'{label}.log'}")


def verify_backtest(output: Path, *, classical: bool) -> dict:
    progress = json.loads((output / "backtest.json").read_text())
    frame = pd.read_csv(output / "temporal_comparison.csv")
    assert len(frame) == (24 if classical else 2)
    assert frame.groupby("fold")["num-targets"].nunique().eq(1).all()
    assert not frame.final_holdout.any() and frame.exploratory.all()
    restored, streams, fresh_runs, flora_history = 0, [], [], []
    for entry in progress["results"]:
        comparison = Path(entry["comparison"])
        for path in json.loads((comparison / "comparison.json").read_text())["runs"]:
            run_path = Path(path)
            run = json.loads((run_path / "run.json").read_text())
            package = Path(__file__).resolve().parents[1] / "src" / "fd_airquality"
            assert run["application_sources"] == {
                item.name: file_digest(item) for item in sorted(package.glob("*.py"))
            }
            settings = RunSettings(**run["settings"])
            assert run["initialization"]["fold"] == "fresh"
            assert not run["initialization"]["trained_checkpoint_reused"]
            fresh_runs.append(str(run_path))
            for checkpoint in run_path.glob("*model.pt"):
                partition = int(checkpoint.name.split("_")[1]) if checkpoint.name.startswith("client_") else 0
                manifest, artifact = load_client_data(settings.manifest, partition)
                artifact = normalization_artifact(artifact, manifest["geometry"], settings)
                dataset = WindowDataset(artifact, "test", manifest["geometry"]["context"],
                                        manifest["geometry"]["horizon"])
                row = next((i for i in range(len(dataset)) if torch.isnan(dataset[i][0]).any()), 0)
                x, _ = dataset[row]
                model = restore_checkpoint(checkpoint)
                with torch.no_grad():
                    prediction = forecast(model, x.unsqueeze(0), settings)
                assert prediction.shape == (1, dataset.horizon) and torch.isfinite(prediction).all()
                restored += 1
            if run["mode"] == "federated":
                assert (run_path / "flower.log").is_file()
                local = pd.read_csv(run_path / "station_metrics.csv")
                assert run["arrival_protocol"]["history"] == "new-only"
                for station, train in local.loc[local.split == "train"].groupby("station"):
                    train = train.sort_values("round")
                    cutoffs = train["available-through-ns"].to_numpy()
                    assert np.all(np.diff(cutoffs) > 0)
                    active = train.loc[train["num-examples"] > 0]
                    assert (active["train-last-target-ns"] < active["available-through-ns"]).all()
                    partition = next(client["partition_id"] for client in manifest["clients"]
                                     if client["station"] == station)
                    _, artifact = load_client_data(settings.manifest, partition)
                    artifact = normalization_artifact(artifact, manifest["geometry"], settings)
                    seen_windows, seen_targets = set(), set()
                    for row in train.to_dict("records"):
                        number = int(row["round"])
                        dataset = training_view(artifact, manifest["geometry"], settings, number)
                        assert len(dataset) == row["num-examples"]
                        assert dataset.candidate_windows == row["candidate-windows"]
                        assert dataset.windows_before_cap == row["windows-before-cap"]
                        positions = dataset.starts[:, None] + dataset.context + np.arange(dataset.horizon)
                        observed = np.isfinite(artifact["values"][positions])
                        assert int(observed.sum()) == row["num-targets"]
                        targets = set(artifact["timestamps"][positions][observed])
                        assert not seen_windows.intersection(dataset.starts)
                        assert not seen_targets.intersection(targets)
                        seen_windows.update(dataset.starts)
                        seen_targets.update(targets)
                        if number > 1:
                            assert row["available-from-ns"] == dataset.arrival_lower.value
                            if len(dataset):
                                assert row["train-first-target-ns"] >= row["available-from-ns"]
                    streams.append({"run": str(run_path), "station": station,
                                    "arrival_history": "new-only",
                                     "cutoffs": [int(value) for value in cutoffs],
                                    "block_starts": [int(value) if pd.notna(value) else None
                                                     for value in train.get("available-from-ns", pd.Series([np.nan] * len(train)))],
                                    "training_windows": train["num-examples"].astype(int).tolist(),
                                     "candidate_windows": train["candidate-windows"].astype(int).tolist()})
                validation = local.loc[local.split == "validation"]
                assert set(validation["round"]) == {settings.num_server_rounds}
                assert len(local.loc[local.split == "test"]) == len(manifest["clients"])
                if settings.adaptation == "flora":
                    payload = torch.load(run_path / "last_model.pt", weights_only=True, map_location="cpu")
                    rounds = sorted({int(key.split(".")[1]) for key in payload["model_state_dict"]
                                     if key.startswith("history.")})
                    assert rounds == list(range(1, settings.num_server_rounds + 1))
                    flora_history.append({"fold": entry["fold"], "rounds": rounds})
    if classical:
        quantile_rows = frame.loc[frame.model.isin(["mlp", "nlinear", "nhits"])]
        assert np.isfinite(quantile_rows.crps).all()
        assert (quantile_rows["crps-count"] == quantile_rows["num-targets"]).all()
    return {"output": str(output), "runs": len(frame), "restored_checkpoints": restored,
            "fresh_fold_runs": fresh_runs, "round_streams": streams,
            "flora_history": flora_history,
            "fold_target_counts": frame.groupby("fold")["num-targets"].first().to_dict()}


def verify_foundation_quantiles(root: Path) -> dict:
    from fd_airquality.missing import forecast_distribution

    comparison, = (root / "foundation_quantiles").glob("benchmark_*/comparison/comparison.csv")
    scores = pd.read_csv(comparison)
    available = scores.loc[scores["probabilistic-available"] > 0]
    assert set(available.model) == {"ttm-r3", "patchtst-fm-r2", "flowstate-r1"}
    assert np.isfinite(available.crps).all()
    assert (available["crps-count"] == available["num-targets"]).all()
    restored = 0
    metadata = json.loads((comparison.parent / "comparison.json").read_text())
    for run_path in map(Path, metadata["runs"]):
        run = json.loads((run_path / "run.json").read_text())
        settings = RunSettings(**run["settings"])
        manifest, artifact = load_client_data(settings.manifest, 0)
        artifact = normalization_artifact(artifact, manifest["geometry"], settings)
        dataset = WindowDataset(artifact, "test", manifest["geometry"]["context"], manifest["geometry"]["horizon"])
        row = next((i for i in range(len(dataset)) if torch.isnan(dataset[i][0]).any()), 0)
        inputs, _ = dataset[row]
        for checkpoint in run_path.glob("*model.pt"):
            model = restore_checkpoint(checkpoint)
            with torch.no_grad():
                prediction, quantiles, levels = forecast_distribution(model, inputs.unsqueeze(0), settings)
            assert torch.isfinite(prediction).all() and torch.isfinite(quantiles).all()
            assert quantiles.shape == (1, dataset.horizon, len(levels))
            restored += 1
    return {"comparison": str(comparison), "runs": len(scores),
            "restored_checkpoints": restored,
            "crps_models": available.model.tolist(),
            "quantile_counts": dict(zip(available.model, (
                available["quantile-count"] / available["num-targets"]
            ).astype(int), strict=True))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundations", action="store_true")
    parser.add_argument("--verify-existing", type=Path,
                        help="Reauditar ejecuciones completadas sin repetir entrenamientos.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.verify_existing:
        root = args.verify_existing.resolve()
        original = json.loads((root / "validation.json").read_text())
        report = {
            "status": "passed", "purpose": original["purpose"],
            "manifest": original["manifest"], "reverified": datetime.now(UTC).isoformat(),
            "classical": verify_backtest(Path(original["classical"]["output"]), classical=True),
            "foundation": verify_backtest(Path(original["foundation"]["output"]), classical=False),
            "foundation_quantiles": verify_foundation_quantiles(root),
            "original_runner_status": original["status"],
            "note": "CV con reinicio por fold; no se reutilizan checkpoints entrenados.",
        }
        path = root / "verification.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Auditoría completada: {path}", flush=True)
        return
    root = create_run_dir(Path("reports/temporal_cv_validation").resolve(),
                          datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    report = {"status": "running", "purpose": "bounded_integration_not_scientific_holdout",
              "started": datetime.now(UTC).isoformat(), "output": str(root)}
    report_path = root / "validation.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        data = root / "prepared"
        run_cli(root, "prepare", "prepare-backtest", "--config", "experiments/smoke.toml",
                "--output", str(data), "--folds", "2")
        family_path, = data.glob("*/backtest_manifest.json")
        report["manifest"] = str(family_path)
        results = root / "classical"
        run_cli(root, "classical", "backtest", "--config", "experiments/smoke.toml",
                "--manifest", str(family_path), "--output-dir", str(results), "--stage", "all")
        backtest, = results.glob("backtest_*")
        report["classical"] = verify_backtest(backtest, classical=True)
        save()
        if args.foundations:
            results = root / "foundation"
            run_cli(root, "foundation", "backtest", "--config", "experiments/smoke.toml",
                    "--manifest", str(family_path), "--output-dir", str(results), "--stage", "all",
                    "--models", "ttm-r3", "--modes", "federated", "--adaptations", "flora",
                    "--num-server-rounds", "2", "--probabilistic", "0")
            backtest, = results.glob("backtest_*")
            report["foundation"] = verify_backtest(backtest, classical=False)
            family = json.loads(family_path.read_text())
            results = root / "foundation_quantiles"
            run_cli(root, "foundation_quantiles", "benchmark", "--config", "experiments/smoke.toml",
                    "--manifest", family["folds"][0]["manifest"], "--output-dir", str(results),
                    "--models", "ttm-r3", "patchtst-fm-r2", "flowstate-r1", "--modes", "zero-shot")
            report["foundation_quantiles"] = verify_foundation_quantiles(root)
        report.update(status="passed", finished=datetime.now(UTC).isoformat())
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print(f"Validación completada: {report_path}", flush=True)


if __name__ == "__main__":
    main()
