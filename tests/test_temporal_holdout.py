import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from flwr.supercore.task_identity import TaskIdentity
from test_backtesting import temporal_family  # noqa: F401
from test_federation import LocalTestGrid

from fd_airquality.backtesting import prepare_backtest, run_backtest
from fd_airquality.data import read_manifest
from fd_airquality.server_app import run_server
from fd_airquality.settings import RunSettings
from fd_airquality.temporal_results import select_configuration, summarize_cv


def test_external_cv_pools_counts_not_fold_means_and_roots_after_pooling(tmp_path):
    records = []
    for fold, count, error in ((1, 1, 0.), (2, 9, 10.)):
        records.append({
            "temporal_configuration": "a", "seed": 42, "model": "mlp", "mode": "local",
            "adaptation": "full", "strategy": "fedavg", "station": "a", "fold": fold,
            "num-targets": count, "absolute-error-sum": count * error,
            "squared-error-sum": count * error**2,
            "mase-scaled-error-sum": count * error / 2, "mase-target-count": count,
            "rmsse-scaled-error-sum": count * error**2 / 4, "rmsse-target-count": count,
            "crps-sum": count * error / 3, "crps-count": count,
        })
    summary = summarize_cv(records, records, tmp_path).iloc[0]
    assert summary.mae == summary["macro-mae"] == 9.
    assert summary.rmse == pytest.approx(np.sqrt(90))
    assert summary.mase == summary["macro-mase"] == 4.5
    assert summary.rmsse == pytest.approx(np.sqrt(90 / 4))
    assert summary.crps == 3.
    means = pd.read_csv(tmp_path / "cv_fold_mean.csv").iloc[0]
    assert means.mae == 5. and means.rmse == 5.
    assert means["mae-defined-folds"] == 2


def test_seed_is_not_selected_as_a_hyperparameter():
    frame = pd.DataFrame({"temporal_configuration": ["a", "a", "b", "b"],
                          "seed": [42, 43, 42, 43], "mae": [0., 20., 5., 5.]})
    assert select_configuration(frame, "mae") == "b"
    with pytest.raises(ValueError, match="indefinida"):
        select_configuration(frame.assign(mae=np.nan), "mae")


@pytest.mark.parametrize("folds", [1, 3])
def test_default_evaluation_uses_all_folds_without_selecting_configuration(
    temporal_family, tmp_path, monkeypatch, folds,  # noqa: F811
):
    raw, geometry = temporal_family
    family = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=folds, step_hours=48)
    assert json.loads(family.read_text())["final_holdout_fold"] is None
    monkeypatch.setattr("fd_airquality.backtesting.select_configuration",
                        lambda *_: pytest.fail("La evaluación normal no selecciona configuraciones"))
    settings = RunSettings(manifest=str(family), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=1, hidden_size=8, mlflow_tracking_uri="disabled")
    output = run_backtest(settings, ["mlp", "nlinear"], ["local"])
    progress = json.loads((output / "backtest.json").read_text())
    assert progress["stage"] == "evaluation"
    assert progress["final_holdout_fold"] is None
    assert not progress["configuration_selection"]
    assert len(progress["results"]) == folds
    assert not (output / "selection.json").exists()
    assert (output / "cv_summary.csv").is_file()
    means = pd.read_csv(output / "cv_fold_mean.csv")
    assert means["num-folds"].eq(folds).all()
    assert len(means) == 2
    # A later tuning workflow must not relabel already evaluated dates as sealed.
    assert (family.parent / "holdout_access.json").is_file()


@pytest.mark.parametrize("mode", ["local", "centralized", "federated"])
def test_development_does_not_score_holdout_and_final_uses_only_frozen_selection(
    temporal_family, tmp_path, mode, monkeypatch,  # noqa: F811
):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry,
                            folds=3, step_hours=48)
    for name in ("_run_id", "_task_id", "_node_id"):
        monkeypatch.setattr(TaskIdentity, name, 1)
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda settings, dry_run=False:
                        run_server(LocalTestGrid(settings, len(read_manifest(settings.manifest)["clients"])), settings))
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=2, hidden_size=8, batch_size=8,
                           probabilistic=1, mlflow_tracking_uri="disabled")
    development = run_backtest(settings, ["mlp", "nlinear"], [mode], seeds=[42, 43], stage="development")
    progress = json.loads((development / "backtest.json").read_text())
    assert [entry["fold"] for entry in progress["results"]] == [1, 2]
    assert not progress["final_holdout_scored"]
    assert not (path.parent / "holdout_access.json").exists()
    selection = development / "selection.json"
    recipe = json.loads(selection.read_text())
    assert recipe["seeds"] == [42, 43]
    assert recipe["fold_initialization"] == "fresh"
    assert "last_runs" not in recipe
    # Final evaluation must not even need the trained development checkpoints.
    for checkpoint in development.rglob("*model.pt"):
        checkpoint.write_bytes(b"not reusable")
    final = run_backtest(replace(settings, output_dir=str(tmp_path / "final")),
                         ["nhits"], ["local"], stage="final", selection=selection)
    frame = pd.read_csv(final / "temporal_comparison.csv")
    assert set(frame.fold) == {3} and frame.final_holdout.all()
    assert len(frame) == 2
    assert set(frame.model) == {recipe["settings"]["model"]}
    assert set(frame["mode"]) == {recipe["mode"]}
    assert (path.parent / "holdout_access.json").exists()
    result = json.loads((final / "backtest.json").read_text())
    comparison = Path(result["results"][0]["comparison"])
    for run_path in json.loads((comparison / "comparison.json").read_text())["runs"]:
        metadata = json.loads((Path(run_path) / "run.json").read_text())
        assert "initial_checkpoint" not in metadata["settings"]
        assert metadata["initialization"]["fold"] == "fresh"
        assert "crps" in json.loads((Path(run_path) / "test_metrics.json").read_text())["test"]
    with pytest.raises(ValueError, match="ya fue consultado"):
        run_backtest(settings, ["mlp"], ["local"], stage="final", selection=selection)


def test_final_stage_requires_selection_before_any_training(temporal_family, tmp_path):  # noqa: F811
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, step_hours=48)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"))
    with pytest.raises(ValueError, match="selección"):
        run_backtest(settings, ["mlp"], ["local"], stage="final")
    assert not (tmp_path / "runs").exists()
