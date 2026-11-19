import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fd_airquality.backtesting import fold_configurations, prepare_backtest, run_backtest
from fd_airquality.data import load_client_data
from fd_airquality.settings import RunSettings
from fd_airquality.uncertainty import compare_calendar_blocks, paired_block_interval


def test_fold_calendar_and_fold_local_normalization(temporal_family, tmp_path):
    raw, geometry = temporal_family
    configs = fold_configurations(geometry, 3, 48)
    assert configs[0].validation_start == "2025-01-06 00:00:00"
    assert configs[-1].test_end == "2025-01-14 00:00:00"
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=3, step_hours=48)
    family = json.loads(path.read_text())
    assert family["pollutant"] == "PM25"
    assert family["protocol"] == "temporal_cv"
    assert family["fold_initialization"] == "fresh"
    artifacts = [load_client_data(entry["manifest"], 0)[1] for entry in family["folds"]]
    for artifact in artifacts:
        values = artifact["values"][artifact["train_mask"]].astype(float)
        assert float(artifact["mean"]) == pytest.approx(values.mean(), rel=1e-6)
    assert len({float(artifact["mean"]) for artifact in artifacts}) > 1
    assert np.max(artifacts[0]["train_starts"]) < np.max(artifacts[-1]["train_starts"])


def test_only_data_history_changes_between_folds(temporal_family):
    _, geometry = temporal_family
    expanding = fold_configurations(geometry, 3, 48)
    assert all(config.train_start == "" for config in expanding)
    sliding = fold_configurations(geometry, 3, 48, train_history="sliding", history_hours=72)
    for config in sliding:
        assert pd.Timestamp(config.train_start) == pd.Timestamp(config.validation_start) - pd.Timedelta(hours=72)
    with pytest.raises(ValueError, match="new-only"):
        fold_configurations(geometry, 3, 48, train_history="new-only")
    with pytest.raises(ValueError, match="paso"):
        fold_configurations(geometry, 3, 24)


@pytest.mark.parametrize("history", ["expanding", "sliding"])
def test_fixed_scalers_fit_only_each_folds_authorized_past(temporal_family, tmp_path, history):
    from fd_airquality.arrival import normalization_artifact, round_cutoff

    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=3, step_hours=48,
                            train_history=history, history_hours=72)
    settings = RunSettings(normalization="fixed", num_server_rounds=2)
    hashes = []
    for entry in json.loads(path.read_text())["folds"]:
        manifest, artifact = load_client_data(entry["manifest"], 0)
        geometry = manifest["geometry"]
        local = normalization_artifact(artifact, geometry, settings)
        mask = np.isfinite(artifact["values"]) & (artifact["timestamps"] < round_cutoff(geometry, settings, 1).value)
        if geometry["train_start"]:
            mask &= artifact["timestamps"] >= pd.Timestamp(geometry["train_start"]).value
        values = artifact["values"][mask].astype(float)
        assert float(local["mean"]) == values.mean()
        assert float(local["scale"]) == values.std()
        hashes.append(local["normalization_sha256"])
    assert len(set(hashes)) == 3


def test_sliding_cv_compares_modes_on_the_same_fold_support(temporal_family, tmp_path, in_process_federation):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry,
                            folds=2, step_hours=48, train_history="sliding")
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "results"),
                            num_server_rounds=1, batch_size=8, mlflow_tracking_uri="disabled")
    output = run_backtest(settings, ["mlp"], ["centralized", "federated"])
    report = pd.read_csv(output / "temporal_comparison.csv")
    assert len(report) == 4
    assert report.groupby("fold")["num-targets"].nunique().eq(1).all()
    progress = json.loads((output / "backtest.json").read_text())
    assert all(entry["status"] == "completed" for entry in progress["results"])
    for entry in progress["results"]:
        runs = json.loads((Path(entry["comparison"]) / "comparison.json").read_text())["runs"]
        for run in runs:
            metadata = json.loads((Path(run) / "run.json").read_text())
            assert "initial_checkpoint" not in metadata["settings"]
            assert metadata["initialization"]["trained_checkpoint_reused"] is False
            assert metadata["settings"]["manifest"] != str(path)
            assert (Path(run) / "last_model.pt").is_file()


@pytest.mark.parametrize("mode", ["local", "centralized", "federated"])
def test_initial_weights_are_fresh_even_if_previous_checkpoints_are_poisoned(
    temporal_family, tmp_path, monkeypatch, in_process_federation, mode,
):
    from fd_airquality.cli import run_benchmark
    from fd_airquality.models import make_model, shared_state

    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, step_hours=48)
    initial_states = []

    def fresh(geometry, settings, **kwargs):
        model = make_model(geometry, settings, **kwargs)
        initial_states.append({key: value.detach().clone() for key, value in shared_state(model).items()})
        return model

    monkeypatch.setattr(f"fd_airquality.{'server_app' if mode == 'federated' else 'baselines'}.make_model", fresh)
    def benchmark(*args, **kwargs):
        result = run_benchmark(*args, **kwargs)
        for run in json.loads((result / "comparison.json").read_text())["runs"]:
            for checkpoint in Path(run).glob("*model.pt"):
                checkpoint.write_bytes(b"poisoned: cannot be restored")
        return result

    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=2, batch_size=8, hidden_size=8,
                           mlflow_tracking_uri="disabled")
    run_backtest(settings, ["mlp"], [mode], benchmark=benchmark, stage="all")
    assert len(initial_states) == (4 if mode == "local" else 2)
    for state in initial_states[1:]:
        for key, value in initial_states[0].items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("protocol", ["streaming", "independent"])
def test_legacy_fold_protocol_requires_fresh_preparation(temporal_family, tmp_path, protocol):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, step_hours=48)
    family = json.loads(path.read_text())
    family.update(schema_version=1, protocol=protocol)
    path.write_text(json.dumps(family))
    with pytest.raises(ValueError, match="reinicio por fold"):
        run_backtest(RunSettings(manifest=str(path)), ["mlp"], ["local"])


@pytest.mark.parametrize("arguments", [
    ["prepare-backtest", "--protocol", "streaming"],
    ["prepare-backtest", "--protocol", "independent"],
    ["prepare-backtest", "--train-history", "new-only"],
    ["backtest", "--initial-checkpoint", "previous.pt"],
])
def test_removed_cross_fold_options_are_rejected(arguments):
    from fd_airquality.cli import _parser

    parser, _ = _parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)
    assert error.value.code == 2


def test_normal_cv_has_no_protocol_switch_or_checkpoint_initialization():
    from fd_airquality.cli import _parser

    parser, _ = _parser()
    arguments = parser.parse_args(["prepare-backtest"])
    assert arguments.train_history == "expanding"
    assert not hasattr(arguments, "protocol")
    assert "initial_checkpoint" not in RunSettings().flower_config()


def test_joint_block_interval_is_paired_and_deterministic():
    a = np.arange(1., 31.)
    b = a + 2
    counts = np.ones(30)
    assert paired_block_interval(a, b, counts) == (-2., -2.)
    assert paired_block_interval(a, a, counts) == (0., 0.)
    with pytest.raises(ValueError):
        paired_block_interval(a, b[:-1], counts)
    with pytest.raises(ValueError, match="dos bloques"):
        paired_block_interval(a[:3], b[:3], counts[:3])


def test_calendar_bootstrap_uses_pooled_scaled_errors_and_common_support(tmp_path):
    dates = pd.date_range("2025-01-01", periods=14)
    a, b = {"run": "a"}, {"run": "b"}
    for count, day in enumerate(dates, start=1):
        prefix = f"block-{day:%Y%m%d}-mase"
        a.update({f"{prefix}-scaled-error-sum": count * 2., f"{prefix}-target-count": count})
        b.update({f"{prefix}-scaled-error-sum": count * 3., f"{prefix}-target-count": count})
    compare_calendar_blocks([a, b], tmp_path)
    row = pd.read_csv(tmp_path / "block_bootstrap_comparison.csv").iloc[0]
    assert row.metric == "pooled-mase"
    assert row.delta_a_minus_b == row.ci95_lower == row.ci95_upper == -1.
    b["block-20250101-mase-target-count"] = 2
    with pytest.raises(ValueError, match="Máscaras de targets distintas"):
        compare_calendar_blocks([a, b], tmp_path)


def test_backtest_can_follow_a_single_federated_candidate(temporal_family, tmp_path, in_process_federation):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, step_hours=48)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=1,
                            batch_size=8, mlflow_tracking_uri="disabled")
    output = run_backtest(settings, ["mlp"], ["federated"], stage="all")
    assert len(pd.read_csv(output / "temporal_comparison.csv")) == 2
    assert len(pd.read_csv(output / "temporal_seed_summary.csv")) == 2


def test_flora_history_restarts_at_each_fold(temporal_family, tmp_path, in_process_federation, mock_granite):
    from fd_airquality.models import restore_checkpoint

    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, step_hours=48)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), model="flowstate-r1",
                            lora_targets="proj", missing_policy="ffill", batch_size=8,
                            num_server_rounds=2, mlflow_tracking_uri="disabled")
    output = run_backtest(settings, ["flowstate-r1"], ["federated"], adaptations=["flora"])
    progress = json.loads((output / "backtest.json").read_text())
    assert [entry["fold"] for entry in progress["results"]] == [1, 2]
    for entry in progress["results"]:
        run, = json.loads((Path(entry["comparison"]) / "comparison.json").read_text())["runs"]
        last = Path(run) / "last_model.pt"
        state = torch.load(last, weights_only=True)["model_state_dict"]
        rounds = {int(key.split(".")[1]) for key in state if key.startswith("history.")}
        assert rounds == {1, 2}
        restored = restore_checkpoint(last)
        assert torch.isfinite(restored(torch.ones(1, 6))).all()
