import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fd_airquality.artifacts import file_digest
from fd_airquality.backtesting import fold_configurations, prepare_backtest, run_backtest
from fd_airquality.data import DataConfig, load_client_data, prepare, read_manifest
from fd_airquality.settings import RunSettings
from fd_airquality.uncertainty import compare_calendar_blocks, paired_block_interval

TEMPORAL_BOUNDS = [(pd.Timestamp("2024-12-28"), pd.Timestamp("2025-01-21"))]


def test_fold_calendar_and_fold_local_normalization(temporal_family, tmp_path):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=3, min_stations=2)
    family = json.loads(path.read_text())
    assert family["pollutant"] == "PM25"
    assert family["protocol"] == "temporal_cv"
    assert family["fold_initialization"] == "fresh"
    assert family["calendar_start"] == "2024-12-28 00:00:00"
    assert family["initial_history_end"] == "2025-01-06 14:00:00"
    assert family["calendar_end"] == "2025-01-21 00:00:00"
    assert family["folds"][0]["validation_start"] == "2025-01-07 19:00:00"
    assert (family["validation_hours"], family["test_hours"]) == (42, 45)
    assert (family["requested_validation_hours"], family["requested_test_hours"]) == (672, 672)
    artifacts = [load_client_data(entry["manifest"], 0)[1] for entry in family["folds"]]
    for artifact in artifacts:
        values = artifact["values"][artifact["train_mask"]].astype(float)
        assert float(artifact["mean"]) == pytest.approx(values.mean(), rel=1e-6)
    assert len({float(artifact["mean"]) for artifact in artifacts}) > 1
    assert np.max(artifacts[0]["train_starts"]) < np.max(artifacts[-1]["train_starts"])


def test_only_data_history_changes_between_folds(temporal_family):
    _, geometry = temporal_family
    calendar, expanding = fold_configurations(geometry, 3, bounds=TEMPORAL_BOUNDS)
    assert all(config.train_start == "" for config in expanding)
    sliding_calendar, sliding = fold_configurations(geometry, 3, bounds=TEMPORAL_BOUNDS,
                                                    train_history="sliding", history_hours=72)
    assert calendar == sliding_calendar
    assert [replace(config, train_start="") for config in sliding] == expanding
    for config in sliding:
        assert pd.Timestamp(config.train_start) == pd.Timestamp(config.validation_start) - pd.Timedelta(hours=72)
    with pytest.raises(ValueError, match="new-only"):
        fold_configurations(geometry, 3, bounds=TEMPORAL_BOUNDS, train_history="new-only")


@pytest.mark.parametrize("extra", [0, 1, 2])
def test_calendar_is_shared_evenly_without_losing_remainder_hours(extra):
    geometry = DataConfig(context=2, horizon=1, stride=1, min_validation_windows=1,
                          min_test_windows=1, seasonality=1)
    beginning = pd.Timestamp("2025-01-01")
    end = beginning + pd.Timedelta(hours=100 + extra)
    calendar, folds = fold_configurations(geometry, 3, bounds=[(beginning, end)],
                                          validation_hours=5, test_hours=10)
    block_start = pd.Timestamp(calendar["initial_history_end"])
    assert block_start == beginning + pd.Timedelta(hours=40)
    hours = []
    for fold in folds:
        validation, test_start, test_end = map(pd.Timestamp, (fold.validation_start, fold.test_start, fold.test_end))
        assert block_start < validation
        assert test_start - validation == pd.Timedelta(hours=5)
        assert test_end - test_start == pd.Timedelta(hours=10)
        hours.append((test_end - block_start) // pd.Timedelta(hours=1))
        block_start = test_end
    assert hours == [20 + (index < extra) for index in range(3)]
    assert block_start == end


@pytest.mark.parametrize("folds", [1, 3])
def test_calendar_rejects_a_span_too_short_to_fit_the_configured_windows(folds):
    start = pd.Timestamp("2025-01-01")
    with pytest.raises(ValueError, match=rf"no alcanza para {folds} folds"):
        fold_configurations(DataConfig(), folds, bounds=[(start, start + pd.Timedelta(days=1))])


def test_calendar_uses_remaining_space_for_initial_history_and_keeps_eval_on_stride():
    geometry = DataConfig(context=3, horizon=4, stride=3, min_train_windows=3, train_stride=2,
                          min_validation_windows=3, min_test_windows=5, seasonality=2)
    start = pd.Timestamp("2025-01-01")
    end = start + pd.Timedelta(hours=92)
    calendar, configs = fold_configurations(geometry, 3, bounds=[(start, end)])
    assert pd.Timestamp(calendar["initial_history_end"]) == start + pd.Timedelta(hours=11)
    for config in configs:
        validation, test, end = map(pd.Timestamp, (config.validation_start, config.test_start, config.test_end))
        assert test - validation == pd.Timedelta(hours=10)
        assert end - test == pd.Timedelta(hours=16)


def test_calendar_rejects_sliding_train_without_room_for_the_minimum_windows(caplog):
    geometry = DataConfig(context=6, horizon=3, stride=1, train_stride=2, min_train_windows=4)
    with pytest.raises(ValueError, match="train deslizante de 8 h.*mínimo de 9 h.*history-hours"):
        fold_configurations(geometry, 1, bounds=TEMPORAL_BOUNDS, train_history="sliding", history_hours=8)
    assert any(record.levelname == "CRITICAL" and "train deslizante" in record.message
               for record in caplog.records)


@pytest.mark.parametrize("folds", [1, 3])
@pytest.mark.parametrize("requested,effective", [(None, (672, 672)), ((337, 677), (336, 672))])
def test_evaluation_lengths_do_not_scale_with_a_long_calendar(folds, requested, effective):
    begin, end = map(pd.Timestamp, ("2024-01-01", "2026-11-19 20:00"))
    options = dict(zip(("validation_hours", "test_hours"), requested, strict=True)) if requested else {}
    calendar, configs = fold_configurations(DataConfig(), folds, bounds=[(begin, end)], **options)
    assert (calendar["validation_hours"], calendar["test_hours"]) == effective
    assert pd.Timestamp(configs[-1].test_end) == end
    next_block = pd.Timestamp(calendar["initial_history_end"])
    for config in configs:
        validation, test, fold_end = map(pd.Timestamp, (config.validation_start, config.test_start, config.test_end))
        assert validation > next_block
        assert test - validation == pd.Timedelta(hours=effective[0])
        assert fold_end - test == pd.Timedelta(hours=effective[1])
        assert (config.validation_hours, config.test_hours) == effective
        next_block = fold_end


@pytest.mark.parametrize("option,value", [("validation_hours", 47), ("test_hours", 95), ("test_hours", True)])
def test_evaluation_lengths_must_fit_the_required_windows(option, value):
    with pytest.raises(ValueError, match=option.replace("_", "-")):
        fold_configurations(DataConfig(), 1, bounds=TEMPORAL_BOUNDS, **{option: value})


@pytest.mark.parametrize("folds", [1, 3])
def test_preparation_follows_observed_data_as_the_csv_coverage_changes(temporal_family, tmp_path, folds):
    raw, geometry = temporal_family
    long_path = prepare_backtest(raw, tmp_path / "long", "PM25", geometry, folds=folds, min_stations=2)
    for source in raw.glob("*/*.csv"):
        readings = pd.read_csv(source, index_col=0, parse_dates=True)
        readings.loc[readings.index >= pd.Timestamp("2025-01-17"), "PM25"] = np.nan
        readings.to_csv(source)
    short_path = prepare_backtest(raw, tmp_path / "short", "PM25", geometry, folds=folds, min_stations=2)
    long, short = (json.loads(path.read_text()) for path in (long_path, short_path))
    assert long["calendar_start"] == short["calendar_start"] == "2024-12-28 00:00:00"
    for family, end in ((long, "2025-01-21 00:00:00"), (short, "2025-01-17 00:00:00")):
        assert family["calendar_end"] == family["folds"][-1]["test_end"] == end
    assert long["folds"][0]["test_start"] != short["folds"][0]["test_start"]


@pytest.mark.parametrize("first,minimum,common", [({"station-a"}, 1, 0), (None, 2, 1)])
def test_preparation_requires_enough_stations_in_common(
    temporal_family, tmp_path, monkeypatch, caplog, first, minimum, common,
):
    raw, geometry = temporal_family
    selected = {"fold_001": first, "fold_002": {"station-b"}}

    def prepare_disjoint(raw_root, output, pollutant, config, stations, *, reference):
        assert output.name in selected, "Debe abortar antes de preparar el tercer fold"
        return prepare(raw_root, output, pollutant, config, selected[output.name], reference=reference)

    monkeypatch.setattr("fd_airquality.backtesting.prepare", prepare_disjoint)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=f"Panel común insuficiente: {common} estaciones; mínimo exigido: {minimum}"):
        prepare_backtest(raw, output, "PM25", geometry, folds=3, min_stations=minimum)
    assert any(record.levelname == "CRITICAL" and "fold=2/3" in record.message for record in caplog.records)
    assert not list(output.glob("*/backtest_manifest.json"))


def test_previous_test_and_later_targets_become_training_data_in_the_next_fold(temporal_family, tmp_path):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, min_stations=2)
    family = json.loads(path.read_text())
    first, second = family["folds"]
    _, first_artifact = load_client_data(first["manifest"], 0)
    _, second_artifact = load_client_data(second["manifest"], 0)

    def observed_hours(artifact, split):
        positions = artifact[f"{split}_starts"][:, None] + geometry.context + np.arange(geometry.horizon)
        return set(artifact["timestamps"][positions][np.isfinite(artifact["values"][positions])])

    first_test = observed_hours(first_artifact, "test")
    second_train = observed_hours(second_artifact, "train")
    assert first_test and first_test <= second_train
    assert any(pd.Timestamp(first["test_end"]).value <= hour < pd.Timestamp(second["validation_start"]).value
               for hour in second_train)
    used = (second_train | observed_hours(second_artifact, "validation") | observed_hours(second_artifact, "test"))
    selected = ((second_artifact["timestamps"] >= pd.Timestamp(family["initial_history_end"]).value)
                & (second_artifact["timestamps"] < pd.Timestamp(second["test_end"]).value)
                & np.isfinite(second_artifact["values"]))
    assert set(second_artifact["timestamps"][selected]) <= used
    assert first_test.isdisjoint(observed_hours(first_artifact, "train"))
    assert second_train.isdisjoint(observed_hours(second_artifact, "validation"))
    assert second_train.isdisjoint(observed_hours(second_artifact, "test"))


def test_single_fold_uses_the_same_windows_as_a_standalone_preparation(temporal_family, tmp_path):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "backtests", "PM25", geometry, folds=1, min_stations=2)
    fold, = json.loads(path.read_text())["folds"]
    cuts = {key: fold[key] for key in ("validation_start", "test_start", "test_end")}
    fixed = prepare(raw, tmp_path / "prepared", "PM25", replace(geometry, **cuts))
    manifest = read_manifest(fixed)
    assert fold["fold"] == 1
    assert fold["test_end"] == "2025-01-21 00:00:00"
    assert read_manifest(fold["manifest"])["evaluation_sha256"] == manifest["evaluation_sha256"]
    for client in manifest["clients"]:
        partition = client["partition_id"]
        _, expected = load_client_data(fixed, partition)
        _, actual = load_client_data(fold["manifest"], partition)
        for split in ("train", "validation", "test"):
            np.testing.assert_array_equal(actual[f"{split}_starts"], expected[f"{split}_starts"])


@pytest.mark.parametrize("history", ["expanding", "sliding"])
def test_fixed_scalers_fit_only_each_folds_authorized_past(temporal_family, tmp_path, history):
    from fd_airquality.arrival import normalization_artifact, round_cutoff

    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=3, min_stations=2,
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
                             folds=2, min_stations=2, train_history="sliding")
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
    from fd_airquality.backtesting import _run_fold
    from fd_airquality.models import make_model, shared_state

    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, min_stations=2)
    initial_states = []

    def fresh(geometry, settings, **kwargs):
        model = make_model(geometry, settings, **kwargs)
        initial_states.append({key: value.detach().clone() for key, value in shared_state(model).items()})
        return model

    monkeypatch.setattr(f"fd_airquality.{'server_app' if mode == 'federated' else 'baselines'}.make_model", fresh)
    def poison_after_fold(*args, **kwargs):
        result = _run_fold(*args, **kwargs)
        for run in json.loads((result / "comparison.json").read_text())["runs"]:
            for checkpoint in Path(run).glob("*model.pt"):
                checkpoint.write_bytes(b"poisoned: cannot be restored")
        return result

    monkeypatch.setattr("fd_airquality.backtesting._run_fold", poison_after_fold)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"),
                           num_server_rounds=2, batch_size=8, hidden_size=8,
                           mlflow_tracking_uri="disabled")
    run_backtest(settings, ["mlp"], [mode], stage="all")
    assert len(initial_states) == (4 if mode == "local" else 2)
    for state in initial_states[1:]:
        for key, value in initial_states[0].items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("schema,protocol", [
    (1, "streaming"), (1, "independent"), (2, "temporal_cv"), (3, "temporal_cv"), (4, "temporal_cv"),
])
def test_legacy_fold_protocol_requires_fresh_preparation(single_fold_manifest, schema, protocol):
    path = single_fold_manifest
    family = json.loads(path.read_text())
    family.update(schema_version=schema, protocol=protocol)
    path.write_text(json.dumps(family))
    with pytest.raises(ValueError, match="reparto de calendario.*prepare-backtest"):
        run_backtest(RunSettings(manifest=str(path)), ["mlp"], ["local"])


def test_backtest_checks_calendar_against_the_prepared_cuts(temporal_family, tmp_path):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, min_stations=2)
    family = json.loads(path.read_text())
    family["initial_history_end"] = "2025-01-01 06:00:00"
    path.write_text(json.dumps(family))
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"))
    with pytest.raises(ValueError, match="no coinciden con el reparto"):
        run_backtest(settings, ["mlp"], ["local"])
    assert not (path.parent / "holdout_access.json").exists()


@pytest.mark.parametrize("stage", ["evaluation", "all"])
def test_backtest_rechecks_minimum_after_metric_filtering_before_any_training(
    temporal_family, tmp_path, monkeypatch, stage,
):
    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, min_stations=2)
    family = json.loads(path.read_text())
    item = family["folds"][-1]
    manifest_path = Path(item["manifest"])
    manifest = read_manifest(manifest_path)
    manifest["clients"][0]["validation_mase_defined"] = False
    manifest_path.write_text(json.dumps(manifest))
    item["sha256"] = file_digest(manifest_path)
    path.write_text(json.dumps(family))
    monkeypatch.setattr("fd_airquality.backtesting._run_fold", lambda *_: pytest.fail("No debe entrenar"))
    output = tmp_path / "runs"
    with pytest.raises(ValueError, match="Panel común insuficiente: 1 estaciones; mínimo exigido: 2"):
        run_backtest(RunSettings(manifest=str(path), output_dir=str(output)), ["mlp"], ["local"], stage=stage)
    audit, = output.glob("*/temporal_exclusions.csv")
    assert "station-a" in audit.read_text()
    assert "CRITICAL" in audit.with_name("backtest.log").read_text()
    assert not (path.parent / "holdout_access.json").exists()


@pytest.mark.parametrize("arguments", [
    ["prepare-backtest", "--protocol", "streaming"],
    ["prepare-backtest", "--protocol", "independent"],
    ["prepare-backtest", "--train-history", "new-only"],
    ["prepare-backtest", "--step-hours", "48"],
    ["prepare-backtest", "--fold-gap-hours", "6"],
    ["prepare-backtest", "--test-end", "2026-04-26"],
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
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, min_stations=2)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "runs"), num_server_rounds=1,
                            batch_size=8, mlflow_tracking_uri="disabled")
    output = run_backtest(settings, ["mlp"], ["federated"], stage="all")
    assert len(pd.read_csv(output / "temporal_comparison.csv")) == 2
    assert len(pd.read_csv(output / "temporal_seed_summary.csv")) == 2


def test_flora_history_restarts_at_each_fold(temporal_family, tmp_path, in_process_federation, mock_granite):
    from fd_airquality.models import restore_checkpoint

    raw, geometry = temporal_family
    path = prepare_backtest(raw, tmp_path / "prepared", "PM25", geometry, folds=2, min_stations=2)
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
