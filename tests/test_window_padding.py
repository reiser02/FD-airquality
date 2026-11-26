import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fd_airquality.arrival import (
    arrival_metrics,
    normalization_artifact,
    round_validation_bounds,
    training_view,
)
from fd_airquality.artifacts import file_digest
from fd_airquality.data import DataConfig, WindowDataset, load_client_data, prepare, read_manifest
from fd_airquality.models import make_model, shared_state
from fd_airquality.settings import RunSettings
from fd_airquality.splitting import masked_window_starts
from fd_airquality.task import configure_torch, train_local


def padded_artifact(*, dtype=np.float32):
    values = np.array([0, 1, 2, 3, 4, 999, 999, 999], dtype=dtype)
    return {"values": values, "timestamps": pd.date_range("2025-01-01", periods=len(values), freq="h").asi8,
            "train_starts": np.array([-2, 1]), "validation_starts": np.array([0]), "test_starts": np.array([0]),
            "mean": np.array(0.), "scale": np.array(1.), "allow_missing": np.array(True),
            "window_padding": np.array("both"), "train_target_end": np.array(5)}


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_padding_only_masks_virtual_slots_and_preserves_observed_zeros_and_truth(dtype):
    artifact = padded_artifact(dtype=dtype)
    original = {key: value.copy() for key, value in artifact.items()}
    dataset = WindowDataset(artifact, "train", 3, 3)
    first, last = dataset[0], dataset[1]
    np.testing.assert_array_equal(first[0].numpy(), [np.nan, np.nan, 0.])
    np.testing.assert_array_equal(first[1].numpy(), [1., 2., 3.])
    np.testing.assert_array_equal(last[0].numpy(), [1., 2., 3.])
    np.testing.assert_array_equal(last[1].numpy(), [4., np.nan, np.nan])
    assert dataset.padding_counts() == (2, 2)
    np.testing.assert_array_equal(dataset.observation_counts(), [[1, 3], [3, 1]])
    np.testing.assert_array_equal(dataset.target_values(), [[1., 2., 3.], [4., np.nan, np.nan]])
    assert dataset.context_timestamps(0)[-1] == artifact["timestamps"][0]
    assert dataset.context_timestamps(0)[0] == artifact["timestamps"][0] - pd.Timedelta(hours=2).value
    for key, value in original.items():
        np.testing.assert_array_equal(artifact[key], value)
    # Neither evaluation split inherits training padding or its target cutoff.
    for split in ("validation", "test"):
        external = WindowDataset(artifact, split, 3, 3)
        assert external.padding_mode == "none" and external.padding_counts() == (0, 0)
        np.testing.assert_array_equal(external[0][1].numpy(), [3., 4., 999.])


def test_padding_retains_missing_labels_instead_of_imputing_them_or_scaling_them_to_zero():
    artifact = padded_artifact()
    artifact["values"][2] = np.nan
    dataset = WindowDataset(artifact | {"mean": np.array(10.), "scale": np.array(2.)}, "train", 3, 3)
    np.testing.assert_array_equal(dataset[0][0].numpy(), [np.nan, np.nan, -5.])
    np.testing.assert_array_equal(dataset[0][1].numpy(), [-4.5, np.nan, -3.5])
    np.testing.assert_array_equal(dataset[1][1].numpy(), [-3., np.nan, np.nan])
    assert dataset.observation_counts()[1].sum() == 3


def test_context_metadata_keeps_actual_dates_after_a_calendar_gap():
    artifact = padded_artifact()
    artifact["timestamps"][1:] += pd.Timedelta(hours=2).value
    artifact["train_starts"] = np.array([1])
    dataset = WindowDataset(artifact, "train", 3, 3)
    np.testing.assert_array_equal(dataset.context_timestamps(0), artifact["timestamps"][1:4])


def test_padding_cannot_create_an_observed_context_or_target_from_nothing():
    series = pd.Series([0., 1., 2.], index=pd.date_range("2025-01-01", periods=3, freq="h"))
    options = {"context": 6, "horizon": 3, "stride": 3, "validation_start": "2025-01-01 03:00",
               "test_start": "2025-01-02", "test_end": "2025-01-03"}
    assert not len(masked_window_starts(series, **options)["train_starts"])
    np.testing.assert_array_equal(masked_window_starts(series, window_padding="both", **options)["train_starts"], [-5, -4])
    empty = masked_window_starts(series * np.nan, window_padding="both", **options)
    assert not len(empty["train_starts"])
    assert not len(empty["validation_starts"]) and not len(empty["test_starts"])


@pytest.mark.parametrize("mode,start", [("none", -1), ("right", -1), ("both", -3), ("right", 2)])
def test_dataset_rejects_padding_outside_allowed_geometry(mode, start):
    artifact = padded_artifact() | {"window_padding": np.array(mode), "train_starts": np.array([start])}
    with pytest.raises(ValueError, match="temporal"):
        WindowDataset(artifact, "train", 3, 3)


@pytest.mark.parametrize("options", [
    {"window_padding": "left"}, {"window_padding": "both"},
    {"window_padding": "right", "split_mode": "station", "quality_mode": "retrospective"},
])
def test_invalid_or_strict_padding_configuration_is_rejected(options):
    with pytest.raises(ValueError, match="padding"):
        DataConfig(**options)


@pytest.mark.parametrize(("mode", "expected"), [("none", [0, 3]), ("right", [0, 3, 6]), ("both", [-3, 0, 3, 6])])
def test_padding_keeps_the_origin_grid_and_only_changes_training(mode, expected):
    index = pd.date_range("2025-01-01", periods=36, freq="h")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    series.iloc[[1, 7]] = np.nan
    # L=5 gives at most 4 virtual context slots, but -4 is off the stride-3 grid.
    # The origin at 6 has a target at hour 13, just outside the training interval.
    options = {"context": 5, "horizon": 3, "train_stride": 3, "stride": 3,
               "validation_start": "2025-01-01 13:00", "test_start": "2025-01-02",
               "test_end": "2025-01-02 12:00", "window_padding": mode}
    actual = masked_window_starts(series, **options)
    np.testing.assert_array_equal(actual["train_starts"], expected)
    np.testing.assert_array_equal(actual["validation_starts"], [8, 11, 14])
    np.testing.assert_array_equal(actual["test_starts"], [19, 22, 25, 28])
    restricted = masked_window_starts(series, **options, min_context_observed=2, min_target_observed=3)
    np.testing.assert_array_equal(restricted["train_starts"], [3])


def test_masked_padding_does_not_bridge_calendar_gaps():
    index = pd.date_range("2025-01-01", periods=36, freq="h")
    series = pd.Series(np.arange(len(index), dtype=float), index=index).drop(index[7])
    starts = masked_window_starts(series, context=5, horizon=3, stride=3, train_stride=3,
                                  validation_start="2025-01-01 13:00", test_start="2025-01-02",
                                  test_end="2025-01-02 12:00", window_padding="both")
    np.testing.assert_array_equal(starts["train_starts"], [-3])


@pytest.fixture(params=["none", "right", "both"])
def padding_prepared(arrival_raw, tmp_path, request):
    raw, config = arrival_raw
    path = prepare(raw, tmp_path / request.param, "PM25",
                   replace(config, window_padding=request.param, max_train_windows=7))
    return path, read_manifest(path)


def test_padding_preparation_integrity_preserves_canonical_arrays_and_evaluation(padding_prepared, arrival_prepared):
    path, manifest = padding_prepared
    original_path, original = arrival_prepared
    mode = manifest["geometry"]["window_padding"]
    assert manifest["schema_version"] == (5 if mode == "none" else 6)
    assert manifest["evaluation_sha256"] == original["evaluation_sha256"]
    blocks = pd.read_csv(path.parent / "blocks.csv")
    coverage = pd.read_csv(path.parent / "coverage.csv")
    for partition, client in enumerate(manifest["clients"]):
        _, artifact = load_client_data(str(path), partition)
        _, canonical = load_client_data(str(original_path), partition)
        for key in ("values", "timestamps", "train_mask", "useful_readings", "quality_flags",
                    "validation_starts", "test_starts"):
            np.testing.assert_array_equal(artifact[key], canonical[key])
        assert int(artifact["train_target_end"]) == int(np.searchsorted(
            artifact["timestamps"], pd.Timestamp(manifest["geometry"]["validation_start"]).value,
        ))
        train = WindowDataset(artifact, "train", 6, 3)
        assert len(train) == client["num_train_windows"] > 7
        assert train.observation_counts()[1].sum() == client["num_train_targets"]
        for split in ("train", "validation", "test"):
            assert blocks.loc[blocks.station == client["station"], f"num_{split}_windows"].sum() == client[f"num_{split}_windows"]
        local = coverage.loc[coverage.station == client["station"]]
        assert (local.observed_target_hours_used <= local.observed_hours).all()


def test_arrival_padding_keeps_targets_in_their_block_and_outside_the_reserve(padding_prepared):
    path, manifest = padding_prepared
    geometry = manifest["geometry"]
    settings = RunSettings(num_server_rounds=3, local_epochs=3, round_validation_hours=12)
    _, artifact = load_client_data(str(path), 0)
    seen_targets = set()
    for number in (1, 2, 3):
        dataset = training_view(artifact, geometry, settings, number)
        lower, upper = round_validation_bounds(geometry, settings, number)
        assert dataset.round_validation_interval == (lower, upper)
        positions = dataset.starts[:, None] + dataset.context + np.arange(dataset.horizon)
        truth = dataset.target_values()
        observed = np.isfinite(truth)
        times = artifact["timestamps"][positions[observed]]
        assert (times < dataset.available_through.value).all()
        if number > 1:
            assert (times >= dataset.arrival_lower.value).all()
        assert not seen_targets.intersection(times)
        assert not ((times >= lower.value) & (times < upper.value)).any()
        assert dataset.target_time_bounds()[1] < dataset.available_through.value
        np.testing.assert_array_equal(dataset.observation_counts()[1], observed.sum(axis=1))
        seen_targets.update(times)


def test_padding_masks_partial_round_horizons_and_permits_blocks_shorter_than_horizon():
    artifact = padded_artifact() | {"train_starts": np.arange(-2, 4), "train_target_end": np.array(7)}
    geometry = {"context": 3, "horizon": 3, "split_mode": "global", "validation_start": "2025-01-01 07:00"}
    settings = RunSettings(num_server_rounds=3, arrival_period_hours=1)
    first, second, third = [training_view(artifact, geometry, settings, number) for number in (1, 2, 3)]
    assert len(second) == len(third) == 1
    np.testing.assert_array_equal(second[0][1].numpy(), [999., np.nan, np.nan])
    np.testing.assert_array_equal(third[0][1].numpy(), [999., np.nan, np.nan])
    assert first.target_end == 5 and second.target_end == 6 and third.target_end == 7
    assert second.padding_counts() == (0, 2)
    assert second.target_time_bounds() == (artifact["timestamps"][5],) * 2


def test_all_missing_new_block_is_empty_instead_of_replaying_old_targets():
    artifact = padded_artifact() | {"train_starts": np.arange(-2, 4), "train_target_end": np.array(7)}
    artifact["values"][5] = np.nan
    geometry = {"context": 3, "horizon": 3, "split_mode": "global", "validation_start": "2025-01-01 07:00"}
    dataset = training_view(artifact, geometry, RunSettings(num_server_rounds=3, arrival_period_hours=1), 2)
    assert not len(dataset)
    metrics = arrival_metrics(dataset)
    assert metrics["arrival-unobserved-windows"] == 1
    assert dataset.padding_counts() == (0, 0)
    assert "train-last-target-ns" not in metrics


def test_invalid_prediction_is_not_hidden_by_a_padded_target_mask():
    class InvalidForecast(torch.nn.Linear):
        def forward(self, inputs):
            output = super().forward(inputs)
            output[:, -1] = float("nan")
            return output

    artifact = padded_artifact() | {"train_starts": np.array([1])}
    with pytest.raises(ValueError, match="no finitas"):
        train_local(InvalidForecast(3, 3), WindowDataset(artifact, "train", 3, 3),
                    RunSettings(batch_size=8), torch.device("cpu"), 13)


def test_padding_is_counted_missing_by_full_availability_threshold():
    from fd_airquality.availability import filter_training_windows

    artifact = padded_artifact() | {"train_starts": np.array([-2, 0, 1]), "train_target_end": np.array(6)}
    artifact["values"][0] = 0.  # A valid zero is not padding.
    dataset = WindowDataset(artifact, "train", 3, 3)
    full = filter_training_windows(dataset, RunSettings(training_data_availability_threshold="[1.0,1.0]"))
    np.testing.assert_array_equal(full.starts, [0])
    assert full.availability_excluded_windows == 2
    assert len(dataset) == 3


def test_unreleased_values_do_not_change_the_scaler_or_sampled_update(padding_prepared):
    path, manifest = padding_prepared
    geometry = manifest["geometry"]
    _, artifact = load_client_data(str(path), 0)
    settings = RunSettings(hidden_size=8, batch_size=16, num_server_rounds=3, normalization="fixed")
    cutoff = pd.Timestamp(geometry["validation_start"]) - pd.Timedelta(hours=48)
    changed = artifact | {"values": artifact["values"].copy()}
    changed["values"][artifact["timestamps"] >= cutoff.value] = np.nan
    states, scalers = [], []
    for source in (artifact, changed):
        local = normalization_artifact(source, geometry, settings)
        scalers.append(local["normalization_sha256"])
        dataset = training_view(local, geometry, settings, 1)
        device = configure_torch(settings, 13)
        model = make_model(geometry, settings)
        metrics = train_local(model, dataset, settings, device, 19,
                              max_train_windows=geometry["max_train_windows"])
        assert metrics["admissible-windows"] == len(dataset) > 7
        assert metrics["num-examples"] == metrics["unique-training-windows"] == 7
        assert metrics["sampled-window-occurrences"] == 16
        assert metrics["replacement-window-occurrences"] == 9
        states.append({key: value.detach().clone() for key, value in shared_state(model).items()})
    assert scalers[0] == scalers[1]
    torch.testing.assert_close(*states, rtol=0, atol=0)


@pytest.mark.parametrize(("corruption", "reason"), [
    ("padding", "política de padding"), ("target_end", "extremo de targets"),
    ("missing_end", "extremo de targets"), ("schema", "Esquema de padding"),
    ("target_count", "recuento de targets"),
])
def test_padded_schema_rejects_incompatible_artifacts(arrival_raw, tmp_path, corruption, reason):
    raw, config = arrival_raw
    path = prepare(raw, tmp_path / "padded", "PM25", replace(config, window_padding="both"))
    manifest = read_manifest(path)
    _, artifact = load_client_data(str(path), 0)
    client = manifest["clients"][0]
    if corruption == "padding":
        artifact["window_padding"] = np.array("none")
    elif corruption == "target_end":
        artifact["train_target_end"] += 1
    elif corruption == "missing_end":
        del artifact["train_target_end"]
    elif corruption == "schema":
        manifest["schema_version"] = 5
    else:
        client["num_train_targets"] += 1
    if corruption in {"padding", "target_end", "missing_end"}:
        np.savez_compressed(client["dataset"], **artifact)
        client["sha256"] = file_digest(Path(client["dataset"]))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=reason):
        load_client_data(str(path), 0)
