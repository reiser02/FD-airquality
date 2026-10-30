import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fd_airquality.data import (
    DataConfig,
    WindowDataset,
    load_client_data,
    prepare,
    read_manifest,
    split_station,
)
from fd_airquality.metrics import seasonal_error_scales
from fd_airquality.preprocessing import hourly_mean


@pytest.fixture
def global_geometry():
    return DataConfig(
        context=6, horizon=2, stride=2, seasonality=3,
        validation_start="2025-01-04", test_start="2025-01-06", test_end="2025-01-09",
        min_validation_windows=2, min_test_windows=2,
    )


def test_default_geometry_is_global_context90_and_horizon12():
    config = DataConfig()
    assert (config.context, config.horizon, config.split_mode, config.quality_mode) == (90, 12, "global", "causal")
    assert (config.validation_start, config.test_start, config.test_end) == ("2026-03-01", "2026-03-29", "2026-04-26")
    assert config.error_scale_hours == 28 * 24


def test_global_geometry_ignores_legacy_station_durations():
    config = DataConfig(horizon=30, stride=10)
    assert (config.validation_hours, config.test_hours) == (48, 96)
    with pytest.raises(ValueError, match="múltiplo"):
        replace(config, split_mode="station")


@pytest.mark.parametrize("hours", [0, True, 1.5, 24])
def test_error_scale_reference_must_allow_seasonal_pairs(hours):
    with pytest.raises(ValueError):
        DataConfig(error_scale_hours=hours)


@pytest.mark.parametrize("kwargs", [
    {"validation_start": "not-a-date"}, {"test_start": "2026-02-01"},
    {"test_end": "2026-03-29"}, {"validation_start": "2026-03-01 00:30"},
    {"validation_start": "2026-03-01T00:00:00Z"},
    {"quality_mode": "unknown"}, {"split_mode": "unknown"},
    {"min_validation_windows": 0}, {"min_test_windows": True},
])
def test_global_configuration_rejects_invalid_dates_and_rules(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


def test_global_cuts_prevent_future_information_between_stations(global_geometry):
    splits = []
    for offset in (0, -48):
        index = pd.date_range("2025-01-01", periods=300, freq="h") + pd.Timedelta(hours=offset)
        values = np.sin(np.arange(300) / 5) + np.arange(300) / 100
        values[30:35] = np.nan
        series = pd.Series(values, index=index)
        split = split_station(series, global_geometry)
        assert split is not None
        splits.append(split)
        for name, lower, upper in (
            ("train", None, global_geometry.validation_start),
            ("validation", global_geometry.validation_start, global_geometry.test_start),
            ("test", global_geometry.test_start, global_geometry.test_end),
        ):
            for start in split[f"{name}_starts"]:
                window = series.iloc[start:start + 8]
                assert window.notna().all()
                targets = window.index[6:]
                assert targets[-1] < pd.Timestamp(upper)
                if lower is not None:
                    assert targets[0] >= pd.Timestamp(lower)
    assert max(pd.Timestamp(s["train_end"]) for s in splits) < min(pd.Timestamp(s["validation_target_start"]) for s in splits)
    assert max(pd.Timestamp(s["validation_target_end"]) for s in splits) < min(pd.Timestamp(s["test_target_start"]) for s in splits)


def test_global_split_keeps_all_past_blocks_and_fragmented_evaluation(global_geometry):
    index = pd.date_range("2025-01-01", periods=220, freq="h")
    series = pd.Series(np.arange(220, dtype=float), index=index)
    series.iloc[[20, 40, 80, 105, 130, 155, 180, 205]] = np.nan
    split = split_station(series, global_geometry)
    assert split is not None
    assert {0, 21, 41}.issubset(set(split["train_starts"]))
    assert len(split["validation_starts"]) > 2
    assert len(split["test_starts"]) > 2
    assert series.index[split["test_starts"][0]] < pd.Timestamp(global_geometry.test_start)
    assert split["training_hours"] == int(split["train_mask"].sum())


def test_global_normalization_and_error_scales_ignore_future_values(global_geometry):
    series = pd.Series(np.sin(np.arange(240) / 9), index=pd.date_range("2025-01-01", periods=240, freq="h"))
    original = split_station(series, global_geometry)
    series.loc[global_geometry.validation_start:] += 1_000_000
    changed = split_station(series, global_geometry)
    for key in ("mean", "scale", "validation_mase_scale", "validation_rmsse_scale", "validation_scale_pairs"):
        assert original[key] == changed[key]
    assert original["test_mase_scale"] != changed["test_mase_scale"]
    np.testing.assert_array_equal(original["train_starts"], changed["train_starts"])
    capped = split_station(series, replace(global_geometry, max_train_windows=4))
    assert capped["mean"] == changed["mean"]
    assert capped["training_hours"] == changed["training_hours"]


def test_capped_training_end_still_reports_all_history_used_for_scaling(global_geometry):
    series = pd.Series(np.sin(np.arange(240) / 9), index=pd.date_range("2025-01-01", periods=240, freq="h"))
    split = split_station(series, replace(global_geometry, max_train_windows=1))
    assert pd.Timestamp(split["train_end"]) == series.index[split["train_mask"]][-1]


def test_constant_reference_scale_is_not_replaced_by_a_fake_error_denominator():
    series = pd.Series(np.ones(100))
    absolute, squared, count = seasonal_error_scales(series, 24)
    assert (absolute, squared, count) == (0., 0., 76)
    from fd_airquality.settings import RunSettings
    from fd_airquality.task import selection_value

    with pytest.raises(ValueError, match="indefinida"):
        selection_value({"mae": 1.0}, RunSettings())


def test_seasonal_error_scales_use_observed_endpoints_despite_intermediate_gap():
    series = pd.Series([1., 2., 4., 8., np.nan, 100., 102., 106., 114., 9999.])
    mae_scale, mse_scale, count = seasonal_error_scales(series.iloc[:9], 2)
    differences = np.array([3., 6., 92., 6., 12.])
    assert count == 5
    assert mae_scale == pytest.approx(np.abs(differences).mean())
    assert mse_scale == pytest.approx(np.square(differences).mean())


def test_reference_windows_exclude_old_and_future_values_and_do_not_follow_origins(global_geometry):
    config = replace(global_geometry, error_scale_hours=48)
    index = pd.date_range("2025-01-01", periods=240, freq="h")
    series = pd.Series(np.arange(240, dtype=float) ** 2, index=index)
    original = split_station(series, config)
    for name, end in (("validation", config.validation_start), ("test", config.test_start)):
        end = pd.Timestamp(end)
        assert pd.Timestamp(original[f"{name}_scale_start"]) == end - pd.Timedelta(hours=48)
        assert pd.Timestamp(original[f"{name}_scale_end"]) == end
        reference = series.loc[(index >= end - pd.Timedelta(hours=48)) & (index < end)]
        differences = reference.to_numpy()[3:] - reference.to_numpy()[:-3]
        assert original[f"{name}_scale_pairs"] == 45
        assert original[f"{name}_mase_scale"] == pytest.approx(np.abs(differences).mean())
        assert original[f"{name}_rmsse_scale"] == pytest.approx(np.square(differences).mean())
    changed = series.copy()
    changed.loc[index < pd.Timestamp(original["validation_scale_start"])] += 100_000
    changed.loc[config.test_start:] *= 100_000
    later = split_station(changed, config)
    for name in ("validation", "test"):
        for key in ("mase_scale", "rmsse_scale", "scale_pairs"):
            assert later[f"{name}_{key}"] == original[f"{name}_{key}"]
    capped = split_station(series, replace(config, max_train_windows=1))
    assert capped["test_mase_scale"] == original["test_mase_scale"]


def test_causal_quality_is_prefix_invariant_at_completed_hours():
    index = pd.date_range("2025-01-01", periods=48, freq="5min")
    frame = pd.DataFrame({"station": np.r_[np.arange(10.), np.repeat(17., 38)]}, index=index)
    full = hourly_mean(frame, "NO2", quality_mode="causal")
    for hours in (1, 2, 3):
        prefix = hourly_mean(frame.iloc[:hours * 12], "NO2", quality_mode="causal")
        pd.testing.assert_frame_equal(prefix, full.iloc[:hours])
    retrospective = hourly_mean(frame, "NO2", quality_mode="retrospective")
    assert not full.equals(retrospective)


def test_global_preparation_has_audited_retention_and_immutable_calendar(tmp_path, global_geometry):
    raw = tmp_path / "raw"
    for name, hours in (("active-a", 250), ("active-b", 230), ("retired", 60)):
        folder = raw / name
        folder.mkdir(parents=True)
        values = 10 + np.sin(np.arange(hours * 12) / 31)
        pd.DataFrame({"NO2": values}, index=pd.date_range("2025-01-01", periods=len(values), freq="5min")).to_csv(folder / f"{name}_NO2.csv")
    path = prepare(raw, tmp_path / "prepared", "NO2", global_geometry)
    manifest = read_manifest(path)
    assert manifest["schema_version"] == 4
    assert manifest["protocol"] == "global_temporal_forecasting_v3"
    assert len(manifest["clients"]) == 2
    audit = json.loads((path.parent / "retention.json").read_text())
    assert audit["candidate_stations"] == 3
    assert audit["eligible_stations"] == 2
    assert 0 < audit["train_window_retention_pct"] <= 100
    assert "retired" in pd.read_csv(path.parent / "excluded_stations.csv").station.to_list()
    for client in manifest["clients"]:
        _, artifact = load_client_data(str(path), client["partition_id"])
        assert "validation_mase_scale" in artifact and "test_mase_scale" in artifact
        assert "mase_scale" not in artifact
        assert "scale" not in client and "test_mase_scale" not in client
        assert client["test_scale_pairs"] == int(artifact["test_scale_pairs"])
        for name in ("train", "validation", "test"):
            assert len(WindowDataset(artifact, name, 6, 2)) == client[f"num_{name}_windows"]
    corrupted = json.loads(path.read_text())
    corrupted["clients"][0]["train_end"] = "2025-01-07"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="temporal"):
        read_manifest(path)


def test_missing_timestamps_also_delimit_global_blocks(global_geometry):
    index = pd.date_range("2025-01-01", periods=240, freq="h")
    series = pd.Series(np.sin(np.arange(240) / 5), index=index).drop(index[20])
    split = split_station(series, global_geometry)
    for name in ("train", "validation", "test"):
        for start in split[f"{name}_starts"]:
            assert (np.diff(series.index[start:start + 8].asi8) == pd.Timedelta(hours=1).value).all()


def test_client_checks_actual_window_dates_not_just_manifest_labels(global_prepared):
    from fd_airquality.artifacts import file_digest

    path, manifest = global_prepared
    client = manifest["clients"][0]
    dataset = Path(client["dataset"])
    with np.load(dataset, allow_pickle=False) as archive:
        artifact = {key: archive[key] for key in archive.files}
    artifact["train_starts"] = artifact["train_starts"].copy()
    artifact["train_starts"][-1] = artifact["test_starts"][0]
    np.savez_compressed(dataset, **artifact)
    manifest["clients"][0]["sha256"] = file_digest(dataset)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="temporal|ventanas"):
        load_client_data(str(path), 0)
