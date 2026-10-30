import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fd_airquality.data import DataConfig, WindowDataset, load_client_data, prepare, split_station


def test_split_never_crosses_gaps_and_targets_are_ordered(fragmented_series, geometry):
    split = split_station(fragmented_series, geometry)
    assert split is not None
    width = geometry.context + geometry.horizon
    for name in ("train_starts", "validation_starts", "test_starts"):
        for start in split[name]:
            window = fragmented_series.iloc[start : start + width]
            assert len(window) == width
            assert np.isfinite(window).all()
            assert (window.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    last_train = max(split["train_starts"]) + width - 1
    first_val_target = min(split["validation_starts"]) + geometry.context
    last_val_target = max(split["validation_starts"]) + width - 1
    first_test_target = min(split["test_starts"]) + geometry.context
    assert last_train < first_val_target <= last_val_target < first_test_target


def test_scaler_does_not_inspect_validation_or_test_values(fragmented_series, geometry):
    initial = split_station(fragmented_series, geometry)
    changed = fragmented_series.copy()
    changed.loc[initial["validation_target_start"] :] += 100_000
    later = split_station(changed, geometry)
    assert initial["mean"] == later["mean"]
    assert initial["scale"] == later["scale"]
    np.testing.assert_array_equal(initial["train_starts"], later["train_starts"])


def test_missing_timestamp_is_not_compacted(fragmented_series, geometry):
    missing_time = fragmented_series.index[100]
    fragmented_series = fragmented_series.drop(missing_time)
    from fd_airquality.preprocessing import ensure_datetime_series

    regular = ensure_datetime_series(fragmented_series, "h", "station")
    assert np.isnan(regular.loc[missing_time])
    split = split_station(regular, geometry)
    for start in split["train_starts"]:
        assert missing_time not in regular.iloc[start : start + geometry.context + geometry.horizon].index


def test_all_earlier_eligible_blocks_are_reused(fragmented_series, geometry):
    split = split_station(fragmented_series, geometry)
    assert 0 in split["train_starts"]
    assert 27 in split["train_starts"]
    assert 67 in split["train_starts"]


def test_capped_training_windows_are_deterministic(fragmented_series, geometry):
    capped = replace(geometry, max_train_windows=5)
    one = split_station(fragmented_series, capped)
    two = split_station(fragmented_series, capped)
    assert len(one["train_starts"]) == 5
    np.testing.assert_array_equal(one["train_starts"], two["train_starts"])


def test_preparation_keeps_data_local_and_reports_excluded_stations(prepared):
    path, manifest = prepared
    assert len(manifest["clients"]) == 2
    assert "too-short" in pd.read_csv(path.parent / "excluded_stations.csv").station.to_list()
    assert {row["station"] for row in manifest["clients"]} == {"station-a", "station-b"}
    serialized = json.dumps(manifest)
    assert '"values"' not in serialized
    assert '"mean"' not in serialized
    assert '"scale"' not in serialized
    blocks = pd.read_csv(path.parent / "blocks.csv")
    for client in manifest["clients"]:
        _, artifact = load_client_data(str(path), client["partition_id"])
        for split in ("train", "validation", "test"):
            dataset = WindowDataset(artifact, split, 6, 2)
            assert len(dataset) == client[f"num_{split}_windows"]
            inputs, targets = dataset[0]
            assert inputs.shape == (6,)
            assert targets.shape == (2,)
            station_blocks = blocks.loc[blocks.station == client["station"]]
            assert station_blocks[f"num_{split}_windows"].sum() == len(dataset)


def test_changed_local_artifact_is_rejected(prepared):
    path, manifest = prepared
    load_client_data(str(path), 0)  # changes must also be detected after a previous successful load
    dataset = manifest["clients"][0]["dataset"]
    with open(dataset, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="cambió"):
        load_client_data(str(path), 0)


@pytest.mark.parametrize("missing_timestamp", [False, True])
def test_dataset_rejects_nan_and_missing_hours(missing_timestamp):
    values = np.arange(10, dtype=np.float32)
    times = pd.date_range("2025-01-01", periods=10, freq="h").asi8.copy()
    if missing_timestamp:
        times[4:] += pd.Timedelta(hours=1).value
    else:
        values[4] = np.nan
    artifact = {
        "values": values, "timestamps": times, "train_starts": np.array([0]),
        "mean": np.array(0.0), "scale": np.array(1.0),
    }
    with pytest.raises(ValueError, match="hueco|ausentes"):
        WindowDataset(artifact, "train", 6, 2)


@pytest.mark.parametrize("kwargs", [
    {"context": 0}, {"stride": 13}, {"split_mode": "station", "validation_hours": 13},
    {"max_train_windows": 0}, {"min_train_windows": 5, "max_train_windows": 3},
    {"context": 1.5}, {"context": True},
])
def test_invalid_geometry_fails_closed(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


def test_no_eligible_station_still_writes_exclusions(tmp_path, geometry):
    raw = tmp_path / "raw" / "short"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {"NO2": np.arange(12, dtype=float)},
        index=pd.date_range("2025-01-01", periods=12, freq="5min"),
    ).to_csv(raw / "short_NO2.csv")
    with pytest.raises(ValueError, match="Ninguna estación"):
        prepare(raw.parent, tmp_path / "prepared", "NO2", geometry)
    reports = list((tmp_path / "prepared").glob("*/excluded_stations.csv"))
    assert len(reports) == 1
    assert pd.read_csv(reports[0]).station.to_list() == ["short"]


def test_preparation_preserves_frozen_and_negative_intervals(tmp_path, geometry):
    raw = tmp_path / "raw" / "station"
    raw.mkdir(parents=True)
    index = pd.date_range("2025-01-01", periods=120 * 12, freq="5min")
    values = 10 + np.sin(np.arange(len(index)) / 30)
    values[:12] = 50.0  # a frozen sensor, not twelve valid observations
    values[24:36] = -1.0  # below the pollutant detection limit
    pd.DataFrame({"NO2": values}, index=index).to_csv(raw / "station_NO2.csv")
    path = prepare(raw.parent, tmp_path / "prepared", "NO2", geometry)
    _, artifact = load_client_data(str(path), 0)
    assert np.isnan(artifact["values"][0])
    assert np.isnan(artifact["values"][2])
    assert np.isfinite(artifact["values"][1])


def test_special_upct_format_preserves_station_identity(tmp_path, geometry):
    raw = tmp_path / "raw" / "UPCT"
    raw.mkdir(parents=True)
    index = pd.date_range("2025-01-01 00:00:03", periods=120 * 12, freq="5min")
    frame = pd.DataFrame({
        "Datetime": index, "NO2 (ug/m3)": 10 + np.sin(np.arange(len(index)) / 30),
        "Flag NO2": 0,
    })
    path = raw / "UPCT_NO2.csv"
    path.write_text("UPCT export\n" + frame.to_csv(index=False, sep=";", decimal=","))
    manifest_path = prepare(raw.parent, tmp_path / "prepared", "NO2", geometry)
    manifest, artifact = load_client_data(str(manifest_path), 0)
    assert manifest["clients"][0]["station"] == "UPCT_NO2"
    assert len(manifest["clients"]) == 1
    assert len(artifact["values"]) == 120
