import numpy as np
import pandas as pd
import pytest

from fd_airquality.data import DataConfig, prepare, read_manifest


@pytest.fixture
def geometry():
    return DataConfig(context=6, horizon=2, stride=1, validation_hours=4, test_hours=6, seasonality=3,
                      split_mode="station", quality_mode="retrospective")


@pytest.fixture
def fragmented_series():
    index = pd.date_range("2025-01-01", periods=150, freq="h")
    values = 10 + np.sin(np.arange(len(index)) / 4) + np.arange(len(index)) / 100
    values[22:27] = np.nan
    values[62:67] = np.nan
    return pd.Series(values, index=index, name="station")


@pytest.fixture
def prepared(tmp_path, geometry):
    raw = tmp_path / "raw"
    for name, hours in (("station-a", 160), ("station-b", 185), ("too-short", 5)):
        directory = raw / name
        directory.mkdir(parents=True)
        index = pd.date_range("2025-01-01", periods=hours * 12, freq="5min")
        values = 10 + np.sin(np.arange(len(index)) / 30) + np.arange(len(index)) / 10000
        if hours > 10:
            values[20 * 12 : 25 * 12] = np.nan
            values[60 * 12 : 65 * 12] = np.nan
        pd.DataFrame({"NO2": values}, index=index).to_csv(directory / f"{name}_NO2.csv")
    path = prepare(raw, tmp_path / "prepared", "NO2", geometry)
    return path, read_manifest(path)


@pytest.fixture
def global_prepared(tmp_path):
    raw = tmp_path / "global_raw"
    for name, hours in (("station-a", 500), ("station-b", 520)):
        folder = raw / name
        folder.mkdir(parents=True)
        values = 10 + np.sin(np.arange(hours * 12) / 31) + np.arange(hours * 12) / 10000
        if name == "station-a":
            values[20 * 12:25 * 12] = np.nan
            values[60 * 12:65 * 12] = np.nan
        pd.DataFrame({"NO2": values}, index=pd.date_range("2025-01-01", periods=len(values), freq="5min")).to_csv(folder / f"{name}_NO2.csv")
    config = DataConfig(validation_start="2025-01-10", test_start="2025-01-12", test_end="2025-01-16")
    path = prepare(raw, tmp_path / "global_prepared", "NO2", config)
    return path, read_manifest(path)


@pytest.fixture
def selection_prepared(tmp_path, request):
    raw = tmp_path / "selection_raw"
    for offset, name in enumerate(getattr(request, "param", ("station-a", "station-b"))):
        folder = raw / name
        folder.mkdir(parents=True)
        index = pd.date_range("2025-01-01", periods=300 * 12, freq="5min")
        values = 10 + offset + np.sin(np.arange(len(index)) / 31) + np.arange(len(index)) / 10000
        if name == "station-a":
            values[96 * 12:144 * 12] = np.nan
        pd.DataFrame({"NO2": values}, index=index).to_csv(folder / f"{name}_NO2.csv")
    config = DataConfig(context=6, horizon=2, stride=2, error_scale_hours=48,
                        validation_start="2025-01-07", test_start="2025-01-09", test_end="2025-01-11")
    path = prepare(raw, tmp_path / "selection_prepared", "NO2", config)
    return path, read_manifest(path)
