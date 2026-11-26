import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fd_airquality.artifacts import file_digest
from fd_airquality.backtesting import prepare_backtest, run_backtest
from fd_airquality.cli import _parser
from fd_airquality.configuration import parse_configured_args
from fd_airquality.data import prepare, read_manifest
from fd_airquality.loaders import (
    filename_matches_pollutant,
    load_pollutant_file,
    normalize_pollutant,
)
from fd_airquality.preprocessing import hourly_mean, preprocess
from fd_airquality.settings import RunSettings, resolve_manifest


def test_preparation_defaults_to_pm25_and_accepts_lowercase(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser, commands = _parser()
    assert parse_configured_args(parser, commands, ["prepare-backtest"]).pollutant == "PM25"
    args = parse_configured_args(parser, commands, ["prepare-backtest", "--pollutant", "pm25"])
    assert normalize_pollutant(args.pollutant) == "PM25"


@pytest.mark.parametrize("pollutant", ["NO2", "O3", "CO", "PM10"])
def test_other_pollutants_are_rejected_by_cli_and_toml(tmp_path, monkeypatch, pollutant):
    monkeypatch.chdir(tmp_path)
    parser, commands = _parser()
    with pytest.raises(SystemExit) as exc:
        parse_configured_args(parser, commands, ["prepare-backtest", "--pollutant", pollutant])
    assert exc.value.code == 2
    (tmp_path / "fd-airquality.toml").write_text(f'[prepare-backtest]\npollutant = "{pollutant}"\n')
    with pytest.raises(ValueError, match="PM25"):
        parse_configured_args(parser, commands, ["prepare-backtest"])


@pytest.mark.parametrize("filename,expected", [
    ("station_PM25.csv", True), ("station_pm25.csv", True), ("PM25.csv", True),
    ("station_PM10.csv", False), ("station_NO2.csv", False),
    ("station_XPM25.csv", False), ("station_PM250.csv", False),
])
def test_csv_discovery_selects_only_pm25(filename, expected):
    assert filename_matches_pollutant(Path(filename), "pm25") is expected


@pytest.mark.parametrize("pollutant", ["NO2", "O3", "CO", "PM10", "", None])
def test_data_entrypoints_reject_other_pollutants_before_writing(tmp_path, geometry, pollutant):
    output = tmp_path / "prepared"
    frame = pd.DataFrame({"PM25": [1.]}, index=pd.date_range("2025-01-01", periods=1, freq="5min"))
    for operation in (
        lambda: prepare(tmp_path, output, pollutant, geometry),
        lambda: prepare_backtest(tmp_path, output, pollutant, geometry),
        lambda: load_pollutant_file(str(tmp_path / "missing.csv"), pollutant),
        lambda: hourly_mean(frame, pollutant),
        lambda: preprocess([], pollutant),
    ):
        with pytest.raises(ValueError, match="PM25"):
            operation()
    assert not output.exists()


def test_lowercase_preparation_ignores_other_csvs_and_records_canonical_name(tmp_path, geometry):
    raw = tmp_path / "raw" / "station"
    raw.mkdir(parents=True)
    index = pd.date_range("2025-01-01", periods=120 * 12, freq="5min")
    values = 10 + np.sin(np.arange(len(index)) / 30)
    pd.DataFrame({"PM25": values}, index=index).to_csv(raw / "station_pm25.csv")
    (raw / "station_NO2.csv").write_text("not a usable PM25 source")
    path = prepare(raw.parent, tmp_path / "prepared", "pm25", geometry)
    manifest = read_manifest(path)
    assert manifest["pollutant"] == "PM25"
    assert path.parent.name.startswith("PM25_")
    assert len(manifest["clients"]) == 1
    assert len(pd.read_csv(path.parent / "stations.csv")) == 1


@pytest.mark.parametrize("pollutant", ["NO2", "O3", "CO", "PM10", None])
def test_non_pm25_manifests_are_rejected_without_being_modified(prepared, pollutant):
    path, manifest = prepared
    manifest["pollutant"] = pollutant
    content = json.dumps(manifest)
    path.write_text(content)
    with pytest.raises(ValueError, match="PM25"):
        read_manifest(path)
    assert path.read_text() == content


@pytest.mark.parametrize("backtest", [False, True])
def test_latest_manifest_selection_ignores_other_pollutants(tmp_path, monkeypatch, backtest):
    monkeypatch.chdir(tmp_path)
    root, filename = "data/backtests", "backtest_manifest.json"
    for timestamp, pollutant in enumerate(("PM25", "PM25", "NO2", "O3", "CO", "PM10"), start=1):
        path = Path(root) / f"preparation_{timestamp}" / filename
        path.parent.mkdir(parents=True)
        fold = path.with_name("manifest.json")
        fold.write_text(json.dumps({"pollutant": pollutant}))
        path.write_text(json.dumps({"pollutant": pollutant, "folds": [
            {"manifest": fold.name, "sha256": file_digest(fold)},
        ]}))
        os.utime(path, ns=(timestamp, timestamp))
    expected = (Path(root) / "preparation_2" / (filename if backtest else "manifest.json")).resolve()
    assert resolve_manifest(None, backtest=backtest) == expected


@pytest.mark.parametrize("backtest", [False, True])
def test_missing_pm25_preparation_requests_the_correct_command(tmp_path, monkeypatch, backtest):
    monkeypatch.chdir(tmp_path)
    root, filename = "data/backtests", "backtest_manifest.json"
    path = Path(root) / "legacy" / filename
    path.parent.mkdir(parents=True)
    path.write_text('{"pollutant": "NO2"}')
    with pytest.raises(FileNotFoundError, match="PM25") as exc:
        resolve_manifest(None, backtest=backtest)
    assert str(exc.value).endswith("prepare-backtest." if backtest else "prepare-backtest --folds 1.")


def test_backtest_rejects_other_pollutants_before_creating_results(tmp_path):
    path = tmp_path / "backtest_manifest.json"
    path.write_text('{"pollutant": "NO2"}')
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "results"))
    with pytest.raises(ValueError, match="PM25"):
        run_backtest(settings, ["mlp"], ["local"])
    assert not Path(settings.output_dir).exists()
