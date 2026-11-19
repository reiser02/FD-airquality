import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from fd_airquality.data import (
    DataConfig,
    WindowDataset,
    load_client_data,
    prepare,
    read_manifest,
    split_station,
)
from fd_airquality.loaders import load_pollutant_file
from fd_airquality.metrics import compute_mase, compute_rmsse, effective_error_scales
from fd_airquality.missing import impute_context, resolve_missing_policy
from fd_airquality.models import ForecastAutoencoder, make_model
from fd_airquality.preprocessing import ensure_datetime_series, preprocess
from fd_airquality.quality import target_signature
from fd_airquality.settings import RunSettings
from fd_airquality.task import evaluate_local, train_local


@pytest.mark.parametrize("policy,expected", [
    ("ffill", [0., 2., 2., 6., 6.]), ("linear", [0., 2., 4., 6., 6.]),
    ("zero", [0., 2., 0., 6., 0.]), ("mean", [4., 2., 4., 6., 4.]),
])
def test_imputers_use_only_the_current_history_without_modifying_it(policy, expected):
    inputs = torch.tensor([[float("nan"), 2., float("nan"), 6., float("nan")]])
    original = inputs.clone()
    torch.testing.assert_close(impute_context(inputs, policy), torch.tensor([expected]))
    torch.testing.assert_close(inputs, original, equal_nan=True)
    joined = torch.cat([inputs, inputs + 100])
    torch.testing.assert_close(impute_context(joined, policy)[0], torch.tensor(expected))


def test_native_and_strict_policies_are_not_silently_substituted():
    inputs = torch.tensor([[1., float("nan")]])
    assert torch.isnan(impute_context(inputs, "native")[0, 1])
    with pytest.raises(ValueError, match="strict"):
        impute_context(inputs, "strict")
    with pytest.raises(ValueError, match="soportada"):
        resolve_missing_policy("autoencoder", "native")
    assert resolve_missing_policy("ttm-r3") == "native"
    assert resolve_missing_policy("mlp", overrides='{"mlp":"mean"}') == "mean"


def test_nan_aware_metrics_keep_calendar_and_mask_only_labels():
    reference = pd.Series([1., np.nan, 3., 9., 5.], index=pd.date_range("2025-01-01", periods=5, freq="h"))
    actual = pd.Series([10., np.nan, 14.], index=pd.date_range("2025-01-02", periods=3, freq="h"))
    predicted = pd.Series([12., 999., 10.], index=actual.index)
    assert compute_mase(actual, predicted, reference, seasonality_m=2) == 1.5
    assert compute_rmsse(actual, predicted, reference, seasonality_m=2) == pytest.approx(np.sqrt(2.5))
    predicted.iloc[1] = np.nan
    with pytest.raises(ValueError, match="no finitas"):
        compute_mase(actual, predicted, reference, seasonality_m=2)
    assert effective_error_scales(0., 0., 4) == (0., 0.)
    assert effective_error_scales(0., 0., 4, policy="impermanent") == (.01, .0001)
    assert effective_error_scales(0., 0., 0, policy="impermanent") == (0., 0.)


@pytest.fixture
def masked_geometry():
    return DataConfig(context=6, horizon=3, stride=3, seasonality=3, window_policy="masked",
                      validation_start="2025-01-04", test_start="2025-01-06", test_end="2025-01-08",
                      min_validation_windows=2, min_test_windows=2, hourly_repeat_filter=0)


def test_masked_windows_keep_holes_and_training_statistics_ignore_future(masked_geometry):
    index = pd.date_range("2025-01-01", periods=240, freq="h")
    series = pd.Series(np.arange(240, dtype=float), index=index)
    series.iloc[::5] = np.nan
    split = split_station(series, masked_geometry)
    assert split is not None
    assert split_station(series, replace(masked_geometry, window_policy="strict")) is None
    changed = series.copy()
    changed.loc[masked_geometry.validation_start:] += 1e6
    later = split_station(changed, masked_geometry)
    assert split["mean"] == later["mean"]
    assert split["scale"] == later["scale"]
    for name in ("train", "validation", "test"):
        for start in split[f"{name}_starts"]:
            assert series.iloc[start:start + 9].isna().any()
            assert series.iloc[start + 6:start + 9].notna().any()


@pytest.fixture
def masked_prepared(tmp_path, masked_geometry):
    raw = tmp_path / "raw"
    for offset, station in enumerate(("a", "b")):
        directory = raw / station
        directory.mkdir(parents=True)
        values = 10 + offset + np.sin(np.arange(240 * 12) / 7)
        values.reshape(-1, 12)[offset::5] = np.nan
        pd.DataFrame({"PM25": values}, index=pd.date_range("2025-01-01", periods=len(values), freq="5min")).to_csv(directory / "PM25.csv")
    path = prepare(raw, tmp_path / "prepared", "PM25", masked_geometry)
    return path, read_manifest(path)


def test_masked_artifacts_and_evaluation_use_real_target_counts(masked_prepared):
    path, manifest = masked_prepared
    assert manifest["schema_version"] == 5
    _, artifact = load_client_data(str(path), 0)
    dataset = WindowDataset(artifact, "test", 6, 3)
    model = make_model({"context": 6, "horizon": 3}, RunSettings(hidden_size=8))
    stats = evaluate_local(model, dataset, RunSettings(), torch.device("cpu"), artifact, 3)
    assert 0 < stats["num-targets"] < stats["num-target-slots"]
    assert stats["num-targets"] == manifest["clients"][0]["num_test_targets"]
    assert stats["missing-context-values"] > 0
    assert sum(stats[f"lead-{i:02d}-count"] for i in range(1, 4)) == stats["num-targets"]
    assert stats["seasonal-baseline-mae"] >= 0


@pytest.mark.parametrize("microbatch", [1, 3])
@pytest.mark.parametrize("model", ["mlp", "autoencoder"])
def test_masked_microbatches_match_full_batch_gradients(monkeypatch, microbatch, model):
    torch.manual_seed(12)
    inputs, actual = torch.randn(7, 4, dtype=torch.float64), torch.randn(7, 3, dtype=torch.float64)
    actual[::2, :2] = float("nan")
    inputs[::2, 1] = float("nan")
    original = (make_model({"context": 4, "horizon": 3}, RunSettings(hidden_size=8))
                if model == "mlp" else ForecastAutoencoder(4, 3, 8, 2)).double()
    full, micro = deepcopy(original), deepcopy(original)
    settings = RunSettings(model=model, batch_size=7, local_epochs=2,
                           early_stopping=0, lr_scheduler="none")
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 7)
    reference = train_local(full, TensorDataset(inputs, actual), settings, torch.device("cpu"), 17, .1)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: microbatch)
    result = train_local(micro, TensorDataset(inputs, actual), settings, torch.device("cpu"), 17, .1)
    assert result["num-targets"] == int(torch.isfinite(actual).sum())
    assert result["optimizer-steps"] == reference["optimizer-steps"] == 2
    for key, value in full.state_dict().items():
        torch.testing.assert_close(micro.state_dict()[key], value, atol=1e-10, rtol=1e-9)


def test_wrong_horizon_is_rejected_before_broadcasting(masked_prepared):
    path, _ = masked_prepared
    _, artifact = load_client_data(str(path), 0)
    dataset = WindowDataset(artifact, "test", 6, 3)
    with pytest.raises(ValueError, match="Forma"):
        evaluate_local(torch.nn.Linear(6, 1), dataset, RunSettings(missing_policy="ffill"), torch.device("cpu"), artifact, 3)


def test_evaluation_scores_canonical_labels_and_counts_all_negative_forecasts():
    values = np.array([.1234567, np.nan, .3456789, .456789, .5678901, np.nan,
                       .7890123, .8901234, .9012345], dtype=np.float32)
    artifact = {"values": values, "timestamps": pd.date_range("2025-01-01", periods=9, freq="h").asi8,
                "test_starts": np.array([0, 3]), "mean": np.array(1e8), "scale": np.array(3.),
                "allow_missing": np.array(True), "test_mase_scale": np.array(2.),
                "test_rmsse_scale": np.array(5.), "test_scale_pairs": np.array(4)}
    model = torch.nn.Linear(3, 3)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    settings = RunSettings(batch_size=1, missing_policy="ffill")
    dataset = WindowDataset(artifact, "test", 3, 3)
    stats = evaluate_local(model, dataset, settings, torch.device("cpu"), artifact, 2)
    truth = values[3:].astype(np.float64)
    observed = np.isfinite(truth)
    errors = 1e8 - truth[observed]
    assert stats["num-targets"] == 5
    assert stats["mae"] == np.abs(errors).mean()
    assert stats["mse"] == np.square(errors).mean()

    artifact = artifact | {"mean": np.array(0.), "scale": np.array(1.)}
    torch.nn.init.constant_(model.bias, -1.)
    dataset = WindowDataset(artifact, "test", 3, 3)
    stats = evaluate_local(model, dataset, settings, torch.device("cpu"), artifact, 2)
    assert stats["negative-predictions"] == stats["num-target-slots"] == 6
    assert stats["negative-prediction-fraction"] == 1.


def test_ordinary_loader_aligns_small_drift_and_audits_collisions(tmp_path):
    folder = tmp_path / "station"
    folder.mkdir()
    path = folder / "PM25.csv"
    pd.DataFrame({"PM25": [1., 2., 3.]}, index=pd.to_datetime([
        "2025-01-01 00:00:00", "2025-01-01 00:05:01", "2025-01-01 00:10:00",
    ])).to_csv(path)
    _, frame = load_pollutant_file(str(path), "PM25")
    assert len(frame) == 3
    assert frame.attrs["source_quality"]["off_grid_readings"] == 1
    assert load_pollutant_file(str(path), "PM25", grid_tolerance_seconds=0) is None
    with pytest.raises(ValueError, match="rejilla"):
        ensure_datetime_series(pd.Series([1.], index=pd.to_datetime(["2025-01-01 00:00:01"])), "5min", "test")


def test_equal_means_can_be_kept_without_disabling_raw_freeze_detection():
    frame = pd.DataFrame({"PM25": np.tile(np.arange(1., 13.), 2)}, index=pd.date_range("2025-01-01", periods=24, freq="5min"))
    (hourly,), _ = preprocess([frame], "PM25", hourly_repeat_filter=0, quality_mode="causal")
    assert hourly.iloc[:, 0].notna().all()
    assert hourly.attrs["quality_counts"]["useful_readings"] == 24


def test_explicit_upct_flags_are_respected_without_guessing_their_meaning(tmp_path):
    directory = tmp_path / "UPCT"
    directory.mkdir()
    path = directory / "PM25.csv"
    path.write_text("export\nDatetime;PM25 (ug/m3);Flag PM25\n2025-01-01 00:00:00;1;0\n2025-01-01 00:05:00;2;9\n")
    _, unchanged = load_pollutant_file(str(path), "PM25")
    _, filtered = load_pollutant_file(str(path), "PM25", upct_valid_flags="0")
    assert unchanged.iloc[:, 0].notna().sum() == 2
    assert filtered.iloc[:, 0].notna().sum() == 1
    assert filtered.attrs["source_quality"]["invalid_flags"] == 1


def test_missing_toml_mapping_and_macro_mae_are_configurable(tmp_path, monkeypatch):
    from fd_airquality.cli import _parser
    from fd_airquality.configuration import parse_configured_args

    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text('[missing]\nmlp = "mean"\n[run]\nselection-metric = "macro-mae"\n')
    parser, commands = _parser()
    args = parse_configured_args(parser, commands, ["backtest"])
    assert json.loads(args.missing_policies) == {"mlp": "mean"}
    assert args.selection_metric == "macro-mae"


def test_prepared_windows_and_evaluation_do_not_depend_on_epoch_budget(masked_prepared, tmp_path, masked_geometry):
    path, original = masked_prepared
    changed_path = prepare(Path(original["raw_root"]), tmp_path / "other", "PM25",
                            replace(masked_geometry, max_train_windows=4))
    changed = read_manifest(changed_path)
    assert original["evaluation_sha256"] == changed["evaluation_sha256"]
    for partition, client in enumerate(changed["clients"]):
        _, full = load_client_data(str(path), partition)
        _, budgeted = load_client_data(str(changed_path), partition)
        assert len(full["train_starts"]) == client["num_train_windows"] > 4
        for key in ("train_starts", "train_mask", "validation_starts", "test_starts", "values"):
            np.testing.assert_array_equal(full[key], budgeted[key])
    assert (path.parent / "coverage.csv").is_file()


def test_evaluation_signature_preserves_precision_of_the_scale_reference():
    series = pd.Series(np.linspace(10., 20., 30), index=pd.date_range("2025-01-01", periods=30, freq="h"))
    changed = series.copy()
    changed.iloc[1] += 1e-9
    np.testing.assert_array_equal(series.to_numpy(dtype=np.float32), changed.to_numpy(dtype=np.float32))
    parameters = (np.array([10]), 2, 2, str(series.index[0]), str(series.index[8]))
    assert target_signature(series, *parameters) != target_signature(changed, *parameters)
