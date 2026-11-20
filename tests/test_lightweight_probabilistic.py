import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import MetricRecord, RecordDict

from fd_airquality.backtesting import _candidate_plan, prepare_backtest, run_backtest
from fd_airquality.calibration import (
    point_calibration,
    residual_calibration,
    seasonal_calibration,
    validation_view,
)
from fd_airquality.data import load_client_data
from fd_airquality.models import make_model, model_metadata, validate_run_mode
from fd_airquality.probabilistic import quantile_statistics
from fd_airquality.settings import RunSettings
from fd_airquality.strategy import aggregate_evaluation


def test_seasonal_naive_uses_same_hour_in_previous_cycles_then_last_value():
    model = make_model({"context": 90, "horizon": 4, "seasonality": 24},
                       RunSettings(model="seasonal-naive"))
    values = torch.arange(90.).reshape(1, -1)
    values[0, [67, 68, 44, 69, 45, 21]] = torch.nan
    torch.testing.assert_close(model(values), torch.tensor([[66., 43., 20., 89.]]))
    assert not list(model.parameters())


def test_seasonal_intervals_match_statsforecast_on_complete_history():
    from statsforecast.models import SeasonalNaive

    times = pd.date_range("2025-01-01", periods=96, freq="h").as_unit("ns")
    values = np.sin(np.arange(96) / 4) + np.arange(96) / 18
    client = {"validation_scale_start": str(times[0]), "validation_scale_end": "2025-01-05"}
    geometry = {"context": 90, "horizon": 26, "seasonality": 24}
    scale = seasonal_calibration({"values": values, "timestamps": times.asi8}, geometry, client, 20)
    model = make_model(geometry, RunSettings(model="seasonal-naive"))
    point = model(torch.as_tensor(values[-90:][None]))[0].numpy()
    reference = SeasonalNaive(season_length=24).forecast(values, h=26, level=[80])
    np.testing.assert_allclose(point, reference["mean"])
    np.testing.assert_allclose(point + scale.offsets[:, 0], reference["lo-80"])
    np.testing.assert_allclose(point + scale.offsets[:, -1], reference["hi-80"])


def test_holt_advances_through_missing_hours_instead_of_compressing_time():
    model = make_model({"context": 4, "horizon": 2}, RunSettings(model="holt"))
    model.set_coefficients(1., 1.)
    values = torch.tensor([[1., 2., torch.nan, torch.nan], [torch.nan, torch.nan, 3., 4.]])
    torch.testing.assert_close(model(values), torch.tensor([[5., 6.], [5., 6.]]))
    assert not list(model.parameters())


def test_linear_lags_and_nlinear_point_have_small_point_heads_even_with_probabilistic_scoring():
    geometry = {"context": 90, "horizon": 12}
    linear = RunSettings(model="linear-6lags", normalization="none")
    model = make_model(geometry, linear)
    assert sum(parameter.numel() for parameter in model.parameters()) == 84
    values = torch.arange(90.).reshape(1, -1)
    with torch.no_grad():
        model.linear.weight.fill_(1)
    expected = sum(90 - lag for lag in (1, 2, 3, 24, 48, 72))
    torch.testing.assert_close(model(values), torch.full((1, 12), float(expected)))
    with pytest.raises(ValueError, match="72 horas"):
        make_model(geometry | {"context": 71}, linear)

    point = RunSettings(model="nlinear-point", probabilistic=1)
    fitted = make_model(geometry, point)
    assert fitted.forecast_distribution(values)[1:] == (None, None)
    assert model_metadata(geometry, point)["training_loss"] == "masked_mse"
    assert sum(parameter.numel() for parameter in fitted.parameters()) == 1092
    probabilistic = make_model(geometry, replace(point, model="nlinear"))
    assert len(probabilistic.forecast_distribution(values)[2]) == 19


def test_station_controls_are_not_duplicated_by_seed_or_federated_modes():
    settings = RunSettings()
    plan = _candidate_plan(settings, ["seasonal-naive", "holt", "linear-6lags", "nlinear-point"],
                           ["local", "centralized", "federated", "zero-shot"], None, [5, 6], None, None)
    for name in ("seasonal-naive", "holt"):
        assert [(c["mode"], c["seed"]) for c in plan if c["model"] == name] == [("local", 5)]
        with pytest.raises(ValueError, match="solo en local"):
            validate_run_mode(replace(settings, model=name), "federated")
    assert len(plan) == 14


def test_scaled_crps_preserves_standard_crps_and_equals_mase_for_degenerate_forecasts():
    actual = np.array([[3., 6.], [np.nan, 8.]])
    point = np.array([[1., 2.], [5., 7.]])
    levels = (.13, .4, .5, .82)
    quantiles = np.repeat(point[..., None], len(levels), axis=-1)
    kwargs = {"origin_days": np.array(["20261122", "20261123"])}
    original = quantile_statistics(actual, quantiles, levels, seasonal_scale=2., **kwargs)
    rescaled = quantile_statistics(5 * actual, 5 * quantiles, levels, seasonal_scale=10., **kwargs)
    assert original["crps"] == pytest.approx(np.nanmean(np.abs(actual - point)))
    assert original["scaled-crps"] == pytest.approx(original["crps"] / 2)
    assert rescaled["crps"] == pytest.approx(5 * original["crps"])
    assert rescaled["scaled-crps"] == pytest.approx(original["scaled-crps"])
    assert original["block-20261123-scaled-crps-target-count"] == 1
    assert original["block-20261122-scaled-crps-error-sum"] == pytest.approx(3.)
    assert "scaled-crps" not in quantile_statistics(actual, quantiles, levels)


def test_panel_scaled_crps_is_macro_and_requires_complete_common_support():
    def entry(value, count, eligible=1, scored=True):
        common = dict.fromkeys(("mase-sum", "mase-count", "rmsse-sum", "rmsse-count"), 0)
        fields = common | {"num-examples": count, "num-targets": count, "scaled-crps-eligible": eligible}
        if scored and eligible:
            fields.update({"scaled-crps": value, "scaled-crps-count": count})
        return RecordDict({"metrics": MetricRecord(fields)})

    result = aggregate_evaluation([entry(2., 10), entry(4., 100), entry(8., 200, 0)], "num-examples")
    assert result["macro-scaled-crps"] == 3
    assert result["scaled-crps-count"] == 110
    assert result["macro-scaled-crps-stations"] == 2
    partial = aggregate_evaluation([entry(2., 10), entry(4., 100, scored=False)], "num-examples")
    assert "macro-scaled-crps" not in partial
    assert partial["scaled-crps-eligible-targets"] == 110
    assert partial["scaled-crps-count"] == 10


def test_sparse_lead_fallback_counts_physical_hours_once():
    actual = np.array([[1., 2., 3.], [np.nan, 4., np.nan]])
    positions = np.array([[1, 2, 3], [3, 4, 5]])
    point = np.zeros_like(actual)
    without = residual_calibration(actual, point, 3)
    assert without.offsets is None
    with_neighbors = residual_calibration(actual, point, 3, target_positions=positions)
    assert with_neighbors.counts.tolist() == [1, 2, 1]
    assert with_neighbors.effective_counts.tolist() == [3, 3, 3]
    assert (np.diff(with_neighbors.offsets, axis=1) >= 0).all()
    repeated = residual_calibration(
        actual, point, 3, target_positions=np.ones_like(positions),
    )
    assert repeated.offsets is None


def test_calibration_and_selection_cannot_share_targets_or_read_test(global_prepared):
    path, manifest = global_prepared
    _, artifact = load_client_data(str(path), 0)
    geometry = manifest["geometry"]
    settings = RunSettings(model="autoencoder-tiny", probabilistic=1, calibrate_point_forecasts=1,
                           calibration_min_pairs=2)
    selection = validation_view(artifact, geometry, settings)
    calibration = validation_view(artifact, geometry, settings, calibration=True)
    assert selection.target_time_bounds()[1] < calibration.target_time_bounds()[0]
    model = make_model(geometry, settings)
    before = point_calibration(model, artifact, geometry, settings, torch.device("cpu"))
    changes = artifact["values"].copy()
    changes[artifact["timestamps"] >= pd.Timestamp(geometry["test_start"]).value] = 1e6
    after = point_calibration(model, artifact | {"values": changes}, geometry, settings, torch.device("cpu"))
    assert before.offsets is not None
    np.testing.assert_array_equal(before.offsets, after.offsets)


def test_light_models_backtest_publishes_calibrated_metrics(
    temporal_family, in_process_federation, mlflow_uri, tmp_path,
):
    raw, config = temporal_family
    config = replace(config, context=90, horizon=12, seasonality=24, stride=6, train_stride=6,
                     error_scale_reference="training")
    manifest = prepare_backtest(raw, tmp_path / "prepared", "PM25", config, folds=1,
                                validation_hours=72, test_hours=72, min_stations=2)
    settings = RunSettings(manifest=str(manifest), output_dir=str(tmp_path / "results"),
                           num_server_rounds=1, batch_size=8, probabilistic=1, calibrate_point_forecasts=1,
                           calibration_min_pairs=3, mlflow_tracking_uri=mlflow_uri, mlflow_log_models=0)
    output = run_backtest(settings, ["linear-6lags", "seasonal-naive", "holt", "nlinear-point", "mlp"],
                          ["local", "centralized", "federated"])
    scores = pd.read_csv(output / "temporal_comparison.csv")
    assert len(scores) == 11
    assert scores["macro-scaled-crps"].notna().all()
    assert scores["macro-scaled-crps-stations"].eq(2).all()
    assert scores["interval-80-coverage"].between(0, 1).all()
    assert sorted(scores["quantile_levels"].map(json.loads).map(len).unique()) == [9, 19]
