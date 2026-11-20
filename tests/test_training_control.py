import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from fd_airquality.arrival import (
    normalization_artifact,
    round_cutoff,
    round_validation_bounds,
    round_validation_view,
    training_view,
)
from fd_airquality.baselines import run_baseline
from fd_airquality.data import load_client_data, prepare
from fd_airquality.server_app import run_server
from fd_airquality.settings import RunSettings
from fd_airquality.task import train_local, training_validation_mse
from fd_airquality.training_control import ContextSupervisionDataset, TrainingControl


@pytest.mark.parametrize("options", [
    {"early_stopping": 2}, {"early_stopping_patience": 0}, {"round_validation_hours": 0},
    {"early_stopping_min_delta": -1.}, {"lr_scheduler": "unknown"},
    {"lr_scheduler_patience": -1}, {"lr_scheduler_factor": 1.},
    {"lr_scheduler_factor": 0.}, {"min_learning_rate": 0.}, {"min_learning_rate": .1},
])
def test_training_control_settings_reject_invalid_values(options):
    with pytest.raises(ValueError):
        RunSettings(**options)


def test_patience_scheduler_min_delta_and_best_weight_restoration():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.)
    settings = RunSettings(early_stopping_patience=2, lr_scheduler_patience=0,
                           lr_scheduler_factor=.5, early_stopping_min_delta=.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    control = TrainingControl(model, optimizer, settings)
    assert not control.observe(0, 10.)
    model.weight.data.fill_(2.)
    assert not control.observe(1, 9.)
    model.weight.data.fill_(3.)
    assert not control.observe(2, 8.95)  # < min_delta: not an improvement
    assert optimizer.param_groups[0]["lr"] == settings.learning_rate / 2
    model.weight.data.fill_(4.)
    assert control.observe(3, 9.1)
    assert control.best_epoch == 1 and control.reductions == 2
    control.restore()
    assert model.weight.item() == 2.
    with pytest.raises(ValueError, match="no es finita"):
        control.observe(4, float("nan"))


def test_actual_epoch_loop_stops_reduces_lr_and_preserves_aggregation_counts(monkeypatch):
    data = TensorDataset(torch.ones(5, 1), torch.tensor([[0.], [1.], [2.], [3.], [float("nan")]]))
    # The generic dataset has one row with no observed target.
    model = torch.nn.Linear(1, 1)
    scores = iter([10., 9., 9., 9., 9.])
    states = []

    def monitor(model, *_):
        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return next(scores)

    monkeypatch.setattr("fd_airquality.task.training_validation_mse", monitor)
    settings = RunSettings(local_epochs=20, batch_size=5, early_stopping_patience=3, lr_scheduler_patience=0)
    metrics = train_local(model, data, settings, torch.device("cpu"), 42, validation_data=data)
    assert metrics["epochs-completed"] == metrics["optimizer-steps"] == 4
    assert metrics["early-stopped"] == 1 and metrics["best-epoch"] == 1
    assert metrics["num-targets"] == 4
    assert metrics["learning-rate-reductions"] == 3
    assert metrics["learning-rate-final"] == settings.learning_rate / 8
    for key, expected in states[1].items():
        torch.testing.assert_close(model.state_dict()[key], expected)


def test_controls_require_explicit_validation_and_can_be_disabled():
    data = TensorDataset(torch.ones(2, 1), torch.ones(2, 1))
    model = torch.nn.Linear(1, 1)
    settings = RunSettings(local_epochs=2)
    with pytest.raises(ValueError, match="validación interna"):
        train_local(model, data, settings, torch.device("cpu"), 42)
    metrics = train_local(model, data, replace(settings, early_stopping=0, lr_scheduler="none"),
                          torch.device("cpu"), 42)
    assert metrics["epochs-completed"] == 2
    assert not metrics["early-stopped"]


def test_monitor_pools_observed_errors_and_station_scales():
    model = torch.nn.Linear(1, 9, bias=False)
    model.weight.data.zero_()
    y = torch.full((2, 9), float("nan"))
    y[0, 0], y[1] = 0., 10.
    data = TensorDataset(torch.ones(2, 1), y)
    assert training_validation_mse(model, data, RunSettings(batch_size=1), torch.device("cpu")) == 90.
    a = TensorDataset(torch.ones(1, 1), torch.ones(1, 9))
    b = TensorDataset(torch.ones(3, 1), torch.full((3, 9), 3.))
    a.scale, b.scale = 2., 10.
    assert training_validation_mse(model, ConcatDataset([a, b]), RunSettings(), torch.device("cpu")) == 676.
    assert model.training


def test_federated_monitoring_reserve_is_past_disjoint_and_protects_fixed_scaler(
    arrival_prepared,
):
    path, manifest = arrival_prepared
    _, original = load_client_data(str(path), 0)
    geometry = manifest["geometry"]
    settings = RunSettings(local_epochs=8, num_server_rounds=3, normalization="fixed")
    lower, upper = round_validation_bounds(geometry, settings)
    assert upper == round_cutoff(geometry, settings, 1)
    artifact = normalization_artifact(original, geometry, settings)
    fit = np.isfinite(original["values"]) & (original["timestamps"] < lower.value)
    assert float(artifact["mean"]) == pytest.approx(original["values"][fit].astype(float).mean())
    validation = round_validation_view(artifact, geometry, settings)
    positions = validation.starts[:, None] + validation.context + np.arange(validation.horizon)
    val_times = set(original["timestamps"][positions].flat)
    assert min(val_times) >= lower.value and max(val_times) < upper.value
    for number in (1, 2, 3):
        train = training_view(artifact, geometry, settings, number)
        targets = train.starts[:, None] + train.context + np.arange(train.horizon)
        assert not val_times.intersection(original["timestamps"][targets].flat)
    changed = original | {"values": original["values"].copy()}
    changed["values"][original["timestamps"] >= upper.value] = 1e8
    other = normalization_artifact(changed, geometry, settings)
    assert artifact["normalization_sha256"] == other["normalization_sha256"]
    other_validation = round_validation_view(other, geometry, settings)
    for index in range(len(validation)):
        torch.testing.assert_close(validation[index][0], other_validation[index][0], equal_nan=True)
        torch.testing.assert_close(validation[index][1], other_validation[index][1], equal_nan=True)
    with pytest.raises(ValueError, match="invade"):
        normalization_artifact(original, geometry, replace(settings, normalization_reference_end=str(upper)))


def test_federated_monitor_requires_observed_past_not_future_fallback(arrival_raw, tmp_path):
    raw, config = arrival_raw
    settings = RunSettings(local_epochs=8, num_server_rounds=3)
    # The first arrival is Jan 7 and its 24-hour reserve is Jan 6.
    source = raw / "a" / "PM25.csv"
    frame = pd.read_csv(source, index_col=0, parse_dates=True)
    frame.loc[(frame.index >= "2025-01-06") & (frame.index < "2025-01-07")] = np.nan
    frame.to_csv(source)
    path = prepare(raw, tmp_path / "prepared", "PM25", config)
    manifest, artifact = load_client_data(str(path), 0)
    assert len(training_view(artifact, manifest["geometry"], settings, 1)) > 0
    assert len(artifact["validation_starts"]) > 0
    with pytest.raises(ValueError, match="Sin ventanas observadas"):
        round_validation_view(artifact, manifest["geometry"], settings)


@pytest.mark.parametrize("padding", ["none", "both"])
def test_monitoring_labels_are_not_reused_by_autoencoder_auxiliary_loss(arrival_raw, tmp_path, padding):
    raw, config = arrival_raw
    path = prepare(raw, tmp_path / "prepared", "PM25", replace(config, window_padding=padding))
    manifest, artifact = load_client_data(str(path), 0)
    settings = RunSettings(model="autoencoder", local_epochs=8, num_server_rounds=3)
    geometry = manifest["geometry"]
    train = training_view(artifact, geometry, settings, 2)
    # The first new target is at the upper edge of the reserve, the next starts
    # three hours later. Their contexts are respectively all/partly reserved.
    protected = ContextSupervisionDataset(train)
    for index, allowed_labels in ((0, [False] * 6), (1, [False] * 3 + [True] * 3)):
        inputs, actual, allowed = protected[index]
        expected_inputs, expected_actual = train[index]
        torch.testing.assert_close(inputs, expected_inputs, equal_nan=True)
        torch.testing.assert_close(actual, expected_actual, equal_nan=True)
        assert allowed.tolist() == allowed_labels


@pytest.mark.parametrize(("mode", "model_name", "padding"), [
    ("local", "mlp", "none"), ("centralized", "nhits", "none"), ("centralized", "nlinear", "both"),
    ("federated", "autoencoder", "both"), ("federated", "autoencoder-tiny", "both"),
    ("federated", "flowstate-r1", "right"),
])
def test_training_paths_apply_patience_lr_and_preserve_all_arrivals(
    arrival_raw, tmp_path, flower_grid, mock_granite, mode, model_name, padding,
):
    raw, config = arrival_raw
    path = prepare(raw, tmp_path / "prepared", "PM25", replace(config, window_padding=padding, max_train_windows=5))
    settings = RunSettings(
        manifest=str(path), output_dir=str(tmp_path / mode), num_server_rounds=3,
        local_epochs=8, hidden_size=8, batch_size=8, early_stopping_patience=2,
        early_stopping_min_delta=1e6, lr_scheduler_patience=0, mlflow_tracking_uri="disabled",
        model=model_name, probabilistic=int(model_name in {"mlp", "nlinear", "nhits"}),
        missing_policy="ffill" if model_name == "flowstate-r1" else "auto",
        lora_targets="proj" if model_name == "flowstate-r1" else "auto",
    )
    output = (run_server(flower_grid(settings, 2), settings) if mode == "federated" else
              run_baseline(settings, mode))
    frame = pd.read_csv(output / ("round_metrics.csv" if mode == "centralized" else "station_metrics.csv"))
    train = frame.loc[frame.split == "train"]
    assert set(train["round"]) == {1, 2, 3}
    assert train["early-stopped"].gt(0).all()
    assert train["epochs-completed"].eq(2).all()
    assert train["learning-rate-reductions"].eq(2).all()
    assert train["num-examples"].eq(10 if mode == "centralized" else 5).all()
    assert train["replacement-window-occurrences"].eq(12 if mode == "centralized" else 6).all()
    if mode != "centralized":
        assert train["learning-rate-final"].eq(settings.learning_rate / 4).all()
    run = json.loads((output / "run.json").read_text())
    assert run["training_controls"]["monitor"] == "pooled_observed_mse_original_units"
    assert (output / "test_metrics.json").is_file()
