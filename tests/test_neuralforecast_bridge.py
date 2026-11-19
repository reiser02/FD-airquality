import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from fd_airquality.availability import availability_minima, availability_threshold, parse_threshold
from fd_airquality.models import make_model, restore_checkpoint, shared_state
from fd_airquality.neural import NeuralForecaster
from fd_airquality.settings import RunSettings
from fd_airquality.task import configure_torch, train_local


def test_darts_is_not_a_dependency_or_imported():
    import tomllib

    dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
    assert not any(item.startswith("darts") for item in dependencies)
    assert importlib.util.find_spec("darts") is None


@pytest.mark.parametrize("model_name", ["mlp", "nlinear", "nhits"])
@pytest.mark.parametrize("probabilistic", [0, 1])
@pytest.mark.parametrize("scaler", ["identity", "standard", "robust", "minmax"])
def test_package_models_masks_losses_and_scalers_are_used(model_name, probabilistic, scaler):
    from neuralforecast.losses.pytorch import MSE, MQLoss

    settings = RunSettings(model=model_name, hidden_size=8, probabilistic=probabilistic, scaler_type=scaler)
    model = make_model({"context": 6, "horizon": 3}, settings)
    assert isinstance(model, NeuralForecaster)
    assert type(model.module).__module__.startswith("neuralforecast.models.")
    assert isinstance(model.module.loss, MQLoss if probabilistic else MSE)
    assert model.module.scaler.scaler_type == scaler
    x = torch.tensor([[float("nan"), 0., 2., float("nan"), 4., float("nan")],
                      [1., 2., 3., 4., 5., 6.]])
    y = torch.tensor([[7., float("nan"), 9.], [float("nan"), 6., 0.]])
    original = x.clone()
    batch, targets, mask = model._prepare(x, y)
    assert batch["insample_mask"][0, :, 0].tolist() == [0., 1., 1., 0., 1., 0.]
    assert mask[..., 0].tolist() == [[1., 0., 1.], [0., 1., 1.]]
    assert torch.isfinite(batch["insample_y"]).all() and torch.isfinite(targets).all()
    output = model._predict_normalized(batch)
    expected = model.module.loss(y=targets, y_hat=output, mask=mask)
    if probabilistic:
        expected /= len(model.quantile_levels)
    objective, squared = model.training_objective(x, y)
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    point, quantiles, levels = model.forecast_distribution(x)
    assert point.shape == y.shape and torch.isfinite(point).all()
    if probabilistic:
        assert quantiles.shape == (2, 3, 19) and len(levels) == 19
        assert not any("loss" in name for name in shared_state(model))
    else:
        assert quantiles is levels is None
    torch.testing.assert_close(x, original, equal_nan=True)
    assert squared >= 0


@pytest.mark.parametrize("model_name", ["mlp", "nlinear", "nhits"])
def test_context_normalization_does_not_use_targets_batch_neighbors_or_missing_placeholders(model_name):
    settings = RunSettings(model=model_name, hidden_size=8)
    model = make_model({"context": 4, "horizon": 2}, settings).double()
    x = torch.tensor([[10., float("nan"), 14., 0.]], dtype=torch.float64)
    y = torch.tensor([[12., float("nan")]], dtype=torch.float64)
    batch, _, mask = model._prepare(x, y)
    shift, scale = model.module.scaler.x_shift.clone(), model.module.scaler.x_scale.clone()
    expected = x[0, torch.isfinite(x[0])]
    assert shift.item() == pytest.approx(expected.mean().item())
    assert scale.item() == pytest.approx(expected.std(unbiased=False).item() + 1e-6)
    changed, _, changed_mask = model._prepare(x, y + 1e8)
    torch.testing.assert_close(batch["insample_y"], changed["insample_y"])
    torch.testing.assert_close(mask, changed_mask)
    combined, _, _ = model._prepare(torch.cat([x, x * 1e6]))
    torch.testing.assert_close(batch["insample_y"][0], combined["insample_y"][0])
    torch.testing.assert_close(model.module.scaler.x_shift[0], shift[0])


@pytest.mark.parametrize("policy", ["native", "ffill", "linear", "mean", "zero"])
def test_context_policy_never_relabels_imputed_values_as_observed(policy):
    model = make_model({"context": 4, "horizon": 2}, RunSettings(hidden_size=8))
    x = torch.tensor([[float("nan"), 0., 2., float("nan")]])
    batch, _, _ = model._prepare(x, missing_policy=policy)
    assert batch["insample_mask"][0, :, 0].tolist() == [0., 1., 1., 0.]
    assert torch.isfinite(batch["insample_y"]).all()


@pytest.mark.parametrize("threshold", [0, 1, [0., 1], [0, 1.], -0.1, 1.1, float("nan"), [0.1], True])
def test_threshold_validation_rejects_integers_and_invalid_values(threshold):
    with pytest.raises(ValueError):
        parse_threshold(threshold)


@pytest.mark.parametrize("threshold,expected", [(0., (1, 1)), (.5, (12, 3)),
                                                ([.75, 1.], (18, 6)), (1., (24, 6))])
def test_availability_minima_match_documented_truncation(threshold, expected):
    assert availability_minima(24, 6, threshold) == expected
    assert availability_minima(24, 3, [.0, .8]) == (1, 2)


def test_model_specific_availability_configuration():
    settings = RunSettings(training_data_availability_threshold="[0.5, 1.0]",
                           training_availability='{"nlinear":[0.25,0.5]}')
    assert availability_threshold(settings) == (.5, 1.)
    assert availability_threshold(replace(settings, model="nlinear")) == (.25, .5)


@pytest.mark.parametrize("scaler", ["standard", "robust", "minmax"])
def test_single_observation_and_constant_contexts_remain_finite(scaler):
    model = make_model({"context": 4, "horizon": 2}, RunSettings(scaler_type=scaler, hidden_size=8))
    inputs = torch.tensor([[float("nan"), float("nan"), 3., float("nan")], [3., 3., 3., 3.]])
    loss, _ = model.training_objective(inputs, torch.tensor([[2., float("nan")], [3., 4.]]))
    loss.backward()
    assert torch.isfinite(model(inputs)).all() and torch.isfinite(loss)


def test_external_training_filter_matches_native_package_availability():
    from fd_airquality.availability import filter_training_windows
    from fd_airquality.data import WindowDataset

    settings = RunSettings(hidden_size=8, training_data_availability_threshold="[0.75,1.0]")
    model = make_model({"context": 4, "horizon": 2}, settings)
    values = np.array([1., np.nan, 2., 3., 4., 5., np.nan, 6., 7., 8., 9., 10.], dtype=np.float32)
    starts = np.arange(7)
    artifact = {"values": values, "timestamps": pd.date_range("2025-01-01", periods=12, freq="h").asi8,
                "mean": np.array(0.), "scale": np.array(1.), "allow_missing": np.array(True),
                "train_starts": starts}
    dataset = WindowDataset(artifact, "train", 4, 2)
    filtered = filter_training_windows(dataset, settings)
    rows = np.stack([values[start:start + 6] for start in starts])
    batch = {"temporal": torch.tensor(np.stack([np.nan_to_num(rows), np.isfinite(rows)], axis=1)).float(),
             "temporal_cols": pd.Index(["y", "available_mask"])}
    windows, _, _, indices = model.module._create_windows(batch, step="train")
    # Each supplied example produces 1 + H candidate origins after right padding.
    native = indices[indices % 3 == 0].numpy() // 3
    np.testing.assert_array_equal(filtered.starts, starts[native])
    assert len(windows) == len(starts) * 3


@pytest.mark.parametrize("probabilistic", [0, 1])
def test_neural_losses_have_no_gradient_on_missing_targets(probabilistic):
    model = make_model({"context": 4, "horizon": 3}, RunSettings(hidden_size=8, probabilistic=probabilistic))
    x = torch.tensor([[0., float("nan"), 2., 3.]])
    y = torch.tensor([[2., float("nan"), 4.]])
    _batch, target, mask = model._prepare(x, y)
    shape = (1, 3, 1, 19) if probabilistic else (1, 3, 1)
    predictions = torch.randn(*shape, requires_grad=True)
    loss = model.module.loss(y=target, y_hat=predictions, mask=mask)
    loss.backward()
    assert torch.count_nonzero(predictions.grad[:, 1]) == 0
    assert torch.count_nonzero(predictions.grad[:, [0, 2]]) > 0


@pytest.mark.parametrize("name", ["mlp", "nlinear", "nhits"])
@pytest.mark.parametrize("probabilistic", [0, 1])
def test_microbatch_weighting_uses_observed_targets_for_actual_package_losses(name, probabilistic, monkeypatch):
    settings = RunSettings(model=name, hidden_size=8, batch_size=5, probabilistic=probabilistic)
    configure_torch(settings, 17)
    model = make_model({"context": 4, "horizon": 3}, settings).double()
    inputs = torch.randn(5, 4, dtype=torch.float64)
    actual = torch.randn(5, 3, dtype=torch.float64)
    inputs[::2, 0] = float("nan")
    actual[::2, :2] = float("nan")
    full, micro = deepcopy(model), deepcopy(model)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 5)
    a = train_local(full, TensorDataset(inputs, actual), settings, torch.device("cpu"), 7)
    monkeypatch.setattr("fd_airquality.task._microbatch_size", lambda *_: 2)
    b = train_local(micro, TensorDataset(inputs, actual), settings, torch.device("cpu"), 7)
    assert a["num-targets"] == b["num-targets"] == int(torch.isfinite(actual).sum())
    for key, value in shared_state(full).items():
        torch.testing.assert_close(shared_state(micro)[key], value, atol=1e-10, rtol=1e-8)


def test_package_cross_validation_and_adapter_produce_the_same_forecast():
    from neuralforecast import NeuralForecast

    model = make_model({"context": 6, "horizon": 2}, RunSettings(model="nlinear", hidden_size=8))
    model.module.max_steps = 2
    model.module.val_check_steps = 2
    model.module.trainer_kwargs.update(enable_model_summary=False, accelerator="cpu", devices=1)
    times = pd.date_range("2025-01-01", periods=28, freq="h")
    values = np.arange(28, dtype=np.float32)
    values[::5] = np.nan
    observed = np.isfinite(values)
    df = pd.DataFrame({"unique_id": "a", "ds": times,
                       "y": np.nan_to_num(values), "available_mask": observed.astype(np.float32)})
    nf = NeuralForecast(models=[model.module], freq="h", local_scaler_type=None)
    cv = nf.cross_validation(df=df, n_windows=2, step_size=2, val_size=0, refit=False, verbose=False)
    # NeuralForecast keeps fitted models in .models; its CV output's y is a
    # placeholder at gaps, so external truth must always be rejoined by timestamp.
    model.module = nf.models[0]
    model.eval()
    for cutoff, group in cv.groupby("cutoff"):
        origin = times.get_loc(pd.Timestamp(cutoff)) + 1
        context = torch.tensor(values[origin - 6:origin]).unsqueeze(0)
        prediction = model(context).detach().numpy()[0]
        np.testing.assert_allclose(prediction, group.sort_values("ds")["NLinear"], rtol=1e-5, atol=1e-5)
    assert len(cv) == 4


def test_probabilistic_checkpoint_restores_exact_package_configuration(tmp_path):
    from fd_airquality.models import checkpoint_payload

    settings = RunSettings(model="nlinear", probabilistic=1)
    geometry = {"context": 6, "horizon": 2}
    model = make_model(geometry, settings)
    path = tmp_path / "model.pt"
    torch.save(checkpoint_payload(geometry, settings, shared_state(model), 1), path)
    restored = restore_checkpoint(path)
    x = torch.tensor([[1., 2., float("nan"), 4., 5., 6.]])
    torch.testing.assert_close(model.forecast_distribution(x)[1], restored.forecast_distribution(x)[1])
    assert json.loads(json.dumps(restored.quantile_levels)) == list(model.quantile_levels)
