from dataclasses import replace

import pytest
import torch

from fd_airquality.backtesting import _candidate_plan
from fd_airquality.models import make_model, model_metadata, validate_run_mode
from fd_airquality.settings import RunSettings


def test_seasonal_naive_uses_same_hour_in_previous_cycles_then_last_value():
    model = make_model({"context": 90, "horizon": 4, "seasonality": 24},
                       RunSettings(model="seasonal-naive"))
    values = torch.arange(90.).reshape(1, -1)
    values[0, [67, 68, 44, 69, 45, 21]] = torch.nan
    torch.testing.assert_close(model(values), torch.tensor([[66., 43., 20., 89.]]))
    assert not list(model.parameters())


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
