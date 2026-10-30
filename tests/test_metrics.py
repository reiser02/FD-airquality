import numpy as np
import pandas as pd
import pytest
import torch
from darts import TimeSeries
from darts.metrics import mase, rmsse

from fd_airquality.data import DataConfig, WindowDataset, split_station
from fd_airquality.metrics import seasonal_error_scales
from fd_airquality.settings import RunSettings
from fd_airquality.task import evaluate_local


@pytest.mark.parametrize("gap", [False, True])
def test_recent_scales_match_published_seasonal_formula_and_darts(gap):
    index = pd.date_range("2026-03-01", periods=672, freq="h")
    values = np.arange(672, dtype=float) ** 2
    if gap:
        values[100] = np.nan
    reference = pd.Series(values, index=index)
    absolute, squared, pairs = seasonal_error_scales(reference, 24)
    differences = values[24:] - values[:-24]
    differences = differences[np.isfinite(differences)]
    assert absolute == pytest.approx(np.abs(differences).mean())
    assert squared == pytest.approx(np.square(differences).mean())
    assert pairs == 648 - (2 if gap else 0)
    times = pd.date_range("2026-03-29", periods=12, freq="h")
    actual = TimeSeries.from_times_and_values(times, np.full(12, 10.))
    prediction = TimeSeries.from_times_and_values(times, np.full(12, 9.))
    insample = TimeSeries.from_series(reference)
    assert mase(actual, prediction, insample, m=24) == pytest.approx(1 / absolute)
    assert rmsse(actual, prediction, insample, m=24) == pytest.approx(np.sqrt(1 / squared))


def test_missing_timestamps_are_not_treated_as_adjacent_hours():
    index = pd.date_range("2026-03-01", periods=9, freq="h")
    reference = pd.Series([1., 2., 4., 8., np.nan, 100., 102., 106., 114.], index=index)
    expected = seasonal_error_scales(reference, 2)
    assert seasonal_error_scales(reference.drop(index[4]), 2) == expected
    assert expected == (23.8, 1737.8, 5)


def test_default_reference_is_the_previous_28_calendar_days_for_each_period():
    index = pd.date_range("2026-01-01", "2026-04-26", freq="h", inclusive="left")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    split = split_station(series, DataConfig())
    assert split["validation_scale_start"] == "2026-02-01 00:00:00"
    assert split["validation_scale_end"] == "2026-03-01 00:00:00"
    assert split["test_scale_start"] == "2026-03-01 00:00:00"
    assert split["test_scale_end"] == "2026-03-29 00:00:00"
    for name in ("validation", "test"):
        assert split[f"{name}_scale_pairs"] == 648
        assert split[f"{name}_mase_scale"] == 24.
        assert split[f"{name}_rmsse_scale"] == 576.


def test_reference_is_not_extended_to_replace_missing_calendar_hours():
    index = pd.date_range("2026-01-01", "2026-04-26", freq="h", inclusive="left")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    series.loc["2026-03-02"] = np.nan
    split = split_station(series, DataConfig())
    assert split["test_scale_start"] == "2026-03-01 00:00:00"
    assert split["test_scale_pairs"] == 648 - 48


@pytest.mark.parametrize("reference", [[], [np.nan] * 48, [1.] * 48])
def test_empty_missing_and_constant_references_have_no_fake_denominator(reference):
    absolute, squared, pairs = seasonal_error_scales(pd.Series(reference, dtype=float), 24)
    assert (absolute, squared) == (0., 0.)
    assert pairs == (24 if reference and reference[0] == 1. else 0)


@pytest.mark.parametrize("legacy", [False, True])
def test_evaluation_uses_split_specific_scale_but_preserves_legacy_training_scales(legacy):
    artifact = {
        "values": np.arange(20, dtype=np.float32),
        "timestamps": pd.date_range("2026-03-01", periods=20, freq="h").asi8,
        "validation_starts": np.array([0]), "test_starts": np.array([4]),
        "mean": np.array(0.), "scale": np.array(1.),
        "mase_scale": np.array(8.), "rmsse_scale": np.array(64.),
    }
    if not legacy:
        for name, scale in (("validation", 2.), ("test", 4.)):
            artifact.update({f"{name}_mase_scale": np.array(scale),
                             f"{name}_rmsse_scale": np.array(scale ** 2),
                             f"{name}_scale_pairs": np.array(648)})

    class Persistence(torch.nn.Module):
        def forward(self, inputs):
            return inputs[:, -1:].repeat(1, 2)

    for name, scale in (("validation", 2.), ("test", 4.)):
        dataset = WindowDataset(artifact, name, 6, 2)
        stats = evaluate_local(Persistence(), dataset, RunSettings(), torch.device("cpu"), artifact, 24)
        denominator = 8. if legacy else scale
        assert stats["mase"] == pytest.approx(1.5 / denominator)
        assert stats["rmsse"] == pytest.approx(np.sqrt(2.5) / denominator)
        assert stats["mase-count"] == 1
        if not legacy:
            assert stats["error-scale-pairs"] == 648
