from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fd_airquality.data import DataConfig, load_client_data, prepare
from fd_airquality.metrics import seasonal_error_scales
from fd_airquality.preprocessing import frozen_mask, preprocess


def zero_frame(hours):
    return pd.DataFrame({"NO2": np.zeros(hours * 12)},
                        index=pd.date_range("2026-02-01", periods=hours * 12, freq="5min"))


@pytest.mark.parametrize("hours", [6, 23])
@pytest.mark.parametrize("quality_mode", ["causal", "retrospective"])
def test_zero_runs_shorter_than_a_day_remain_valid(hours, quality_mode):
    (hourly,), counts = preprocess([zero_frame(hours)], "NO2", quality_mode=quality_mode)
    assert len(hourly) == hours
    assert hourly.iloc[:, 0].eq(0).all()
    assert counts == [0]


def test_causal_zero_freeze_starts_at_the_24th_hour_and_stays_frozen():
    (hourly,), counts = preprocess([zero_frame(48)], "NO2", quality_mode="causal")
    assert hourly.iloc[:23, 0].eq(0).all()
    assert hourly.iloc[23:, 0].isna().all()
    assert counts == [25]
    assert seasonal_error_scales(hourly.iloc[:, 0], 24) == (0., 0., 0)


def test_retrospective_mode_marks_the_entire_zero_run_once_it_reaches_a_day():
    (hourly,), counts = preprocess([zero_frame(24)], "NO2", quality_mode="retrospective")
    assert hourly.iloc[:, 0].isna().all()
    assert counts == [24]


def test_observed_nonzero_hour_resets_zero_freeze_detection():
    frame = zero_frame(48)
    frame.iloc[20 * 12:21 * 12, 0] = np.arange(1., 13.)
    (hourly,), _ = preprocess([frame], "NO2", quality_mode="causal")
    assert hourly.iloc[:20, 0].eq(0).all()
    assert hourly.iloc[20, 0] > 0
    assert hourly.iloc[21:44, 0].eq(0).all()
    assert hourly.iloc[44:, 0].isna().all()


@pytest.mark.parametrize("gap_hours", [1, 5, 48])
def test_missing_hours_pause_the_counter_without_resetting_or_counting_as_zeros(gap_hours):
    frame = zero_frame(30 + gap_hours)
    frame.iloc[10 * 12:(10 + gap_hours) * 12, 0] = np.nan
    (hourly,), _ = preprocess([frame], "NO2", quality_mode="causal")
    assert hourly.iloc[:10, 0].eq(0).all()
    assert hourly.iloc[10:10 + gap_hours, 0].isna().all()
    assert hourly.iloc[10 + gap_hours:23 + gap_hours, 0].eq(0).all()
    assert hourly.iloc[23 + gap_hours:, 0].isna().all()


def test_retrospective_zero_plateau_detection_also_preserves_missing_hours():
    frame = zero_frame(25)
    frame.iloc[12 * 12:13 * 12, 0] = np.nan
    (hourly,), counts = preprocess([frame], "NO2", quality_mode="retrospective")
    assert hourly.iloc[:, 0].isna().all()
    assert counts == [24]


def test_missing_five_minute_samples_do_not_restart_a_valid_hourly_zero_run():
    frame = zero_frame(48)
    frame.iloc[[100, 288, 500], 0] = np.nan
    (hourly,), _ = preprocess([frame], "NO2", quality_mode="causal")
    assert hourly.iloc[:23, 0].eq(0).all()
    assert hourly.iloc[23:, 0].isna().all()


def test_causal_zero_detection_does_not_rewrite_completed_hours():
    frame = zero_frame(48)
    (full,), _ = preprocess([frame], "NO2", quality_mode="causal")
    for hours in (6, 23, 24, 30, 40):
        (prefix,), _ = preprocess([frame.iloc[:hours * 12]], "NO2", quality_mode="causal")
        pd.testing.assert_frame_equal(prefix, full.iloc[:hours])


def test_nonzero_frozen_readings_keep_the_existing_six_sample_rule():
    series = pd.Series([7.] * 12)
    mask = frozen_mask(series, quality_mode="causal")
    assert not mask.iloc[:5].any()
    assert mask.iloc[5:].all()
    assert not frozen_mask(pd.Series([0.] * 12), quality_mode="causal").any()


@pytest.mark.parametrize("hours", [0, True, 1.5])
def test_zero_run_threshold_must_be_a_positive_integer(hours):
    with pytest.raises(ValueError, match="positivo"):
        preprocess([zero_frame(1)], "NO2", zero_run_hours=hours)
    with pytest.raises(ValueError):
        DataConfig(zero_run_hours=hours)


def test_preparation_records_and_applies_zero_run_threshold(tmp_path, geometry):
    folder = tmp_path / "raw" / "station"
    folder.mkdir(parents=True)
    index = pd.date_range("2025-01-01", periods=160 * 12, freq="5min")
    values = 10 + np.sin(np.arange(len(index)) / 30)
    values[24 * 12:32 * 12] = 0
    pd.DataFrame({"NO2": values}, index=index).to_csv(folder / "station_NO2.csv")
    for hours in (4, 24):
        path = prepare(folder.parent, tmp_path / "prepared", "NO2", replace(geometry, zero_run_hours=hours))
        manifest, artifact = load_client_data(str(path), 0)
        assert manifest["geometry"]["zero_run_hours"] == hours
        if hours == 4:
            assert np.isnan(artifact["values"][24:32]).all()
        else:
            assert (artifact["values"][24:32] == 0).all()
