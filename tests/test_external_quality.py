from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fd_airquality.backtesting import prepare_backtest
from fd_airquality.data import DataConfig, read_manifest
from fd_airquality.loaders import MompeanReference, load_mompean
from fd_airquality.preprocessing import (
    INCIDENT_END,
    INCIDENT_FLAG,
    INCIDENT_START,
    REFERENCE_LEVEL_FLAG,
    REFERENCE_MIN_PAIRS,
    REFERENCE_PEARSON_FLAG,
    apply_quality_filters,
)
from fd_airquality.settings import resolve_manifest


def _hourly(values, index):
    frame = pd.DataFrame({"PM25": values}, index=index)
    frame.attrs["quality_flags"] = np.zeros(len(index), dtype=np.uint16)
    frame.attrs["quality_counts"] = {}
    return frame


def test_reference_uses_interval_start_in_utc_and_leaves_dst_gaps(tmp_path):
    path = tmp_path / "mompean.csv"
    path.write_text("fecha,PM25\n2026-01-20 04:00:00,12\n2026-07-20 04:00:00,17\n"
                    "2026-10-25 02:00:00,234\n2026-03-29 02:00:00,345\n")
    result = load_mompean(str(path)).values
    assert result.to_dict() == {pd.Timestamp("2026-01-20 02:00"): 12, pd.Timestamp("2026-07-20 01:00"): 17}


def test_reference_masks_supported_hours_and_never_revises_a_prefix():
    index = pd.date_range("2025-01-01", periods=400, freq="h")
    y = 10 + np.sin(np.arange(len(index)))
    reference = MompeanReference(Path("mompean.csv"), "", pd.Series(y, index=index))
    poor_dynamic = _hourly(20 - y, index)
    poor_level = _hourly(y * 0.05, index)
    prefix = poor_dynamic.iloc[:250].copy()
    prefix.attrs["quality_flags"] = prefix.attrs["quality_flags"][:len(prefix)]

    for frame, flag in ((poor_dynamic, REFERENCE_PEARSON_FLAG), (poor_level, REFERENCE_LEVEL_FLAG),
                        (prefix, REFERENCE_PEARSON_FLAG)):
        audit = apply_quality_filters(frame, "estación", exclude_incident=False, reference=reference)
        first = REFERENCE_MIN_PAIRS - 1
        assert frame.iloc[:first, 0].notna().all()
        assert frame.iloc[first:, 0].isna().all()
        assert (frame.attrs["quality_flags"][first:] == flag).all()
        assert len(audit) == 1
        assert audit[0]["removed_hours"] == len(frame) - first

    pd.testing.assert_series_equal(prefix.iloc[:, 0], poor_dynamic.iloc[:len(prefix), 0])


def test_incident_mask_applies_before_fold_windows_and_is_audited(tmp_path, caplog):
    raw = tmp_path / "raw" / "estación"
    raw.mkdir(parents=True)
    index = pd.date_range(INCIDENT_START - pd.Timedelta(days=4), INCIDENT_END + pd.Timedelta(days=9), freq="5min")
    pd.DataFrame({"PM25": 10 + np.sin(np.arange(len(index)))}, index=index).to_csv(raw / "station_PM25.csv")
    config = replace(DataConfig(), context=12, horizon=4, stride=2, window_policy="masked",
                     exclude_may_june_2026=True)
    family = prepare_backtest(raw.parent, tmp_path / "backtests", "PM25", config, folds=1,
                              validation_hours=48, test_hours=48, min_stations=1)
    fold = resolve_manifest(str(family))
    audit_path = family.with_name("quality_exclusions.csv")
    audit = pd.read_csv(audit_path)
    assert audit["reason"].tolist() == ["may_june_2026_incident"]
    assert "TFG_NO_PUBLICADO" in caplog.text
    assert "source" not in audit.columns
    for artifact in (family, fold, audit_path):
        assert "TFG_NO_PUBLICADO" not in artifact.read_text()
    manifest = read_manifest(fold)
    with np.load(manifest["clients"][0]["dataset"]) as dataset:
        hours = pd.to_datetime(dataset["timestamps"])
        inside = (hours >= INCIDENT_START) & (hours < INCIDENT_END)
        assert np.isnan(dataset["values"][inside]).all()
        assert np.isfinite(dataset["values"][~inside]).all()
        assert (dataset["quality_flags"][inside] == INCIDENT_FLAG).all()
        assert int(inside.sum()) == audit["removed_hours"].sum()
