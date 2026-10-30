"""Station CSV discovery and loading, including five-minute UPCT exports."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def filename_matches_pollutant(path: Path, pollutant: str) -> bool:
    token = re.escape(pollutant.strip())
    return re.search(rf"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])", path.stem, re.IGNORECASE) is not None


def _load_upct_pollutant_df(file_path: str, pollutant: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";", decimal=",", skiprows=1)
    if "Datetime" not in df.columns:
        raise ValueError("el CSV UPCT no contiene la columna 'Datetime'")

    pollutant_name = pollutant.strip().upper()
    value_columns = [
        column
        for column in df.columns
        if str(column).strip().upper() == pollutant_name
        or str(column).strip().upper().startswith(f"{pollutant_name} ")
    ]
    value_columns = [
        column
        for column in value_columns
        if not str(column).strip().upper().startswith(("FLAG ", "DESCRIPTION "))
    ]
    if len(value_columns) != 1:
        raise ValueError(
            f"no se pudo identificar una única columna de {pollutant_name}: {value_columns}"
        )

    index = pd.to_datetime(df["Datetime"], errors="coerce")
    values = pd.to_numeric(df[value_columns[0]], errors="coerce")
    out = pd.DataFrame({pollutant_name: values.to_numpy()}, index=index)
    out = out[~out.index.isna()]
    # Samples can drift by a few seconds around the nominal five-minute grid.
    out.index = out.index.round("5min")
    return out.groupby(level=0).mean().sort_index()


def load_pollutant_file(
    file_path: str,
    pollutant: str,
    *,
    target_column_index: int = 0,
) -> tuple[str, pd.DataFrame] | None:
    path = Path(file_path)
    is_upct = path.parent.name.strip().upper() == "UPCT"
    try:
        if is_upct:
            df = _load_upct_pollutant_df(file_path, pollutant)
            series_name = f"UPCT_{pollutant.strip().upper()}"
        else:
            raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if target_column_index < 0 or target_column_index >= len(raw.columns):
                raise ValueError(
                    f"target_column_index={target_column_index} fuera de rango; "
                    f"columnas disponibles ({len(raw.columns)}): {list(raw.columns)}"
                )
            df = raw.iloc[:, [target_column_index]].copy()
            series_name = path.parent.name.strip()

        if df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        if df.empty:
            return None
        return series_name, df.sort_index()
    except Exception as exc:  # noqa: BLE001 -- malformed station exports are logged and excluded.
        logger.warning("Error processing %s (file dropped from dataset): %s", file_path, exc)
        return None
