"""Station CSV discovery and loading, including five-minute UPCT exports."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
POLLUTANT = "PM25"


def normalize_pollutant(pollutant: str) -> str:
    if not isinstance(pollutant, str) or pollutant.strip().upper() != POLLUTANT:
        raise ValueError(f"El único contaminante admitido es {POLLUTANT} (PM2.5); recibido: {pollutant!r}.")
    return POLLUTANT


def filename_matches_pollutant(path: Path, pollutant: str) -> bool:
    token = re.escape(normalize_pollutant(pollutant))
    return re.search(rf"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])", path.stem, re.IGNORECASE) is not None


def _load_upct_pollutant_df(file_path: str, pollutant: str, valid_flags: str = "") -> pd.DataFrame:
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
    flag_columns = [col for col in df if str(col).strip().upper() == f"FLAG {pollutant_name}"]
    invalid_flags = 0
    if valid_flags:
        if len(flag_columns) != 1:
            raise ValueError("Se configuraron flags UPCT válidos pero no existe una única columna de flags.")
        allowed = {item.strip() for item in valid_flags.split(",")}
        accepted = df[flag_columns[0]].astype(str).str.strip().isin(allowed)
        numeric_flags = pd.to_numeric(df[flag_columns[0]], errors="coerce")
        for code in allowed:
            try:
                number = float(code)
            except ValueError:
                continue
            if math.isfinite(number):
                accepted |= numeric_flags.eq(number)
        invalid_flags = int((~accepted).sum())
        values = values.where(accepted)
    out = pd.DataFrame({pollutant_name: values.to_numpy()}, index=index)
    out.attrs["source_quality"] = {"invalid_flags": invalid_flags, "flag_columns": len(flag_columns),
                                   "invalid_timestamps": int(index.isna().sum())}
    out = out[~out.index.isna()]
    return out


def load_pollutant_file(
    file_path: str,
    pollutant: str,
    *,
    target_column_index: int = 0,
    grid_tolerance_seconds: int = 30,
    upct_valid_flags: str = "",
) -> tuple[str, pd.DataFrame] | None:
    pollutant = normalize_pollutant(pollutant)
    path = Path(file_path)
    is_upct = path.parent.name.strip().upper() == "UPCT"
    try:
        if is_upct:
            df = _load_upct_pollutant_df(file_path, pollutant, upct_valid_flags)
            series_name = f"UPCT_{pollutant.strip().upper()}"
        else:
            raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if target_column_index < 0 or target_column_index >= len(raw.columns):
                raise ValueError(
                    f"target_column_index={target_column_index} fuera de rango; "
                    f"columnas disponibles ({len(raw.columns)}): {list(raw.columns)}"
                )
            df = raw.iloc[:, [target_column_index]].copy()
            numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            df.attrs["source_quality"] = {"nonnumeric_readings": int((df.iloc[:, 0].notna() & numeric.isna()).sum())}
            df.iloc[:, 0] = numeric
            series_name = path.parent.name.strip()

        if df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df.attrs["source_quality"] = df.attrs.get("source_quality", {}) | {"invalid_timestamps": int(df.index.isna().sum())}
            df = df[~df.index.isna()]
        if df.empty:
            return None
        if df.index.tz is not None:
            raise ValueError("Fuente con zona horaria: normaliza y documenta su calendario antes de prepare.")
        rounded = df.index.round("5min")
        drift = abs((rounded - df.index).total_seconds())
        if (drift > grid_tolerance_seconds).any():
            raise ValueError("Desviación de la rejilla de cinco minutos superior a la tolerancia configurada.")
        audit = df.attrs.get("source_quality", {}) | {
            "off_grid_readings": int((drift > 0).sum()),
            "duplicate_timestamps": int(rounded.duplicated().sum()),
        }
        df.index = rounded
        if is_upct:
            df = df.groupby(level=0).mean()
        else:
            df = df[~df.index.duplicated(keep="last")]
        df.attrs["source_quality"] = audit
        return series_name, df.sort_index()
    except Exception as exc:  # noqa: BLE001 -- malformed station exports are logged and excluded.
        logger.warning("Error processing %s (file dropped from dataset): %s", file_path, exc)
        return None
