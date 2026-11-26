"""Download the public hourly series for the CARM station of Mompeán."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

STATION = "Mompeán"
URL = "https://sinqlair.carm.es/calidadaire/obtener_datos.aspx?" + urlencode({
    "tipo": "tablaDatosDiarios", "estacion": STATION,
})
COLUMNS = ["fecha", "PM25", "PM10"]


def _download_day(day: date, cache: Path) -> pd.DataFrame:
    path = cache / f"{day.isoformat()}.csv"
    if path.is_file():
        return _validate_day(pd.read_csv(path, parse_dates=["fecha"]), day)
    request = Request(URL, data=urlencode({
        "fechaDatosDiarios": day.strftime("%d/%m/%Y"),
        "estacionDatosDiarios": STATION,
        "contaminantesDatosDiarios": "todos",
        "tipodatos": "grid",
    }).encode())
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8-sig")
            records = [dict(re.findall(r"(\w+):'([^']*)'", row)) for row in re.findall(r"\{([^{}]*)\}", text)]
            frame = pd.DataFrame(records)
            if not set(COLUMNS) <= set(frame):
                raise ValueError(f"Respuesta horaria incompleta de CARM para {day}.")
            frame = frame[COLUMNS].copy()
            frame["fecha"] = pd.to_datetime(frame["fecha"], format="%d/%m/%Y %H:%M")
            for pollutant in COLUMNS[1:]:
                frame[pollutant] = pd.to_numeric(frame[pollutant].str.replace(",", ".", regex=False), errors="coerce")
            frame = _validate_day(frame, day)
            _write_csv(frame, path)
            return frame
        except (OSError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"No se pudo descargar Mompeán para {day}: {exc}") from exc
            sleep(2 ** attempt)


def _validate_day(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    frame = frame.drop_duplicates()
    index = pd.DatetimeIndex(frame["fecha"])
    if not index.is_unique or not index.isin(pd.date_range(day, periods=24, freq="h")).all():
        raise ValueError(f"Calendario inesperado de CARM para {day}.")
    return frame.sort_values("fecha")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as target:
        temporary = Path(target.name)
        try:
            frame.to_csv(target, index=False)
            target.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desde", required=True, type=date.fromisoformat)
    parser.add_argument("--hasta", required=True, type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=Path("data/raw/mompean/mompean_horario.csv"))
    args = parser.parse_args()
    if args.desde > args.hasta:
        parser.error("--desde debe ser anterior o igual a --hasta.")
    cache = args.output.parent / "dias"
    cache.mkdir(parents=True, exist_ok=True)
    days = [args.desde + timedelta(days=offset) for offset in range((args.hasta - args.desde).days + 1)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        series = pd.concat(pool.map(lambda day: _download_day(day, cache), days), ignore_index=True)
    _write_csv(series, args.output)
    print(f"Mompeán: {len(series)} horas · {series['fecha'].min()} → {series['fecha'].max()} · {args.output}")


if __name__ == "__main__":
    main()
