"""Station-local preparation; no window crosses a missing hour or split target."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fd_airquality.artifacts import create_run_dir, file_digest, source_fingerprints
from fd_airquality.loaders import filename_matches_pollutant, load_pollutant_file
from fd_airquality.metrics import seasonal_error_scales
from fd_airquality.preprocessing import ensure_datetime_series, observed_blocks, preprocess
from fd_airquality.splitting import (
    global_window_starts,
    select_holdout_window,
    split_train_val_subseries,
)


@dataclass(frozen=True)
class DataConfig:
    context: int = 90
    horizon: int = 12
    stride: int = 6
    validation_hours: int = 48
    test_hours: int = 96
    train_stride: int = 1
    min_train_windows: int = 1
    max_train_windows: int | None = None
    seasonality: int = 24
    error_scale_hours: int = 672
    min_run: int = 6
    min_useful: int = 1
    zero_run_hours: int = 24
    split_mode: str = "global"
    quality_mode: str = "causal"
    validation_start: str = "2026-03-01"
    test_start: str = "2026-03-29"
    test_end: str = "2026-04-26"
    min_validation_windows: int = 7
    min_test_windows: int = 15

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for key, value in asdict(self).items() if value is not None and key not in {
                "split_mode", "quality_mode", "validation_start", "test_start", "test_end"
            }
        ):
            raise ValueError("Todas las longitudes y mínimos deben ser enteros positivos.")
        if self.stride > self.horizon:
            raise ValueError("stride no puede superar horizon (dejaría targets sin evaluar).")
        if self.split_mode == "station":
            for length in (self.validation_hours, self.test_hours):
                if length < self.horizon or (length - self.horizon) % self.stride:
                    raise ValueError("Validación y test deben cubrir horizon + un múltiplo de stride.")
        if self.error_scale_hours <= self.seasonality:
            raise ValueError("error_scale_hours debe superar seasonality para formar diferencias.")
        if self.max_train_windows is not None and self.max_train_windows < self.min_train_windows:
            raise ValueError("max_train_windows debe alcanzar min_train_windows.")
        if self.split_mode not in {"global", "station"} or self.quality_mode not in {"causal", "retrospective"}:
            raise ValueError("Modo temporal o quality_mode desconocido.")
        if self.split_mode == "global":
            try:
                dates = [pd.Timestamp(value) for value in (self.validation_start, self.test_start, self.test_end)]
                if any(pd.isna(date) or date.tz is not None or date != date.floor("h") for date in dates):
                    raise ValueError("Fechas horarias sin zona requeridas; conserva el calendario de los CSV.")
                if not dates[0] < dates[1] < dates[2]:
                    raise ValueError("Los cortes temporales deben estar estrictamente ordenados.")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Cortes temporales inválidos: {exc}") from exc
            if self.quality_mode != "causal":
                raise ValueError("El protocolo global exige quality_mode=causal.")


def split_station(series: pd.Series, config: DataConfig) -> dict[str, Any] | None:
    split = _global_split(series, config) if config.split_mode == "global" else _station_split(series, config)
    if split is not None:
        for name in ("validation", "test"):
            end = pd.Timestamp(getattr(config, f"{name}_start") if config.split_mode == "global"
                               else split[f"{name}_target_start"])
            start = end - pd.Timedelta(hours=config.error_scale_hours)
            reference = series.loc[(series.index >= start) & (series.index < end)]
            mase, rmsse, pairs = seasonal_error_scales(reference, config.seasonality)
            split.update({
                f"{name}_mase_scale": mase, f"{name}_rmsse_scale": rmsse,
                f"{name}_scale_pairs": pairs,
                f"{name}_scale_start": str(start), f"{name}_scale_end": str(end),
            })
    return split


def _station_split(series: pd.Series, config: DataConfig) -> dict[str, Any] | None:
    """Select the fixed station holdout and strictly earlier training targets."""
    from darts import TimeSeries

    width = config.context + config.horizon
    window = select_holdout_window(
        series,
        holdout=config.test_hours,
        context_len=config.context,
        train_min_len=width,
        validation_len=config.validation_hours,
        host_min_len=width + config.validation_hours,
    )
    if window is None:
        return None
    prefix = series.loc[window["train_index"]]
    split = split_train_val_subseries(
        TimeSeries.from_series(prefix),
        input_chunk=config.context,
        size_k=config.horizon,
        validation_len=config.validation_hours,
        validation_stride=config.stride,
    )
    if split is None:
        return None
    train, validation = split
    index = series.index
    starts = []
    train_points = []
    for block in train:
        start = int(index.get_loc(block.start_time()))
        starts.extend(range(start, start + len(block) - width + 1, config.train_stride))
        train_points.append(block.univariate_values())
    starts = np.asarray(starts, dtype=np.int64)
    if config.max_train_windows is not None and len(starts) > config.max_train_windows:
        # Deterministic coverage of all retained blocks, rather than only the last one.
        starts = starts[np.linspace(0, len(starts) - 1, config.max_train_windows, dtype=int)]
    if len(starts) < config.min_train_windows:
        return None
    train_values = np.concatenate(train_points)
    mean = float(train_values.mean())
    scale = float(train_values.std())
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    validation_starts = np.array(
        [index.get_loc(block.start_time()) for block in validation], dtype=np.int64
    )
    test_start = int(index.get_loc(window["test_context_start"]))
    test_starts = np.arange(
        test_start,
        test_start + config.test_hours - config.horizon + 1,
        config.stride,
        dtype=np.int64,
    )
    train_mask = np.zeros(len(series), dtype=bool)
    for block in train:
        first = int(index.get_loc(block.start_time()))
        train_mask[first:first + len(block)] = True
    return {
        "train_starts": starts,
        "validation_starts": validation_starts,
        "test_starts": test_starts,
        "mean": mean,
        "scale": scale,
        "train_mask": train_mask,
        "training_hours": len(train_values),
        "training_blocks": len(train),
        "train_end": str(train[-1].end_time()),
        "validation_target_start": str(validation[0].time_index[config.context]),
        "validation_target_end": str(validation[-1].end_time()),
        "test_target_start": str(window["test_target_start"]),
        "test_target_end": str(window["test_target_end"]),
    }


def _global_candidates(series: pd.Series, config: DataConfig) -> dict:
    return global_window_starts(
        series, context=config.context, horizon=config.horizon,
        validation_start=config.validation_start, test_start=config.test_start, test_end=config.test_end,
        train_stride=config.train_stride, stride=config.stride,
    )


def _global_split(series: pd.Series, config: DataConfig) -> dict | None:
    if not len(series):
        return None
    split = _global_candidates(series, config)
    if any(len(split[f"{name}_starts"]) < minimum for name, minimum in (
        ("train", config.min_train_windows), ("validation", config.min_validation_windows), ("test", config.min_test_windows),
    )):
        return None
    if config.max_train_windows is not None and len(split["train_starts"]) > config.max_train_windows:
        split["train_starts"] = split["train_starts"][np.linspace(0, len(split["train_starts"]) - 1, config.max_train_windows, dtype=int)]
    values = series.to_numpy()[split["train_mask"]]
    scale = float(values.std())
    split.update(
        mean=float(values.mean()), scale=scale if np.isfinite(scale) and scale >= 1e-8 else 1.0,
        training_hours=len(values), training_blocks=len(observed_blocks(series.where(split["train_mask"]))),
        train_end=str(series.index[split["train_mask"]][-1]),
    )
    for name in ("validation", "test"):
        targets = series.index[split[f"{name}_starts"] + config.context]
        split[f"{name}_target_start"] = str(targets[0])
        split[f"{name}_target_end"] = str(targets[-1] + pd.Timedelta(hours=config.horizon - 1))
        split[f"{name}_days"] = len(targets.normalize().unique())
    return split


def _source_files(root: Path, pollutant: str, stations: set[str] | None) -> list[Path]:
    return [
        path
        for path in sorted(root.glob("*/*.csv"))
        if filename_matches_pollutant(path, pollutant)
        and (stations is None or path.parent.name in stations)
    ]


def prepare(
    raw_root: Path,
    output: Path,
    pollutant: str,
    config: DataConfig,
    stations: set[str] | None = None,
) -> Path:
    """Prepare one local artifact per station and a metadata-only server manifest."""
    pollutant = pollutant.upper()
    files = _source_files(raw_root, pollutant, stations)
    if not files:
        raise FileNotFoundError(f"No hay CSV de {pollutant} bajo {raw_root} para las estaciones elegidas.")
    if stations is not None:
        missing = stations - {path.parent.name for path in files}
        if missing:
            raise ValueError(f"Estaciones sin CSV de {pollutant}: {sorted(missing)}")
    # Each preparation is immutable: do not silently replace datasets used by a running federation.
    run = create_run_dir(output.resolve(), f"{pollutant}_{datetime.now(UTC):%Y%m%d_%H%M%S}")
    clients, inventory, exclusions, block_rows = [], [], [], []
    candidate_train_windows = candidate_training_hours = retained_train_windows = 0
    names: set[str] = set()
    for path in files:
        loaded = load_pollutant_file(str(path), pollutant)
        if loaded is None:
            exclusions.append({"station": path.parent.name, "reason": "load_failed"})
            continue
        station, raw = loaded
        if station in names:
            raise ValueError(f"Más de un CSV para el mismo cliente: {station}")
        names.add(station)
        (hourly,), frozen = preprocess(
            [raw], pollutant, min_run=config.min_run, min_useful=config.min_useful,
            zero_run_hours=config.zero_run_hours, quality_mode=config.quality_mode,
        )
        series = ensure_datetime_series(hourly.iloc[:, 0], "h", station)
        series = series.where(np.isfinite(series))
        blocks = observed_blocks(series)
        row = {
            "station": station,
            "hourly_points": len(series),
            "observed_hours": int(series.notna().sum()),
            "missing_hours": int(series.isna().sum()),
            "observed_blocks": len(blocks),
            "longest_block": int(blocks.hours.max()) if len(blocks) else 0,
            "hourly_repeats_removed": frozen[0],
            "last_raw_timestamp": str(raw.index.max()) if len(raw) else "",
        }
        split = split_station(series, config) if len(series) else None
        candidates = _global_candidates(series, config) if config.split_mode == "global" and len(series) else None
        if candidates is not None:
            candidate_train_windows += len(candidates["train_starts"])
            candidate_training_hours += int(candidates["train_mask"].sum())
        last_observed = series.last_valid_index()
        row["last_observation"] = str(last_observed) if last_observed is not None else ""
        row["inactive_days_before_validation"] = max(0.0, (pd.Timestamp(config.validation_start) - last_observed).total_seconds() / 86400) if last_observed is not None and config.split_mode == "global" else None
        if candidates is not None:
            row.update({f"candidate_{name}_windows": len(candidates[f"{name}_starts"]) for name in ("train", "validation", "test")})
        for block in blocks.itertuples(index=False):
            first = int(series.index.get_loc(block.start))
            last = int(series.index.get_loc(block.end))
            counts = {
                f"num_{name}_windows": int(np.sum(
                    (split[f"{name}_starts"] >= first)
                    & (split[f"{name}_starts"] + config.context + config.horizon - 1 <= last)
                )) if split is not None else 0
                for name in ("train", "validation", "test")
            }
            block_rows.append({"station": station, "start": block.start, "end": block.end,
                               "hours": block.hours, **counts})
        inventory.append(row | {"eligible": split is not None} | ({
            "training_hours": split["training_hours"],
            "training_blocks": split["training_blocks"],
            "num_train_windows": len(split["train_starts"]),
        } if split is not None else {}))
        if split is None:
            reason = "no_fixed_test_or_training_host_or_windows"
            if candidates is not None:
                reason = ",".join(f"insufficient_{name}_windows" for name, minimum in (
                    ("train", config.min_train_windows), ("validation", config.min_validation_windows), ("test", config.min_test_windows),
                ) if len(candidates[f"{name}_starts"]) < minimum)
            exclusions.append(row | {"reason": reason})
            continue
        retained_train_windows += len(candidates["train_starts"]) if candidates is not None else len(split["train_starts"])
        partition = len(clients)
        local = run / "clients" / f"{partition:03d}"
        local.mkdir(parents=True)
        dataset = local / "series.npz"
        np.savez_compressed(
            dataset,
            values=series.to_numpy(dtype=np.float32),
            timestamps=series.index.asi8,
            **{key: split[key] for key in (
                "train_starts", "validation_starts", "test_starts", "mean", "scale",
                "train_mask",
            )},
            **{f"{name}_{key}": split[f"{name}_{key}"]
               for name in ("validation", "test") for key in ("mase_scale", "rmsse_scale", "scale_pairs")},
        )
        # Neither normalization statistics nor measurement arrays enter the server manifest.
        clients.append({
            "partition_id": partition,
            "station": station,
            "dataset": str(dataset),
            "sha256": file_digest(dataset),
            "source_sha256": file_digest(path),
            "num_train_windows": len(split["train_starts"]),
            "num_validation_windows": len(split["validation_starts"]),
            "num_test_windows": len(split["test_starts"]),
            "validation_days": split.get("validation_days", 0), "test_days": split.get("test_days", 0),
            "robust_evaluation": split.get("validation_days", 0) >= 3 and split.get("test_days", 0) >= 7,
            **{f"{name}_scale_pairs": split[f"{name}_scale_pairs"] for name in ("validation", "test")},
            **{f"{name}_{metric}_defined": split[f"{name}_{metric}_scale"] > 0
               for name in ("validation", "test") for metric in ("mase", "rmsse")},
            **{key: value for key, value in split.items() if key.endswith(("_start", "_end"))},
        })
    pd.DataFrame(inventory).to_csv(run / "stations.csv", index=False)
    pd.DataFrame(block_rows, columns=[
        "station", "start", "end", "hours", "num_train_windows", "num_validation_windows", "num_test_windows"
    ]).to_csv(run / "blocks.csv", index=False)
    pd.DataFrame(exclusions, columns=list(dict.fromkeys(
        ["station", "reason"] + [key for row in exclusions for key in row]
    ))).to_csv(run / "excluded_stations.csv", index=False)
    for exclusion in exclusions:
        logging.getLogger(__name__).warning("Estación descartada en prepare: %s; %s. Informe: %s",
                                            exclusion["station"], exclusion["reason"], run / "excluded_stations.csv")
    (run / "retention.json").write_text(json.dumps({
        "candidate_stations": len(files), "loaded_stations": len(inventory), "eligible_stations": len(clients),
        "candidate_pre_cutoff_train_windows": candidate_train_windows,
        "retained_pre_cap_train_windows": retained_train_windows,
        "used_train_windows": sum(client["num_train_windows"] for client in clients),
        "candidate_pre_cutoff_training_hours": candidate_training_hours,
        "retained_training_hours": sum(row.get("training_hours", 0) for row in inventory),
        "train_window_retention_pct": 100 * retained_train_windows / candidate_train_windows if candidate_train_windows else 0.0,
        "training_hour_retention_pct": 100 * sum(row.get("training_hours", 0) for row in inventory) / candidate_training_hours if candidate_training_hours else 0.0,
        "robust_evaluation_stations": sum(client["robust_evaluation"] for client in clients),
        "note": "Ventanas solapadas, no muestras independientes. Retención anterior al corte; evaluación no es desperdicio.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not clients:
        raise ValueError(f"Ninguna estación es elegible. Consulta {run / 'excluded_stations.csv'}")
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 4,
        "protocol": "global_temporal_forecasting_v3" if config.split_mode == "global" else "station_local_retrospective_forecasting_v2",
        "pollutant": pollutant,
        "raw_root": str(raw_root.resolve()),
        "preparation_sources": source_fingerprints(),
        "preparation_versions": {name: version(name) for name in ("darts", "numpy", "pandas")},
        "geometry": asdict(config),
        "clients": clients,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {2, 3, 4} or not manifest.get("clients"):
        raise ValueError("Manifest federado vacío o incompatible; vuelve a ejecutar prepare.")
    clients = manifest["clients"]
    if [client["partition_id"] for client in clients] != list(range(len(clients))):
        raise ValueError("Los partition_id deben ser únicos y consecutivos desde cero.")
    config = DataConfig(**(manifest["geometry"] if manifest["schema_version"] >= 3 else manifest["geometry"] | {
        "split_mode": "station", "quality_mode": "retrospective",
    }))
    if config.split_mode == "global":
        for client in clients:
            if not (pd.Timestamp(client["train_end"]) < pd.Timestamp(config.validation_start)
                    <= pd.Timestamp(client["validation_target_start"]) <= pd.Timestamp(client["validation_target_end"])
                    < pd.Timestamp(config.test_start) <= pd.Timestamp(client["test_target_start"])
                    <= pd.Timestamp(client["test_target_end"]) < pd.Timestamp(config.test_end)):
                raise ValueError("Violación temporal global en el manifest.")
    if manifest["schema_version"] == 4:
        for client in clients:
            for name in ("validation", "test"):
                end = pd.Timestamp(getattr(config, f"{name}_start") if config.split_mode == "global"
                                   else client[f"{name}_target_start"])
                if (pd.Timestamp(client[f"{name}_scale_end"]) != end
                        or pd.Timestamp(client[f"{name}_scale_start"]) != end - pd.Timedelta(hours=config.error_scale_hours)):
                    raise ValueError("Referencia temporal de MASE/RMSSE incompatible con el manifest.")
    return manifest


class WindowDataset(Dataset):
    """Slice a single station lazily; starts, not compacted values, define windows."""

    def __init__(self, artifact: dict[str, np.ndarray], split: str, context: int, horizon: int):
        self.values = artifact["values"]
        self.starts = artifact[f"{split}_starts"]
        self.mean = float(artifact["mean"])
        self.scale = float(artifact["scale"])
        self.context, self.horizon = context, horizon
        self.split = split
        if not len(self.starts):
            raise ValueError(f"No hay ventanas de {split}.")
        width = context + horizon
        if np.any(self.starts < 0) or np.any(self.starts + width > len(self.values)):
            raise ValueError("Ventana fuera de la serie.")
        # Prefix sums validate all windows in O(series length + window count).
        missing = np.r_[0, np.cumsum(~np.isfinite(self.values))]
        if np.any(missing[self.starts + width] - missing[self.starts]):
            raise ValueError("Una ventana cruza un hueco o contiene valores no finitos.")
        times = artifact["timestamps"]
        breaks = np.r_[0, np.cumsum(np.diff(times) != pd.Timedelta(hours=1).value)]
        if np.any(breaks[self.starts + width - 1] - breaks[self.starts]):
            raise ValueError("Una ventana cruza horas ausentes.")
        if not np.isfinite(self.mean) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Normalización local inválida.")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        values = (self.values[start : start + self.context + self.horizon] - self.mean) / self.scale
        return torch.from_numpy(values[: self.context].copy()), torch.from_numpy(
            values[self.context :].copy()
        )


def load_client_data(manifest_path: str, partition: int) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = read_manifest(manifest_path)
    if not 0 <= partition < len(manifest["clients"]):
        raise ValueError(f"Cliente sin estación: partition-id={partition}")
    client = manifest["clients"][partition]
    dataset = Path(client["dataset"])
    if file_digest(dataset) != client["sha256"]:
        raise ValueError(f"El dataset local cambió después de prepare: {dataset}")
    with np.load(dataset, allow_pickle=False) as archive:
        artifact = {key: archive[key] for key in archive.files}
    if manifest["schema_version"] >= 3:
        geometry = manifest["geometry"]
        for split in ("train", "validation", "test"):
            windows = WindowDataset(artifact, split, geometry["context"], geometry["horizon"])
            if len(windows) != client[f"num_{split}_windows"] or np.any(np.diff(windows.starts) <= 0):
                raise ValueError("Número u orden de ventanas incompatible con el manifest.")
            if geometry["split_mode"] == "global":
                first = pd.to_datetime(artifact["timestamps"][windows.starts + geometry["context"]])
                last = first + pd.Timedelta(hours=geometry["horizon"] - 1)
                upper = geometry["validation_start"] if split == "train" else geometry["test_start"] if split == "validation" else geometry["test_end"]
                lower = None if split == "train" else geometry["validation_start"] if split == "validation" else geometry["test_start"]
                if (last >= pd.Timestamp(upper)).any() or (lower is not None and (first < pd.Timestamp(lower)).any()):
                    raise ValueError("Violación temporal en las ventanas reales del cliente.")
        scale_keys = ([f"{name}_{metric}_scale" for name in ("validation", "test") for metric in ("mase", "rmsse")]
                      if manifest["schema_version"] == 4 else ["mase_scale", "rmsse_scale"])
        for key in scale_keys:
            if key not in artifact or not np.isfinite(artifact[key]) or float(artifact[key]) < 0:
                raise ValueError("Escala de error de referencia inválida; vuelve a ejecutar prepare.")
        if manifest["schema_version"] == 4:
            for name in ("validation", "test"):
                pairs = artifact.get(f"{name}_scale_pairs")
                if (pairs is None or pairs.dtype.kind not in "iu" or pairs.ndim != 0
                        or not 0 <= int(pairs) <= geometry["error_scale_hours"] - geometry["seasonality"]
                        or int(pairs) != client[f"{name}_scale_pairs"]):
                    raise ValueError("Recuento de pares MASE/RMSSE incompatible con el manifest.")
    return manifest, artifact
