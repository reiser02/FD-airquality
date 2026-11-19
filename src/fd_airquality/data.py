"""Station-local preparation; calendar gaps and target boundaries stay explicit."""

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
from fd_airquality.loaders import (
    filename_matches_pollutant,
    load_pollutant_file,
    normalize_pollutant,
)
from fd_airquality.metrics import effective_error_scales, seasonal_error_scales
from fd_airquality.preprocessing import ensure_datetime_series, observed_blocks, preprocess
from fd_airquality.quality import coverage_rows, evaluation_identity, target_signature
from fd_airquality.splitting import (
    global_window_starts,
    masked_window_starts,
    select_holdout_window,
    split_train_val_subseries,
    window_observation_counts,
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
    window_policy: str = "strict"
    min_context_observed: int = 1
    min_target_observed: int = 1
    error_scale_policy: str = "unfloored"
    error_scale_reference: str = "pre-evaluation"
    error_scale_epsilon: float = 0.01
    min_scale_pairs: int = 1
    hourly_repeat_filter: int = 1
    grid_tolerance_seconds: int = 30
    upct_valid_flags: str = ""
    train_start: str = ""
    window_padding: str = "none"

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for key, value in asdict(self).items() if value is not None and key not in {
                "split_mode", "quality_mode", "validation_start", "test_start", "test_end",
                "window_policy", "error_scale_policy", "error_scale_epsilon", "hourly_repeat_filter",
                "grid_tolerance_seconds", "upct_valid_flags", "train_start", "error_scale_reference", "window_padding",
            }
        ):
            raise ValueError("Todas las longitudes y mínimos deben ser enteros positivos.")
        if self.stride > self.horizon:
            raise ValueError("stride no puede superar horizon (dejaría targets sin evaluar).")
        if self.window_policy not in {"strict", "masked"} or (self.split_mode == "station" and self.window_policy != "strict"):
            raise ValueError("window_policy debe ser strict o masked; station requiere strict.")
        if self.window_padding not in {"none", "right", "both"}:
            raise ValueError("window_padding debe ser none, right o both.")
        if self.window_padding != "none" and (self.window_policy != "masked" or self.split_mode != "global"):
            raise ValueError("El padding de entrenamiento requiere window_policy=masked y split_mode=global.")
        if self.min_context_observed > self.context or self.min_target_observed > self.horizon:
            raise ValueError("Los mínimos observados no pueden superar contexto/horizonte.")
        effective_error_scales(1., 1., 1, policy=self.error_scale_policy, epsilon=self.error_scale_epsilon)
        if self.error_scale_reference not in {"pre-evaluation", "training"}:
            raise ValueError("error_scale_reference debe ser pre-evaluation o training.")
        if type(self.hourly_repeat_filter) is not int or self.hourly_repeat_filter not in {0, 1}:
            raise ValueError("hourly_repeat_filter debe ser 0/1.")
        if type(self.grid_tolerance_seconds) is not int or not 0 <= self.grid_tolerance_seconds < 150:
            raise ValueError("grid_tolerance_seconds debe estar entre 0 y 149.")
        if self.min_useful > 12:
            raise ValueError("min_useful no puede superar las doce lecturas horarias.")
        for name in ("train_start",):
            value = getattr(self, name)
            if value:
                date = pd.Timestamp(value)
                upper = pd.Timestamp(self.validation_start)
                if (pd.isna(date) or date.tz is not None or date != date.floor("h")
                        or date > upper or (name == "train_start" and date == upper)):
                    raise ValueError(f"{name} debe ser una fecha horaria anterior al corte de entrenamiento.")
        if self.split_mode == "station":
            for length in (self.validation_hours, self.test_hours):
                if length < self.horizon or (length - self.horizon) % self.stride:
                    raise ValueError("Validación y test deben cubrir horizon + un múltiplo de stride.")
        if self.error_scale_hours <= self.seasonality:
            raise ValueError("error_scale_hours debe superar seasonality para formar diferencias.")
        if self.min_scale_pairs > self.error_scale_hours - self.seasonality:
            raise ValueError("min_scale_pairs supera los pares posibles de la referencia.")
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
            reference = "validation" if config.error_scale_reference == "training" else name
            end = pd.Timestamp(getattr(config, f"{reference}_start") if config.split_mode == "global"
                               else split[f"{reference}_target_start"])
            start = end - pd.Timedelta(hours=config.error_scale_hours)
            reference = series.loc[(series.index >= start) & (series.index < end)]
            mase, rmsse, pairs = seasonal_error_scales(reference, config.seasonality)
            mase, rmsse = effective_error_scales(mase, rmsse, pairs, policy=config.error_scale_policy,
                                                 epsilon=config.error_scale_epsilon, min_pairs=config.min_scale_pairs)
            split.update({
                f"{name}_mase_scale": mase, f"{name}_rmsse_scale": rmsse,
                f"{name}_scale_pairs": pairs,
                f"{name}_scale_start": str(start), f"{name}_scale_end": str(end),
            })
    return split


def _station_split(series: pd.Series, config: DataConfig) -> dict[str, Any] | None:
    """Select the fixed station holdout and strictly earlier training targets."""
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
        prefix,
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
        start = int(index.get_loc(block.index[0]))
        starts.extend(range(start, start + len(block) - width + 1, config.train_stride))
        train_points.append(block.to_numpy())
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) < config.min_train_windows:
        return None
    train_values = np.concatenate(train_points)
    mean = float(train_values.mean())
    scale = float(train_values.std())
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    validation_starts = np.array(
        [index.get_loc(block.index[0]) for block in validation], dtype=np.int64
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
        first = int(index.get_loc(block.index[0]))
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
        "train_end": str(train[-1].index[-1]),
        "validation_target_start": str(validation[0].index[config.context]),
        "validation_target_end": str(validation[-1].index[-1]),
        "test_target_start": str(window["test_target_start"]),
        "test_target_end": str(window["test_target_end"]),
    }


def _global_candidates(series: pd.Series, config: DataConfig) -> dict:
    function = masked_window_starts if config.window_policy == "masked" else global_window_starts
    result = function(
        series, context=config.context, horizon=config.horizon,
        validation_start=config.validation_start, test_start=config.test_start, test_end=config.test_end,
        train_stride=config.train_stride, stride=config.stride,
        **({"min_context_observed": config.min_context_observed,
             "min_target_observed": config.min_target_observed, "train_start": config.train_start,
             "window_padding": config.window_padding}
           if config.window_policy == "masked" else {}),
    )
    if config.train_start and config.window_policy == "strict":
        starts = result["train_starts"]
        result["train_starts"] = starts[series.index[starts + config.context] >= pd.Timestamp(config.train_start)]
        result["train_mask"] &= series.index >= pd.Timestamp(config.train_start)
    return result


def _global_split(series: pd.Series, config: DataConfig) -> dict | None:
    if not len(series):
        return None
    split = _global_candidates(series, config)
    if any(len(split[f"{name}_starts"]) < minimum for name, minimum in (
        ("train", config.min_train_windows), ("validation", config.min_validation_windows), ("test", config.min_test_windows),
    )):
        return None
    normalization_mask = split["train_mask"]
    values = series.to_numpy()[normalization_mask]
    if not len(values):
        return None
    scale = float(values.std())
    split.update(
        mean=float(values.mean()), scale=scale if np.isfinite(scale) and scale >= 1e-8 else 1.0,
        training_hours=int(split["train_mask"].sum()), normalization_hours=len(values),
        training_blocks=len(observed_blocks(series.where(split["train_mask"]))),
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


def window_target_count(values: np.ndarray, starts: np.ndarray, context: int, horizon: int,
                        target_end: int | None = None) -> int:
    return int(window_observation_counts(values, starts, context, horizon, target_end)[1].sum())


def prepare(
    raw_root: Path,
    output: Path,
    pollutant: str,
    config: DataConfig,
    stations: set[str] | None = None,
) -> Path:
    """Prepare one local artifact per station and a metadata-only server manifest."""
    pollutant = normalize_pollutant(pollutant)
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
    coverage = []
    candidate_train_windows = candidate_training_hours = retained_train_windows = 0
    names: set[str] = set()
    for path in files:
        loaded = load_pollutant_file(str(path), pollutant, grid_tolerance_seconds=config.grid_tolerance_seconds,
                                     upct_valid_flags=config.upct_valid_flags)
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
            hourly_repeat_filter=config.hourly_repeat_filter,
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
            **raw.attrs.get("source_quality", {}), **hourly.attrs.get("quality_counts", {}),
        }
        split = split_station(series, config) if len(series) else None
        candidates = _global_candidates(series, config) if config.split_mode == "global" and len(series) else None
        if candidates is not None:
            candidate_train_windows += len(candidates["train_starts"])
            candidate_training_hours += int(candidates["train_mask"].sum())
            coverage.extend(coverage_rows(series, config, candidates, station))
        last_observed = series.last_valid_index()
        row["last_observation"] = str(last_observed) if last_observed is not None else ""
        prior = series.loc[series.index < pd.Timestamp(config.validation_start)].last_valid_index()
        row["last_observation_before_validation"] = str(prior) if prior is not None else ""
        row["inactive_days_before_validation"] = ((pd.Timestamp(config.validation_start) - prior).total_seconds() / 86400
                                                   if prior is not None and config.split_mode == "global" else None)
        if candidates is not None:
            row.update({f"candidate_{name}_windows": len(candidates[f"{name}_starts"]) for name in ("train", "validation", "test")})
        for block in blocks.itertuples(index=False):
            first = int(series.index.get_loc(block.start))
            last = int(series.index.get_loc(block.end))
            counts = {
                f"num_{name}_windows": int(np.sum(
                    (split[f"{name}_starts"] >= first)
                    & (split[f"{name}_starts"] + config.context + config.horizon - 1 <= (
                        min(last, int(series.index.searchsorted(pd.Timestamp(config.validation_start))) - 1)
                        if name == "train" and config.window_padding != "none" else last))
                )) if split is not None else 0
                for name in ("train", "validation", "test")
            }
            block_rows.append({"station": station, "start": block.start, "end": block.end,
                               "hours": block.hours, "kind": "fully_observed_block", **counts})
        if split is not None and config.window_policy == "masked":
            contained = {name: sum(row[f"num_{name}_windows"] for row in block_rows if row["station"] == station)
                         for name in ("train", "validation", "test")}
            block_rows.append({
                "station": station, "start": series.index[0], "end": series.index[-1],
                "hours": int(series.isna().sum()), "kind": ("windows_with_missing_or_padding"
                         if config.window_padding != "none" else "windows_crossing_missing_values"),
                **{f"num_{name}_windows": len(split[f"{name}_starts"]) - contained[name]
                   for name in ("train", "validation", "test")},
            })
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
            allow_missing=np.array(config.window_policy == "masked"),
            window_padding=np.array(config.window_padding),
            train_target_end=np.array(int(series.index.searchsorted(pd.Timestamp(config.validation_start)))
                                      if config.split_mode == "global" else len(series)),
            useful_readings=hourly.attrs["useful_readings"],
            quality_flags=hourly.attrs["quality_flags"],
            # Keep all origins here. The epoch sampler runs after the arrival
            # cutoff and the model-specific availability filters.
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
            "num_train_candidate_windows": len(candidates["train_starts"]) if candidates is not None else len(split["train_starts"]),
            "num_validation_windows": len(split["validation_starts"]),
            "num_test_windows": len(split["test_starts"]),
            **{f"num_{name}_targets": window_target_count(series.to_numpy(), split[f"{name}_starts"], config.context, config.horizon,
                                                        int(series.index.searchsorted(pd.Timestamp(config.validation_start)))
                                                        if name == "train" and config.split_mode == "global" else None)
               for name in ("train", "validation", "test")},
            **{f"{name}_signature": target_signature(series, split[f"{name}_starts"], config.context, config.horizon,
                                                     split[f"{name}_scale_start"], split[f"{name}_scale_end"])
               for name in ("validation", "test")},
            "validation_days": split.get("validation_days", 0), "test_days": split.get("test_days", 0),
            "robust_evaluation": split.get("validation_days", 0) >= 3 and split.get("test_days", 0) >= 7,
            **{f"{name}_scale_pairs": split[f"{name}_scale_pairs"] for name in ("validation", "test")},
            **{f"{name}_{metric}_defined": split[f"{name}_{metric}_scale"] > 0
               for name in ("validation", "test") for metric in ("mase", "rmsse")},
            **{key: value for key, value in split.items() if key.endswith(("_start", "_end"))},
        })
    pd.DataFrame(inventory).to_csv(run / "stations.csv", index=False)
    pd.DataFrame(coverage).to_csv(run / "coverage.csv", index=False)
    pd.DataFrame(block_rows, columns=[
        "station", "start", "end", "hours", "kind", "num_train_windows", "num_validation_windows", "num_test_windows"
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
        "retained_pre_cutoff_train_windows": retained_train_windows,
        "train_window_budget_per_station_per_epoch": config.max_train_windows,
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
    document = {
        "schema_version": 6 if config.window_padding != "none" else 5 if config.window_policy == "masked" else 4,
        "protocol": ("global_masked_forecasting_v1" if config.window_policy == "masked" else "global_temporal_forecasting_v3") if config.split_mode == "global" else "station_local_retrospective_forecasting_v2",
        "pollutant": pollutant,
        "raw_root": str(raw_root.resolve()),
        "preparation_sources": source_fingerprints(),
        "preparation_versions": {name: version(name) for name in ("numpy", "pandas")},
        "geometry": asdict(config),
        "clients": clients,
    }
    document["evaluation_sha256"] = evaluation_identity(document)
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {2, 3, 4, 5, 6} or not manifest.get("clients"):
        raise ValueError("Manifest federado vacío o incompatible; vuelve a ejecutar prepare.")
    normalize_pollutant(manifest.get("pollutant", ""))
    clients = manifest["clients"]
    if [client["partition_id"] for client in clients] != list(range(len(clients))):
        raise ValueError("Los partition_id deben ser únicos y consecutivos desde cero.")
    # Preserve inspection of historical artifacts without reviving shared scalers.
    geometry = {key: value for key, value in manifest["geometry"].items() if key != "normalization_end"}
    config = DataConfig(**(geometry if manifest["schema_version"] >= 3 else geometry | {
        "split_mode": "station", "quality_mode": "retrospective",
    }))
    if (manifest["schema_version"] == 6) != (config.window_padding != "none"):
        raise ValueError("Esquema de padding incompatible; vuelve a ejecutar prepare.")
    if config.split_mode == "global":
        for client in clients:
            if not (pd.Timestamp(client["train_end"]) < pd.Timestamp(config.validation_start)
                    <= pd.Timestamp(client["validation_target_start"]) <= pd.Timestamp(client["validation_target_end"])
                    < pd.Timestamp(config.test_start) <= pd.Timestamp(client["test_target_start"])
                    <= pd.Timestamp(client["test_target_end"]) < pd.Timestamp(config.test_end)):
                raise ValueError("Violación temporal global en el manifest.")
    if manifest["schema_version"] >= 4:
        for client in clients:
            for name in ("validation", "test"):
                reference = "validation" if config.error_scale_reference == "training" else name
                end = pd.Timestamp(getattr(config, f"{reference}_start") if config.split_mode == "global"
                                   else client[f"{reference}_target_start"])
                if (pd.Timestamp(client[f"{name}_scale_end"]) != end
                        or pd.Timestamp(client[f"{name}_scale_start"]) != end - pd.Timedelta(hours=config.error_scale_hours)):
                    raise ValueError("Referencia temporal de MASE/RMSSE incompatible con el manifest.")
    return manifest


class WindowDataset(Dataset):
    """Slice a single station lazily; starts, not compacted values, define windows."""

    def __init__(self, artifact: dict[str, np.ndarray], split: str, context: int, horizon: int,
                 *, allow_empty: bool = False):
        self.values = artifact["values"]
        self.starts = artifact[f"{split}_starts"]
        self.mean = float(artifact["mean"])
        self.scale = float(artifact["scale"])
        self.context, self.horizon = context, horizon
        self.split = split
        self.allow_missing = bool(artifact.get("allow_missing", False))
        self.padding_mode = str(artifact.get("window_padding", "none")) if split == "train" else "none"
        self.target_end = (min(int(artifact.get("train_target_end", len(self.values))), len(self.values))
                           if split == "train" else len(self.values))
        if self.padding_mode not in {"none", "right", "both"} or self.target_end < 0:
            raise ValueError("Padding o extremo autorizado de entrenamiento inválido.")
        if not len(self.starts) and not allow_empty:
            raise ValueError(f"No hay ventanas de {split}.")
        width = context + horizon
        minimum_start = 1 - context if self.padding_mode == "both" else 0
        if (np.any(self.starts < minimum_start)
                or np.any(self.starts + context >= self.target_end)
                or (self.padding_mode == "none" and np.any(self.starts + width > self.target_end))):
            raise ValueError("Ventana fuera de la serie o del intervalo temporal autorizado.")
        # Prefix sums validate all windows in O(series length + window count).
        missing = np.r_[0, np.cumsum(~np.isfinite(self.values))]
        left = np.clip(self.starts, 0, self.target_end)
        right = np.clip(self.starts + width, 0, self.target_end)
        if not self.allow_missing and (self.padding_mode != "none" or np.any(missing[right] - missing[left])):
            raise ValueError("Una ventana cruza un hueco o contiene valores no finitos.")
        if self.allow_missing:
            if np.isinf(self.values).any():
                raise ValueError("Los infinitos no son una ausencia válida.")
            context_counts, target_counts = self.observation_counts()
            if np.any(context_counts == 0) or np.any(target_counts == 0):
                raise ValueError("Una ventana no contiene contexto o targets observados.")
        times = artifact["timestamps"]
        self.timestamps = times
        breaks = np.r_[0, np.cumsum(np.diff(times) != pd.Timedelta(hours=1).value)]
        if np.any(breaks[np.maximum(right - 1, 0)] - breaks[left]):
            raise ValueError("Una ventana cruza horas ausentes.")
        if not np.isfinite(self.mean) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Normalización local inválida.")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        width = self.context + self.horizon
        if self.padding_mode == "none":
            values = self.values[start:start + width].copy()
        else:
            values = np.full(width, np.nan, dtype=np.result_type(self.values.dtype, np.float32))
            left, right = max(0, start), min(start + width, self.target_end)
            values[left - start:right - start] = self.values[left:right]
        values = (values - self.mean) / self.scale
        return torch.from_numpy(values[: self.context].copy()), torch.from_numpy(
            values[self.context :].copy()
        )

    def observation_counts(self):
        return window_observation_counts(self.values, self.starts, self.context, self.horizon, self.target_end)

    def target_values(self, starts: np.ndarray | None = None) -> np.ndarray:
        """Original labels, with unauthorized/virtual positions left missing."""
        starts = self.starts if starts is None else starts
        positions = starts[:, None] + self.context + np.arange(self.horizon)
        actual = np.full(positions.shape, np.nan, dtype=np.result_type(self.values.dtype, np.float32))
        real = (positions >= 0) & (positions < self.target_end)
        actual[real] = self.values[positions[real]]
        return actual

    def padding_counts(self, starts: np.ndarray | None = None) -> tuple[int, int]:
        starts = self.starts if starts is None else starts
        return (int(np.maximum(-starts, 0).sum()),
                int(np.maximum(starts + self.context + self.horizon - self.target_end, 0).sum()))

    def target_time_bounds(self, starts: np.ndarray | None = None) -> tuple[int, int]:
        """First and last real target slots, excluding the virtual horizon."""
        starts = self.starts if starts is None else starts
        first = int(starts.min()) + self.context
        last = min(int(starts.max()) + self.context + self.horizon - 1, self.target_end - 1)
        return int(self.timestamps[first]), int(self.timestamps[last])

    def context_timestamps(self, index: int):
        """Virtual context dates are metadata only; their input mask is zero."""
        positions = int(self.starts[index]) + np.arange(self.context)
        times = self.timestamps[np.maximum(positions, 0)].copy()
        times[positions < 0] += positions[positions < 0] * pd.Timedelta(hours=1).value
        return times


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
        if str(artifact.get("window_padding", "none")) != geometry.get("window_padding", "none"):
            raise ValueError("La política de padding del NPZ no coincide con el manifest.")
        if manifest["schema_version"] == 6:
            expected_end = int(np.searchsorted(artifact["timestamps"], pd.Timestamp(geometry["validation_start"]).value))
            if int(artifact.get("train_target_end", -1)) != expected_end:
                raise ValueError("El extremo de targets de entrenamiento no coincide con su corte.")
        for split in ("train", "validation", "test"):
            windows = WindowDataset(artifact, split, geometry["context"], geometry["horizon"])
            if windows.allow_missing != (geometry.get("window_policy", "strict") == "masked"):
                raise ValueError("La política de ausencias del NPZ no coincide con el manifest.")
            if len(windows) != client[f"num_{split}_windows"] or np.any(np.diff(windows.starts) <= 0):
                raise ValueError("Número u orden de ventanas incompatible con el manifest.")
            if manifest["schema_version"] >= 5:
                expected = window_target_count(artifact["values"], windows.starts, geometry["context"],
                                               geometry["horizon"], windows.target_end)
                if not windows.allow_missing or expected != client[f"num_{split}_targets"]:
                    raise ValueError("Máscara o recuento de targets incompatible con el manifest.")
                context_counts, target_counts = windows.observation_counts()
                if (np.any(context_counts < geometry["min_context_observed"])
                        or np.any(target_counts < geometry["min_target_observed"])):
                    raise ValueError("Una ventana incumple los mínimos observados del manifest.")
            if geometry["split_mode"] == "global":
                first = pd.to_datetime(artifact["timestamps"][windows.starts + geometry["context"]])
                last = pd.to_datetime(artifact["timestamps"][np.minimum(
                    windows.starts + geometry["context"] + geometry["horizon"] - 1, windows.target_end - 1)])
                upper = geometry["validation_start"] if split == "train" else geometry["test_start"] if split == "validation" else geometry["test_end"]
                lower = None if split == "train" else geometry["validation_start"] if split == "validation" else geometry["test_start"]
                if (last >= pd.Timestamp(upper)).any() or (lower is not None and (first < pd.Timestamp(lower)).any()):
                    raise ValueError("Violación temporal en las ventanas reales del cliente.")
        scale_keys = ([f"{name}_{metric}_scale" for name in ("validation", "test") for metric in ("mase", "rmsse")]
                      if manifest["schema_version"] >= 4 else ["mase_scale", "rmsse_scale"])
        for key in scale_keys:
            if key not in artifact or not np.isfinite(artifact[key]) or float(artifact[key]) < 0:
                raise ValueError("Escala de error de referencia inválida; vuelve a ejecutar prepare.")
        if manifest["schema_version"] >= 4:
            for name in ("validation", "test"):
                pairs = artifact.get(f"{name}_scale_pairs")
                if (pairs is None or pairs.dtype.kind not in "iu" or pairs.ndim != 0
                        or not 0 <= int(pairs) <= geometry["error_scale_hours"] - geometry["seasonality"]
                        or int(pairs) != client[f"{name}_scale_pairs"]):
                    raise ValueError("Recuento de pares MASE/RMSSE incompatible con el manifest.")
    return manifest, artifact
