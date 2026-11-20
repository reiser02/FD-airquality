"""Temporal cross-validation: fresh models per fold, expanding or sliding data."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fd_airquality.artifacts import create_run_dir, file_digest
from fd_airquality.data import DataConfig, _source_files, prepare, read_manifest
from fd_airquality.execution import execute_plan
from fd_airquality.lightweight import STATION_CONTROLS
from fd_airquality.loaders import load_pollutant_file, normalize_pollutant
from fd_airquality.models import (
    GRANITE,
    MODEL_NAMES,
    model_metadata,
    validate_model_geometry,
    validate_run_mode,
)
from fd_airquality.panels import EXCLUSION_COLUMNS, prepare_runtime_panel, selection_exclusions
from fd_airquality.preprocessing import preprocess
from fd_airquality.quality import evaluation_identity
from fd_airquality.run_logging import RunLogging, report_progress
from fd_airquality.settings import RunSettings
from fd_airquality.temporal_results import (
    select_configuration,
    summarize_cv,
    temporal_run_rows,
)
from fd_airquality.tracking import initialize_tracking, log_leaderboard, tracking_uri

LOGGER = logging.getLogger(__name__)
BACKTEST_SCHEMA_VERSION = 5
DEFAULT_EVALUATION_HOURS = 28 * 24
DEFAULT_MIN_STATIONS = 15
_HOUR = pd.Timedelta(hours=1)
_CUTS = ("validation_start", "test_start", "test_end")


def _require_panel_size(count: int, minimum: int, *, stage: str, detail: str) -> None:
    if count < minimum:
        message = f"Panel común insuficiente: {count} estaciones; mínimo exigido: {minimum}. {detail}"
        LOGGER.critical("stage=%s %s", stage, message)
        raise ValueError(message)


def _hourly_date(value: str, option: str) -> pd.Timestamp:
    try:
        date = pd.Timestamp(value) if isinstance(value, str) and value else pd.NaT
        if not pd.isna(date) and date.tz is None and date == date.floor("h"):
            return date
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError(f"{option} debe ser una fecha horaria sin zona.")


def _block_cuts(start: pd.Timestamp, end: pd.Timestamp, folds: int,
                validation: pd.Timedelta, test: pd.Timedelta) -> list[tuple[pd.Timestamp, ...]]:
    shortest, extra_hours = divmod((end - start) // _HOUR, folds)
    cuts = []
    for number in range(1, folds + 1):
        # Spread the remainder across folds; the last block ends exactly at end.
        test_end = start + (number * shortest + min(number, extra_hours)) * _HOUR
        test_start = test_end - test
        cuts.append((test_start - validation, test_start, test_end))
    return cuts


def _observed_calendar(raw_root: Path, pollutant: str, config: DataConfig,
                        stations: set[str] | None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    files = _source_files(raw_root, pollutant, stations)
    if not files:
        raise FileNotFoundError(f"No hay CSV de {pollutant} bajo {raw_root} para las estaciones elegidas.")
    if stations and (missing := stations - {path.parent.name for path in files}):
        raise ValueError(f"Estaciones sin CSV de {pollutant}: {sorted(missing)}")
    report_progress("calendar", f"Analizando {len(files)} CSV de {pollutant} para detectar horas observadas.")
    bounds = []
    for path in files:
        loaded = load_pollutant_file(str(path), pollutant, grid_tolerance_seconds=config.grid_tolerance_seconds,
                                     upct_valid_flags=config.upct_valid_flags)
        if loaded is None:
            continue
        _, raw = loaded
        (hourly,), _ = preprocess(
            [raw], pollutant, min_run=config.min_run, min_useful=config.min_useful,
            zero_run_hours=config.zero_run_hours, quality_mode=config.quality_mode,
            hourly_repeat_filter=config.hourly_repeat_filter,
        )
        observed = hourly.index[np.isfinite(hourly.iloc[:, 0])]
        if len(observed):
            bounds.append((observed[0], observed[-1] + _HOUR))
            LOGGER.debug("stage=calendar estación=%s; %d horas observadas; primera=%s; última=%s",
                         path.parent.name, len(observed), observed[0], observed[-1])
        else:
            LOGGER.warning("stage=calendar estación=%s sin horas observadas tras el filtrado.", path.parent.name)
    if not bounds:
        message = "No hay horas observadas tras el filtrado para preparar el backtest."
        LOGGER.critical("stage=calendar %s", message)
        raise ValueError(message)
    return bounds


def fold_configurations(config: DataConfig, folds: int, *,
                         bounds: list[tuple[pd.Timestamp, pd.Timestamp]],
                         validation_hours: int = DEFAULT_EVALUATION_HOURS,
                         test_hours: int = DEFAULT_EVALUATION_HOURS,
                         train_history: str = "expanding",
                         history_hours: int = 672) -> tuple[dict[str, str | int], list[DataConfig]]:
    if config.split_mode != "global":
        raise ValueError("La CV temporal requiere cortes globales.")
    if any(type(value) is not int or value < 1 for value in (folds, history_hours)):
        raise ValueError("Número de folds e historial deben ser enteros positivos.")
    if train_history not in {"expanding", "sliding"}:
        raise ValueError(f"train-history={train_history!r} no admitido: usa expanding o sliding; los modelos reinician en cada fold.")
    start = min(lower for lower, _ in bounds)
    end = max(upper for _, upper in bounds)
    total = (end - start) // _HOUR
    min_history = max(config.context + config.horizon + (config.min_train_windows - 1) * config.train_stride,
                      config.seasonality + config.min_scale_pairs)
    min_validation = config.horizon + (config.min_validation_windows - 1) * config.stride
    min_test = config.horizon + (config.min_test_windows - 1) * config.stride
    for name, requested, minimum in (
        ("validation-hours", validation_hours, min_validation),
        ("test-hours", test_hours, min_test),
    ):
        if type(requested) is not int or requested < minimum:
            message = f"{name} debe ser un entero de al menos {minimum} h con estos mínimos de ventanas."
            LOGGER.critical("stage=calendar %s", message)
            raise ValueError(message)
    min_evaluation = min_validation + min_test
    min_block = min_evaluation + 1
    required = min_history + folds * min_block
    max_folds = max(0, (total - min_history) // min_block)
    report_progress(
        "calendar", f"Calendario de {total} h; folds solicitados: {folds}; máximo teórico: {max_folds}. "
        f"Mínimos de calendario: histórico de train {min_history} h; "
        f"validación {min_validation} h, test {min_test} h y train nuevo 1 h por fold.",
    )
    min_target_span = config.horizon if config.window_padding == "none" else config.min_target_observed
    min_sliding = min_target_span + (config.min_train_windows - 1) * config.train_stride
    if train_history == "sliding" and history_hours < min_sliding:
        message = (f"El train deslizante de {history_hours} h no alcanza el mínimo de {min_sliding} h "
                   f"para {config.min_train_windows} ventanas de entrenamiento. Aumenta history-hours.")
        LOGGER.critical("stage=calendar %s", message)
        raise ValueError(message)
    if total < required:
        message = (f"El calendario de {total} h no alcanza para {folds} folds: se necesitan {required} h "
                   f"(train inicial {min_history} h; cada fold: validación {min_validation} h, "
                   f"test {min_test} h y train nuevo 1 h). Máximo teórico: {max_folds} folds "
                   "antes de comprobar la cobertura real. Usa menos folds o más datos.")
        LOGGER.critical("stage=calendar %s", message)
        raise ValueError(message)
    # Give later stations room to train when their span can accommodate the schedule.
    latest_start = max((lower for lower, upper in bounds if (upper - lower) // _HOUR >= required),
                       default=start)
    preferred_reserve = max(total * 2 // 5, (latest_start - start) // _HOUR + min_history)
    reserve = min(preferred_reserve, total - folds * min_block)
    if reserve < preferred_reserve:
        LOGGER.warning("stage=calendar Histórico inicial de train reducido de %d a %d h para que quepan %d folds.",
                       preferred_reserve, reserve, folds)
    history_end = start + reserve * _HOUR
    block = (total - reserve) // folds
    # Leave time for new training in every block, even when evaluation must shrink.
    new_training = min(max(1, block // 4), block - min_evaluation)
    # Align to stride and share the space above the two minima proportionally.
    validation_extra = (validation_hours - min_validation) // config.stride
    test_extra = (test_hours - min_test) // config.stride
    requested_extra = validation_extra + test_extra
    available_extra = (block - new_training - min_evaluation) // config.stride
    if requested_extra > available_extra:
        validation_extra = validation_extra * available_extra // requested_extra
        test_extra = available_extra - validation_extra
    validation = min_validation + validation_extra * config.stride
    test = min_test + test_extra * config.stride
    if requested_extra > available_extra:
        LOGGER.warning(
            "stage=calendar No caben las duraciones solicitadas con %d folds y un bloque mínimo de %d h. "
            "Por fold: validación %d → %d h, test %d → %d h; train nuevo al menos %d h. "
            "Usa menos folds o más datos si necesitas conservar las duraciones.",
            folds, block, validation_hours, validation, test_hours, test, block - validation - test,
        )
    elif (validation, test) != (validation_hours, test_hours):
        LOGGER.debug("stage=calendar Ajuste al stride de %d h: validación %d → %d h, test %d → %d h.",
                     config.stride, validation_hours, validation, test_hours, test)
    cuts = _block_cuts(history_end, end, folds, validation * _HOUR, test * _HOUR)
    result = []
    for fold in cuts:
        dates = dict(zip(_CUTS, map(str, fold), strict=True))
        train_start = str(fold[0] - history_hours * _HOUR) if train_history == "sliding" else ""
        result.append(replace(config, **dates, train_start=train_start,
                              validation_hours=validation, test_hours=test))
    return {
        "calendar_start": str(start), "calendar_end": str(end), "initial_history_end": str(history_end),
        "validation_hours": validation, "test_hours": test,
    }, result


def prepare_backtest(raw_root: Path, output: Path, pollutant: str, config: DataConfig,
                     stations: set[str] | None = None, *, folds: int = 3,
                     validation_hours: int = DEFAULT_EVALUATION_HOURS,
                     test_hours: int = DEFAULT_EVALUATION_HOURS,
                     min_stations: int = DEFAULT_MIN_STATIONS,
                     train_history: str = "expanding", history_hours: int = 672) -> Path:
    pollutant = normalize_pollutant(pollutant)
    if config.split_mode != "global":
        raise ValueError("La CV temporal requiere cortes globales.")
    if type(min_stations) is not int or min_stations < 1:
        raise ValueError("min-stations debe ser un entero positivo.")
    minimum = min_stations if stations is None else 1
    selection = "all" if stations is None else "subset"
    report_progress("panel", f"Mínimo de estaciones comunes: {minimum} "
                    f"({'todas las estaciones' if stations is None else 'selección explícita de estaciones'}).")
    bounds = _observed_calendar(raw_root, pollutant, config, stations)
    _require_panel_size(
        len(bounds), minimum, stage="calendar",
        detail="Estaciones con datos observados antes de preparar los folds. "
               "Revisa los CSV y min-stations, o usa --station para una prueba con menos estaciones.",
    )
    calendar, geometries = fold_configurations(
        config, folds, bounds=bounds, validation_hours=validation_hours, test_hours=test_hours,
        train_history=train_history, history_hours=history_hours,
    )
    report_progress("calendar", f"{calendar['calendar_start']} → {calendar['calendar_end']}; "
                    f"histórico inicial hasta {calendar['initial_history_end']}; "
                    f"validación {calendar['validation_hours']}/{validation_hours} h y "
                    f"test {calendar['test_hours']}/{test_hours} h por fold (efectivo/solicitado)")
    root = create_run_dir(output.resolve(), f"{pollutant}_{datetime.now(UTC):%Y%m%d_%H%M%S}")
    entries = []
    shared_stations: set[str] | None = None
    calendar_start = pd.Timestamp(calendar["calendar_start"])
    for index, geometry in enumerate(geometries):
        label = f"fold={index + 1}/{folds}"
        train_start = max(calendar_start, pd.Timestamp(geometry.train_start)) if geometry.train_start else calendar_start
        train_hours = (pd.Timestamp(geometry.validation_start) - train_start) // _HOUR
        report_progress(
            "prepare-fold", f"{label} train [{train_start}, {geometry.validation_start}) ({train_hours} h); "
            f"validación [{geometry.validation_start}, {geometry.test_start}) ({geometry.validation_hours} h); "
            f"test [{geometry.test_start}, {geometry.test_end}) ({geometry.test_hours} h)",
        )
        try:
            path = prepare(raw_root, root / f"fold_{index + 1:03d}", pollutant, geometry, stations)
        except (ValueError, FileNotFoundError) as exc:
            LOGGER.critical("stage=prepare-fold %s No se pudo preparar el fold: %s", label, exc)
            raise
        clients = read_manifest(path)["clients"]
        prepared = {client["station"] for client in clients}
        shared_stations = prepared if shared_stations is None else shared_stations & prepared
        support = ", ".join(f"{name}={min(client[f'num_{name}_windows'] for client in clients)}"
                            for name in ("train", "validation", "test"))
        report_progress("prepare-fold", f"{label} estaciones elegibles={len(clients)}; "
                        f"mínimo de ventanas por estación: {support}; cobertura={path.parent / 'coverage.csv'}")
        entries.append({"fold": index + 1, "manifest": str(path), "sha256": file_digest(path),
                        "validation_start": geometry.validation_start, "test_start": geometry.test_start,
                        "test_end": geometry.test_end, "train_start": geometry.train_start})
        _require_panel_size(
            len(shared_stations), minimum, stage="prepare-fold",
            detail=f"{label}: estaciones que cumplen los mínimos de ventanas en los folds preparados. "
                   f"Revisa los informes de cobertura y exclusiones en {root} o usa menos folds.",
        )
    report_progress("prepare-fold", f"Panel preparado: {len(shared_stations)} estaciones con ventanas en todos los folds.")
    path = root / "backtest_manifest.json"
    path.write_text(json.dumps({"schema_version": BACKTEST_SCHEMA_VERSION, "pollutant": pollutant,
                                "protocol": "temporal_cv", "fold_initialization": "fresh",
                                "train_history": train_history,
                                "min_stations": minimum, "station_selection": selection,
                                "requested_validation_hours": validation_hours, "requested_test_hours": test_hours,
                                **calendar, "folds": entries,
                                "final_holdout_fold": None, "potential_holdout_fold": folds,
                                "note": "Simulación retrospectiva. El último fold solo se reserva al solicitar development/final."},
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _validate_calendar(family: dict) -> None:
    start = _hourly_date(family.get("calendar_start", ""), "calendar_start del manifest")
    history_end = _hourly_date(family.get("initial_history_end", ""), "initial_history_end del manifest")
    end = _hourly_date(family.get("calendar_end", ""), "calendar_end del manifest")
    cuts = [tuple(_hourly_date(entry.get(key, ""), key) for key in _CUTS) for entry in family["folds"]]
    validation_start, test_start, test_end = cuts[-1]
    validation, test = test_start - validation_start, test_end - test_start
    if not (start < history_end < end and validation_start < test_start < test_end == end
            and (end - history_end) // len(cuts) >= validation + test + _HOUR):
        raise ValueError("Los cortes no coinciden con el reparto del calendario en el manifest.")
    if cuts != _block_cuts(history_end, end, len(cuts), validation, test):
        raise ValueError("Los cortes no coinciden con el reparto del calendario en el manifest.")
    for name, duration in (("validation", validation), ("test", test)):
        hours = duration // _HOUR
        if f"{name}_hours" in family and family[f"{name}_hours"] != hours:
            raise ValueError("Las duraciones no coinciden con los cortes del manifest.")


def _effective_folds(family: dict, settings: RunSettings, root: Path, *, panel_folds: int,
                     scored_indices: set[int],
                     stations: list[str] | None = None) -> list[Path]:
    manifests, exclusions = [], []
    for entry in family["folds"]:
        path = Path(entry["manifest"])
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Un manifest temporal cambió después de preparar los folds.")
        manifest = read_manifest(path)
        if (manifest["geometry"]["split_mode"] != "global"
                or any(pd.Timestamp(manifest["geometry"][name]) != pd.Timestamp(entry[name])
                       for name in _CUTS)):
            raise ValueError("El calendario del fold no coincide con su manifest.")
        rejected = {row["station"]: row for row in selection_exclusions(manifest, settings.selection_metric)}
        if rejected and (entry["fold"] <= panel_folds or entry["fold"] - 1 in scored_indices):
            LOGGER.warning("stage=panel fold=%d: %d estaciones sin %s de validación definida.",
                           entry["fold"], len(rejected), settings.selection_metric)
        manifests.append(manifest)
        exclusions.append(rejected)
    common = (set(stations) if stations is not None else
              set.intersection(*({client["station"] for client in manifest["clients"]} - rejected.keys()
                                for manifest, rejected in list(zip(manifests, exclusions, strict=True))[:panel_folds])))
    audit = []
    for index, (manifest, rejected) in enumerate(zip(manifests, exclusions, strict=True), start=1):
        for client in manifest["clients"]:
            station = client["station"]
            if station not in common or (index - 1 in scored_indices and station in rejected):
                audit.append({"fold": index, **rejected.get(station, {
                    "station": station, "reason": "outside_common_temporal_panel",
                    "source_partition_id": client.get("source_partition_id", client["partition_id"]),
                    "selection_metric": settings.selection_metric,
                })})
    audit_path = root / "temporal_exclusions.csv"
    pd.DataFrame(audit, columns=["fold", *EXCLUSION_COLUMNS]).to_csv(audit_path, index=False)
    minimum = family["min_stations"]
    _require_panel_size(len(common), minimum, stage="panel", detail=f"Consulta {audit_path}")
    report_progress("panel", f"Estaciones en el panel común: {len(common)} (mínimo {minimum}); informe={audit_path}")
    paths = []
    for index, (manifest, rejected) in enumerate(zip(manifests, exclusions, strict=True)):
        if not common <= {client["station"] for client in manifest["clients"]}:
            message = "Faltan estaciones del panel de desarrollo en un fold; no se cambia el panel mirando el holdout."
            LOGGER.critical("stage=panel fold=%d %s", index + 1, message)
            raise ValueError(message)
        if index in scored_indices and index >= panel_folds:
            _require_panel_size(
                len(common - rejected.keys()), minimum, stage="panel",
                detail=f"Fold {index + 1}, tras comprobar la métrica de validación. Consulta {audit_path}",
            )
        retained = []
        for client in manifest["clients"]:
            if client["station"] in common:
                retained.append(client | {"source_partition_id": client["partition_id"], "partition_id": len(retained)})
        path = root / f"fold_{index + 1:03d}_manifest.json"
        effective = manifest | {"clients": retained, "temporal_panel": {
            "fold": index + 1, "selection_metric": settings.selection_metric,
             "note": ("Panel fijado con todos los folds de evaluación, sin usar sus errores."
                      if panel_folds == len(manifests) else
                      "Panel fijado con los folds de desarrollo; el holdout no decide estaciones ni se usan sus errores.")}}
        effective["evaluation_sha256"] = evaluation_identity(effective)
        path.write_text(json.dumps(effective, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def build_candidate_matrix(models: list[str], modes: list[str], adaptations: list[str],
                           strategies: tuple | list = ("fedavg",)) -> list[dict]:
    for values in (models, modes, adaptations, strategies):
        if not values or len(set(values)) != len(values):
            raise ValueError("La matriz no admite listas vacías ni valores repetidos.")
    if (set(models) - set(MODEL_NAMES) or set(modes) - {"federated", "local", "centralized", "zero-shot"}
            or set(adaptations) - {"lora", "ffa-lora", "flora", "lora-fair"}
            or set(strategies) - {"fedavg", "fedprox"}):
        raise ValueError("Combinación desconocida en la matriz.")
    matrix = []
    controls = (["lora"] if set(adaptations) - {"ffa-lora"} else []) + (["ffa-lora"] if "ffa-lora" in adaptations else [])
    for mode in modes:
        for model in models:
            if model in STATION_CONTROLS:
                if mode == "local":
                    matrix.append({"model": model, "mode": mode, "adaptation": "none", "strategy": "fedavg"})
                continue
            foundation = model in GRANITE
            if mode == "zero-shot":
                if foundation:
                    matrix.append({"model": model, "mode": mode, "adaptation": "none", "strategy": "fedavg"})
                continue
            algorithms = adaptations if foundation and mode == "federated" else controls if foundation else ["auto"]
            for algorithm in algorithms:
                if algorithm in {"flora", "lora-fair"} and "fedavg" not in strategies:
                    raise ValueError("FLoRA/LoRA-FAIR requieren su agregación sin FedProx; incluye fedavg en --strategies.")
                choices = ["fedavg"] if mode != "federated" or algorithm in {"flora", "lora-fair"} else strategies
                for strategy in choices:
                    matrix.append({"model": model, "mode": mode, "adaptation": algorithm, "strategy": strategy})
    return matrix


def _candidate_plan(settings: RunSettings, models, modes, adaptations, seeds, lora_ranks, strategies) -> list[dict]:
    matrix = build_candidate_matrix(models, modes, adaptations or ["lora", "ffa-lora", "flora", "lora-fair"],
                                    strategies or [settings.strategy])
    seeds, ranks = seeds or [settings.seed], lora_ranks or [settings.lora_rank]
    if len(set(seeds)) != len(seeds) or len(set(ranks)) != len(ranks):
        raise ValueError("No repitas semillas ni rangos.")
    plan = []
    for case in matrix:
        once = case["mode"] == "zero-shot" or case["model"] in STATION_CONTROLS
        for seed in seeds[:1] if once else seeds:
            for rank in ranks if case["model"] in GRANITE and case["mode"] != "zero-shot" else ranks[:1]:
                plan.append(case | {"seed": seed, "lora_rank": rank})
    if not plan:
        raise ValueError("La matriz del backtest no contiene candidatos válidos.")
    return plan


def _run_fold(settings: RunSettings, plan: list[dict], max_parallel_models: int | None) -> Path:
    from fd_airquality.baselines import run_baseline
    from fd_airquality.cli import launch_simulation
    from fd_airquality.comparison import compare_runs

    root = Path(settings.output_dir)
    with RunLogging(root, f"backtest {root.name}", filename="fold.log", level=settings.log_level,
                    heartbeat_seconds=settings.heartbeat_seconds) as progress_log:
        settings, manifest = prepare_runtime_panel(settings, root)
        for case in plan:
            current = replace(settings, **{key: value for key, value in case.items() if key != "mode"})
            validate_model_geometry(manifest["geometry"], current)
            validate_run_mode(current, case["mode"])
        checked = set()
        for case in plan:
            key = (case["model"], case["lora_rank"])
            if case["model"] in GRANITE and case["mode"] != "zero-shot" and key not in checked:
                report_progress("preflight", f"loading model={case['model']} rank={case['lora_rank']}")
                current = replace(settings, **{key: value for key, value in case.items() if key != "mode"})
                model_metadata(manifest["geometry"], current)
                checked.add(key)
        report_progress("tracking", "initializing MLflow before concurrent candidates")
        initialize_tracking(settings, root.parent / "mlflow-artifacts")
        paths = execute_plan(settings, plan, root,
                             max_parallel_models=max_parallel_models, baseline=run_baseline,
                             launch=launch_simulation, progress_log=progress_log)
        report_progress("comparison", f"completed_candidates={len(paths)}")
        return compare_runs(paths, root / "comparison", allow_single=True)


def run_backtest(settings: RunSettings, models: list[str], modes: list[str], *, adaptations=None,
                  seeds=None, lora_ranks=None, strategies=None, max_parallel_models=None,
                  stage: str = "evaluation", selection: Path | None = None, weekly_leaderboard: bool = False) -> Path:
    family = json.loads(Path(settings.manifest).read_text(encoding="utf-8"))
    holdout_fold = family.get("potential_holdout_fold", family.get("final_holdout_fold"))
    normalize_pollutant(family.get("pollutant", ""))
    minimum = family.get("min_stations")
    if (family.get("schema_version") != BACKTEST_SCHEMA_VERSION or not family.get("folds")
            or type(minimum) is not int or minimum < 1
            or family.get("protocol") != "temporal_cv"
            or family.get("fold_initialization") != "fresh"
            or family.get("train_history") not in {"expanding", "sliding"}
            or [entry.get("fold") for entry in family["folds"]] != list(range(1, len(family["folds"]) + 1))
            or holdout_fold != len(family["folds"])):
        raise ValueError(
            "Se requiere una preparación con reparto de calendario, reinicio por fold y mínimo de estaciones; "
            "vuelve a ejecutar prepare-backtest."
        )
    _validate_calendar(family)
    if stage not in {"evaluation", "development", "final", "all"}:
        raise ValueError("stage debe ser evaluation, development, final o all (exploratorio).")
    if stage in {"development", "final"} and len(family["folds"]) < 2:
        raise ValueError("La separación desarrollo/holdout requiere al menos dos folds.")
    recipe = None
    source_hash = file_digest(Path(settings.manifest))
    if stage == "final":
        if selection is None:
            raise ValueError("El holdout requiere --selection con una selección de CV congelada.")
        recipe = json.loads(Path(selection).read_text())
        if (recipe.get("family_sha256") != source_hash or recipe.get("schema_version") != 2
                or recipe.get("fold_initialization") != "fresh"):
            raise ValueError("La selección no pertenece a esta familia temporal.")
        package = Path(__file__).parent
        sources = {path.name: file_digest(path) for path in sorted(package.glob("*.py"))}
        if recipe["application_sources"] != sources:
            raise ValueError("Cambió el código desde la selección; repite desarrollo antes del holdout.")
        summary_path = Path(recipe["cv_summary"])
        if file_digest(summary_path) != recipe["cv_summary_sha256"]:
            raise ValueError("Cambió el resumen de CV desde que se congeló la selección.")
        if select_configuration(pd.read_csv(summary_path), recipe["selection_metric"]) != recipe["temporal_configuration"]:
            raise ValueError("La configuración no coincide con la selección por CV.")
        for record in recipe["run_records"]:
            if file_digest(Path(record["run_json"])) != record["sha256"]:
                raise ValueError("Cambió una ejecución de desarrollo desde la selección.")
        original = json.loads(Path(recipe["run_records"][0]["run_json"]).read_text())
        if recipe["settings"] != original["settings"]:
            raise ValueError("La configuración congelada no coincide con el desarrollo.")
        if recipe["mode"] != original["mode"]:
            raise ValueError("El modo congelado no coincide con el desarrollo.")
        settings = RunSettings(**(recipe["settings"] | {
            "manifest": settings.manifest, "output_dir": settings.output_dir,
        }))
        models, modes = [settings.model], [recipe["mode"]]
        adaptations = [settings.adaptation] if settings.adaptation in {"lora", "ffa-lora", "flora", "lora-fair"} else None
        seeds, lora_ranks, strategies = recipe["seeds"], [settings.lora_rank], [settings.strategy]
    elif selection is not None:
        raise ValueError("--selection solo se utiliza con --stage final.")
    if max_parallel_models is not None and (type(max_parallel_models) is not int or max_parallel_models < 1):
        raise ValueError("max-parallel-models debe ser un entero positivo.")
    plan = _candidate_plan(settings, models, modes, adaptations, seeds, lora_ranks, strategies)
    settings = replace(settings, mlflow_tracking_uri=tracking_uri(settings))
    root = create_run_dir(Path(settings.output_dir).resolve(), f"backtest_{datetime.now(UTC):%Y%m%d_%H%M%S}")
    with RunLogging(root, f"backtest {stage}", filename="backtest.log", level=settings.log_level,
                    heartbeat_seconds=settings.heartbeat_seconds) as log:
        report_progress("backtest", f"manifest={settings.manifest}; "
                        f"{len(family['folds'])} folds preparados; {len(plan)} candidatos por fold.")
        return _run_prepared_backtest(
            settings, family, root, plan, stage=stage, recipe=recipe, source_hash=source_hash,
            holdout_fold=holdout_fold, max_parallel_models=max_parallel_models, log=log,
            weekly_leaderboard=weekly_leaderboard,
        )


def _run_prepared_backtest(settings: RunSettings, family: dict, root: Path, plan: list[dict], *,
                            stage: str, recipe: dict | None, source_hash: str, holdout_fold: int,
                            max_parallel_models: int | None, log: RunLogging,
                            weekly_leaderboard: bool = False) -> Path:
    num_folds = len(family["folds"])
    indices = ([num_folds - 1] if stage == "final" else
               list(range(num_folds if stage in {"evaluation", "all"} else num_folds - 1)))
    manifests = _effective_folds(
        family, settings, root, panel_folds=num_folds if stage == "evaluation" else max(1, num_folds - 1),
        scored_indices=set(indices), stations=recipe["stations"] if recipe else None,
    )
    progress = family | {"source_sha256": source_hash, "results": [], "stage": stage,
                          "final_holdout_scored": stage in {"final", "all"},
                          "final_holdout_fold": None if stage == "evaluation" else holdout_fold,
                          "configuration_selection": stage in {"development", "all"},
                          "primary_summary": "cv_fold_mean.csv" if stage == "evaluation" else "cv_summary.csv",
                          "exploratory": stage == "all"}
    rows, summaries, weekly = [], [], []
    cv_rows, cv_stations = [], []
    # Record access before running: a failed partial holdout is not untouched.
    if stage in {"evaluation", "final", "all"}:
        ledger = Path(settings.manifest).parent / "holdout_access.json"
        if stage == "final" and ledger.exists():
            raise ValueError("El holdout ya fue consultado por este protocolo; no puede presentarse como sellado.")
        if not ledger.exists():
            with ledger.open("x") as stream:
                json.dump({"stage": stage, "output": str(root),
                           "note": "Control de acceso del software, no prueba de desconocimiento humano."}, stream, indent=2)
    log.update(completed=0, total=len(indices))
    for completed, index in enumerate(indices, start=1):
        manifest = manifests[index]
        entry = {"fold": index + 1, "status": "running"}
        progress["results"].append(entry)
        _write_progress(root, progress)
        try:
            report_progress("fold", f"Iniciando fold {index + 1} ({completed}/{len(indices)}).")
            current = replace(settings, manifest=str(manifest), output_dir=str(root / f"fold_{index + 1:03d}"))
            comparison = _run_fold(current, plan, max_parallel_models)
            metadata = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
            run_paths = [Path(path) for path in metadata["runs"]]
            frame = pd.read_csv(comparison / "comparison.csv")
            frame.insert(0, "fold", index + 1)
            frame["test_start"], frame["test_end"] = family["folds"][index]["test_start"], family["folds"][index]["test_end"]
            frame["final_holdout"] = stage == "final"
            frame["exploratory"] = stage == "all"
            rows.append(frame)
            summary = pd.read_csv(comparison / "seed_summary.csv")
            summary.insert(0, "fold", index + 1)
            summaries.append(summary)
            if (comparison / "weekly_metrics.csv").is_file():
                weeks = pd.read_csv(comparison / "weekly_metrics.csv")
                weeks.insert(0, "fold", index + 1)
                weekly.append(weeks)
            entry.update(status="completed", comparison=str(comparison))
            if stage == "evaluation" or index + 1 != holdout_fold:
                run_rows, station_rows = temporal_run_rows(run_paths, index + 1)
                cv_rows.extend(run_rows)
                cv_stations.extend(station_rows)
        except BaseException:
            entry["status"] = "failed"
            _write_progress(root, progress)
            raise
        _write_progress(root, progress)
        log.update(completed=completed)
        report_progress("fold", f"Fold {index + 1} completado ({completed}/{len(indices)}).")
    report_progress("summary", f"Consolidando resultados de {len(indices)} folds.")
    pd.concat(rows, ignore_index=True).to_csv(root / "temporal_comparison.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(root / "temporal_seed_summary.csv", index=False)
    if weekly:
        # Never publish a historical or partially populated collection as a complete race.
        if len(weekly) != len(indices):
            LOGGER.warning("No todos los folds publicaron estadísticas semanales; no se mezclan.")
        else:
            pd.concat(weekly, ignore_index=True).to_csv(root / "weekly_metrics.csv", index=False)
    if cv_rows:
        summary = summarize_cv(cv_rows, cv_stations, root)
    if cv_rows and stage != "evaluation":
        winner = select_configuration(summary, settings.selection_metric)
        last_fold = max(row["fold"] for row in cv_rows)
        selected_runs = [Path(row["run_path"]) for row in cv_rows
                         if row["temporal_configuration"] == winner and row["fold"] == last_fold]
        selected_run = json.loads((selected_runs[0] / "run.json").read_text())
        selected_settings = selected_run["settings"]
        (root / "selection.json").write_text(json.dumps({
            "schema_version": 2, "family_sha256": source_hash, "fold_initialization": "fresh",
            "temporal_configuration": winner, "selection_metric": settings.selection_metric,
            "settings": selected_settings, "mode": selected_run["mode"],
            "seeds": [json.loads((path / "run.json").read_text())["settings"]["seed"] for path in selected_runs],
            "stations": [client["station"] for client in selected_run["manifest"]["clients"]],
            "run_records": [{
                "run_json": str(path / "run.json"), "sha256": file_digest(path / "run.json"),
            } for path in selected_runs],
            "cv_summary": str(root / "cv_summary.csv"),
            "cv_summary_sha256": file_digest(root / "cv_summary.csv"),
            "application_sources": selected_run["application_sources"],
            "aggregation": "pool_observed_origin_target_errors_within_station_then_macro; mean_over_seeds",
            "holdout_was_scored": stage == "all",
        }, indent=2, allow_nan=False) + "\n")
    if weekly_leaderboard and (root / "weekly_metrics.csv").is_file():
        from fd_airquality.leaderboard import plot_weekly

        report_progress("leaderboard", "Generando gráficas semanales sin reentrenar.")
        output, artifacts = plot_weekly(root)
        progress["weekly_leaderboard"] = str(output)
        run_id = log_leaderboard(settings, root, family["pollutant"], stage, artifacts)
        if run_id:
            progress["weekly_leaderboard_mlflow_run_id"] = run_id
        _write_progress(root, progress)
    return root


def _write_progress(root: Path, progress: dict) -> None:
    temporary = root / "backtest.tmp"
    temporary.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(root / "backtest.json")
