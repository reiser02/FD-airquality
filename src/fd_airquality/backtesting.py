"""Temporal cross-validation: fresh models per fold, expanding or sliding data."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fd_airquality.artifacts import create_run_dir, file_digest
from fd_airquality.data import DataConfig, prepare, read_manifest
from fd_airquality.loaders import normalize_pollutant
from fd_airquality.panels import selection_exclusions
from fd_airquality.quality import evaluation_identity
from fd_airquality.temporal_results import (
    select_configuration,
    summarize_cv,
    temporal_run_rows,
)


def fold_configurations(config: DataConfig, folds: int, step_hours: int, *,
                         train_history: str = "expanding",
                         history_hours: int = 672) -> list[DataConfig]:
    if config.split_mode != "global":
        raise ValueError("La CV temporal requiere cortes globales.")
    if any(type(value) is not int or value < 1 for value in (folds, step_hours, history_hours)):
        raise ValueError("Número de folds, paso e historial deben ser enteros positivos.")
    if train_history not in {"expanding", "sliding"}:
        raise ValueError(f"train-history={train_history!r} no admitido: usa expanding o sliding; los modelos reinician en cada fold.")
    test_hours = int((pd.Timestamp(config.test_end) - pd.Timestamp(config.test_start)).total_seconds() / 3600)
    if step_hours < test_hours:
        raise ValueError("El paso debe evitar tests solapados.")
    shift = pd.Timedelta(hours=step_hours)
    result = []
    for index in range(folds):
        delta = (folds - 1 - index) * shift
        dates = {name: str(pd.Timestamp(getattr(config, name)) - delta)
                 for name in ("validation_start", "test_start", "test_end")}
        cutoff = pd.Timestamp(dates["validation_start"])
        train_start = config.train_start
        if train_history == "sliding":
            train_start = str(cutoff - pd.Timedelta(hours=history_hours))
        result.append(replace(config, **dates, train_start=train_start))
    return result


def prepare_backtest(raw_root: Path, output: Path, pollutant: str, config: DataConfig,
                     stations: set[str] | None = None, *, folds: int = 3, step_hours: int = 672,
                     train_history: str = "expanding",
                      history_hours: int = 672) -> Path:
    pollutant = normalize_pollutant(pollutant)
    geometries = fold_configurations(config, folds, step_hours,
                                     train_history=train_history, history_hours=history_hours)
    root = create_run_dir(output.resolve(), f"{pollutant}_{datetime.now(UTC):%Y%m%d_%H%M%S}")
    entries = []
    for index, geometry in enumerate(geometries):
        path = prepare(raw_root, root / f"fold_{index + 1:03d}", pollutant, geometry, stations)
        entries.append({"fold": index + 1, "manifest": str(path), "sha256": file_digest(path),
                        "validation_start": geometry.validation_start, "test_start": geometry.test_start,
                        "test_end": geometry.test_end, "train_start": geometry.train_start})
    path = root / "backtest_manifest.json"
    path.write_text(json.dumps({"schema_version": 2, "pollutant": pollutant,
                                "protocol": "temporal_cv", "fold_initialization": "fresh",
                                "train_history": train_history,
                                "step_hours": step_hours, "folds": entries,
                                "final_holdout_fold": folds,
                                "note": "Simulación retrospectiva; no certifica predicciones emitidas en tiempo real."},
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _effective_folds(family: dict, settings, root: Path, *, panel_folds: int | None = None,
                     stations: list[str] | None = None) -> list[Path]:
    manifests, exclusions = [], []
    for entry in family["folds"]:
        path = Path(entry["manifest"])
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Un manifest temporal cambió después de preparar los folds.")
        manifest = read_manifest(path)
        if (manifest["geometry"]["split_mode"] != "global"
                or any(str(pd.Timestamp(manifest["geometry"][name])) != str(pd.Timestamp(entry[name]))
                       for name in ("validation_start", "test_start", "test_end"))):
            raise ValueError("El calendario del fold no coincide con su manifest.")
        if manifests:
            previous = manifests[-1]["geometry"]
            current = manifest["geometry"]
            if (pd.Timestamp(current["test_start"]) < pd.Timestamp(previous["test_end"])
                    or pd.Timestamp(current["validation_start"]) <= pd.Timestamp(previous["validation_start"])):
                raise ValueError("Folds desordenados o tests solapados.")
        rejected = {row["station"] for row in selection_exclusions(manifest, settings.selection_metric)}
        manifests.append(manifest)
        exclusions.append(rejected)
    common = (set(stations) if stations is not None else
              set.intersection(*({client["station"] for client in manifest["clients"]} - rejected
                                for manifest, rejected in list(zip(manifests, exclusions, strict=True))[:panel_folds])))
    if not common:
        raise ValueError("No hay un panel común elegible en los folds; revisa cobertura, cortes y escalas.")
    paths, audit = [], []
    for index, manifest in enumerate(manifests):
        if not common <= {client["station"] for client in manifest["clients"]}:
            raise ValueError("Faltan estaciones del panel de desarrollo en un fold; no se cambia el panel mirando el holdout.")
        retained = []
        for client in manifest["clients"]:
            if client["station"] in common:
                retained.append(client | {"source_partition_id": client["partition_id"], "partition_id": len(retained)})
            else:
                audit.append({"fold": index + 1, "station": client["station"], "reason": "outside_common_temporal_panel"})
        path = root / f"fold_{index + 1:03d}_manifest.json"
        effective = manifest | {"clients": retained, "temporal_panel": {
            "fold": index + 1, "selection_metric": settings.selection_metric,
            "note": "Panel fijado con los folds de desarrollo; el holdout no decide estaciones ni se usan sus errores."}}
        effective["evaluation_sha256"] = evaluation_identity(effective)
        path.write_text(json.dumps(effective, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        paths.append(path)
    pd.DataFrame(audit, columns=["fold", "station", "reason"]).to_csv(root / "temporal_exclusions.csv", index=False)
    return paths


def run_backtest(settings, models: list[str], modes: list[str], *, adaptations=None, seeds=None,
                  lora_ranks=None, strategies=None, max_parallel_models=None, benchmark=None,
                  stage: str = "development", selection: Path | None = None) -> Path:
    if benchmark is None:
        from fd_airquality.cli import run_benchmark

        benchmark = run_benchmark
    family = json.loads(Path(settings.manifest).read_text(encoding="utf-8"))
    normalize_pollutant(family.get("pollutant", ""))
    if (family.get("schema_version") != 2 or not family.get("folds")
            or family.get("protocol") != "temporal_cv"
            or family.get("fold_initialization") != "fresh"
            or family.get("train_history") not in {"expanding", "sliding"}
            or [entry.get("fold") for entry in family["folds"]] != list(range(1, len(family["folds"]) + 1))
            or family.get("final_holdout_fold") != len(family["folds"])):
        raise ValueError("Se requiere una preparación de CV con reinicio por fold; vuelve a ejecutar prepare-backtest.")
    if stage not in {"development", "final", "all"}:
        raise ValueError("stage debe ser development, final o all (exploratorio).")
    if stage != "all" and len(family["folds"]) < 2:
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
        from fd_airquality.settings import RunSettings

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
    root = create_run_dir(Path(settings.output_dir).resolve(), f"backtest_{datetime.now(UTC):%Y%m%d_%H%M%S}")
    manifests = _effective_folds(
        family, settings, root, panel_folds=max(1, len(family["folds"]) - 1),
        stations=recipe["stations"] if recipe else None,
    )
    progress = family | {"source_sha256": source_hash, "results": [], "stage": stage,
                         "final_holdout_scored": stage in {"final", "all"},
                         "exploratory": stage == "all"}
    rows, summaries = [], []
    cv_rows, cv_stations = [], []
    indices = ([len(manifests) - 1] if stage == "final" else
               list(range(len(manifests) if stage == "all" else len(manifests) - 1)))
    # Record access before running: a failed partial holdout is not untouched.
    if stage in {"final", "all"}:
        ledger = Path(settings.manifest).parent / "holdout_access.json"
        if stage == "final" and ledger.exists():
            raise ValueError("El holdout ya fue consultado por este protocolo; no puede presentarse como sellado.")
        if not ledger.exists():
            with ledger.open("x") as stream:
                json.dump({"stage": stage, "output": str(root),
                           "note": "Control de acceso del software, no prueba de desconocimiento humano."}, stream, indent=2)
    for index in indices:
        manifest = manifests[index]
        entry = {"fold": index + 1, "status": "running"}
        progress["results"].append(entry)
        _write_progress(root, progress)
        try:
            current = replace(settings, manifest=str(manifest), output_dir=str(root / f"fold_{index + 1:03d}"))
            comparison = benchmark(current, models, modes, adaptations, seeds, lora_ranks, strategies,
                                   max_parallel_models, allow_single=True)
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
            entry.update(status="completed", comparison=str(comparison))
            if index + 1 != family["final_holdout_fold"]:
                run_rows, station_rows = temporal_run_rows(run_paths, index + 1)
                cv_rows.extend(run_rows)
                cv_stations.extend(station_rows)
        except BaseException:
            entry["status"] = "failed"
            _write_progress(root, progress)
            raise
        _write_progress(root, progress)
    pd.concat(rows, ignore_index=True).to_csv(root / "temporal_comparison.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(root / "temporal_seed_summary.csv", index=False)
    if cv_rows:
        summary = summarize_cv(cv_rows, cv_stations, root)
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
    return root


def _write_progress(root: Path, progress: dict) -> None:
    temporary = root / "backtest.tmp"
    temporary.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(root / "backtest.json")
