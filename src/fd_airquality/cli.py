"""uv-friendly preparation, auditing, and official Flower simulation launcher."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tomllib
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path

from fd_airquality.backtesting import DEFAULT_EVALUATION_HOURS, DEFAULT_MIN_STATIONS
from fd_airquality.configuration import INTERNAL_FOLD_FIELDS, parse_configured_args
from fd_airquality.data import DataConfig, read_manifest
from fd_airquality.execution import usable_gpu_slots
from fd_airquality.lightweight import STATION_CONTROLS
from fd_airquality.loaders import POLLUTANT
from fd_airquality.models import (
    GRANITE,
    MODEL_NAMES,
    TINY_AUTOENCODER_LATENT_SIZE,
    validate_model_geometry,
    validate_run_mode,
)
from fd_airquality.panels import prepare_runtime_panel
from fd_airquality.processes import ProcessRunner
from fd_airquality.run_logging import RunLogging, report_progress, run_label
from fd_airquality.settings import RunSettings, resolve_manifest
from fd_airquality.task import validate_selection_coverage
from fd_airquality.tracking import tracking_uri

DEFAULT_MATRIX_MODELS = tuple(name for name in MODEL_NAMES if name != "patchtst-fm-r2")


def _parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                        help="Archivo TOML; por defecto fd-airquality.toml en el directorio actual.")
    commands = parser.add_subparsers(dest="command", required=True)
    temporal_data = commands.add_parser("prepare-backtest", help="Repartir el calendario observado en folds globales.")
    temporal_data.add_argument("--raw-root", type=Path, default=Path("data/raw/datos_estaciones_5m"))
    temporal_data.add_argument("--output", type=Path, default=Path("data/backtests"))
    temporal_data.add_argument("--pollutant", choices=[POLLUTANT, POLLUTANT.lower()], default=POLLUTANT,
                               help="Único contaminante admitido: PM25 (PM2.5).")
    temporal_data.add_argument("--station", action="append", help="Nombre exacto del directorio; se puede repetir.")
    defaults = DataConfig()
    for field in fields(DataConfig):
        if field.name in INTERNAL_FOLD_FIELDS:
            continue
        flag = f"--{field.name.replace('_', '-')}"
        value = getattr(defaults, field.name)
        options = {}
        if field.name == "window_padding":
            options = {"choices": ["none", "right", "both"],
                       "help": "Padding solo de entrenamiento: none lo desactiva; right a la derecha; both también a la izquierda."}
        elif field.name == "max_train_windows":
            options = {"help": "Presupuesto de ventanas distintas por estación y época; se sortea del conjunto admisible."}
        temporal_data.add_argument(flag, type=type(value) if value is not None else int, **options)
    temporal_data.add_argument("--heartbeat-seconds", type=float, default=60.0)
    temporal_data.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    temporal_data.add_argument("--folds", type=int, default=3,
                               help="Número de cortes temporales; usa 1 para evaluar un solo corte.")
    temporal_data.add_argument("--min-stations", type=int, default=DEFAULT_MIN_STATIONS,
                                help="Mínimo de estaciones comunes al usar todas (15); con selección explícita basta una.")
    temporal_data.add_argument("--validation-hours", type=int, default=DEFAULT_EVALUATION_HOURS,
                               help="Horas objetivo de validación por fold (672 por defecto).")
    temporal_data.add_argument("--test-hours", type=int, default=DEFAULT_EVALUATION_HOURS,
                               help="Horas objetivo de test por fold (672 por defecto).")
    temporal_data.add_argument("--train-history", choices=["expanding", "sliding"], default="expanding")
    temporal_data.add_argument("--history-hours", type=int, default=672)
    audit = commands.add_parser("inspect", help="Resumen del soporte y las exclusiones, sin entrenar.")
    audit.add_argument("--manifest")
    simulation = commands.add_parser("simulate", help="Ejecutar flwr run en el simulador local oficial.")
    simulation.add_argument("--manifest", help="Por defecto: la preparación más reciente.")
    simulation.add_argument("--output-dir", default="reports/federated")
    simulation.add_argument("--print-command", action=argparse.BooleanOptionalAction, default=False,
                            help="Preparar el comando sin lanzarlo.")
    baseline = commands.add_parser("baseline", help="Control sin Flower, con las mismas ventanas y métricas.")
    baseline.add_argument("--mode", choices=["local", "centralized", "zero-shot"], default="local")
    baseline.add_argument("--manifest")
    baseline.add_argument("--output-dir", default="reports/non_federated")
    backtest = commands.add_parser("backtest", help="Comparar modelos y métodos en uno o varios folds temporales.")
    backtest.add_argument("--stage", choices=["evaluation", "development", "final", "all"], default="evaluation",
                          help="evaluation evalúa todos los folds sin seleccionar configuración; development/final son opt-in; all es exploratorio.")
    backtest.add_argument("--selection", type=Path, help="selection.json del desarrollo para consultar el holdout.")
    backtest.add_argument("--manifest", help="backtest_manifest.json de prepare-backtest; por defecto el más reciente.")
    backtest.add_argument("--output-dir", default="reports/backtests")
    backtest.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(DEFAULT_MATRIX_MODELS))
    backtest.add_argument("--modes", nargs="+", choices=["federated", "local", "centralized", "zero-shot"],
                          default=["local", "centralized", "federated", "zero-shot"])
    backtest.add_argument("--adaptations", nargs="+", choices=["lora", "ffa-lora", "flora", "lora-fair"],
                          default=["lora", "ffa-lora", "flora", "lora-fair"])
    backtest.add_argument("--seeds", nargs="+", type=int)
    backtest.add_argument("--lora-ranks", nargs="+", type=int)
    backtest.add_argument("--strategies", nargs="+", choices=["fedavg", "fedprox"])
    backtest.add_argument("--max-parallel-models", type=int,
                          help="Límite de candidatos por modo no federado; por defecto uno por GPU. CPU: secuencial.")
    backtest.add_argument("--weekly-leaderboard", action=argparse.BooleanOptionalAction, default=False,
                          help="Publicar las gráficas semanales y un resumen en MLflow, si está activo.")
    comparison = commands.add_parser("compare", help="Comparar resultados completos del mismo manifest.")
    comparison.add_argument("runs", nargs="*", type=Path)
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--allow-training-differences", action="store_true",
                            help="Permitir distinto entrenamiento solo con firma de evaluación idéntica.")
    plots = commands.add_parser("plot", help="Graficar resultados de test federados y centralizados sin entrenar.")
    plots.add_argument("source", type=Path, help="Directorio del fold o de la comparación, o comparison.csv.")
    plots.add_argument("--output", type=Path, help="Directorio nuevo; por defecto <comparación>/plots.")
    plots.add_argument("--weekly", action="store_true", help="Graficar la clasificación por semana de origen.")
    commands.add_parser("models", help="Listar modelos, adaptación y requisitos sin descargarlos.")
    for command in commands.choices.values():
        command.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                             help="Archivo TOML; por defecto fd-airquality.toml en el directorio actual.")
    defaults = RunSettings()
    for field in fields(RunSettings):
        if field.name in {"manifest", "output_dir"}:
            continue
        value = getattr(defaults, field.name)
        options = {}
        if field.name == "num_server_rounds":
            options["help"] = "Número de rondas federadas; no repite el ajuste local ni centralizado."
        for command in (simulation, baseline, backtest):
            command.add_argument(f"--{field.name.replace('_', '-')}", type=type(value), default=value,
                                 **options)
    return parser, commands.choices


def _toml_scalar(value: str | float) -> str:
    return json.dumps(value, ensure_ascii=False)


def launch_simulation(settings: RunSettings, dry_run: bool) -> Path | None:
    project = Path.cwd()
    with (project / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    if config["project"]["name"] != "fd-airquality":
        raise ValueError("Ejecuta simulate desde la raíz de FD-airquality.")
    home = project / ".flwr"
    launches = home / "launches"
    launches.mkdir(parents=True, exist_ok=True)
    overrides = launches / f"{datetime.now(UTC):%Y%m%d_%H%M%S_%f}.toml"
    with RunLogging(overrides.with_suffix(""), run_label(settings, "launcher"), level=settings.log_level,
                    heartbeat_seconds=settings.heartbeat_seconds):
        return _launch_simulation(settings, dry_run, home, overrides)


def _launch_simulation(settings, dry_run, home, overrides):
    validate_run_mode(settings, "federated")
    slots = usable_gpu_slots(settings.device)
    if settings.device == "cuda" and not slots:
        logging.getLogger(__name__).warning("Ninguna GPU utilizable; simulación secuencial en CPU.")
        settings = replace(settings, device="cpu")
    manifest = read_manifest(settings.manifest)
    validate_model_geometry(manifest["geometry"], settings)
    settings, manifest = prepare_runtime_panel(settings, overrides.with_suffix(""))
    validate_selection_coverage(manifest, settings, "federated")
    overrides.write_text("\n".join(
        f"{key} = {_toml_scalar(value)}" for key, value in settings.flower_config().items()
    ) + "\n", encoding="utf-8")
    nodes = len(manifest["clients"])
    parallel_clients = min(len(slots) if slots else 1, nodes)
    client_gpus = 1.0 if slots else 0.0
    federation = (
        f"num-supernodes={nodes} client-resources-num-cpus=1 "
        f"client-resources-num-gpus={client_gpus} init-args-num-cpus={parallel_clients} "
        f"init-args-num-gpus={len(slots)}"
    )
    command = [
        "uv", "run", "--frozen", "flwr", "run", ".", "local", "--stream",
        "--run-config", str(overrides), "--federation-config", federation,
    ]
    print(f"{nodes} estaciones; {len(slots)} GPU utilizables; "
          f"hasta {parallel_clients} clientes federados simultáneos.", flush=True)
    display_command = ["env", f"CUDA_VISIBLE_DEVICES={','.join(slots)}", *command] if slots else command
    print(shlex.join(display_command), flush=True)
    if not dry_run:
        results_root = Path(settings.output_dir)
        previous_runs = set(results_root.glob("*/run.json"))
        env = os.environ.copy()
        env.update(
            FLWR_HOME=str(home), FLWR_TELEMETRY_ENABLED="0", FLWR_DISABLE_UPDATE_CHECK="1",
            RAY_TMPDIR=str(Path("/tmp/opencode/fd-airquality").resolve()),
            OMP_NUM_THREADS=str(settings.torch_threads),
            MKL_NUM_THREADS=str(settings.torch_threads),
        )
        if slots:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(slots)
        Path(env["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=True)
        env["PYTHONUNBUFFERED"] = "1"
        flower_log = overrides.with_suffix("") / "flower.log"
        run_id = None

        def on_line(line):
            nonlocal run_id
            match = re.fullmatch(r"Successfully started run\s+(\d+)", line.strip())
            if run_id is None and match:
                run_id = match.group(1)

        report_progress("flower", f"parallel_clients={parallel_clients} gpu_slots={slots} log={flower_log}")
        try:
            ProcessRunner().stream(command, env, flower_log, "flower", on_line=on_line)
            # A successful logstream alone does not prove the ServerApp succeeded.
            completed = [
                path for path in set(results_root.glob("*/run.json")) - previous_runs
                if json.loads(path.read_text(encoding="utf-8"))["settings"] == asdict(settings)
                and (path.parent / "test_metrics.json").is_file()
            ]
            if not completed:
                raise RuntimeError(
                    "Flower terminó sin resultados de test completos para esta configuración. "
                    "Revisa los errores del logstream y .flwr/local-superlink/superlink.log."
                )
            if len(completed) != 1:
                raise RuntimeError("Se encontraron varias ejecuciones completas; no se puede atribuir el resultado.")
            shutil.copyfile(flower_log, completed[0].parent / "flower.log")
            return completed[0].parent
        except BaseException:
            if run_id is not None:
                try:
                    stopped = subprocess.run(
                        ["uv", "run", "--frozen", "flwr", "stop", run_id, "local"],
                        check=True, env=env, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=30,
                    )
                    logging.getLogger(__name__).warning("Flower run %s detenido: %s", run_id, stopped.stdout)
                except (OSError, subprocess.SubprocessError):
                    logging.getLogger(__name__).exception("No se pudo detener Flower run %s", run_id)
            raise
    return None


def main(argv: list[str] | None = None) -> None:
    parser, commands = _parser()
    preparation_log = None
    try:
        args = parse_configured_args(parser, commands, argv)
        if getattr(args, "config", None) is not None:
            print(f"Configuración: {args.config}", flush=True)
        if args.command == "prepare-backtest":
            from fd_airquality.backtesting import prepare_backtest

            overrides = {field.name: value for field in fields(DataConfig) if field.name not in INTERNAL_FOLD_FIELDS
                         and (value := getattr(args, field.name, None)) is not None}
            log_dir = args.output / "logs" / f"prepare_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}"
            with RunLogging(log_dir, f"{args.command} pollutant={args.pollutant}", level=args.log_level,
                             heartbeat_seconds=args.heartbeat_seconds) as log:
                preparation_log = log.path
                report_progress("prepare", f"raw_root={args.raw_root} pollutant={args.pollutant}")
                manifest = prepare_backtest(
                    args.raw_root.expanduser().resolve(), args.output, args.pollutant,
                    DataConfig(**overrides), set(args.station) if args.station else None,
                    folds=args.folds, validation_hours=args.validation_hours, test_hours=args.test_hours,
                    min_stations=args.min_stations, train_history=args.train_history, history_hours=args.history_hours,
                )
                report_progress("prepare", f"manifest={manifest}")
            if manifest.is_file():
                shutil.copyfile(log.path, manifest.parent / "prepare.log")
            print(f"Manifest: {manifest}")
            _inspect(manifest)
        elif args.command == "inspect":
            _inspect(resolve_manifest(args.manifest, backtest=True))
        elif args.command == "models":
            for name in MODEL_NAMES:
                if name in GRANITE:
                    details = "LoRA / FFA-LoRA / FLoRA / LoRA-FAIR internos; zero-shot disponible"
                    if name not in DEFAULT_MATRIX_MODELS:
                        details += "; fuera de las matrices predeterminadas"
                elif name == "autoencoder-tiny":
                    details = f"cuello de botella de {TINY_AUTOENCODER_LATENT_SIZE}; reconstrucción solo al entrenar"
                elif name in STATION_CONTROLS:
                    details = "referencia por estación; solo modo local"
                else:
                    details = "entrenamiento completo"
                print(f"{name}: {details}")
            print("TTM-R3 (90-30-dec-90-r3): context=90, horizon<=30; PatchTST-FM-R2: context=16..8192; FlowState-R1: context<=2048.")
            print("FlowState: adaptación interna experimental, validada por las pruebas de integración reales, no soporte oficial de fine-tuning.")
        elif args.command == "compare":
            from fd_airquality.comparison import compare_runs

            compare_runs(args.runs, args.output or Path("reports/comparisons") / f"comparison_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}",
                         allow_training_differences=args.allow_training_differences)
        elif args.command == "plot":
            if args.weekly:
                from fd_airquality.leaderboard import plot_weekly

                path, _ = plot_weekly(args.source, args.output)
                print(f"Clasificación semanal: {path / 'index.html'}")
            else:
                from fd_airquality.plotting import plot_comparison

                plot_comparison(args.source, args.output)
        else:
            manifest = resolve_manifest(args.manifest, backtest=args.command == "backtest")
            settings = RunSettings(**{
                field.name: (
                    str(manifest) if field.name == "manifest"
                    else str(Path(args.output_dir).resolve()) if field.name == "output_dir"
                    else getattr(args, field.name)
                ) for field in fields(RunSettings)
            })
            if args.command != "backtest":
                settings = replace(settings, mlflow_tracking_uri=tracking_uri(settings))
            if args.command == "simulate":
                launch_simulation(settings, args.print_command)
            elif args.command == "baseline":
                from fd_airquality.baselines import run_baseline

                run_baseline(settings, args.mode)
            elif args.command == "backtest":
                from fd_airquality.backtesting import run_backtest

                result = run_backtest(settings, args.models, args.modes, adaptations=args.adaptations,
                                       seeds=args.seeds, lora_ranks=args.lora_ranks, strategies=args.strategies,
                                       max_parallel_models=args.max_parallel_models,
                                       stage=args.stage, selection=args.selection,
                                       weekly_leaderboard=args.weekly_leaderboard)
                print(f"Resultados temporales: {result}")
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        log_hint = f"\nLog de preparación: {preparation_log}" if preparation_log is not None else ""
        parser.exit(2, f"Error: {exc}{log_hint}\n")


def _inspect(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if "folds" in document:
        print(f"Protocolo temporal: {document['protocol']}; historial={document['train_history']}")
        block_start = document.get("initial_history_end")
        if block_start is not None:
            print(f"  Calendario observado: [{document['calendar_start']}, {document['calendar_end']}); "
                  f"histórico inicial hasta {block_start}")
        if "requested_validation_hours" in document:
            print(f"  Por fold: validación {document['validation_hours']}/{document['requested_validation_hours']} h; "
                  f"test {document['test_hours']}/{document['requested_test_hours']} h (efectivo/solicitado)")
        if "min_stations" in document:
            print(f"  Mínimo de estaciones comunes: {document['min_stations']}")
        for entry in document["folds"]:
            validation, test, end = (entry[key] for key in ("validation_start", "test_start", "test_end"))
            block = f"bloque [{block_start}, {end}); " if block_start is not None else ""
            print(f"  Fold {entry['fold']}: {block}train < {validation}; "
                  f"validación [{validation}, {test}); test [{test}, {end})")
            if block_start is not None:
                block_start = end
        return
    manifest = read_manifest(path)
    print(f"{manifest['pollutant']}: {len(manifest['clients'])} estaciones elegibles")
    print(f"Geometría: {manifest['geometry']}")
    for client in manifest["clients"]:
        print(
            f"  [{client['partition_id']:02d}] {client['station']}: "
            f"train={client['num_train_windows']}, validation={client['num_validation_windows']}, "
            f"test={client['num_test_windows']} ventanas"
        )
        if manifest["schema_version"] >= 4:
            print(f"       Pares para MASE/RMSSE: validation={client['validation_scale_pairs']}, "
                  f"test={client['test_scale_pairs']}")
            undefined = [name for name in ("validation", "test") if not client[f"{name}_mase_defined"]]
            if undefined:
                print(f"       AVISO: MASE/RMSSE indefinidas en {', '.join(undefined)} (escala cero o pares insuficientes).")
    print(f"Inventario y exclusiones: {path.parent}")
    if (path.parent / "retention.json").is_file():
        print(f"Retención: {json.loads((path.parent / 'retention.json').read_text())}")


if __name__ == "__main__":
    main()
