"""uv-friendly preparation, auditing, and official Flower simulation launcher."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tomllib
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path

from fd_airquality.configuration import parse_configured_args
from fd_airquality.data import DataConfig, prepare, read_manifest
from fd_airquality.models import (
    GRANITE,
    MODEL_NAMES,
    model_metadata,
    validate_model_geometry,
    validate_run_mode,
)
from fd_airquality.panels import prepare_runtime_panel
from fd_airquality.settings import RunSettings, resolve_manifest
from fd_airquality.task import resolve_device, validate_selection_coverage
from fd_airquality.tracking import tracking_uri


def _parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                        help="Archivo TOML; por defecto fd-airquality.toml en el directorio actual.")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare", help="Preprocesar y auditar cada estación por separado.")
    data.add_argument("--raw-root", type=Path, default=Path("data/raw/datos_estaciones_5m"),
                      help="Por defecto: data/raw/datos_estaciones_5m, dentro de este proyecto.")
    data.add_argument("--output", type=Path, default=Path("data/federated"))
    data.add_argument("--pollutant", choices=["NO2", "O3", "CO"], default="NO2")
    data.add_argument("--station", action="append", help="Nombre exacto del directorio; se puede repetir.")
    for field in fields(DataConfig):
        data.add_argument(f"--{field.name.replace('_', '-')}", type=str if isinstance(getattr(DataConfig(), field.name), str) else int)
    audit = commands.add_parser("inspect", help="Resumen del soporte y las exclusiones, sin entrenar.")
    audit.add_argument("--manifest")
    simulation = commands.add_parser("simulate", help="Ejecutar flwr run en el simulador local oficial.")
    simulation.add_argument("--manifest", help="Por defecto: la preparación más reciente.")
    simulation.add_argument("--output-dir", default="reports/federated")
    simulation.add_argument("--workers", type=int, default=2, help="Máximo de clientes CPU simultáneos.")
    simulation.add_argument("--client-gpus", type=float, default=0.0)
    simulation.add_argument("--print-command", action=argparse.BooleanOptionalAction, default=False,
                            help="Preparar el comando sin lanzarlo.")
    baseline = commands.add_parser("baseline", help="Control sin Flower, con las mismas ventanas y métricas.")
    baseline.add_argument("--mode", choices=["local", "centralized", "zero-shot"], default="local")
    baseline.add_argument("--manifest")
    baseline.add_argument("--output-dir", default="reports/non_federated")
    benchmark = commands.add_parser("benchmark", help="Matriz de modelos y métodos sobre un único manifest.")
    benchmark.add_argument("--manifest")
    benchmark.add_argument("--output-dir", default="reports/benchmarks")
    benchmark.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    benchmark.add_argument("--modes", nargs="+", choices=["federated", "local", "centralized", "zero-shot"],
                           default=["federated", "local", "centralized", "zero-shot"])
    benchmark.add_argument("--workers", type=int, default=2)
    benchmark.add_argument("--client-gpus", type=float, default=0.0)
    benchmark.add_argument("--adaptations", nargs="+", choices=["lora", "ffa-lora", "flora", "lora-fair"],
                           default=["lora", "ffa-lora", "flora", "lora-fair"])
    benchmark.add_argument("--seeds", nargs="+", type=int)
    benchmark.add_argument("--lora-ranks", nargs="+", type=int)
    benchmark.add_argument("--strategies", nargs="+", choices=["fedavg", "fedprox"])
    comparison = commands.add_parser("compare", help="Comparar resultados completos del mismo manifest.")
    comparison.add_argument("runs", nargs="*", type=Path)
    comparison.add_argument("--output", type=Path)
    commands.add_parser("models", help="Listar modelos, adaptación y requisitos sin descargarlos.")
    for command in commands.choices.values():
        command.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                             help="Archivo TOML; por defecto fd-airquality.toml en el directorio actual.")
    defaults = RunSettings()
    for field in fields(RunSettings):
        if field.name in {"manifest", "output_dir"}:
            continue
        value = getattr(defaults, field.name)
        for command in (simulation, baseline, benchmark):
            command.add_argument(f"--{field.name.replace('_', '-')}", type=type(value), default=value)
    return parser, commands.choices


def _toml_scalar(value: str | float) -> str:
    return json.dumps(value, ensure_ascii=False)


def launch_simulation(settings: RunSettings, workers: int, client_gpus: float, dry_run: bool) -> Path | None:
    validate_run_mode(settings, "federated")
    if workers < 1 or client_gpus < 0:
        raise ValueError("workers debe ser positivo y client-gpus no negativo.")
    if settings.device == "cuda":
        device = resolve_device(settings.device)
        if device.type == "cpu":
            settings = replace(settings, device="cpu")
            client_gpus = 0.0
        elif client_gpus <= 0:
            raise ValueError("Con device=cuda reserva recursos mediante --client-gpus.")
    manifest = read_manifest(settings.manifest)
    validate_model_geometry(manifest["geometry"], settings)
    project = Path.cwd()
    with (project / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    if config["project"]["name"] != "fd-airquality":
        raise ValueError("Ejecuta simulate desde la raíz de FD-airquality.")
    home = project / ".flwr"
    launches = home / "launches"
    launches.mkdir(parents=True, exist_ok=True)
    overrides = launches / f"{datetime.now(UTC):%Y%m%d_%H%M%S_%f}.toml"
    settings, manifest = prepare_runtime_panel(settings, overrides.with_suffix(""))
    validate_selection_coverage(manifest, settings, "federated")
    overrides.write_text("\n".join(
        f"{key} = {_toml_scalar(value)}" for key, value in settings.flower_config().items()
    ) + "\n", encoding="utf-8")
    nodes = len(manifest["clients"])
    federation = (
        f"num-supernodes={nodes} client-resources-num-cpus=1 "
        f"client-resources-num-gpus={client_gpus} init-args-num-cpus={min(workers, nodes)}"
    )
    command = [
        "uv", "run", "--frozen", "flwr", "run", ".", "local", "--stream",
        "--run-config", str(overrides), "--federation-config", federation,
    ]
    print(f"{nodes} estaciones; hasta {min(workers, nodes)} clientes CPU simultáneos.", flush=True)
    print(shlex.join(command), flush=True)
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
        Path(env["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, env=env)
        # A successful CLI submission/logstream alone does not prove the ServerApp succeeded.
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
        return completed[0].parent
    return None


def build_benchmark_matrix(models: list[str], modes: list[str], adaptations: list[str],
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
    for model in models:
        foundation = model in GRANITE
        for mode in modes:
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


def run_benchmark(settings: RunSettings, models: list[str], modes: list[str], workers: int, client_gpus: float,
                  adaptations: list[str] | None = None, seeds: list[int] | None = None,
                  lora_ranks: list[int] | None = None, strategies: list[str] | None = None) -> Path:
    from fd_airquality.baselines import run_baseline
    from fd_airquality.comparison import compare_runs

    manifest = read_manifest(settings.manifest)
    matrix = build_benchmark_matrix(models, modes, adaptations or ["lora", "ffa-lora", "flora", "lora-fair"], strategies or [settings.strategy])
    seeds, ranks = seeds or [settings.seed], lora_ranks or [settings.lora_rank]
    if len(set(seeds)) != len(seeds) or len(set(ranks)) != len(ranks):
        raise ValueError("No repitas semillas ni rangos.")
    plan = []
    for case in matrix:
        for seed in seeds[:1] if case["mode"] == "zero-shot" else seeds:
            for rank in ranks if case["model"] in GRANITE and case["mode"] != "zero-shot" else ranks[:1]:
                plan.append(case | {"seed": seed, "lora_rank": rank})
    if len(plan) < 2:
        raise ValueError("benchmark requiere al menos dos combinaciones válidas.")
    for case in plan:
        current = replace(settings, **{key: value for key, value in case.items() if key != "mode"})
        validate_model_geometry(manifest["geometry"], current)
        validate_run_mode(current, case["mode"])
    # Validate ALL geometries first, then all requested target/rank inventories.
    # No classic run starts before discovering an invalid foundation setting.
    checked = set()
    for case in plan:
        if case["model"] in GRANITE and case["mode"] != "zero-shot":
            key = (case["model"], case["lora_rank"])
            if key not in checked:
                current = replace(settings, **{key: value for key, value in case.items() if key != "mode"})
                model_metadata(manifest["geometry"], current)
                checked.add(key)
    root = Path(settings.output_dir).resolve() / f"benchmark_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}"
    root.mkdir(parents=True, exist_ok=False)
    settings, manifest = prepare_runtime_panel(settings, root)
    settings = replace(settings, output_dir=str(root / "runs"))
    paths = []
    progress = [case | {"status": "pending"} for case in plan]
    progress_path = root / "benchmark.json"
    for index, case in enumerate(plan):
        mode = case["mode"]
        print(f"\nCaso {index + 1}/{len(plan)}: {case}", flush=True)
        current = replace(settings, **{key: value for key, value in case.items() if key != "mode"})
        progress[index]["status"] = "running"
        progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        try:
            path = launch_simulation(current, workers, client_gpus, dry_run=False) if mode == "federated" else run_baseline(current, mode)
        except Exception as exc:
            progress[index].update(status="failed", error=str(exc))
            progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            raise
        paths.append(path)
        progress[index].update(status="completed", output=str(path))
        progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    return compare_runs(paths, root / "comparison")


def main(argv: list[str] | None = None) -> None:
    parser, commands = _parser()
    try:
        args = parse_configured_args(parser, commands, argv)
        if getattr(args, "config", None) is not None:
            print(f"Configuración: {args.config}", flush=True)
        if args.command == "prepare":
            overrides = {
                field.name: getattr(args, field.name)
                for field in fields(DataConfig) if getattr(args, field.name) is not None
            }
            manifest = prepare(
                args.raw_root.expanduser().resolve(),
                args.output, args.pollutant, DataConfig(**overrides),
                set(args.station) if args.station else None,
            )
            print(f"Manifest: {manifest}")
            _inspect(manifest)
        elif args.command == "inspect":
            _inspect(resolve_manifest(args.manifest))
        elif args.command == "models":
            for name in MODEL_NAMES:
                details = "LoRA / FFA-LoRA / FLoRA / LoRA-FAIR internos; zero-shot disponible" if name in GRANITE else "entrenamiento completo"
                print(f"{name}: {details}")
            print("TTM-R3 (90-30-dec-90-r3): context=90, horizon<=30; PatchTST-FM-R2: context=16..8192; FlowState-R1: context<=2048.")
            print("FlowState: adaptación interna experimental, validada por las pruebas de integración reales, no soporte oficial de fine-tuning.")
        elif args.command == "compare":
            from fd_airquality.comparison import compare_runs

            compare_runs(args.runs, args.output or Path("reports/comparisons") / f"comparison_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}")
        else:
            manifest = resolve_manifest(args.manifest)
            settings = RunSettings(**{
                field.name: (
                    str(manifest) if field.name == "manifest"
                    else str(Path(args.output_dir).resolve()) if field.name == "output_dir"
                    else getattr(args, field.name)
                ) for field in fields(RunSettings)
            })
            settings = replace(settings, mlflow_tracking_uri=tracking_uri(settings))
            if args.command == "simulate":
                launch_simulation(settings, args.workers, args.client_gpus, args.print_command)
            elif args.command == "baseline":
                from fd_airquality.baselines import run_baseline

                run_baseline(settings, args.mode)
            else:
                run_benchmark(settings, args.models, args.modes, args.workers, args.client_gpus,
                              args.adaptations, args.seeds, args.lora_ranks, args.strategies)
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Error: {exc}\n")


def _inspect(path: Path) -> None:
    manifest = read_manifest(path)
    print(f"{manifest['pollutant']}: {len(manifest['clients'])} estaciones elegibles")
    print(f"Geometría: {manifest['geometry']}")
    for client in manifest["clients"]:
        print(
            f"  [{client['partition_id']:02d}] {client['station']}: "
            f"train={client['num_train_windows']}, validation={client['num_validation_windows']}, "
            f"test={client['num_test_windows']} ventanas"
        )
        if manifest["schema_version"] == 4:
            print(f"       Pares para MASE/RMSSE: validation={client['validation_scale_pairs']}, "
                  f"test={client['test_scale_pairs']}")
            undefined = [name for name in ("validation", "test") if not client[f"{name}_mase_defined"]]
            if undefined:
                print(f"       AVISO: MASE/RMSSE indefinidas en {', '.join(undefined)} (escala cero o sin pares).")
    print(f"Inventario y exclusiones: {path.parent}")
    if (path.parent / "retention.json").is_file():
        print(f"Retención: {json.loads((path.parent / 'retention.json').read_text())}")


if __name__ == "__main__":
    main()
