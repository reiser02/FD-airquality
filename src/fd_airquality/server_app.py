"""Central Flower server: receives model weights and aggregate statistics, not data."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from fd_airquality.data import read_manifest
from fd_airquality.models import (
    effective_adaptation,
    shared_state,
    validate_model_geometry,
    validate_run_mode,
)
from fd_airquality.panels import prepare_runtime_panel
from fd_airquality.run_logging import RunLogging, report_progress, run_label
from fd_airquality.settings import RunSettings
from fd_airquality.strategy import AuditedStrategy
from fd_airquality.task import configure_torch, make_model, validate_selection_coverage
from fd_airquality.tracking import RunTracker, write_run


def run_server(grid: Grid, settings: RunSettings) -> Path:
    validate_run_mode(settings, "federated")
    manifest = read_manifest(settings.manifest)
    validate_model_geometry(manifest["geometry"], settings)
    output = Path(settings.output_dir).resolve() / (
        f"{manifest['pollutant']}_{settings.model}_{effective_adaptation(settings)}_{settings.strategy}_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(parents=True, exist_ok=False)
    with RunLogging(output, run_label(settings, "federated"), level=settings.log_level,
                    heartbeat_seconds=settings.heartbeat_seconds):
        return _run_server(grid, settings, output)


def _run_server(grid, settings, output):
    report_progress("panel", f"manifest={settings.manifest}")
    settings, manifest = prepare_runtime_panel(settings, output)
    validate_selection_coverage(manifest, settings, "federated")
    settings = replace(settings, device=configure_torch(settings, settings.seed).type)
    report_progress("loading-model", f"device={settings.device} stations={len(manifest['clients'])}")
    write_run(output, settings, manifest, "federated")
    with RunTracker(settings, output, "federated", manifest) as tracker:
        strategy = AuditedStrategy(settings, manifest, output, tracker)
        metrics = _fit(grid, settings, manifest, strategy)
    print(f"Resultados: {output}", flush=True)
    print(f"Test federado (ronda {strategy.best_round}): {dict(metrics)}", flush=True)
    return output


def _fit(grid: Grid, settings: RunSettings, manifest: dict, strategy: AuditedStrategy):
    configure_torch(settings, settings.seed)
    arrays = ArrayRecord(shared_state(make_model(manifest["geometry"], settings, pretrained=False)))
    # Round zero may be persistence, an untrained Darts model, or a foundation model.
    validation_config = ConfigRecord({"split": "validation"})
    initial_messages = strategy.configure_evaluate(0, arrays, validation_config, grid)
    strategy.aggregate_evaluate(0, grid.send_and_receive(initial_messages, timeout=3600))
    strategy.start(
        grid=grid, initial_arrays=arrays,
        num_rounds=settings.num_server_rounds,
        train_config=ConfigRecord(),
        evaluate_config=ConfigRecord({"split": "validation"}),
    )
    return strategy.evaluate_test(grid)


def build_server_app(run_config: dict | None = None) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        if run_config is not None:
            context.run_config.update(run_config)
        run_server(grid, RunSettings.from_mapping(context.run_config))

    return app


app = build_server_app()
