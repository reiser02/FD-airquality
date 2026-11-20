import json
import tomllib
import zipfile
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import pytest

from fd_airquality.artifacts import file_digest
from fd_airquality.cli import _parser, launch_simulation, main
from fd_airquality.settings import RunSettings


@pytest.mark.parametrize("command", ["simulate", "baseline", "backtest"])
def test_cli_defaults_and_flower_mapping_use_shared_settings(command):
    parser, _ = _parser()
    args = parser.parse_args([command])
    settings = RunSettings()
    for name, value in asdict(settings).items():
        if name not in {"manifest", "output_dir"}:
            assert getattr(args, name) == value
    assert RunSettings.from_mapping(settings.flower_config()) == settings


@pytest.mark.parametrize("command", ["simulate", "backtest"])
@pytest.mark.parametrize("option", ["--workers", "--client-gpus"])
def test_manual_federated_resource_options_are_removed(command, option, capsys):
    parser, _ = _parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([command, option, "1"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_benchmark_is_not_a_separate_command(capsys):
    parser, commands = _parser()
    assert "benchmark" not in commands
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["benchmark"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_simulation_command_uses_manifest_station_count(prepared, capsys, tmp_path, monkeypatch):
    path, _ = prepared
    project_config = Path("pyproject.toml").read_bytes()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_bytes(project_config)
    main(["simulate", "--manifest", str(path), "--print-command"])
    output = capsys.readouterr().out
    assert "uv run --frozen flwr run . local --stream" in output
    assert "num-supernodes=2" in output
    assert "init-args-num-cpus=1" in output
    assert "client-resources-num-gpus=0.0" in output
    assert "init-args-num-gpus=0" in output
    assert "hasta 1 clientes federados simultáneos" in output


def test_inspect_uses_no_training(prepared, capsys):
    path, _ = prepared
    main(["inspect", "--manifest", str(path)])
    assert "2 estaciones elegibles" in capsys.readouterr().out


def test_models_lists_granite_without_downloading(capsys, monkeypatch):
    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: pytest.fail("No debe descargar"))
    main(["models"])
    output = capsys.readouterr().out
    for name in ("autoencoder", "nlinear", "nhits", "ttm-r3", "patchtst-fm-r2", "flowstate-r1"):
        assert name in output
    assert "patchtst-fm-r2:" in output and "fuera de las matrices predeterminadas" in output


def test_backtest_requires_a_backtest_manifest(global_prepared, tmp_path, capsys):
    path, _ = global_prepared
    with pytest.raises(SystemExit) as exc:
        main(["backtest", "--manifest", str(path), "--output-dir", str(tmp_path / "runs")])
    assert exc.value.code == 2
    assert "prepare-backtest" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_prepare_backtest_requires_15_stations_by_default_but_allows_a_test_subset(temporal_family, tmp_path, capsys):
    raw, _ = temporal_family
    output = tmp_path / "prepared"
    args = ["prepare-backtest", "--raw-root", str(raw), "--output", str(output),
            "--folds", "1", "--window-policy", "masked"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert "2 estaciones; mínimo exigido: 15" in capsys.readouterr().err
    assert not list(output.glob("*/backtest_manifest.json"))

    main([*args, "--station", "station-a"])
    path, = output.glob("*/backtest_manifest.json")
    family = json.loads(path.read_text())
    assert (family["station_selection"], family["min_stations"]) == ("subset", 1)
    assert "selección explícita de estaciones" in path.with_name("prepare.log").read_text()


def test_prepare_backtest_logs_fold_sizes_and_warns_when_targets_are_reduced(temporal_family, tmp_path):
    raw, _ = temporal_family
    output = tmp_path / "prepared"
    main(["prepare-backtest", "--raw-root", str(raw), "--output", str(output),
          "--folds", "3", "--min-stations", "2", "--window-policy", "masked"])
    path, = output.glob("*/backtest_manifest.json")
    log = path.with_name("prepare.log").read_text()
    assert "INFO" in log and "Analizando 2 CSV" in log
    assert "WARNING" in log and "Histórico inicial de train reducido" in log
    assert "No caben las duraciones solicitadas con 3 folds" in log
    assert "validación 672 → 48 h, test 672 → 96 h" in log
    for number in range(1, 4):
        assert f"fold={number}/3 train [" in log
        assert f"fold={number}/3 estaciones elegibles=2; mínimo de ventanas" in log
    assert "Panel preparado: 2 estaciones" in log


def test_prepare_backtest_logs_critical_calendar_failure_before_building_folds(temporal_family, tmp_path, capsys):
    raw, _ = temporal_family
    output = tmp_path / "prepared"
    with pytest.raises(SystemExit) as exc:
        main(["prepare-backtest", "--raw-root", str(raw), "--output", str(output),
              "--folds", "10", "--min-stations", "2", "--log-level", "ERROR"])
    assert exc.value.code == 2
    log_path, = (output / "logs").glob("*/run.log")
    log = log_path.read_text()
    assert "CRITICAL" in log and "El calendario de 576 h no alcanza para 10 folds" in log
    assert "train inicial 102 h" in log and "validación 48 h, test 96 h" in log
    assert "Máximo teórico: 3 folds" in log
    assert "Analizando 2 CSV" not in log
    assert f"Log de preparación: {log_path}" in capsys.readouterr().err
    assert not list(output.glob("*/fold_*"))


def test_prepare_backtest_distinguishes_lack_of_observed_windows_from_calendar_space(temporal_family, tmp_path):
    raw, _ = temporal_family
    output = tmp_path / "prepared"
    # The strict protocol cannot make a 90 h context from the short observed runs.
    with pytest.raises(SystemExit) as exc:
        main(["prepare-backtest", "--raw-root", str(raw), "--output", str(output), "--folds", "1",
              "--min-stations", "2"])
    assert exc.value.code == 2
    log_path, = (output / "logs").glob("*/run.log")
    log = log_path.read_text()
    assert "CRITICAL" in log and "fold=1/1 No se pudo preparar el fold" in log
    assert "Ninguna estación es elegible" in log and "excluded_stations.csv" in log
    assert "El calendario de 576 h no alcanza" not in log
    assert not list(output.glob("*/backtest_manifest.json"))


def test_backtest_validates_all_contexts_before_starting_any_run(single_fold_manifest, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *_: pytest.fail("No debe entrenar"))
    monkeypatch.setattr("fd_airquality.baselines.run_baseline", lambda *_: pytest.fail("No debe entrenar"))
    with pytest.raises(SystemExit) as exc:
        main(["backtest", "--manifest", str(single_fold_manifest), "--models", "mlp", "ttm-r3",
              "--output-dir", str(tmp_path / "runs")])
    assert exc.value.code == 2
    assert "exige contexto 90" in capsys.readouterr().err
    report, = (tmp_path / "runs").glob("*/backtest.json")
    assert json.loads(report.read_text())["results"] == [{"fold": 1, "status": "failed"}]
    assert not list((tmp_path / "runs").glob("**/test_metrics.json"))


def test_backtest_validates_adapter_ranks_before_training_classics(
    single_fold_manifest, tmp_path, monkeypatch, mock_granite,
):
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *_: pytest.fail("No debe entrenar"))
    monkeypatch.setattr("fd_airquality.baselines.run_baseline", lambda *_: pytest.fail("No debe entrenar"))
    with pytest.raises(SystemExit) as exc:
        main(["backtest", "--manifest", str(single_fold_manifest), "--models", "mlp", "flowstate-r1",
              "--lora-targets", "proj", "--lora-ranks", "99", "--output-dir", str(tmp_path / "runs")])
    assert exc.value.code == 2
    log, = (tmp_path / "runs").glob("*/fold_001/fold.log")
    assert "Run failed" in log.read_text()
    assert not list((tmp_path / "runs").glob("**/test_metrics.json"))


def test_backtest_rejects_empty_effective_panel_before_any_training(single_fold_manifest, tmp_path, monkeypatch):
    family = json.loads(single_fold_manifest.read_text())
    fold, = family["folds"]
    path = Path(fold["manifest"])
    manifest = json.loads(path.read_text())
    for client in manifest["clients"]:
        client["validation_mase_defined"] = False
    path.write_text(json.dumps(manifest))
    fold["sha256"] = file_digest(path)
    single_fold_manifest.write_text(json.dumps(family))
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *_: pytest.fail("No debe entrenar"))
    monkeypatch.setattr("fd_airquality.baselines.run_baseline", lambda *_: pytest.fail("No debe entrenar"))
    with pytest.raises(SystemExit) as exc:
        main(["backtest", "--manifest", str(single_fold_manifest), "--models", "mlp",
              "--modes", "federated", "local", "--output-dir", str(tmp_path / "runs")])
    assert exc.value.code == 2
    report, = (tmp_path / "runs").glob("*/temporal_exclusions.csv")
    for client in manifest["clients"]:
        assert client["station"] in report.read_text()
    log = report.with_name("backtest.log").read_text()
    assert "WARNING" in log and "2 estaciones sin macro-mase de validación definida" in log
    assert "CRITICAL" in log and "Panel común insuficiente: 0 estaciones; mínimo exigido: 2" in log


def test_inspect_reports_zero_denominators_even_when_pairs_exist(global_prepared, capsys):
    path, manifest = global_prepared
    manifest["clients"][0]["validation_mase_defined"] = False
    assert manifest["clients"][0]["validation_scale_pairs"] > 0
    path.write_text(json.dumps(manifest))
    main(["inspect", "--manifest", str(path)])
    assert "MASE/RMSSE indefinidas en validation" in capsys.readouterr().out


@pytest.mark.parametrize("slots", [["GPU-B"], ["3", "7"], ["0", "1", "2"]])
def test_simulation_automatically_reserves_one_gpu_per_client(prepared, tmp_path, monkeypatch, capsys, slots):
    path, _ = prepared
    project_config = Path("pyproject.toml").read_bytes()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_bytes(project_config)
    monkeypatch.setattr("fd_airquality.cli.usable_gpu_slots", lambda _: slots)
    main(["simulate", "--manifest", str(path), "--device", "cuda", "--print-command"])
    output = capsys.readouterr().out
    assert "client-resources-num-gpus=1.0" in output
    assert f"init-args-num-gpus={len(slots)}" in output
    assert f"init-args-num-cpus={min(len(slots), 2)}" in output
    assert f"hasta {min(len(slots), 2)} clientes federados simultáneos" in output
    assert f"CUDA_VISIBLE_DEVICES={','.join(slots)}" in output
    overrides, = (tmp_path / ".flwr" / "launches").glob("*.toml")
    with overrides.open("rb") as stream:
        assert tomllib.load(stream)["device"] == "cuda"


def test_cuda_fallback_does_not_reserve_ray_gpu_resources(prepared, tmp_path, monkeypatch, capsys):
    path, _ = prepared
    project_config = Path("pyproject.toml").read_bytes()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_bytes(project_config)
    monkeypatch.setattr("fd_airquality.cli.usable_gpu_slots", lambda _: [])
    settings = RunSettings(manifest=str(path), device="cuda")
    launch_simulation(settings, dry_run=True)
    output = capsys.readouterr().out
    assert "client-resources-num-gpus=0.0" in output
    assert "init-args-num-gpus=0" in output
    assert "init-args-num-cpus=1" in output
    overrides, = (tmp_path / ".flwr" / "launches").glob("*.toml")
    with overrides.open("rb") as stream:
        assert tomllib.load(stream)["device"] == "cpu"


def test_backtest_model_concurrency_cap_does_not_limit_federated_clients(prepared, tmp_path, monkeypatch, capsys):
    from io import StringIO

    from fd_airquality.execution import execute_plan
    from fd_airquality.run_logging import RunLogging

    path, _ = prepared
    project_config = Path("pyproject.toml").read_bytes()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_bytes(project_config)
    monkeypatch.setattr("fd_airquality.cli.usable_gpu_slots", lambda _: ["3", "7"])
    monkeypatch.setattr("fd_airquality.execution.usable_gpu_slots", lambda _: ["3", "7"])

    def launch(settings, dry_run):
        launch_simulation(settings, dry_run=True)
        return tmp_path / "federated-result"

    plan = [{"model": "mlp", "mode": "federated"}]
    with RunLogging(tmp_path, "test", stream=StringIO()) as log:
        execute_plan(RunSettings(manifest=str(path), device="cuda"), plan, tmp_path,
                     max_parallel_models=1, baseline=None, launch=launch, progress_log=log)
    output = capsys.readouterr().out
    assert "init-args-num-cpus=2" in output
    assert "init-args-num-gpus=2" in output
    assert "client-resources-num-gpus=1.0" in output


def test_flower_bundle_never_uploads_datasets_reports_or_skills():
    from flwr.cli.build import build_fab_from_files

    files = {
        "pyproject.toml": Path("pyproject.toml"),
        "data/clients/000/measurements.json": b'{"private": true}',
        "reports/run.json": b'{"private": true}',
        ".agents/private_skill.py": b"raise RuntimeError('not application code')",
        **{str(path): path for path in Path("src/fd_airquality").glob("*.py")},
    }
    bundle, _ = build_fab_from_files(files)
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        paths = archive.namelist()
    assert "src/fd_airquality/client_app.py" in paths
    assert "src/fd_airquality/server_app.py" in paths
    assert not any(path.startswith(("data/", "reports/", ".agents/")) for path in paths)


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("slots", [[], ["GPU-B", "GPU-C"]])
def test_launch_requires_successful_server_results(prepared, tmp_path, monkeypatch, completed, slots):
    path, _ = prepared
    project_config = Path("pyproject.toml").read_bytes()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_bytes(project_config)
    monkeypatch.setattr("fd_airquality.cli.usable_gpu_slots", lambda _: slots)
    settings = RunSettings(manifest=str(path), output_dir=str(tmp_path / "reports"),
                           device="cuda" if slots else "cpu")

    def successful_submission(self, command, env, path, label, **kwargs):
        if slots:
            assert env["CUDA_VISIBLE_DEVICES"] == ",".join(slots)
        path.write_text("Flower output streamed dynamically\n")
        if completed:
            result = tmp_path / "reports" / "new_run"
            result.mkdir(parents=True)
            (result / "run.json").write_text(json.dumps({"settings": asdict(settings)}))
            (result / "test_metrics.json").write_text("{}")

    monkeypatch.setattr("fd_airquality.cli.ProcessRunner.stream", successful_submission)
    if completed:
        result = launch_simulation(settings, dry_run=False)
        assert "streamed dynamically" in (result / "flower.log").read_text()
    else:
        with pytest.raises(RuntimeError, match="sin resultados de test completos"):
            launch_simulation(settings, dry_run=False)


def test_interrupted_launcher_requests_native_flower_stop_and_preserves_logs(prepared, tmp_path, monkeypatch):
    path, _ = prepared
    project_config = Path("pyproject.toml").read_bytes()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_bytes(project_config)
    stopped = []

    def stream(self, command, env, path, label, on_line):
        path.write_text("Successfully started run 123456\n")
        on_line("Successfully started run 123456\n")
        on_line("Some unrelated log: Successfully started run 999999\n")
        on_line("Successfully started run 999999\n")
        raise KeyboardInterrupt

    def stop(command, **kwargs):
        from subprocess import CompletedProcess

        stopped.append(command)
        return CompletedProcess(command, 0, stdout="stopped")

    monkeypatch.setattr("fd_airquality.cli.ProcessRunner.stream", stream)
    monkeypatch.setattr("fd_airquality.cli.subprocess.run", stop)
    with pytest.raises(KeyboardInterrupt):
        launch_simulation(RunSettings(manifest=str(path)), dry_run=False)
    assert stopped == [["uv", "run", "--frozen", "flwr", "stop", "123456", "local"]]
    log, = (tmp_path / ".flwr" / "launches").glob("*/run.log")
    assert "Run interrupted" in log.read_text()
    assert "123456" in log.with_name("flower.log").read_text()
