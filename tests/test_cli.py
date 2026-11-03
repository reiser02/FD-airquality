import json
import tomllib
import zipfile
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import pytest

from fd_airquality.cli import _parser, launch_simulation, main
from fd_airquality.settings import RunSettings


def test_flower_and_cli_defaults_are_identical():
    with Path("pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    assert RunSettings.from_mapping(config["tool"]["flwr"]["app"]["config"]) == RunSettings()


@pytest.mark.parametrize("command", ["simulate", "benchmark"])
@pytest.mark.parametrize("option", ["--workers", "--client-gpus"])
def test_manual_federated_resource_options_are_removed(command, option, capsys):
    parser, _ = _parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([command, option, "1"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


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
    for name in ("autoencoder", "darts-nlinear", "ttm-r3", "patchtst-fm-r2", "flowstate-r1"):
        assert name in output


def test_benchmark_validates_all_contexts_before_starting_any_run(prepared, tmp_path, monkeypatch):
    path, _ = prepared
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *_: pytest.fail("No debe entrenar"))
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--manifest", str(path), "--models", "mlp", "ttm-r3", "--output-dir", str(tmp_path / "runs")])
    assert exc.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_benchmark_validates_adapter_ranks_before_training_classics(global_prepared, tmp_path, monkeypatch):
    from test_models import DummyBackbone

    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: DummyBackbone())
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *_: pytest.fail("No debe entrenar"))
    monkeypatch.setattr("fd_airquality.baselines.run_baseline", lambda *_: pytest.fail("No debe entrenar"))
    path, _ = global_prepared
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--manifest", str(path), "--models", "mlp", "ttm-r3", "--lora-targets", "proj",
              "--lora-ranks", "99", "--output-dir", str(tmp_path / "runs")])
    assert exc.value.code == 2
    log, = (tmp_path / "runs").glob("*/benchmark.log")
    assert "Run failed" in log.read_text()
    assert not list((tmp_path / "runs").glob("**/test_metrics.json"))


def test_benchmark_rejects_empty_effective_panel_before_any_training(global_prepared, tmp_path, monkeypatch):
    path, manifest = global_prepared
    for client in manifest["clients"]:
        client["validation_mase_defined"] = False
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *_: pytest.fail("No debe entrenar"))
    monkeypatch.setattr("fd_airquality.baselines.run_baseline", lambda *_: pytest.fail("No debe entrenar"))
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--manifest", str(path), "--models", "mlp", "--modes", "federated", "local",
              "--output-dir", str(tmp_path / "runs")])
    assert exc.value.code == 2
    report, = (tmp_path / "runs").glob("*/excluded_stations.csv")
    assert "station-a" in report.read_text() and "station-b" in report.read_text()


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


def test_benchmark_model_concurrency_cap_does_not_limit_federated_clients(prepared, tmp_path, monkeypatch, capsys):
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
