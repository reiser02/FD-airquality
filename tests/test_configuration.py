import json
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from fd_airquality.cli import _parser, main
from fd_airquality.configuration import parse_configured_args
from fd_airquality.data import DataConfig
from fd_airquality.settings import RunSettings


def arguments(values):
    parser, commands = _parser()
    return parse_configured_args(parser, commands, values)


def test_toml_defaults_and_preparation_overrides_are_valid(tmp_path, monkeypatch):
    # The user's experiment file is editable; defaults must use a controlled TOML.
    monkeypatch.chdir(tmp_path)
    preparation = asdict(DataConfig()) | {"train_stride": 6}
    (tmp_path / "fd-airquality.toml").write_text(
        "[prepare]\n" + "\n".join(
            f"{key.replace('_', '-')} = {json.dumps(value)}"
            for key, value in preparation.items() if value is not None
        ) + '\nstation = ["AQN1 - Puerto", "AQN3 - Paseo Alfonso XIII"]\n'
    )
    data = arguments(["prepare"])
    for key, value in asdict(DataConfig()).items():
        assert getattr(data, key) == (6 if key == "train_stride" else value)
    assert data.station == ["AQN1 - Puerto", "AQN3 - Paseo Alfonso XIII"]
    for command in ("simulate", "baseline", "benchmark"):
        args = arguments([command])
        for key, value in asdict(RunSettings()).items():
            if key not in {"manifest", "output_dir"}:
                assert getattr(args, key) == value


def test_project_toml_is_valid_without_requiring_builtin_defaults():
    path = Path(__file__).resolve().parents[1] / "fd-airquality.toml"
    data = arguments(["prepare", "--config", str(path)])
    DataConfig(**{field.name: getattr(data, field.name)
                  for field in fields(DataConfig) if getattr(data, field.name) is not None})
    for command in ("simulate", "baseline", "benchmark"):
        args = arguments([command, "--config", str(path)])
        RunSettings(**{field.name: getattr(args, field.name) for field in fields(RunSettings)})


def test_no_file_uses_builtin_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert arguments(["simulate"]).batch_size == RunSettings().batch_size
    assert arguments(["prepare"]).context is None


def test_auto_config_precedence_and_explicit_builtin_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text(
        '[run]\nbatch-size = 32\nseed = 17\n[baseline]\nbatch-size = 64\nmode = "centralized"\n'
    )
    assert arguments(["simulate"]).batch_size == 32
    baseline = arguments(["baseline"])
    assert (baseline.batch_size, baseline.seed, baseline.mode) == (64, 17, "centralized")
    explicit = arguments(["baseline", "--batch-size", "128", "--seed", "42", "--mode", "local"])
    assert (explicit.batch_size, explicit.seed, explicit.mode) == (128, 42, "local")


@pytest.mark.parametrize("before", [True, False])
def test_explicit_config_works_before_and_after_command(tmp_path, monkeypatch, before):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text('[run]\nseed = 5\n')
    path = tmp_path / "custom.toml"
    path.write_text('[run]\nseed = 99\n')
    args = ["--config", str(path), "simulate"] if before else ["simulate", "--config", str(path)]
    assert arguments(args).seed == 99


def test_cli_lists_and_repeated_stations_replace_file_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text(
        '[prepare]\nstation = ["old-a", "old-b"]\n'
        '[benchmark]\nmodels = ["mlp", "autoencoder"]\nseeds = [1, 2]\n'
    )
    assert arguments(["prepare"]).station == ["old-a", "old-b"]
    assert arguments(["prepare", "--station", "new"]).station == ["new"]
    args = arguments(["benchmark", "--models", "darts-nlinear", "--seeds", "42"])
    assert (args.models, args.seeds) == (["darts-nlinear"], [42])


@pytest.mark.parametrize("text", [
    '[unknown]\nseed = 1', '[run]\nsead = 1', '[run]\nworkers = 1',
    '[simulate]\nworkers = 2', '[simulate]\nclient-gpus = 1.0',
    '[benchmark]\nworkers = 2', '[benchmark]\nclient-gpus = 1.0',
    '[run]\nbatch-size = "32"', '[run]\nseed = true', '[run]\nbatch-size = 1.5',
    '[benchmark]\nmodels = "mlp"', '[benchmark]\nmodels = ["unknown"]',
    '[benchmark]\nseeds = []', '[simulate]\nprint-command = 1',
    '[run]\nlearning-rate = true', '[prepare]\nraw-root = 1',
    '[run]\nlora-rank = 4\nlora_rank = 8', 'run = 1', '[run',
])
def test_invalid_file_fails_with_clear_error(tmp_path, monkeypatch, capsys, text):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text(text)
    with pytest.raises(SystemExit) as exc:
        main(["simulate", "--print-command"])
    assert exc.value.code == 2
    assert "Error:" in capsys.readouterr().err


def test_missing_explicit_file_is_not_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="configuración"):
        arguments(["models", "--config", "missing.toml"])


def test_config_paths_and_scalar_types_follow_cli_semantics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "config"
    nested.mkdir()
    path = nested / "custom.toml"
    path.write_text('[prepare]\nraw-root = "raw"\n[run]\nlearning-rate = 1\n[simulate]\nprint-command = true\n')
    assert arguments(["prepare", "--config", str(path)]).raw_root == Path("raw")
    args = arguments(["simulate", "--config", str(path)])
    assert args.print_command is True
    assert args.learning_rate == 1.0
    assert arguments(["simulate", "--config", str(path), "--no-print-command"]).print_command is False


def test_compare_runs_can_come_from_file_or_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text('[compare]\nruns = ["a", "b"]\n')
    assert arguments(["compare"]).runs == [Path("a"), Path("b")]
    assert arguments(["compare", "c", "d"]).runs == [Path("c"), Path("d")]


def test_main_passes_merged_configuration_to_the_launcher(prepared, tmp_path, monkeypatch):
    path, _ = prepared
    config = tmp_path / "experiment.toml"
    config.write_text('[run]\nseed = 17\nnum-server-rounds = 9\n'
                      '[simulate]\nprint-command = true\n')
    received = []
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *args: received.append(args))
    main(["simulate", "--config", str(config), "--manifest", str(path), "--num-server-rounds", "1"])
    settings, dry_run = received[0]
    assert (settings.seed, settings.num_server_rounds, settings.manifest) == (17, 1, str(path))
    assert dry_run is True


def test_main_passes_file_geometry_and_cli_station_to_preparation(tmp_path, monkeypatch):
    config = tmp_path / "experiment.toml"
    config.write_text('[prepare]\ncontext = 16\nhorizon = 4\nstation = ["old"]\n'
                      'raw-root = "raw"\npollutant = "CO"\n')
    received = []

    def prepare(*args):
        received.append(args)
        return tmp_path / "manifest.json"

    monkeypatch.setattr("fd_airquality.cli.prepare", prepare)
    monkeypatch.setattr("fd_airquality.cli._inspect", lambda *_: None)
    main(["prepare", "--config", str(config), "--station", "new", "--stride", "2"])
    raw_root, _, pollutant, geometry, stations = received[0]
    assert (geometry.context, geometry.horizon, geometry.stride) == (16, 4, 2)
    assert (raw_root, pollutant, stations) == (Path("raw").resolve(), "CO", {"new"})


def test_logging_and_parallelism_options_respect_toml_and_cli_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text(
        '[run]\nheartbeat-seconds = 15\nlog-level = "DEBUG"\n'
        '[benchmark]\nmax-parallel-models = 2\n'
        'modes = ["local", "centralized", "federated", "zero-shot"]\n'
    )
    args = arguments(["benchmark", "--max-parallel-models", "1", "--heartbeat-seconds", "30"])
    assert (args.max_parallel_models, args.heartbeat_seconds, args.log_level) == (1, 30.0, "DEBUG")
    assert args.modes == ["local", "centralized", "federated", "zero-shot"]
