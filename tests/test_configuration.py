import json
import tomllib
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
    for command in ("simulate", "baseline", "backtest"):
        args = arguments([command])
        for key, value in asdict(RunSettings()).items():
            if key not in {"manifest", "output_dir"}:
                assert getattr(args, key) == value


def test_explicit_toml_is_valid_without_requiring_builtin_defaults(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text('[prepare]\npollutant = "PM25"\ncontext = 24\nhorizon = 6\nstride = 3\n'
                    '[run]\nseed = 17\nbatch-size = 16\n')
    data = arguments(["prepare", "--config", str(path)])
    geometry = DataConfig(**{field.name: getattr(data, field.name)
                             for field in fields(DataConfig) if getattr(data, field.name) is not None})
    assert (geometry.context, geometry.horizon, geometry.stride) == (24, 6, 3)
    for command in ("simulate", "baseline", "backtest"):
        args = arguments([command, "--config", str(path)])
        settings = RunSettings(**{field.name: getattr(args, field.name) for field in fields(RunSettings)})
        assert (settings.seed, settings.batch_size) == (17, 16)


@pytest.mark.parametrize("command", ["prepare", "prepare-backtest"])
def test_window_padding_can_be_enabled_or_disabled_by_toml_and_cli(tmp_path, command):
    path = tmp_path / "padding.toml"
    path.write_text('[prepare]\nwindow-policy = "masked"\nwindow-padding = "both"\n')
    assert arguments([command, "--config", str(path)]).window_padding == "both"
    for mode in ("none", "right"):
        assert arguments([command, "--config", str(path), "--window-padding", mode]).window_padding == mode


def test_no_file_uses_builtin_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert arguments(["simulate"]).batch_size == RunSettings().batch_size
    assert arguments(["prepare"]).context is None
    assert arguments(["backtest"]).stage == "evaluation"
    assert arguments(["backtest", "--stage", "development"]).stage == "development"


def test_default_matrix_includes_the_tiny_model_and_requires_opt_in_for_patchtst():
    root_config = Path(__file__).resolve().parents[1] / "fd-airquality.toml"
    for config in ([], ["--config", str(root_config)]):
        args = arguments(["backtest", *config])
        assert args.models == [
            "mlp", "autoencoder", "autoencoder-tiny", "nlinear", "nlinear-point", "nhits",
            "linear-6lags", "seasonal-naive", "holt", "ttm-r3", "flowstate-r1",
        ]
        assert arguments(["backtest", *config, "--models", "patchtst-fm-r2"]).models == ["patchtst-fm-r2"]


@pytest.mark.parametrize("filename,modes,seeds,folds", [
    ("fd-airquality.toml", ["local", "centralized", "federated", "zero-shot"], [42], 4),
    ("experiments/smoke.toml", ["local", "centralized", "federated"], [42], 2),
])
def test_shipped_backtest_configs_keep_modes_seeds_and_fold_counts(filename, modes, seeds, folds):
    config = Path(__file__).resolve().parents[1] / filename
    evaluation = arguments(["backtest", "--config", str(config)])
    assert evaluation.modes == modes
    assert evaluation.seeds == seeds
    assert evaluation.adaptations == ["lora", "ffa-lora", "flora", "lora-fair"]
    preparation = arguments(["prepare-backtest", "--config", str(config)])
    assert preparation.folds == folds
    assert preparation.min_stations == 15
    assert not hasattr(preparation, "test_end")


def test_main_config_shares_batch_budget_and_controls_in_all_training_modes():
    root = Path(__file__).resolve().parents[1] / "fd-airquality.toml"
    for command in ("baseline", "simulate", "backtest"):
        args = arguments([command, "--config", str(root)])
        assert (args.batch_size, args.pad_last_batch, args.local_epochs, args.early_stopping_patience) == (32, 0, 40, 7)
        assert (args.arrival_period_hours, args.federated_update_epochs) == (240, 20)
        assert (args.round_validation_hours, args.round_validation_max_hours) == (24, 336)
    data = arguments(["prepare-backtest", "--config", str(root)])
    assert data.max_train_windows == 256 and data.station is None
    assert (data.stride, data.train_stride) == (4, 4)


def test_backtest_inherits_geometry_but_not_the_fixed_cuts_or_station_split_lengths(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text('[prepare]\ncontext = 16\nvalidation-start = "2026-03-01"\n'
                    'test-start = "2026-03-29"\ntest-end = "2026-04-26"\nvalidation-hours = 48\ntest-hours = 96\n'
                    'min-train-windows = 16\nmin-validation-windows = 7\nmin-test-windows = 15\n'
                    '[prepare-backtest]\nfolds = 3\ntest-hours = 504\nmin-stations = 12\nmin-validation-windows = 10\n')
    args = ["prepare-backtest", "--config", str(path)]
    temporal = arguments([*args, "--folds", "1", "--test-hours", "672", "--min-stations", "8"])
    assert (temporal.context, temporal.folds) == (16, 1)
    assert not {"validation_start", "test_start", "test_end"} & vars(temporal).keys()
    assert (temporal.validation_hours, temporal.test_hours) == (672, 672)
    assert arguments(args).test_hours == 504
    assert (arguments(args).min_stations, temporal.min_stations) == (12, 8)
    assert (temporal.min_train_windows, temporal.min_validation_windows, temporal.min_test_windows) == (16, 10, 15)
    standalone = arguments(["prepare", "--config", str(path)])
    assert (standalone.test_end, standalone.validation_hours, standalone.test_hours) == ("2026-04-26", 48, 96)


def test_new_blocks_are_default_without_changing_fold_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    removed = {"arrival_history", "arrival_history_hours"}
    assert not removed.intersection(asdict(RunSettings()))
    for command in ("simulate", "baseline", "backtest"):
        assert not removed.intersection(vars(arguments([command])))
        for option in ("--arrival-history", "--arrival-history-hours"):
            with pytest.raises(SystemExit):
                arguments([command, option, "expanding" if option == "--arrival-history" else "24"])
    assert arguments(["prepare-backtest"]).train_history == "expanding"
    root = Path(__file__).resolve().parents[1]
    for filename in ("fd-airquality.toml", "experiments/smoke.toml"):
        config = tomllib.loads((root / filename).read_text())
        assert not {"arrival-history", "arrival-history-hours"}.intersection(config["run"])
        assert config["prepare-backtest"]["train-history"] == "expanding"
    flower = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    assert not {"arrival-history", "arrival-history-hours"}.intersection(flower)
    for name in removed:
        with pytest.raises(ValueError, match="desconocidas"):
            RunSettings.from_mapping({name: "expanding" if name == "arrival_history" else 24})


@pytest.mark.parametrize("text", [
    '[prepare]\npollutant = "NO2"\n',
    '[run]\nbatch-size = -1\n',
    'invalid TOML [',
])
def test_default_config_fixture_excludes_working_directory_experiments(tmp_path, monkeypatch, text):
    user_directory = tmp_path / "user-project"
    user_directory.mkdir()
    (user_directory / "fd-airquality.toml").write_text(text)
    monkeypatch.chdir(user_directory)
    assert arguments(["prepare"]).pollutant == "PM25"
    assert arguments(["simulate"]).batch_size == RunSettings().batch_size


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
        '[backtest]\nmodels = ["mlp", "autoencoder"]\nseeds = [1, 2]\n'
    )
    assert arguments(["prepare"]).station == ["old-a", "old-b"]
    assert arguments(["prepare", "--station", "new"]).station == ["new"]
    args = arguments(["backtest", "--models", "nlinear", "--seeds", "42"])
    assert (args.models, args.seeds) == (["nlinear"], [42])


@pytest.mark.parametrize("text, diagnostic", [
    ('[unknown]\nseed = 1', r"Secciones desconocidas .*unknown"),
    ('[benchmark]\nseeds = [7]', r"\[benchmark\].*Traslada sus opciones a \[backtest\]"),
    ('[prepare-backtest]\nstep-hours = 48', r"\[prepare-backtest\].step-hours.*datos observados"),
    ('[prepare-backtest]\nfold-gap-hours = 0', r"\[prepare-backtest\].fold-gap-hours.*datos observados"),
    ('[prepare-backtest]\ntest-end = "2026-04-26"', r"\[prepare-backtest\].test-end.*datos observados"),
    ('[run]\nsead = 1', r"Opción desconocida .*\[run\]\.sead"),
    ('[run]\nworkers = 1', r"Opción desconocida .*\[run\]\.workers"),
    ('[simulate]\nworkers = 2', r"Opción desconocida .*\[simulate\]\.workers"),
    ('[simulate]\nclient-gpus = 1.0', r"Opción desconocida .*\[simulate\]\.client-gpus"),
    ('[backtest]\nworkers = 2', r"Opción desconocida .*\[backtest\]\.workers"),
    ('[backtest]\nclient-gpus = 1.0', r"Opción desconocida .*\[backtest\]\.client-gpus"),
    ('[run]\nbatch-size = "32"', r"\[run\]\.batch-size: tipo inválido"),
    ('[run]\nseed = true', r"\[run\]\.seed: tipo inválido"),
    ('[run]\nbatch-size = 1.5', r"\[run\]\.batch-size: tipo inválido"),
    ('[backtest]\nmodels = "mlp"', r"\[backtest\]\.models: se requiere una lista"),
    ('[backtest]\nmodels = ["unknown"]', r"\[backtest\]\.models: 'unknown' no está entre"),
    ('[backtest]\nseeds = []', r"\[backtest\]\.seeds: se requiere una lista no vacía"),
    ('[simulate]\nprint-command = 1', r"\[simulate\]\.print-command: tipo inválido"),
    ('[run]\nlearning-rate = true', r"\[run\]\.learning-rate: tipo inválido"),
    ('[prepare]\nraw-root = 1', r"\[prepare\]\.raw-root: tipo inválido"),
    ('[run]\nlora-rank = 4\nlora_rank = 8', r"Opción duplicada .*\[run\]\.lora_rank"),
    ('run = 1', r"\[run\] debe ser una tabla TOML"),
    ('[run', "TOML inválido"),
])
def test_invalid_toml_is_rejected_with_specific_diagnostic(tmp_path, text, diagnostic):
    config = tmp_path / "invalid.toml"
    config.write_text(text)
    with pytest.raises(ValueError, match=diagnostic):
        arguments(["simulate", "--config", str(config)])


def test_main_reports_configuration_error_before_resolving_manifest(tmp_path, monkeypatch, capsys):
    (tmp_path / "fd-airquality.toml").write_text("[run]\nsead = 1")
    monkeypatch.setattr("fd_airquality.cli.resolve_manifest",
                        lambda *args, **kwargs: pytest.fail("El TOML inválido debe abortar antes del manifest"))
    with pytest.raises(SystemExit) as exc:
        main(["simulate", "--print-command"])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert error.startswith("Error: Opción desconocida")
    assert "[run].sead" in error


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


def test_main_passes_merged_configuration_to_the_launcher(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text('{"pollutant": "PM25"}')  # Only manifest resolution is real.
    config = tmp_path / "experiment.toml"
    config.write_text('[run]\nseed = 17\nnum-server-rounds = 9\n'
                      '[simulate]\nprint-command = true\n')
    received = []
    monkeypatch.setattr("fd_airquality.cli.launch_simulation", lambda *args: received.append(args))
    main(["simulate", "--config", str(config), "--manifest", str(path), "--num-server-rounds", "1"])
    (settings, dry_run), = received
    assert (settings.seed, settings.num_server_rounds, settings.manifest) == (17, 1, str(path))
    assert dry_run is True


def test_main_passes_file_geometry_and_cli_station_to_preparation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "prepared"
    config = tmp_path / "experiment.toml"
    config.write_text('[prepare]\ncontext = 16\nhorizon = 4\nstation = ["old"]\n'
                      'raw-root = "raw"\npollutant = "PM25"\n')
    received = []

    def prepare(*args):
        received.append(args)
        return tmp_path / "manifest.json"

    monkeypatch.setattr("fd_airquality.cli.prepare", prepare)
    monkeypatch.setattr("fd_airquality.cli._inspect", lambda *_: None)
    main(["prepare", "--config", str(config), "--station", "new", "--stride", "2", "--output", str(output)])
    (raw_root, prepared_output, pollutant, geometry, stations), = received
    assert (geometry.context, geometry.horizon, geometry.stride) == (16, 4, 2)
    assert (raw_root, prepared_output, pollutant, stations) == (tmp_path / "raw", output, "PM25", {"new"})


def test_logging_and_parallelism_options_respect_toml_and_cli_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fd-airquality.toml").write_text(
        '[run]\nheartbeat-seconds = 15\nlog-level = "DEBUG"\n'
        '[backtest]\nmax-parallel-models = 2\n'
        'modes = ["centralized", "local", "zero-shot", "federated"]\n'
        'lora-ranks = [4, 8]\nstrategies = ["fedprox", "fedavg"]\n'
    )
    args = arguments(["backtest", "--max-parallel-models", "1", "--heartbeat-seconds", "30"])
    assert (args.max_parallel_models, args.heartbeat_seconds, args.log_level) == (1, 30.0, "DEBUG")
    assert args.modes == ["centralized", "local", "zero-shot", "federated"]
    assert args.lora_ranks == [4, 8]
    assert args.strategies == ["fedprox", "fedavg"]
