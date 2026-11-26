import sqlite3
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import Context, RecordDict
from flwr.supercore.task_identity import TaskIdentity
from torch import nn

from fd_airquality.client_app import build_client_app
from fd_airquality.data import DataConfig, prepare, read_manifest
from fd_airquality.settings import GATEWAY_MODE


@pytest.fixture(autouse=True)
def isolated_default_config(tmp_path, monkeypatch):
    # Tests may create this controlled file, but never load the user's experiment.
    monkeypatch.setattr("fd_airquality.configuration.DEFAULT_CONFIG", tmp_path / "fd-airquality.toml")


@pytest.fixture(scope="session", autouse=True)
def cpu_test_environment():
    # CUDA behavior is mocked explicitly in unit tests. In particular, libraries must
    # not initialize every physical GPU just to preserve RNG state on CPU.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        patch.setenv("OMP_NUM_THREADS", "1")
        patch.setenv("MKL_NUM_THREADS", "1")
        yield


@pytest.fixture(scope="session")
def mlflow_schema(tmp_path_factory):
    from mlflow import MlflowClient

    path = tmp_path_factory.mktemp("mlflow-schema") / "mlflow.db"
    # Run the real migrations once; each tracking test gets an isolated database.
    MlflowClient(tracking_uri=f"sqlite:///{path}").search_experiments()
    return path


@pytest.fixture
def mlflow_uri(mlflow_schema, tmp_path):
    path = tmp_path / "mlflow.db"
    with closing(sqlite3.connect(mlflow_schema)) as source, closing(sqlite3.connect(path)) as target:
        source.backup(target)
    return f"sqlite:///{path}"


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
        self.proj = nn.Linear(8, 8, bias=False)
        nn.init.eye_(self.proj.weight)
        self.calls = []
        self.config = SimpleNamespace(context_length=90, prediction_length=30, decoder_patch_len=24,
                                      quantile_levels=[.1, .5, .9])

    def forward(self, past_values, **kwargs):
        self.calls.append(kwargs)
        horizon = kwargs.get("prediction_length", 30)
        value = self.proj(past_values[:, -1:, :].repeat(1, 1, 8)).mean(dim=-1, keepdim=True)
        prediction = value.repeat(1, horizon, 1)
        quantiles = torch.stack([prediction - 1, prediction, prediction + 1], dim=1)
        if "prediction_length" in kwargs and kwargs.get("prediction_type") is None:
            # PatchTST's point prediction is detached, unlike its quantiles.
            prediction = prediction.detach()
        return SimpleNamespace(prediction_outputs=prediction, quantile_outputs=quantiles)


@pytest.fixture
def granite_template():
    return DummyBackbone()


@pytest.fixture
def mock_granite(monkeypatch, granite_template):
    monkeypatch.setattr("fd_airquality.models._load_granite", lambda *_: granite_template)
    return granite_template


@pytest.fixture
def flower_identity(monkeypatch):
    # Normally supplied by the Flower runtime, which the in-process transport replaces.
    for field, value in (("_run_id", 1), ("_task_id", 1), ("_node_id", 0)):
        monkeypatch.setattr(TaskIdentity, field, value)


class LocalTestGrid:
    """Transport-only double; Flower still performs scheduling and aggregation."""

    def __init__(self, settings, count):
        self.app = build_client_app()
        self.contexts = {
            node + 10: Context(
                run_id=1, node_id=node + 10,
                node_config={"partition-id": node, "num-partitions": count},
                state=RecordDict(), run_config=settings.flower_config(),
            ) for node in range(count)
        }
        self.requests = []

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, timeout=None):
        replies = []
        for message in messages:
            self.requests.append((message.metadata.message_type, dict(message.content["config"])))
            replies.append(self.app(message, self.contexts[message.metadata.dst_node_id]))
        return replies


@pytest.fixture
def flower_grid(flower_identity):
    return LocalTestGrid


@pytest.fixture
def in_process_federation(monkeypatch, flower_grid):
    """Backtest with real client/server logic but no Ray startup."""
    from fd_airquality.server_app import run_server

    def launch(settings, dry_run=False):
        assert not dry_run
        count = (settings.gateway_num_clients if settings.federation_mode == GATEWAY_MODE
                 else len(read_manifest(settings.manifest)["clients"]))
        return run_server(flower_grid(settings, count), settings)

    monkeypatch.setattr("fd_airquality.cli.launch_simulation", launch)


@pytest.fixture
def geometry():
    return DataConfig(context=6, horizon=2, stride=1, validation_hours=4, test_hours=6, seasonality=3,
                      split_mode="station", quality_mode="retrospective")


@pytest.fixture
def fragmented_series():
    index = pd.date_range("2025-01-01", periods=150, freq="h")
    values = 10 + np.sin(np.arange(len(index)) / 4) + np.arange(len(index)) / 100
    values[22:27] = np.nan
    values[62:67] = np.nan
    return pd.Series(values, index=index, name="station")


@pytest.fixture
def prepared(tmp_path, geometry):
    raw = tmp_path / "raw"
    for name, hours in (("station-a", 160), ("station-b", 185), ("too-short", 5)):
        directory = raw / name
        directory.mkdir(parents=True)
        index = pd.date_range("2025-01-01", periods=hours * 12, freq="5min")
        values = 10 + np.sin(np.arange(len(index)) / 30) + np.arange(len(index)) / 10000
        if hours > 10:
            values[20 * 12 : 25 * 12] = np.nan
            values[60 * 12 : 65 * 12] = np.nan
        pd.DataFrame({"PM25": values}, index=index).to_csv(directory / f"{name}_PM25.csv")
    path = prepare(raw, tmp_path / "prepared", "PM25", geometry)
    return path, read_manifest(path)


@pytest.fixture
def global_prepared(tmp_path):
    raw = tmp_path / "global_raw"
    for name, hours in (("station-a", 500), ("station-b", 520)):
        folder = raw / name
        folder.mkdir(parents=True)
        values = 10 + np.sin(np.arange(hours * 12) / 31) + np.arange(hours * 12) / 10000
        if name == "station-a":
            values[20 * 12:25 * 12] = np.nan
            values[60 * 12:65 * 12] = np.nan
        pd.DataFrame({"PM25": values}, index=pd.date_range("2025-01-01", periods=len(values), freq="5min")).to_csv(folder / f"{name}_PM25.csv")
    config = DataConfig(validation_start="2025-01-10", test_start="2025-01-12", test_end="2025-01-16")
    path = prepare(raw, tmp_path / "global_prepared", "PM25", config)
    return path, read_manifest(path)


@pytest.fixture
def selection_prepared(tmp_path, request):
    raw = tmp_path / "selection_raw"
    for offset, name in enumerate(getattr(request, "param", ("station-a", "station-b"))):
        folder = raw / name
        folder.mkdir(parents=True)
        index = pd.date_range("2025-01-01", periods=300 * 12, freq="5min")
        values = 10 + offset + np.sin(np.arange(len(index)) / 31) + np.arange(len(index)) / 10000
        if name == "station-a":
            values[96 * 12:144 * 12] = np.nan
        pd.DataFrame({"PM25": values}, index=index).to_csv(folder / f"{name}_PM25.csv")
    config = DataConfig(context=6, horizon=2, stride=2, error_scale_hours=48,
                        validation_start="2025-01-07", test_start="2025-01-09", test_end="2025-01-11")
    path = prepare(raw, tmp_path / "selection_prepared", "PM25", config)
    return path, read_manifest(path)


@pytest.fixture
def arrival_raw(tmp_path):
    raw = tmp_path / "arrival_raw"
    for offset, station in enumerate(("a", "b")):
        folder = raw / station
        folder.mkdir(parents=True)
        values = 10 + offset + np.sin(np.arange(14 * 24 * 12) / 23)
        values.reshape(-1, 12)[offset::11] = np.nan
        pd.DataFrame({"PM25": values}, index=pd.date_range(
            "2025-01-01", periods=len(values), freq="5min",
        )).to_csv(folder / "PM25.csv")
    config = DataConfig(context=6, horizon=3, stride=3, train_stride=3,
                        window_policy="masked", seasonality=3, hourly_repeat_filter=0,
                        min_validation_windows=2, min_test_windows=2,
                        validation_start="2025-01-09", test_start="2025-01-11", test_end="2025-01-13")
    return raw, config


@pytest.fixture
def arrival_prepared(arrival_raw, tmp_path):
    raw, config = arrival_raw
    path = prepare(raw, tmp_path / "prepared", "PM25", config)
    return path, read_manifest(path)


@pytest.fixture
def temporal_family(tmp_path):
    raw = tmp_path / "raw"
    for offset, name in enumerate(("station-a", "station-b")):
        directory = raw / name
        directory.mkdir(parents=True)
        index = pd.date_range("2024-12-28", "2025-01-21", freq="5min", inclusive="left")
        values = 10 + offset + np.sin(np.arange(len(index)) / 23)
        values.reshape(-1, 12)[offset::13] = np.nan
        pd.DataFrame({"PM25": values}, index=index).to_csv(directory / "PM25.csv")
    config = DataConfig(context=6, horizon=3, stride=3, seasonality=3, window_policy="masked",
                        hourly_repeat_filter=0, min_validation_windows=2, min_test_windows=2,
                        validation_start="2025-01-10", test_start="2025-01-12", test_end="2025-01-14",
                        max_train_windows=12)
    return raw, config


@pytest.fixture
def single_fold_manifest(temporal_family, tmp_path):
    from fd_airquality.backtesting import prepare_backtest

    raw, config = temporal_family
    return prepare_backtest(raw, tmp_path / "backtests", "PM25", config, folds=1, min_stations=2)
