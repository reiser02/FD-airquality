import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from fd_airquality.artifacts import file_digest
from fd_airquality.backtesting import prepare_backtest, run_backtest
from fd_airquality.cli import main
from fd_airquality.gateways import (
    TOPOLOGY_FILENAME,
    build_gateway_topology,
    gateway_partitions,
    write_gateway_topology,
)
from fd_airquality.settings import GATEWAY_MODE, RunSettings


def test_gateway_map_uses_the_configured_or_saved_cartagena_topology(tmp_path):
    config = tmp_path / "gateways.toml"
    config.write_text("[gateways]\nnum-clients = 4\nseed = 42\n", encoding="utf-8")
    path = tmp_path / TOPOLOGY_FILENAME
    topology = write_gateway_topology(path, 4, 42)
    assert topology == build_gateway_topology(4, 42)
    assert len(topology["stations"]) == 25
    sizes = Counter(row["gateway_id"] for row in topology["stations"])
    assert sorted(sizes.values()) == [4, 4, 8, 9]
    assert gateway_partitions(topology, {"clients": []}) == [[], [], [], []]
    with pytest.raises(ValueError, match="Estaciones sin coordenadas.*otra"):
        gateway_partitions(topology, {"clients": [{"partition_id": 0, "station": "otra"}]})

    configured = tmp_path / "configured.png"
    saved = tmp_path / "saved.png"
    main(["plot-gateways", "--config", str(config), "--output", str(configured)])
    main(["plot-gateways", "--topology", str(path), "--output", str(saved)])
    assert configured.read_bytes().startswith(b"\x89PNG")
    assert configured.read_bytes() == saved.read_bytes()


def test_gateway_backtest_keeps_the_topology_and_station_support(
    temporal_family, in_process_federation, tmp_path, monkeypatch,
):
    raw, config = temporal_family
    (raw / "station-c").mkdir()
    shutil.copyfile(raw / "station-b" / "PM25.csv", raw / "station-c" / "PM25.csv")
    monkeypatch.setattr("fd_airquality.gateways._station_locations", lambda: {
        "station-a": (37.600, -.99), "station-b": (37.601, -.99),
        "station-c": (37.600, -1.02), "offline": (37.620, -.95),
    })
    manifest = prepare_backtest(raw, tmp_path / "prepared", "PM25", config, folds=2,
                                min_stations=3, validation_hours=48, test_hours=48)
    settings = RunSettings(
        manifest=str(manifest), output_dir=str(tmp_path / "results"), mlflow_tracking_uri="disabled",
        gateway_num_clients=3, batch_size=5, pad_last_batch=1, num_server_rounds=2, local_epochs=2,
        round_validation_fallback_epochs=1, arrival_period_hours=24, normalization="fixed",
        probabilistic=1, calibrate_point_forecasts=1, calibration_min_pairs=2,
    )
    modes = ["local", "centralized", "federated", GATEWAY_MODE]
    output = run_backtest(settings, ["autoencoder-tiny"], modes)
    topology = json.loads((output / TOPOLOGY_FILENAME).read_text())
    assert "offline" in topology["gateways"]
    topology_hash = file_digest(output / TOPOLOGY_FILENAME)
    comparison = pd.read_csv(output / "temporal_comparison.csv")
    assert comparison.groupby("fold")["mode"].agg(list).tolist() == [modes, modes]
    assert comparison.groupby("fold")["num-targets"].nunique().tolist() == [1, 1]

    for fold in (1, 2):
        cases = json.loads((output / f"fold_{fold:03d}" / "candidates.json").read_text())
        run = Path(cases[-1]["output"])
        assert file_digest(run / TOPOLOGY_FILENAME) == topology_hash
        assert json.loads((run / "run.json").read_text())["gateway_topology_sha256"] == topology_hash
        evaluation = json.loads((run / "test_metrics.json").read_text())["test"]
        assert (evaluation["num-clients"], evaluation["num-stations"], evaluation["active-gateways"]) == (3, 3, 2)
        gateways = pd.read_csv(run / "gateway_metrics.csv").query("split == 'train'")
        idle = gateways.query("`num-stations` == 0")
        assert len(idle) == 2 and not idle["uplink-tensor-bytes"].any()
        assert not idle["downlink-tensor-bytes"].any()
        updates = gateways.query("round == 2 and `num-stations` > 0")
        assert len(updates) == 2
        assert updates["round-validation-fallback"].eq(1).all()
        assert updates["round-validation-windows"].eq(0).all()
        assert updates["epochs-completed"].eq(1).all()
        stations = pd.read_csv(run / "station_metrics.csv")
        expected = {row["station"]: row["gateway_id"] for row in topology["stations"]}
        assert stations["station"].map(expected).eq(stations["gateway-id"]).all()
        assert stations.query("split == 'train'")["num-examples"].le(config.max_train_windows).all()
        assert evaluation["macro-mase"] == pytest.approx(stations.query("split == 'test'")["mase"].mean())
