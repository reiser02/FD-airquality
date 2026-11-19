"""Bounded causal-control validation; the deliberately huge min_delta forces stopping."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from validate_neuralforecast import run_cli

from fd_airquality.arrival import round_validation_bounds
from fd_airquality.artifacts import create_run_dir, file_digest
from fd_airquality.settings import RunSettings


@contextmanager
def isolated_superlink(root: Path):
    """Do not submit work to another conversation's already running :local: server."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    key = "FLWR_LOCAL_SUPERLINK_HTTP_API_PORT"
    previous = os.environ.get(key)
    os.environ[key] = str(port)
    env = os.environ.copy()
    env.update(FLWR_HOME=str(Path(".flwr").resolve()), CUDA_VISIBLE_DEVICES="",
               FLWR_TELEMETRY_ENABLED="0", FLWR_DISABLE_UPDATE_CHECK="1",
               RAY_TMPDIR=f"/tmp/opencode/fd-controls-{os.getpid()}",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [
        str(Path(sys.executable).with_name("flower-superlink")),
        "--insecure", "--simulation", "--isolation", "subprocess",
        "--disable-runtime-dependency-installation", "--host", "127.0.0.1",
        "--port", str(port), "--database", str(root / "superlink.db"),
    ]
    with (root / "superlink.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"El SuperLink aislado terminó: consulta {root / 'superlink.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("El SuperLink aislado no respondió al arranque.")
                time.sleep(.1)
            yield {"port": port, "frozen_environment": True}
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def audit(output: Path) -> dict:
    progress = json.loads((output / "backtest.json").read_text())
    assert progress["stage"] == "evaluation"
    assert progress["final_holdout_fold"] is None
    assert not progress["configuration_selection"]
    assert not (output / "selection.json").exists()
    frame = pd.read_csv(output / "temporal_comparison.csv")
    assert len(frame) == 24
    means = pd.read_csv(output / "cv_fold_mean.csv")
    assert len(means) == 12 and means["num-folds"].eq(2).all()
    runs, streams = [], []
    for entry in progress["results"]:
        comparison = Path(entry["comparison"])
        for path in json.loads((comparison / "comparison.json").read_text())["runs"]:
            path = Path(path)
            run = json.loads((path / "run.json").read_text())
            package = Path(__file__).resolve().parents[1] / "src" / "fd_airquality"
            assert run["application_sources"] == {
                source.name: file_digest(source) for source in package.glob("*.py")
            }
            for dependency in ("torch", "neuralforecast", "flwr", "ray"):
                assert run["versions"][dependency] == version(dependency)
            settings = RunSettings(**run["settings"])
            metrics = pd.read_csv(path / ("round_metrics.csv" if run["mode"] == "centralized"
                                          else "station_metrics.csv"))
            train = metrics.loc[(metrics.split == "train") & (metrics["num-examples"] > 0)]
            assert train["epochs-completed"].eq(2).all()
            assert train["early-stopped"].gt(0).all()
            assert train["learning-rate-reductions"].eq(2).all()
            assert set(train["round"]) == {1, 2, 3}
            if run["mode"] != "centralized":
                assert train["learning-rate-final"].eq(settings.learning_rate / 4).all()
            assert (path / "test_metrics.json").is_file()
            if run["mode"] == "federated":
                assert (path / "flower.log").is_file()
                lower, upper = round_validation_bounds(run["manifest"]["geometry"], settings)
                for station, local in train.groupby("station"):
                    local = local.sort_values("round")
                    assert (local.iloc[1:]["train-first-target-ns"] >= local.iloc[1:]["available-from-ns"]).all()
                    assert local.iloc[0]["train-last-target-ns"] < lower.value
                    assert local.iloc[0]["available-through-ns"] == upper.value
                    streams.append({"station": station, "fold": entry["fold"],
                                    "windows": local["num-examples"].astype(int).tolist()})
            runs.append({"run": str(path), "mode": run["mode"], "model": settings.model})
    assert all(np.isfinite(means[name]).all() for name in ("mae", "rmse"))
    return {"runs": runs, "federated_streams": streams, "fold_mean_rows": len(means)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    args = parser.parse_args()
    root = create_run_dir(Path("reports/training_controls_validation").resolve(),
                          datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    config = "experiments/training-controls-smoke.toml"
    report_path = root / "validation.json"
    report = {"status": "running", "purpose": "bounded_forced_plateau_not_predictive_comparison"}
    report_path.write_text(json.dumps(report))
    try:
        run_cli(root, "prepare", "prepare-backtest", "--config", config,
                "--raw-root", str(args.raw_root.resolve()), "--output", str(root / "prepared"))
        family, = (root / "prepared").glob("*/backtest_manifest.json")
        with isolated_superlink(root) as runtime:
            report["runtime"] = runtime
            run_cli(root, "training_controls", "backtest", "--config", config,
                    "--manifest", str(family), "--output-dir", str(root / "results"))
        output, = (root / "results").glob("backtest_*")
        report.update(audit(output), status="passed", manifest=str(family), output=str(output))
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Validación de controles: {report_path}", flush=True)


if __name__ == "__main__":
    main()
