"""Runtime station eligibility shared by every candidate in an experiment."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import replace
from pathlib import Path

from fd_airquality.artifacts import file_digest
from fd_airquality.data import read_manifest
from fd_airquality.settings import RunSettings

LOGGER = logging.getLogger(__name__)
EXCLUSION_COLUMNS = (
    "station", "source_partition_id", "reason", "detail", "selection_metric",
    "reference_start", "reference_end", "reference_pairs",
)


def selection_exclusions(manifest: dict, selection_metric: str) -> list[dict]:
    if manifest["schema_version"] < 4 or selection_metric == "mae":
        return []
    metric = selection_metric.removeprefix("macro-")
    rows = []
    for client in manifest["clients"]:
        if client[f"validation_{metric}_defined"]:
            continue
        no_pairs = client["validation_scale_pairs"] == 0
        rows.append({
            "station": client["station"],
            "source_partition_id": client.get("source_partition_id", client["partition_id"]),
            "reason": "validation_reference_no_pairs" if no_pairs else f"validation_{metric}_scale_zero",
            "detail": ("Sin pares observados en la referencia de validación."
                       if no_pairs else f"Escala {metric.upper()} de validación nula."),
            "selection_metric": selection_metric,
            "reference_start": client["validation_scale_start"],
            "reference_end": client["validation_scale_end"],
            "reference_pairs": client["validation_scale_pairs"],
        })
    return rows


def write_exclusions(manifest: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "excluded_stations.csv"
    rows = manifest.get("runtime_selection", {}).get("excluded_stations", [])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXCLUSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        LOGGER.warning("Estación descartada: %s; %s (%s); referencia [%s, %s), pares=%s.",
                       row["station"], row["detail"], row["reason"],
                       row["reference_start"], row["reference_end"], row["reference_pairs"])
    if rows:
        LOGGER.warning("Panel efectivo: %d estaciones; %d descartadas. Informe: %s",
                       len(manifest["clients"]), len(rows), path)
        if manifest["clients"]:
            (output / "runtime_manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
    return path


def prepare_runtime_panel(settings: RunSettings, output: Path) -> tuple[RunSettings, dict]:
    manifest = read_manifest(settings.manifest)
    previous = manifest.get("runtime_selection")
    if previous and previous["selection_metric"] != settings.selection_metric:
        raise ValueError("El panel ya está filtrado por otra métrica; usa el manifest original de prepare.")
    exclusions = selection_exclusions(manifest, settings.selection_metric)
    if exclusions:
        excluded = {row["source_partition_id"] for row in exclusions}
        retained = [client for client in manifest["clients"]
                    if client.get("source_partition_id", client["partition_id"]) not in excluded]
        manifest = manifest | {
            "clients": [client | {"source_partition_id": client.get("source_partition_id", client["partition_id"]),
                                  "partition_id": partition}
                        for partition, client in enumerate(retained)],
            "runtime_selection": (previous or {
                "source_manifest": str(Path(settings.manifest).resolve()),
                "source_manifest_sha256": file_digest(Path(settings.manifest)),
                "selection_metric": settings.selection_metric,
            }) | {"excluded_stations": (previous or {}).get("excluded_stations", []) + exclusions},
        }
    report = write_exclusions(manifest, output)
    if not manifest["clients"]:
        raise ValueError(f"Ninguna estación utilizable para {settings.selection_metric}. Consulta {report}")
    if exclusions:
        path = output.resolve() / "runtime_manifest.json"
        settings = replace(settings, manifest=str(path))
    return settings, manifest
