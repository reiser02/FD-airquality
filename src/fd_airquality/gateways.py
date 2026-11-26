"""Reproducible geographic topology for Cartagena's municipal PM2.5 network."""

from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path

import numpy as np

STATION_LOCATIONS = Path(__file__).with_name("station_locations.csv")
TOPOLOGY_FILENAME = "gateway_topology.json"
CATALOGUE_SOURCE = "airquality-tfg/Datos_nuevos/station_coords.txt"
EARTH_RADIUS_M = 6_371_000


def _station_locations() -> dict[str, tuple[float, float]]:
    locations = {}
    with STATION_LOCATIONS.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["station"]
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not name or name in locations or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"Nombre repetido o coordenadas geográficas inválidas: {name!r}.")
            locations[name] = (lat, lon)
    return locations


def _haversine(coordinates: list[tuple[float, float]]) -> np.ndarray:
    lat, lon = np.radians(coordinates).T
    a = (np.sin((lat[:, None] - lat) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat) * np.sin((lon[:, None] - lon) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _k_medoids(distances: np.ndarray, count: int, seed: int) -> tuple[int, ...]:
    """PAM swaps from seeded starts; ties use the canonical station order."""
    rng = np.random.default_rng(seed)
    indices = set(range(len(distances)))
    best = (float("inf"), ())
    for _ in range(16):
        medoids = tuple(sorted(rng.choice(len(distances), size=count, replace=False).tolist()))
        score = float(distances[list(medoids)].min(axis=0).sum())
        while True:
            candidate = (score, medoids)
            current = set(medoids)
            for outgoing in medoids:
                retained = current - {outgoing}
                for incoming in sorted(indices - current):
                    swapped = tuple(sorted(retained | {incoming}))
                    total = float(distances[list(swapped)].min(axis=0).sum())
                    candidate = min(candidate, (total, swapped))
            if candidate[0] >= score:
                break
            score, medoids = candidate
        best = min(best, (score, medoids))
    return best[1]


def build_gateway_topology(num_clients: int, seed: int) -> dict:
    locations = _station_locations()
    if type(num_clients) is not int or not 1 <= num_clients <= len(locations):
        raise ValueError(f"El número de gateways debe estar entre 1 y {len(locations)}.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("La semilla de gateways debe estar entre 0 y 2**32 - 1.")
    names = sorted(locations)
    distances = _haversine([locations[name] for name in names])
    medoids = _k_medoids(distances, num_clients, seed)
    assignments = distances[list(medoids)].argmin(axis=0)
    # Each medoid serves its own station, including ties for co-located stations.
    assignments[list(medoids)] = np.arange(num_clients)
    return {
        "schema_version": 1,
        "algorithm": "k-medoids",
        "distance": "haversine",
        "catalogue_source": CATALOGUE_SOURCE,
        "seed": seed,
        "gateways": [names[index] for index in medoids],
        "stations": [
            {"station": name, "latitude": locations[name][0], "longitude": locations[name][1],
             "gateway_id": int(gateway), "distance_m": float(distances[medoids[gateway], index])}
            for index, (name, gateway) in enumerate(zip(names, assignments, strict=True))
        ],
    }


def write_gateway_topology(path: Path, num_clients: int, seed: int) -> dict:
    topology = build_gateway_topology(num_clients, seed)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(topology, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return topology


def read_gateway_topology(path: Path, *, num_clients: int | None = None,
                          seed: int | None = None) -> dict:
    topology = json.loads(path.read_text(encoding="utf-8"))
    try:
        if (topology["schema_version"] != 1 or topology["algorithm"] != "k-medoids"
                or topology["distance"] != "haversine"):
            raise ValueError("Esquema o algoritmo de topología de gateways incompatible.")
        medoids, stations, saved_seed = topology["gateways"], topology["stations"], topology["seed"]
        if (not isinstance(medoids, list) or not medoids
                or any(not isinstance(name, str) for name in medoids)
                or len(set(medoids)) != len(medoids)):
            raise ValueError("La topología debe contener gateways distintos.")
        if not isinstance(stations, list) or type(saved_seed) is not int or not 0 <= saved_seed < 2**32:
            raise ValueError("Inventario o semilla inválidos en la topología de gateways.")
        if ((num_clients is not None and len(medoids) != num_clients)
                or (seed is not None and saved_seed != seed)):
            raise ValueError("La topología guardada no coincide con el número de gateways o la semilla.")
        members = {}
        for row in stations:
            name, gateway = row["station"], row["gateway_id"]
            if (not isinstance(name, str) or not name or name in members
                    or type(gateway) is not int or not 0 <= gateway < len(medoids)
                    or not -90 <= row["latitude"] <= 90 or not -180 <= row["longitude"] <= 180
                    or not isfinite(row["distance_m"]) or row["distance_m"] < 0):
                raise ValueError(f"Asignación o coordenadas inválidas para {name!r} en la topología.")
            members[name] = gateway
        if any(members.get(name) != index for index, name in enumerate(medoids)):
            raise ValueError("Cada medoid debe estar asignado a su propio gateway.")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Topología de gateways inválida: {path}") from exc
    return topology


def gateway_partitions(topology: dict, manifest: dict) -> list[list[int]]:
    """Intersect the immutable topology with the effective station panel."""
    assignments = {row["station"]: row["gateway_id"] for row in topology["stations"]}
    unknown = sorted({client["station"] for client in manifest["clients"]} - assignments.keys())
    if unknown:
        raise ValueError(f"Estaciones sin coordenadas en la topología de gateways: {', '.join(unknown)}.")
    members = [[] for _ in topology["gateways"]]
    for client in manifest["clients"]:
        members[assignments[client["station"]]].append(client["partition_id"])
    return members
