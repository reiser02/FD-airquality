"""Overlay gateway assignments on a local OpenStreetMap extract of Cartagena."""

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patheffects import withStroke

WEB_MERCATOR_RADIUS_M = 6_378_137
CARTAGENA_MAP = Path(__file__).with_name("cartagena_osm.png")
# Geographic bounds of the bundled raster, in (west, south, east, north) order.
CARTAGENA_BOUNDS = (-1.002, 37.589, -0.966, 37.623)


def _web_mercator(latitude, longitude):
    x = WEB_MERCATOR_RADIUS_M * np.radians(longitude)
    y = WEB_MERCATOR_RADIUS_M * np.arcsinh(np.tan(np.radians(latitude)))
    return x, y


def plot_gateway_map(topology: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("El mapa de gateways requiere una salida .png, .pdf o .svg.")
    if output.exists():
        raise FileExistsError(f"La salida ya existe; indica otra ruta con --output: {output}")
    rows = sorted(topology["stations"], key=lambda row: (row["gateway_id"], row["station"]))
    medoids = topology["gateways"]
    colors = colormaps["tab20"].colors
    palette = [*colors[::2], *colors[1::2], *colormaps["tab20b"].colors]
    lat, lon = np.asarray([(row["latitude"], row["longitude"]) for row in rows]).T
    west, south, east, north = CARTAGENA_BOUNDS
    if ((lon < west) | (lon > east) | (lat < south) | (lat > north)).any():
        raise ValueError("La topología contiene estaciones fuera del recorte de Cartagena.")
    left, bottom = _web_mercator(south, west)
    right, top = _web_mercator(north, east)
    x, y = _web_mercator(lat, lon)
    positions = {
        row["station"]: (point_x, point_y)
        for row, point_x, point_y in zip(rows, x, y, strict=True)
    }
    margin = max(180., .20 * max(np.ptp(x), np.ptp(y)))

    figure = Figure(figsize=(12.5, max(9, .27 * (len(rows) + len(medoids)))), layout="constrained")
    FigureCanvasAgg(figure)
    axis, legend = figure.subplots(ncols=2, width_ratios=[2.2, 1])
    figure.suptitle("Estaciones y gateways de Cartagena", fontsize=19, weight="bold")
    axis.set_title(f"{len(medoids)} gateways · {len(rows)} estaciones · semilla {topology['seed']}")
    axis.imshow(imread(CARTAGENA_MAP), extent=(left, right, bottom, top), origin="upper", zorder=0)
    axis.set(xlim=(max(left, x.min() - margin), min(right, x.max() + margin)),
             ylim=(max(bottom, y.min() - margin), min(top, y.max() + margin)),
             aspect="equal", xticks=[], yticks=[])
    credits = axis.set_xlabel("© OpenStreetMap contributors · openstreetmap.org/copyright",
                              loc="left", fontsize=8)
    credits.set_url("https://www.openstreetmap.org/copyright")
    legend.axis("off")
    legend.set_title("Asignación de estaciones", loc="left")
    line = 0
    row_height = .98 / (len(rows) + len(medoids))
    for number, row in enumerate(rows, 1):
        group = row["gateway_id"]
        name, medoid = row["station"], medoids[group]
        color = palette[group % len(palette)]
        x, y = positions[name]
        if name != medoid:
            gx, gy = positions[medoid]
            axis.plot([x, gx], [y, gy], color=color, alpha=.8, lw=1.25, zorder=2)
        axis.scatter(x, y, s=250 if name == medoid else 155, marker="*" if name == medoid else "o",
                     c=[color], edgecolors="white", linewidths=1.1, zorder=3)
        axis.annotate(str(number), (x, y), ha="center", va="center", fontsize=8, zorder=4,
                      path_effects=[withStroke(linewidth=1.4, foreground="white")])
        if number == 1 or group != rows[number - 2]["gateway_id"]:
            legend.text(0, 1 - line * row_height, f"Gateway {group:02d}", weight="bold",
                        va="top", fontsize=10)
            legend.scatter([-.04], [1 - (line + .24) * row_height], marker="*", s=90,
                           c=[color], transform=legend.transAxes, clip_on=False)
            line += 1
        legend.text(0, 1 - line * row_height, f"{number:02d}  {name}", va="top", fontsize=9,
                    weight="bold" if name == medoid else "normal")
        line += 1
    legend.set(xlim=(0, 1), ylim=(0, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        figure.savefig(stream, format=output.suffix[1:].lower(), dpi=180, facecolor="white")
    return output
