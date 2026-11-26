"""Weekly test comparisons from sufficient statistics, not raw forecasts."""

from __future__ import annotations

import html
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from fd_airquality.settings import FEDERATED_MODES, ZERO_SHOT

WEEKLY_METRICS = {
    "macro-mase": ("MASE macro", "mase-scaled-error-sum", "mase-target-count", "mase-count"),
    "macro-rmsse": ("RMSSE macro", "rmsse-scaled-error-sum", "rmsse-target-count", "rmsse-count"),
    "crps": ("CRPS (µg/m³)", "crps-error-sum", "crps-target-count", None),
    "macro-scaled-crps": ("CRPS escalado macro", "scaled-crps-error-sum", "scaled-crps-target-count",
                          "scaled-crps-eligible"),
}
WEEKLY_IDENTITY = ("run", "candidate_configuration", "temporal_configuration", "model", "mode",
                   "adaptation", "strategy", "seed", "lora_rank", "quantile_levels")
WEEKLY_COLUMNS = (*WEEKLY_IDENTITY, "week_start", "metric", "value",
                  "eligible_stations", "expected_targets", "scored_targets", "support_sha256")


def _daily_sum(stations: pd.DataFrame, days: list[str], suffix: str) -> np.ndarray:
    keys = [f"block-{day}-{suffix}" for day in days]
    return stations.reindex(columns=keys).fillna(0).to_numpy(dtype=float).sum(axis=1)


def station_weekly_scores(stations: pd.DataFrame) -> list[dict]:
    """Reduce by origin week; apply scale/support rules station by station first."""
    suffix = "-observed-target-count"
    weeks: dict[str, list[str]] = {}
    for key in sorted(stations):
        if key.startswith("block-") and key.endswith(suffix):
            day = key.removeprefix("block-").removesuffix(suffix)
            date = pd.to_datetime(day, format="%Y%m%d")
            week = (date - pd.Timedelta(days=date.dayofweek)).date().isoformat()
            weeks.setdefault(week, []).append(day)
    result = []
    stations = stations.sort_values("partition-id")
    for week, days in weeks.items():
        observed = _daily_sum(stations, days, "observed-target-count")
        for name, (_, error_key, count_key, eligible_column) in WEEKLY_METRICS.items():
            eligible = observed > 0
            if eligible_column:
                if eligible_column not in stations:
                    continue
                eligible &= stations[eligible_column].fillna(0).to_numpy() > 0
            if not eligible.any():
                continue
            expected = observed[eligible]
            sums = _daily_sum(stations, days, error_key)[eligible]
            counts = _daily_sum(stations, days, count_key)[eligible]
            if (counts > expected).any() or (counts < 0).any() or not np.isfinite(sums).all():
                raise ValueError("Estadísticas semanales incompatibles con los targets observados.")
            value = np.nan
            if np.array_equal(counts, expected):
                per_station = sums / counts
                if name == "macro-rmsse":
                    per_station = np.sqrt(per_station)
                value = float(sums.sum() / counts.sum() if name == "crps" else per_station.mean())
            support = np.column_stack((stations["partition-id"].to_numpy()[eligible], expected))
            result.append({"week_start": week, "metric": name, "value": value,
                           "eligible_stations": int(eligible.sum()), "expected_targets": int(expected.sum()),
                           "scored_targets": int(counts.sum()),
                           "support_sha256": sha256(support.astype("<i8").tobytes()).hexdigest()})
    return result


def rank_weeks(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean of all declared seeds, with a fixed field and common support per race."""
    multi_fold = "fold" in data
    id_col = "temporal_configuration" if multi_fold else "candidate_configuration"
    frame = data.copy()
    frame["fold"] = frame["fold"].astype(int) if multi_fold else 0
    expected_seeds = frame.groupby(id_col, sort=False)["seed"].agg(lambda seeds: frozenset(seeds.dropna()))
    periods = sorted(set(zip(frame["week_start"], frame["fold"], strict=True)))
    period_names = {key: (f"F{key[1]} · {key[0]}" if multi_fold else key[0]) for key in periods}
    configs = frame.groupby(id_col, sort=False).first()
    names = {}
    for config, row in configs.iterrows():
        details = [str(row["model"]), str(row["mode"])]
        if row["adaptation"] == ZERO_SHOT:
            details.append(ZERO_SHOT)
        elif row["adaptation"] not in {"full", "none"}:
            details += [f"{row['adaptation']} r{row['lora_rank']:g}"]
        if row["mode"] in FEDERATED_MODES and row["adaptation"] != ZERO_SHOT:
            details.append(str(row["strategy"]))
        names[config] = " | ".join(details)
    repeated = Counter(names.values())
    names = {key: f"{label} [{key[:8]}]" if repeated[label] > 1 else label for key, label in names.items()}
    scores = []
    for (metric, config, week, fold), group in frame.groupby(
        ["metric", id_col, "week_start", "fold"], sort=False,
    ):
        present = frozenset(group["seed"].dropna())
        complete = (present == expected_seeds[config] and group["value"].notna().all()
                    and group["support_sha256"].nunique() == 1
                    and group["scored_targets"].eq(group["expected_targets"]).all())
        # Duplicate executions of a seed do not count as additional seeds.
        value = group.groupby("seed", dropna=False)["value"].mean().mean() if complete else np.nan
        scores.append({
            "metric": metric, "configuration": config, "candidate": names[config],
            "origin_week": week, "fold": fold, "period": period_names[week, fold],
            "value": value, "num_seeds": len(present), "expected_seeds": len(expected_seeds[config]),
            "expected_targets": int(group["expected_targets"].iloc[0]),
            "eligible_stations": int(group["eligible_stations"].iloc[0]),
            "support_sha256": group["support_sha256"].iloc[0] if complete else None,
        })
    board = pd.DataFrame(scores).sort_values(["origin_week", "fold", "candidate", "metric"]).reset_index(drop=True)
    for column in ("rank", "points", "cumulative_points"):
        board[column] = np.nan
    for _, metric in board.groupby("metric", sort=False):
        # Incomplete configurations remain visible in CSV but never enter the points race.
        full = metric.groupby("configuration")["value"].agg(["count", "size"])
        field = full.index[(full["count"] == len(periods)) & (full["size"] == len(periods))]
        if len(field) < 2:
            continue
        eligible = metric.loc[metric["configuration"].isin(field)]
        for _, comparable in eligible.groupby(["origin_week", "fold"], sort=False):
            if comparable["support_sha256"].nunique() == 1:
                ranks = comparable["value"].rank(method="average")
                board.loc[ranks.index, "rank"] = ranks
                board.loc[ranks.index, "points"] = len(field) + 1 - ranks
    ranked = board["rank"].notna()
    board.loc[ranked, "cumulative_points"] = board.loc[ranked].groupby(
        ["metric", "configuration"], sort=False,
    )["points"].cumsum()
    standings = (board.loc[ranked].groupby(["metric", "configuration", "candidate"], as_index=False)
                 .agg(total_points=("points", "sum"), mean_rank=("rank", "mean"),
                      best_rank=("rank", "min"), worst_rank=("rank", "max"), weeks_ranked=("rank", "size"))
                 .sort_values(["metric", "total_points", "candidate"], ascending=[True, False, True]))
    return board, standings


def _figure(data: pd.DataFrame, metric: str, kind: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    labels = {"value": WEEKLY_METRICS[metric][0], "rank": "Posición semanal",
              "cumulative_points": "Puntos acumulados", "heatmap": "Estabilidad del ranking"}
    column = "rank" if kind == "heatmap" else kind
    matrix = data.pivot(index="candidate", columns="period", values=column).reindex(
        columns=data["period"].drop_duplicates(),
    ).dropna(how="all")
    if matrix.empty:
        return None
    figure = Figure(figsize=(13, max(4, len(matrix) * (.33 if kind == "heatmap" else .17) + 2.2)),
                    layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    caption = labels[kind]
    ax.set_title(f"{WEEKLY_METRICS[metric][0]} · {caption}" if kind != "value" else caption)
    columns = np.arange(len(matrix.columns))
    if kind == "heatmap":
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="Purples_r", aspect="auto", vmin=1,
                          vmax=max(2., float(np.nanmax(matrix))))
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=8)
        figure.colorbar(image, ax=ax, label="Posición (1 = mejor)")
    else:
        for index, (name, values) in enumerate(matrix.iterrows()):
            marker = "o" if kind != "rank" else "."
            ax.plot(columns, values, label=name, marker=marker, markersize=4,
                    linestyle=("-", "--", ":", "-.")[(index // 10) % 4])
        if kind == "rank":
            ax.invert_yaxis()
        ax.set_ylabel(caption)
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7, ncols=1 if len(matrix) < 20 else 2)
    ax.set_xticks(columns, matrix.columns, rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("Semana del origen de pronóstico (lunes)")
    return figure


def plot_weekly(source: Path, output: Path | None = None) -> tuple[Path, list[Path]]:
    source = source.expanduser().resolve()
    data = source if source.is_file() else source / "weekly_metrics.csv"
    if not data.is_file() and (source / "comparison" / "weekly_metrics.csv").is_file():
        data = source / "comparison" / "weekly_metrics.csv"
    frame = pd.read_csv(data)
    if frame.empty or not set(WEEKLY_COLUMNS) <= set(frame):
        raise ValueError("Sin estadísticas semanales; genera una comparación con resultados nuevos.")
    board, standings = rank_weeks(frame)
    output = (output or data.parent / "leaderboard").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = [output / "weekly_ranking.csv", output / "standings.csv"]
    board.to_csv(artifacts[0], index=False)
    standings.to_csv(artifacts[1], index=False)
    sections = []
    for name, rows in board.groupby("metric", sort=False):
        for kind in ("value", "rank", "heatmap", "cumulative_points"):
            figure = _figure(rows, name, kind)
            if figure is None:
                continue
            stem = f"{name}_{kind}"
            for suffix in ("png", "pdf"):
                path = output / f"{stem}.{suffix}"
                figure.savefig(path, dpi=145, facecolor="white")
                artifacts.append(path)
            figure.clear()
            sections.append(f'<p><a href="{stem}.pdf">PDF</a><br><img src="{stem}.png" '
                            f'alt="{html.escape(stem)}" loading="lazy"></p>')
    index = output / "index.html"
    index.write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>Clasificación semanal</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:18px}'
        'table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #ddd}img{max-width:100%}</style>'
        '<h1>Evaluación por semana de origen</h1><p>Media de todas las semillas por configuración. '
        'Solo se dan puntos (N + 1 − posición, rango medio en empates) con un grupo fijo y soporte idéntico. '
        'Las aproximaciones de CRPS conservan las rejillas de cada modelo.</p>'
        '<p><a href="weekly_ranking.csv">Ranking por semana</a> · '
        '<a href="standings.csv">Tabla final</a></p>'
        + (standings.drop(columns="configuration").to_html(index=False, float_format="%.2f")
           if not standings.empty else '<p>Soporte insuficiente para clasificar dos configuraciones completas.</p>')
        + "".join(sections), encoding="utf-8",
    )
    return output, [index, *artifacts]
