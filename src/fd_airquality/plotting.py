"""Descriptive test plots from an audited comparison; never trains or selects models."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

METRICS = {
    "macro-mase": "MASE macro",
    "macro-rmsse": "RMSSE macro",
}
IDENTITY_COLUMNS = ("run", "model", "mode", "adaptation", "strategy", "seed")
CONTEXT_COLUMNS = (
    "lora_rank", "revision", "flowstate_scale_factor", "missing_policy", "selection_metric",
    "selected_round", "num-clients", "num-examples", "num-targets", "target-coverage",
)
MODELS = {
    "mlp": "MLP", "autoencoder": "Autoencoder", "autoencoder-tiny": "Autoencoder Tiny",
    "nlinear": "NLinear", "nhits": "NHITS",
    "nlinear-point": "NLinear puntual", "linear-6lags": "Regresión de 6 retardos",
    "seasonal-naive": "SeasonalNaive", "holt": "Holt",
    "darts-nlinear": "NLinear (Darts)", "ttm-r3": "TTM-R3",
    "patchtst-fm-r2": "PatchTST-FM-R2", "flowstate-r1": "FlowState-R1",
}
ADAPTATIONS = {
    "full": "Completo", "lora": "LoRA", "ffa-lora": "FFA-LoRA",
    "flora": "FLoRA", "lora-fair": "LoRA-FAIR",
}
COLORS = {
    "full": "#64748b", "lora": "#2563eb", "ffa-lora": "#0891b2",
    "flora": "#7c3aed", "lora-fair": "#db2777",
}
FEDERATED, CENTRALIZED = "#2563eb", "#d97706"
NOTE = "Test; menor error es mejor. Comparación descriptiva, sin inferencia estadística."
CONTROL_NOTE = "* FLoRA y LoRA-FAIR frente a LoRA centralizado: control, no adaptación idéntica."


def load_comparison(source: Path) -> tuple[Path, pd.DataFrame, dict]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.name != "comparison.csv":
            raise ValueError("Indica comparison.csv o el directorio de la comparación.")
        source = source.parent
    elif not (source / "comparison.csv").is_file():
        source = source / "comparison"
    metadata = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    if not metadata.get("manifest_sha256"):
        raise ValueError("Falta la identidad del manifest en comparison.json; usa compare primero.")
    rows = pd.read_csv(source / "comparison.csv")
    if missing := set(IDENTITY_COLUMNS) - set(rows):
        raise ValueError(f"Faltan columnas en comparison.csv: {sorted(missing)}")
    # Older reports can contain non-scaled metrics. Never include them in new exports.
    columns = [key for key in (*IDENTITY_COLUMNS, *CONTEXT_COLUMNS, *METRICS) if key in rows]
    rows = rows.loc[rows["mode"].isin(["federated", "centralized"]), columns].copy()
    if set(rows["mode"]) != {"federated", "centralized"}:
        raise ValueError("Se necesitan resultados federados y centralizados en la misma comparación.")
    if rows["run"].duplicated().any():
        raise ValueError("La comparación contiene ejecuciones duplicadas.")
    for metric in METRICS:
        # With masked data, a scaled error may be undefined for the entire panel.
        rows[metric] = pd.to_numeric(rows[metric], errors="raise") if metric in rows else np.nan
        if np.isinf(rows[metric]).any() or (rows[metric].dropna() < 0).any():
            raise ValueError(f"Error negativo o infinito en {metric}.")
    if not rows[list(METRICS)].notna().any().any():
        raise ValueError("No hay resultados definidos de MASE macro ni de RMSSE macro para graficar.")
    # Never pool seeds, ranks, revisions or strategies into an implicit mean.
    labels = []
    for _, row in rows.iterrows():
        label = f"{MODELS.get(row['model'], row['model'])} · {ADAPTATIONS.get(row['adaptation'], row['adaptation'])}"
        if row["mode"] == "federated":
            label += f" · {row['strategy']}"
        details = []
        for key, prefix in (("seed", "s"), ("lora_rank", "r")):
            if (key in rows and rows[key].nunique() > 1 and pd.notna(row[key])
                    and (key != "lora_rank" or row["adaptation"] != "full")):
                details.append(f"{prefix}{row[key]:g}")
        labels.append(label + (f" ({', '.join(details)})" if details else ""))
    rows["label"] = labels
    duplicate = rows.duplicated(["mode", "label"], keep=False)
    rows.loc[duplicate, "label"] += " [" + rows.loc[duplicate, "run"].str.rsplit("/", n=1).str[-1] + "]"
    return source, rows.sort_values([*METRICS, "run"], na_position="last"), metadata


def matched_pairs(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Match configurations, not test winners; FL-only adapters use a labelled LoRA control."""
    centralized = rows.loc[rows["mode"] == "centralized"]
    pairs, warnings = [], []
    for _, fed in rows.loc[rows["mode"] == "federated"].iterrows():
        control = fed["adaptation"] in {"flora", "lora-fair"}
        adaptation = "lora" if control else fed["adaptation"]
        candidates = centralized.loc[
            (centralized["model"] == fed["model"]) & (centralized["adaptation"] == adaptation)
        ]
        keys = ["seed", "revision", "flowstate_scale_factor", "missing_policy", "selection_metric"]
        if fed["adaptation"] != "full":
            keys.append("lora_rank")
        for key in keys:
            if key in rows:
                candidates = candidates.loc[
                    candidates[key].isna() if pd.isna(fed[key]) else candidates[key] == fed[key]
                ]
        if len(candidates) != 1:
            warnings.append(f"Sin pareja única compatible ({len(candidates)} controles): {fed['run']}")
            continue
        central = candidates.iloc[0]
        pair = {
            "model": fed["model"], "federated_run": fed["run"], "centralized_run": central["run"],
            "federated_adaptation": fed["adaptation"], "centralized_adaptation": adaptation,
            "reference_kind": "lora_control" if control else "same_adaptation",
            "label": fed["label"] + (" → LoRA*" if control else ""),
        }
        for metric in METRICS:
            a, b = fed[metric], central[metric]
            pair[f"{metric}_federated"] = a
            pair[f"{metric}_centralized"] = b
            pair[f"{metric}_delta"] = a - b
            pair[f"{metric}_improvement_pct"] = 100 * (1 - a / b) if b > 0 else np.nan
        pairs.append(pair)
    return pd.DataFrame(pairs), warnings


def cross_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """Every federated vs every centralized candidate, without claiming causal attribution."""
    records = []
    for _, fed in rows.loc[rows["mode"] == "federated"].iterrows():
        for _, central in rows.loc[rows["mode"] == "centralized"].iterrows():
            for metric in METRICS:
                a, b = fed[metric], central[metric]
                records.append({
                    "federated_run": fed["run"], "centralized_run": central["run"], "metric": metric,
                    "federated_value": a, "centralized_value": b, "delta_fed_minus_central": a - b,
                    "improvement_pct": 100 * (1 - a / b) if b > 0 else np.nan,
                })
    return pd.DataFrame(records)


def _panels(count: int, title: str, subtitle: str) -> tuple[Figure, np.ndarray]:
    figure = Figure(figsize=(22, max(8, count * .55 + 4)), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(ncols=len(METRICS), squeeze=False)
    figure.suptitle(title, x=.02, y=.98, ha="left", fontsize=21, weight="bold")
    figure.text(.02, .93, subtitle, fontsize=11, color="#475569")
    figure.text(.02, .018, NOTE, fontsize=10, color="#475569")
    figure.subplots_adjust(left=.23, right=.95, top=.84, bottom=.17 if count < 6 else .12,
                           wspace=1.05)
    for axis in axes.flat:
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=.18)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, labelsize=9)
    return figure, axes


def _overview(rows: pd.DataFrame, title: str, subtitle: str) -> Figure:
    figure, axes = _panels(len(rows), title, subtitle + " | Orden por MASE macro y RMSSE macro de test")
    positions = np.arange(len(rows))
    for axis, (metric, name) in zip(axes.flat, METRICS.items(), strict=True):
        bars = axis.barh(positions, rows[metric].fillna(0), height=.68,
                         color=[COLORS.get(value, "#64748b") for value in rows["adaptation"]])
        axis.set_yticks(positions, rows["label"])
        axis.invert_yaxis()
        axis.set_title(f"{name} ↓", loc="left", weight="bold", pad=12)
        axis.set_xlabel("Error escalado")
        axis.bar_label(bars, labels=[f"{v:.4f}" if pd.notna(v) else "N/D" for v in rows[metric]],
                       padding=4, fontsize=9)
        maximum = rows[metric].max()
        axis.set_xlim(0, (max(maximum, .01) if pd.notna(maximum) else 1) * 1.2)
    return figure


def _paired(pairs: pd.DataFrame, title: str, subtitle: str) -> Figure:
    figure, axes = _panels(len(pairs), title, subtitle)
    positions = np.arange(len(pairs))
    for axis, (metric, name) in zip(axes.flat, METRICS.items(), strict=True):
        a, b = pairs[f"{metric}_federated"], pairs[f"{metric}_centralized"]
        axis.hlines(positions, np.minimum(a, b), np.maximum(a, b), color="#94a3b8", linewidth=2)
        axis.scatter(b, positions, color=CENTRALIZED, marker="s", s=48, zorder=3)
        axis.scatter(a, positions, color=FEDERATED, s=48, zorder=4)
        axis.set_yticks(positions, pairs["label"])
        axis.invert_yaxis()
        axis.set_title(f"{name} ↓", loc="left", weight="bold", pad=12)
        axis.set_xlabel("Error; eje ampliado para ver diferencias")
        for position, gain in enumerate(pairs[f"{metric}_improvement_pct"]):
            text = f"{gain:+.2f}%" if pd.notna(gain) else "N/D"
            axis.text(1.02, position, text, transform=axis.get_yaxis_transform(), va="center",
                      color="#047857" if gain > 0 else "#be123c" if gain < 0 else "#475569",
                      fontsize=9)
        axis.margins(x=.18, y=.12)
    figure.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=FEDERATED, label="Federado"),
        Line2D([], [], marker="s", linestyle="none", color=CENTRALIZED, label="Centralizado"),
    ], loc="upper right", bbox_to_anchor=(.97, .94), ncol=2, frameon=False)
    figure.text(.02, .052, "Porcentaje = 100 × (1 − federado / centralizado); positivo favorece federado. "
                + CONTROL_NOTE, fontsize=10, color="#475569")
    return figure


def _matrix(rows: pd.DataFrame, subtitle: str) -> Figure:
    fed = rows.loc[rows["mode"] == "federated"]
    central = rows.loc[rows["mode"] == "centralized"]
    a = fed["macro-mase"].to_numpy()[:, None]
    b = central["macro-mase"].to_numpy()[None, :]
    values = 100 * (1 - np.divide(a, b, out=np.full((len(fed), len(central)), np.nan), where=b > 0))
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(finite.max()) if finite.size else 0, .01)
    figure = Figure(figsize=(max(12, len(central) * 1.05 + 5), max(7, len(fed) * .5 + 3)))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    cmap = colormaps["BrBG"].with_extremes(bad="#e2e8f0")
    image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(central)), central["label"], rotation=45, ha="right", fontsize=9)
    axis.set_yticks(np.arange(len(fed)), fed["label"], fontsize=9)
    axis.set_xlabel("Modelo centralizado (referencia)", labelpad=10)
    axis.set_ylabel("Modelo federado", labelpad=10)
    for (i, j), value in np.ndenumerate(values):
        axis.text(j, i, f"{value:+.2f}%" if np.isfinite(value) else "N/D", ha="center", va="center",
                  fontsize=9, color="white" if abs(value) > limit * .65 else "#0f172a")
    figure.colorbar(image, ax=axis, shrink=.8, label="Mejora MASE macro (%); positivo favorece federado")
    figure.suptitle("Todos los pares federado × centralizado · MASE macro", fontsize=19, weight="bold")
    figure.text(.02, .92, subtitle, fontsize=10, color="#475569")
    figure.text(.02, .015, "100 × (1 − MASE federado / MASE centralizado). " + NOTE, fontsize=9)
    figure.subplots_adjust(left=.28, right=.95, top=.87, bottom=.27)
    return figure


def plot_comparison(source: Path, output: Path | None = None) -> Path:
    """Write PNG/PDF figures, source tables and an offline HTML index into a new directory."""
    source, rows, metadata = load_comparison(source)
    pairs, warnings = matched_pairs(rows)
    warnings = list(metadata.get("warnings", [])) + warnings
    has_scaled_matrix = rows.groupby("mode")["macro-mase"].count().gt(0).all()
    if not has_scaled_matrix:
        warnings.append("No hay MASE macro en ambos modos; no se genera su matriz de cruces.")
    output = (output or source / "plots").expanduser().resolve()
    if output.exists():
        raise ValueError(f"La salida ya existe; indica otra ruta con --output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / "plotted_metrics.csv", index=False)
    pairs.to_csv(output / "matched_pairs.csv", index=False)
    cross_pairs(rows).to_csv(output / "cross_model_pairs.csv", index=False)
    context = []
    for key, name in (("num-clients", "estaciones"), ("num-examples", "ventanas test"),
                      ("seed", "semilla"), ("lora_rank", "rango LoRA")):
        if key in rows:
            values = ", ".join(f"{v:g}" for v in sorted(rows[key].dropna().unique()))
            context.append(f"{name}: {values}")
    subtitle = " | ".join(context)
    figures = []

    def save(figure: Figure, stem: str, title: str) -> None:
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{stem}.{extension}", dpi=160, facecolor="white")
        figure.clear()
        figures.append({"stem": stem, "title": title})

    with rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        for mode, stem, title in (
            ("federated", "01_federado", "Modelos federados · todas las adaptaciones"),
            ("centralized", "02_centralizado", "Modelos centralizados · todas las adaptaciones"),
        ):
            save(_overview(rows.loc[rows["mode"] == mode], title, subtitle), stem, title)
        if not pairs.empty:
            title = "Federado frente a centralizado · mismo modelo"
            save(_paired(pairs, title, subtitle), "03_federado_vs_centralizado", title)
        if has_scaled_matrix:
            save(_matrix(rows, subtitle), "04_pares_entre_modelos", "Todos los pares entre modelos")
        if not pairs.empty:
            models = [*MODELS, *sorted(set(pairs["model"]) - MODELS.keys())]
            for index, model in enumerate(models, 1):
                subset = pairs.loc[pairs["model"] == model]
                if not subset.empty:
                    title = f"{MODELS.get(model, model)} · federado frente a centralizado"
                    # Unrecognized CSV labels may be displayed, but not used as paths.
                    name = model if model in MODELS else f"extra_{index:03d}"
                    save(_paired(subset, title, subtitle), f"modelo_{name}", title)

    manifest = {
        "source": str(source), "manifest_sha256": metadata["manifest_sha256"],
        **{key: metadata[key] for key in ("evaluation_sha256", "allow_training_differences") if key in metadata},
        "metrics": METRICS, "figures": figures, "warnings": warnings,
        "interpretation": NOTE, "control_note": CONTROL_NOTE,
        "aggregation": "Cada ejecución se muestra por separado, sin promediar semillas ni rangos.",
    }
    (output / "plots.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    warning_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    sections = "".join(
        f'<section><h2>{html.escape(item["title"])}</h2><a href="{item["stem"]}.pdf">PDF</a> · '
        f'<a href="{item["stem"]}.png">PNG</a><img src="{item["stem"]}.png" '
        f'alt="{html.escape(item["title"])}" loading="lazy"></section>' for item in figures
    )
    page = (
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>Comparación de modelos</title>'
        '<style>body{font:16px system-ui;max-width:1600px;margin:36px auto;padding:0 24px;'
        'color:#0f172a;background:#f8fafc}section{background:white;padding:24px;margin:24px 0;'
        'border:1px solid #e2e8f0;border-radius:12px}img{display:block;width:100%;height:auto;'
        'margin-top:16px}h2{color:#0f172a}li{margin:8px 0}</style>'
        f'<h1>Rendimiento predictivo: federado y centralizado</h1><p>{html.escape(subtitle)}</p>'
        f'<p>{NOTE} No se muestran intervalos de confianza: las estaciones no son semillas '
        f'y los targets rolling se solapan.</p><p>{CONTROL_NOTE}</p>'
        '<p>MASE y RMSSE macro: igual peso por estación con escala definida. '
        'Las variantes se muestran por separado, sin seleccionar la mejor sobre test. '
        'La matriz cruza todos los candidatos, también modelos y configuraciones diferentes; '
        'sus diferencias no se atribuyen exclusivamente a federar.</p>'
        '<p>Tablas: <a href="plotted_metrics.csv">métricas</a> · '
        '<a href="matched_pairs.csv">parejas del mismo modelo</a> · '
        '<a href="cross_model_pairs.csv">todos los cruces (MASE y RMSSE macro)</a></p>'
        f'<ul>{warning_html}</ul>{sections}</html>'
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    for warning in warnings:
        print(f"AVISO: {warning}")
    print(f"{len(figures)} gráficas PNG/PDF: {output / 'index.html'}")
    return output
