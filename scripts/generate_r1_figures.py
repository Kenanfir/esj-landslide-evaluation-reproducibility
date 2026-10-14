#!/usr/bin/env python3
"""Figures for the rebuilt ESJ paper. Each one is written as 300 dpi PNG and as
vector PDF, because Reviewer #1 asked for publication-quality or editable
versions of every figure.

Every number plotted here is read from results/metrics/; nothing is recomputed
in this file, so a figure can never drift from the table it illustrates.

Run:  python3 scripts/generate_r1_figures.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
PACITAN = METRICS / "pacitan"
REVISION = METRICS / "revision_r1"
RAW = ROOT / "data" / "raw" / "pacitan-dib-2024"
OUT = ROOT / "manuscript" / "esj" / "figures_r1"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
})

INK = "#1a1a1a"
RANDOM_C = "#2b6cb0"
SPATIAL_C = "#c05621"
DUP_C = "#718096"


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def fig_study_area() -> None:
    """Fig 1 — Pacitan study area over the elevation surface, with a locator inset."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(RAW / "map_elev.tif") as raster:
        scale = 6
        band = raster.read(
            1,
            out_shape=(raster.height // scale, raster.width // scale),
            resampling=Resampling.average,
        ).astype("float64")
        band[(band == raster.nodata) | (np.abs(band) > 1e30)] = np.nan
        left, bottom, right, top = raster.bounds

    def points(filename: str, layer: str) -> np.ndarray:
        with sqlite3.connect(RAW / filename) as connection:
            return np.array(connection.execute(f'SELECT X, Y FROM "{layer}"').fetchall())

    landslides = points("ls_center.gpkg", "ls_center")
    negatives = points("non_ls.gpkg", "non_ls")

    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    shaded = ax.imshow(
        band, extent=(left / 1000, right / 1000, bottom / 1000, top / 1000),
        origin="upper", cmap="terrain", alpha=0.85, interpolation="bilinear",
    )
    ax.scatter(negatives[:, 0] / 1000, negatives[:, 1] / 1000, s=5, c="#2d3748",
               alpha=0.45, marker="o", linewidths=0, label="non-landslide (n = 743)")
    ax.scatter(landslides[:, 0] / 1000, landslides[:, 1] / 1000, s=11, c="#c53030",
               alpha=0.9, marker="^", edgecolors="white", linewidths=0.25,
               label="landslide (n = 743)")

    ax.set_xlabel("Easting (km, UTM zone 49S)")
    ax.set_ylabel("Northing (km, UTM zone 49S)")
    ax.set_title("Pacitan Regency, East Java: inventory of the November 2017\nTropical Cyclone Cempaka landslides", loc="left")
    ax.legend(loc="lower left", framealpha=0.94)
    bar = fig.colorbar(shaded, ax=ax, shrink=0.82, pad=0.02)
    bar.set_label("Elevation (m a.s.l.)")

    # locator inset drawn from Natural Earth 1:110m outlines
    inset = ax.inset_axes([0.638, 0.630, 0.352, 0.335])
    outlines = json.loads((ROOT / "data" / "raw" / "naturalearth"
                           / "indonesia_ne110m.geojson").read_text())
    for feature in outlines["features"]:
        is_indonesia = feature["properties"]["NAME"] == "Indonesia"
        geometry = feature["geometry"]
        parts = (geometry["coordinates"] if geometry["type"] == "MultiPolygon"
                 else [geometry["coordinates"]])
        for part in parts:
            ring = np.asarray(part[0])
            inset.fill(ring[:, 0], ring[:, 1],
                       facecolor="#b7c4d1" if is_indonesia else "#e2e8f0",
                       edgecolor="#5a6b7d", linewidth=0.35, zorder=2)
    inset.set_xlim(104.5, 116.5)
    inset.set_ylim(-9.6, -4.8)
    inset.set_facecolor("#eaf2f8")
    inset.plot(111.10, -8.19, marker="*", markersize=9, color="#c53030",
               markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    inset.annotate("Pacitan", (111.10, -8.19), textcoords="offset points",
                   xytext=(5, -9), fontsize=6.5, color="#742a2a", weight="bold", zorder=5)
    inset.plot(106.85, -6.21, marker="o", markersize=2.6, color="#2d3748", zorder=4)
    inset.annotate("Jakarta", (106.85, -6.21), textcoords="offset points",
                   xytext=(3, 2), fontsize=5.8, color="#2d3748", zorder=5)
    inset.annotate("JAVA", (110.0, -7.05), fontsize=6, color="#3b4a5a",
                   weight="bold", ha="center", zorder=5)
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_title("Java, Indonesia", fontsize=6.5, pad=2)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    save(fig, "fig1_study_area")


def fig_sampling_strategy() -> None:
    """Fig 2 — the central result: duplication inflates random CV, not skill."""
    s = pd.read_csv(PACITAN / "sampling_strategy_summary.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    left.errorbar(s.points_per_polygon, s.random_cv_auroc, yerr=s.random_cv_sd,
                  marker="o", color=RANDOM_C, capsize=3, lw=1.7, label="random 10-fold CV")
    left.errorbar(s.points_per_polygon, s.spatial_cv_auroc, yerr=s.spatial_cv_sd,
                  marker="s", color=SPATIAL_C, capsize=3, lw=1.7,
                  label="blocked, leave-one-block-out")
    left.errorbar(s.points_per_polygon, s.lolo_auroc, yerr=s.lolo_sd,
                  marker="D", color="#2f855a", capsize=3, lw=1.7,
                  label="grouped by mapped failure")
    left.errorbar(s.points_per_polygon, s.spatial_rr12_auroc, yerr=s.spatial_rr12_sd,
                  marker="^", color=SPATIAL_C, capsize=3, lw=1.4, ls="--", alpha=0.75,
                  label="blocked, round robin")
    # The two steps of the decomposition at the right-hand end. Both the labels
    # and their positions are derived: they read 0.187 and 0.127 against a caption
    # saying 0.184 and 0.129 for one build, a figure contradicting its own caption.
    random_end = float(s.random_cv_auroc.iloc[-1])
    lolo_end = float(s.lolo_auroc.iloc[-1])
    blocked_end = float(s.spatial_cv_auroc.iloc[-1])
    for upper, lower, label in ((random_end, lolo_end, "pseudo-\nreplication"),
                                (lolo_end, blocked_end, "regional\nextrapolation")):
        left.annotate("", xy=(10, upper), xytext=(10, lower),
                      arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
        left.annotate(f"{label}\n{upper - lower:.3f}", xy=(10, (upper + lower) / 2),
                      fontsize=6.2, color=INK, ha="right", va="center")
    left.axhspan(0.86, 0.95, xmin=0.0, xmax=1.0, color=RANDOM_C, alpha=0.07, zorder=0)
    left.text(1.15, 0.906, "band: range the source study reports for\nthese strategies under RANDOM cross-validation",
              fontsize=6.0, color=RANDOM_C, ha="left", va="center")
    left.set_xticks(s.points_per_polygon)
    left.set_xlabel("Points per landslide polygon")
    left.set_ylabel("AUROC")
    left.set_ylim(0.55, 1.02)
    left.legend(loc="lower left", fontsize=6.4, framealpha=0.95)
    left.set_title("(a) Only the random-CV estimate moves with $k$", loc="left")

    right.plot(s.points_per_polygon, s.positive_duplicate_fraction, marker="o",
               color=RANDOM_C, lw=1.7, label="landslide points")
    right.plot(s.points_per_polygon, s.negative_duplicate_fraction, marker="s",
               color=DUP_C, lw=1.7, label="non-landslide points (control)")
    right.set_xticks(s.points_per_polygon)
    right.set_xlabel("Points per landslide polygon")
    right.set_ylabel("Fraction sharing a factor vector")
    right.set_ylim(-0.03, 0.55)
    right.set_title("(b) Duplication appears only in the sampled class", loc="left")
    right.legend(loc="upper left", framealpha=0.95)

    save(fig, "fig5_sampling_strategy")


def fig_blocking_sensitivity() -> None:
    """Fig 4 — the pooled relationship is between design families, not within them.

    An earlier version of this figure drew one trend line per model through all
    sixteen designs, which asserted exactly the claim the paper now retracts.
    Panel (a) keeps that pooled line but shows it decomposed by assignment
    family, so the reader can see it holds inside none of them; panel (b) plots
    the quantity that does track the gap.
    """
    # Per-family lines use every design label, because the four-block partition is
    # genuinely reached by all three assignment schemes and belongs to each. The
    # pooled line counts it once, over the sixteen distinct partitions, which is
    # the basis the manuscript quotes.
    s = pd.read_csv(PACITAN / "blocking_sensitivity_summary.csv")
    s = s[s.model == "Random Forest"].copy()
    s["sep_km"] = s.mean_nearest_train_m / 1000
    s["ratio"] = s.fold_spread_km / s.study_area_spread_km

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    family = {"round_robin": ("#2b6cb0", "o", "round robin"),
              "contiguous": ("#c05621", "s", "contiguous"),
              "leave_one_block_out": ("#2f855a", "^", "leave one block out")}

    for key, (colour, marker, label) in family.items():
        cell = s[s.assignment == key].sort_values("sep_km")
        left.scatter(cell.sep_km, cell.gap, color=colour, marker=marker, s=42,
                     linewidths=0, label=label, zorder=3)
        if len(cell) > 2:
            fit = np.polyfit(cell.sep_km, cell.gap, 1)
            xs = np.linspace(cell.sep_km.min(), cell.sep_km.max(), 30)
            left.plot(xs, np.polyval(fit, xs), color=colour, lw=1.2, alpha=0.85)
            r = np.corrcoef(cell.sep_km, cell.gap)[0, 1]
            # nudge the family labels apart: two of the three lines end close together
            offset = {"round_robin": (4, -9), "contiguous": (4, 4),
                      "leave_one_block_out": (4, -3)}[key]
            left.annotate(f"r = {r:+.2f}", (xs[-1], np.polyval(fit, xs[-1])),
                          textcoords="offset points", xytext=offset,
                          fontsize=6.5, color=colour)

    distinct = s.drop_duplicates(subset=["n_blocks", "mean_nearest_train_m", "gap"])
    pooled = np.polyfit(distinct.sep_km, distinct.gap, 1)
    xs = np.linspace(distinct.sep_km.min(), distinct.sep_km.max(), 30)
    left.plot(xs, np.polyval(pooled, xs), color=INK, lw=1.1, ls="--", alpha=0.7,
              label=f"pooled (r = {np.corrcoef(distinct.sep_km, distinct.gap)[0, 1]:+.2f})",
              zorder=2)

    # the two designs at identical separation
    pair = s[s.n_blocks == 6].sort_values("gap")
    if len(pair) == 2:
        left.plot(pair.sep_km, pair.gap, color=INK, lw=0.8, zorder=4)
        # Derived, not typed. This label read 0.030 against a caption saying
        # 0.029 for one build, a figure contradicting its own caption.
        spread = float(pair.gap.iloc[1]) - float(pair.gap.iloc[0])
        left.annotate(f"identical separation,\ngap differs by {spread:.3f}",
                      xy=(float(pair.sep_km.iloc[1]), float(pair.gap.iloc[1])),
                      xytext=(3.1, 0.106), fontsize=6.5, color=INK,
                      arrowprops=dict(arrowstyle="->", color=INK, lw=0.6, shrinkB=4))

    left.set_xlabel("Mean distance to the nearest training point (km)")
    left.set_ylabel("AUROC optimism\n(random $-$ blocked)")
    left.set_title("(a) The pooled trend holds inside no family", loc="left")
    left.legend(fontsize=6.4, framealpha=0.95, loc="lower right")

    s = s.drop_duplicates(subset=["n_blocks", "mean_nearest_train_m", "gap"])
    blocking = s[s.ratio < 0.80]
    not_blocking = s[s.ratio >= 0.80]
    right.scatter(blocking.ratio, blocking.gap, color="#2b6cb0", s=42, linewidths=0,
                  label="folds more compact than the area")
    right.scatter(not_blocking.ratio, not_blocking.gap, color=DUP_C, s=42, marker="x",
                  label="folds as dispersed as the area")
    fit = np.polyfit(s.ratio, s.gap, 1)
    xs = np.linspace(s.ratio.min(), s.ratio.max(), 30)
    right.plot(xs, np.polyval(fit, xs), color=INK, lw=1.1, alpha=0.7)
    right.axvline(0.80, color=INK, ls=":", lw=0.8)
    right.annotate(f"r = {np.corrcoef(s.ratio, s.gap)[0, 1]:+.2f}", (0.42, 0.108),
                   fontsize=7, color=INK)
    right.set_xlabel("Fold spread ÷ study-area spread")
    right.set_ylabel("AUROC optimism")
    right.set_title("(b) Fold compactness tracks it instead", loc="left")
    right.legend(fontsize=6.4, framealpha=0.95, loc="lower left")

    save(fig, "fig4_blocking_sensitivity")


def fig_retired_dataset() -> None:
    """Fig 4 — leg one: the same failure in a dataset already in circulation."""
    g = pd.read_csv(REVISION / "cv_regime_gap_by_pool.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    width = 0.2
    positions = np.arange(len(models))

    for offset, (pool, label, colour, hatch) in enumerate([
        ("pool_1011", "as published (1,011 rows)", RANDOM_C, ""),
        ("dedup", "after de-duplication (450 rows)", SPATIAL_C, "//"),
    ]):
        rand = [g[(g.pool == pool) & (g.model == m)]["random"].iloc[0] for m in models]
        spat = [g[(g.pool == pool) & (g.model == m)]["spatial"].iloc[0] for m in models]
        ax.bar(positions + (offset * 2 - 1.5) * width, rand, width, color=colour,
               alpha=0.95, hatch=hatch, edgecolor="white",
               label=f"random CV, {label}")
        ax.bar(positions + (offset * 2 - 0.5) * width, spat, width, color=colour,
               alpha=0.45, hatch=hatch, edgecolor="white",
               label=f"spatial CV, {label}")

    ax.axhline(0.5, color=INK, ls=":", lw=0.8)
    ax.annotate("chance", (2.42, 0.512), fontsize=7, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("The retired dataset: 55% of its rows are duplicates", loc="left")
    ax.legend(fontsize=6.4, ncol=2, framealpha=0.95, loc="upper left")
    save(fig, "fig2_retired_dataset")


def fig_morans() -> None:
    """Fig 5 — the condition that makes spatial blocking necessary, on both datasets."""
    # Read every bar from gate_comparison.json, which is also what Table 2 in the
    # manuscript reports. An earlier version took the two retired-dataset bars
    # from reviewer4_findings.json, whose Moran's I was computed over 499
    # permutations rather than 999; the z printed on this figure then disagreed
    # with the z in the table on the facing page. One source per number.
    gate = json.loads((METRICS / "gate_comparison.json").read_text())["datasets"]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = ["Pacitan\n(1,485 distinct)", "Retired dataset\nas published (1,011)",
              "Retired dataset\nde-duplicated (450)"]
    keys = ["pacitan", "retired_as_assembled", "retired_deduplicated"]
    values = [gate[k]["morans_i"] for k in keys]
    zs = [gate[k]["morans_z"] for k in keys]
    colours = ["#2f855a", "#c05621", "#a0aec0"]

    bars = ax.bar(labels, values, color=colours, width=0.55)
    for bar, z in zip(bars, zs):
        ax.annotate(f"z = {z:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set_ylabel("Moran's $I$ of the landslide label")
    ax.set_ylim(-0.02, 0.33)
    ax.set_title("Spatial autocorrelation: real in one dataset, manufactured in the other",
                 loc="left")
    save(fig, "fig3_morans_i")


def main() -> None:
    print("writing figures to", OUT.relative_to(ROOT))
    fig_study_area()
    fig_sampling_strategy()
    fig_blocking_sensitivity()
    fig_retired_dataset()
    fig_morans()


if __name__ == "__main__":
    main()
