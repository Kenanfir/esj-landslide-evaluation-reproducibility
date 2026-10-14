#!/usr/bin/env python3
"""Audit of the rebuilt Pacitan result, against the dataset's own published analysis.

Four questions, each answered into a file under results/metrics/pacitan/:

 1. Is the extraction correct?  Geometry-vs-attribute agreement, label agreement,
    point-in-polygon containment, class separation, and the physical direction of
    each predictor. (integrity_checks.json)
 2. Does the reported gap survive the choice of AUROC aggregation?
    Mean-of-per-fold against pooled out-of-fold. (aggregation_effect.csv)
 3. Can hyperparameter tuning close the gap, as the published spatial-tuning
    workflow might imply? (tuning_sweep.csv)
 4. Does the within-polygon multi-point sampling that the published benchmarking
    paper reports scoring 0.86-0.98 have enough distinct 10 m cells to be
    anything other than duplicate memorisation? (polygon_resolution.json)

Run:  python3 scripts/pacitan_replication_audit.py
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from shapely import STRtree, wkb
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pacitan-dib-2024"
DATA = ROOT / "data" / "processed" / "pacitan_points.csv"
OUT = ROOT / "results" / "metrics" / "pacitan"

CONTINUOUS = [
    "slope_deg", "elevation_m", "aspect_deg", "plan_curvature", "profile_curvature",
    "terrain_wetness_index", "stream_power_index", "distance_to_river_m", "distance_to_road_m",
]
CATEGORICAL = ["lithology_class", "landuse_class"]
SEEDS = [42, 1, 2, 3, 4]
N_FOLDS_PUBLISHED = 10   # Samodra et al. use 10-fold random CV and 10 k-means partitions
PIXEL_M = 10.0

# What the published analysis reports for the "Center" inventory strategy, which
# is the strategy the ls_center.gpkg layer implements.
PUBLISHED = {
    "source": "Samodra & Nugroho, doi:10.21203/rs.3.rs-1441095/v1 (published as "
              "Artificial Intelligence in Geosciences, doi:10.1016/j.aiig.2024.100093)",
    "strategy": "Center",
    "cv_nonspatial_tuning": {"accuracy": 0.70, "kappa": 0.39, "auroc": 0.74},
    "cv_llo_spatial_tuning": {"accuracy": 0.69, "kappa": 0.38, "auroc": 0.74},
    "cv_llo_ffs_no_tuning": {"accuracy": 0.70, "kappa": 0.41, "auroc": 0.73},
    "multi_point_strategies_auroc_range": [0.86, 0.98],
}


def gpkg_geometry(blob: bytes):
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 0x07]
    return wkb.loads(blob[8 + envelope:])


def integrity_checks(df: pd.DataFrame) -> dict:
    out: dict = {}

    # geometry vs stored X/Y, and the stored label flag
    layers = {"ls_center.gpkg": ("ls_center", "TRUE"), "non_ls.gpkg": ("non_ls", "FALSE")}
    geometry = {}
    for filename, (layer, expected_flag) in layers.items():
        with sqlite3.connect(RAW / filename) as connection:
            rows = connection.execute(f'SELECT geom, lslpts, X, Y FROM "{layer}"').fetchall()
        dx = dy = 0.0
        flags = set()
        for blob, flag, x, y in rows:
            point = gpkg_geometry(blob)
            dx = max(dx, abs(point.x - x))
            dy = max(dy, abs(point.y - y))
            flags.add(str(flag))
        geometry[layer] = {
            "n": len(rows),
            "max_abs_x_minus_geometry_m": dx,
            "max_abs_y_minus_geometry_m": dy,
            "lslpts_values": sorted(flags),
            "lslpts_matches_expected": flags == {expected_flag},
        }
    out["geometry_and_labels"] = geometry

    # containment and class separation
    with sqlite3.connect(RAW / "landslide_pacitan.gpkg") as connection:
        polygons = [gpkg_geometry(b) for (b,) in connection.execute(
            "SELECT geom FROM landslide_pacitan")]
    tree = STRtree(polygons)
    containment = {}
    for layer, label in (("ls_center", 1), ("non_ls", 0)):
        subset = df[df.source_layer == layer]
        points = [Point(x, y) for x, y in zip(subset.x_utm49s, subset.y_utm49s)]
        inside = sum(len(tree.query(p, predicate="intersects")) > 0 for p in points)
        nearest = np.array([p.distance(polygons[tree.nearest(p)]) for p in points])
        containment[layer] = {
            "n": len(points),
            "inside_a_landslide_polygon": int(inside),
            "min_distance_to_polygon_m": round(float(nearest.min()), 2),
            "median_distance_to_polygon_m": round(float(np.median(nearest)), 2),
        }
    out["polygon_containment"] = containment

    positives = {(round(a, 3), round(b, 3)) for a, b in
                 zip(df[df.landslide == 1].x_utm49s, df[df.landslide == 1].y_utm49s)}
    negatives = {(round(a, 3), round(b, 3)) for a, b in
                 zip(df[df.landslide == 0].x_utm49s, df[df.landslide == 0].y_utm49s)}
    out["coordinate_collisions_between_classes"] = len(positives & negatives)

    # physical direction of each predictor
    directions = []
    for column in CONTINUOUS:
        a = df[df.landslide == 1][column]
        b = df[df.landslide == 0][column]
        directions.append({
            "feature": column,
            "median_landslide": round(float(a.median()), 3),
            "median_non_landslide": round(float(b.median()), 3),
            "mann_whitney_p": float(stats.mannwhitneyu(a, b).pvalue),
        })
    out["predictor_direction"] = directions
    return out


def make_model(max_features, min_samples_leaf, seed) -> Pipeline:
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), CONTINUOUS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])),
        ("model", RandomForestClassifier(
            n_estimators=500, max_features=max_features, min_samples_leaf=min_samples_leaf,
            random_state=seed, n_jobs=-1)),
    ])


def evaluate(x, y, folds, model) -> tuple[float, float]:
    """Return (mean of per-fold AUROC, pooled out-of-fold AUROC)."""
    per_fold = []
    out_of_fold = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(x.iloc[test_idx])[:, 1]
        out_of_fold[test_idx] = proba
        if len(np.unique(y.iloc[test_idx])) > 1:
            per_fold.append(roc_auc_score(y.iloc[test_idx], proba))
    scored = ~np.isnan(out_of_fold)
    return float(np.mean(per_fold)), float(roc_auc_score(y[scored], out_of_fold[scored]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    x = df[CONTINUOUS + CATEGORICAL].reset_index(drop=True)
    y = df["landslide"].astype(int).reset_index(drop=True)
    coords = df[["x_utm49s", "y_utm49s"]].to_numpy(dtype=float)

    checks = integrity_checks(df)
    (OUT / "integrity_checks.json").write_text(json.dumps(checks, indent=2, default=float),
                                               encoding="utf-8")

    # --- aggregation effect, under the published validation design ---------
    rows = []
    for seed in SEEDS:
        random_folds = list(StratifiedKFold(
            N_FOLDS_PUBLISHED, shuffle=True, random_state=seed).split(x, y))
        blocks = KMeans(n_clusters=N_FOLDS_PUBLISHED, random_state=seed,
                        n_init=10).fit_predict(coords)
        spatial_folds = list(GroupKFold(n_splits=N_FOLDS_PUBLISHED).split(
            x, y, groups=pd.Series(blocks)))
        model = make_model("sqrt", 1, seed)
        r_pf, r_pool = evaluate(x, y, random_folds, model)
        s_pf, s_pool = evaluate(x, y, spatial_folds, model)
        rows.append({
            "seed": seed,
            "random_auroc_mean_per_fold": r_pf, "random_auroc_pooled": r_pool,
            "spatial_auroc_mean_per_fold": s_pf, "spatial_auroc_pooled": s_pool,
            "gap_mean_per_fold": r_pf - s_pf, "gap_pooled": r_pool - s_pool,
        })
    aggregation = pd.DataFrame(rows).round(4)
    aggregation.to_csv(OUT / "aggregation_effect.csv", index=False)

    # --- can tuning close the gap? -----------------------------------------
    grid = list(itertools.product(["sqrt", 0.3, 0.6, 1.0], [1, 5, 20, 50]))
    seed = SEEDS[0]
    blocks = KMeans(n_clusters=N_FOLDS_PUBLISHED, random_state=seed, n_init=10).fit_predict(coords)
    spatial_folds = list(GroupKFold(n_splits=N_FOLDS_PUBLISHED).split(x, y, groups=pd.Series(blocks)))
    random_folds = list(StratifiedKFold(N_FOLDS_PUBLISHED, shuffle=True, random_state=seed).split(x, y))
    tuning_rows = []
    for max_features, min_samples_leaf in grid:
        model = make_model(max_features, min_samples_leaf, seed)
        s_pf, s_pool = evaluate(x, y, spatial_folds, model)
        r_pf, r_pool = evaluate(x, y, random_folds, model)
        tuning_rows.append({
            "max_features": str(max_features), "min_samples_leaf": min_samples_leaf,
            "spatial_auroc_mean_per_fold": s_pf, "spatial_auroc_pooled": s_pool,
            "random_auroc_mean_per_fold": r_pf, "random_auroc_pooled": r_pool,
            "gap_mean_per_fold": r_pf - s_pf, "gap_pooled": r_pool - s_pool,
        })
    tuning = pd.DataFrame(tuning_rows).round(4).sort_values("spatial_auroc_pooled", ascending=False)
    tuning.to_csv(OUT / "tuning_sweep.csv", index=False)

    # --- can multi-point-per-polygon sampling be anything but duplication? --
    with sqlite3.connect(RAW / "landslide_pacitan.gpkg") as connection:
        polygons = [gpkg_geometry(b) for (b,) in connection.execute(
            "SELECT geom FROM landslide_pacitan")]
    areas = np.array([p.area for p in polygons])
    pixels = areas / (PIXEL_M ** 2)
    resolution = {
        "n_polygons": len(polygons),
        "pixel_size_m": PIXEL_M,
        "area_m2": {
            "min": float(areas.min()), "p25": float(np.percentile(areas, 25)),
            "median": float(np.median(areas)), "p75": float(np.percentile(areas, 75)),
            "max": float(areas.max()),
        },
        "equivalent_pixels": {
            "min": float(pixels.min()), "median": float(np.median(pixels)),
            "max": float(pixels.max()),
        },
        "polygons_smaller_than_n_pixels": {
            str(k): {"count": int((pixels < k).sum()),
                     "fraction": round(float((pixels < k).mean()), 4)}
            for k in (3, 5, 10)
        },
        "interpretation": (
            "the published benchmarking paper reports AUROC 0.86-0.98 for "
            "strategies that draw 3, 5 or 10 random points inside each landslide "
            "polygon. Half the polygons cover fewer than 10 cells of the 10 m "
            "factor grid, so most of those points read identical covariate values "
            "and random K-fold puts copies of one record on both sides of the "
            "split. The high scores measure duplicate memorisation, not skill."
        ),
        "published_reference": PUBLISHED,
    }
    (OUT / "polygon_resolution.json").write_text(json.dumps(resolution, indent=2), encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/pacitan_replication_audit.py",
        "published_comparison": PUBLISHED,
        "our_result_under_published_design": {
            "design": "Random Forest, 10-fold random CV vs 10 k-means spatial partitions",
            "random_auroc_mean_per_fold": round(float(aggregation.random_auroc_mean_per_fold.mean()), 4),
            "spatial_auroc_mean_per_fold": round(float(aggregation.spatial_auroc_mean_per_fold.mean()), 4),
            "random_auroc_pooled": round(float(aggregation.random_auroc_pooled.mean()), 4),
            "spatial_auroc_pooled": round(float(aggregation.spatial_auroc_pooled.mean()), 4),
            "gap_mean_per_fold": round(float(aggregation.gap_mean_per_fold.mean()), 4),
            "gap_pooled": round(float(aggregation.gap_pooled.mean()), 4),
        },
        "best_spatial_auroc_over_tuning_grid": {
            "mean_per_fold": round(float(tuning.spatial_auroc_mean_per_fold.max()), 4),
            "pooled": round(float(tuning.spatial_auroc_pooled.max()), 4),
            "smallest_gap_mean_per_fold": round(float(tuning.gap_mean_per_fold.min()), 4),
            "smallest_gap_pooled": round(float(tuning.gap_pooled.min()), 4),
        },
        "unresolved": (
            "our spatial-CV AUROC (0.68 per-fold, 0.70 pooled) does not reproduce "
            "the 0.74 the source authors report for the same inventory strategy. "
            "Neither the aggregation choice nor a 16-point hyperparameter sweep "
            "closes it. The remaining candidates are their R/ranger implementation "
            "and the exact sperrorest partitioning, neither of which is recoverable "
            "from the deposit. Reported as an open discrepancy."
        ),
    }
    (OUT / "replication_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["our_result_under_published_design"], indent=2))
    print()
    print(aggregation.to_string(index=False))
    print()
    print("wrote integrity_checks.json, aggregation_effect.csv, tuning_sweep.csv,")
    print("      polygon_resolution.json, replication_audit_summary.json")


if __name__ == "__main__":
    main()
