#!/usr/bin/env python3
"""What multi-point-per-polygon inventory sampling actually does to AUROC.

Samodra & Nugroho report AUROC rising from ~0.74 to 0.86-0.98 as the inventory
handling strategy moves from one point per landslide polygon to 3, 5 or 10
random points inside it, and read that as more samples giving a better model.
This reproduces the effect and measures what is behind it: with a median polygon
of ~10.7 cells on a 10 m factor grid, extra points inside one polygon mostly land
in cells that have already been sampled, so random K-fold splits copies of one
record across train and test.

For each strategy the script reports random-CV AUROC, spatially-blocked AUROC,
and the fraction of the positive points that are exact duplicates of another
positive point in the 11-dimensional factor space.

Run:  python3 scripts/pacitan_sampling_experiment.py
"""

from __future__ import annotations

import json
import platform
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
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
OUT = ROOT / "results" / "metrics" / "pacitan"

FACTORS = {
    "map_slope": "slope_deg", "map_elev": "elevation_m", "map_aspect": "aspect_deg",
    "map_planc": "plan_curvature", "map_profc": "profile_curvature",
    "map_twi": "terrain_wetness_index", "map_spi": "stream_power_index",
    "map_distriv": "distance_to_river_m", "map_distroad": "distance_to_road_m",
    "map_geologi": "lithology_class", "map_landuse": "landuse_class",
}
CATEGORICAL = ["lithology_class", "landuse_class"]
CONTINUOUS = [v for v in FACTORS.values() if v not in CATEGORICAL]
ALL_FACTORS = CONTINUOUS + CATEGORICAL

STRATEGIES = [1, 3, 5, 10]
SEEDS = [42, 1, 2, 3, 4]
N_FOLDS = 10


def gpkg_geometry(blob: bytes):
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 0x07]
    return wkb.loads(blob[8 + envelope:])


def load_polygons() -> list:
    with sqlite3.connect(RAW / "landslide_pacitan.gpkg") as connection:
        return [gpkg_geometry(b) for (b,) in connection.execute(
            "SELECT geom FROM landslide_pacitan")]


SLIVER_FALLBACKS = {"count": 0}


def points_in_polygon(polygon, k: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    """k uniformly random points inside one polygon, by rejection sampling.

    Every k uses the same rule, k = 1 included. An earlier version returned
    `representative_point()` for k = 1, which made the base of the sampling
    trend a deterministic draw while every other row was random — the one row
    that had to be comparable was produced differently.
    """
    minx, miny, maxx, maxy = polygon.bounds
    out: list[tuple[float, float]] = []
    guard = 0
    while len(out) < k and guard < 20000:
        xs = rng.uniform(minx, maxx, 64)
        ys = rng.uniform(miny, maxy, 64)
        for x, y in zip(xs, ys):
            if len(out) == k:
                break
            if polygon.contains(Point(x, y)):
                out.append((float(x), float(y)))
        guard += 64
    while len(out) < k:                       # degenerate slivers
        SLIVER_FALLBACKS["count"] += 1
        centre = polygon.representative_point()
        out.append((centre.x, centre.y))
    return out


def sample_negatives(n: int, valid_xy: np.ndarray, tree: STRtree,
                     rng: np.random.Generator) -> np.ndarray:
    """Random cells of the study area that are not inside a landslide polygon."""
    chosen: list[np.ndarray] = []
    have = 0
    while have < n:
        idx = rng.integers(0, len(valid_xy), size=(n - have) * 2 + 64)
        cand = valid_xy[idx]
        keep = [len(tree.query(Point(x, y), predicate="intersects")) == 0 for x, y in cand]
        cand = cand[np.asarray(keep)]
        if len(cand):
            chosen.append(cand)
            have += len(cand)
    return np.vstack(chosen)[:n]


def sample_rasters(xy: np.ndarray) -> pd.DataFrame:
    coords = [(float(a), float(b)) for a, b in xy]
    frame = pd.DataFrame({"x_utm49s": xy[:, 0], "y_utm49s": xy[:, 1]})
    for stem, column in FACTORS.items():
        with rasterio.open(RAW / f"{stem}.tif") as raster:
            values = np.array([v[0] for v in raster.sample(coords)], dtype="float64")
            nodata = raster.nodata
        if nodata is not None:
            values = np.where(np.isclose(values, nodata), np.nan, values)
        values = np.where(np.abs(values) > 1e30, np.nan, values)
        frame[column] = values
    return frame


def make_model(seed: int) -> Pipeline:
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), CONTINUOUS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])),
        ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
    ])


def evaluate(x, y, folds, seed) -> float:
    model = make_model(seed)
    scores = []
    for train_idx, test_idx in folds:
        if len(np.unique(y.iloc[test_idx])) < 2:
            continue
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        scores.append(roc_auc_score(y.iloc[test_idx], model.predict_proba(x.iloc[test_idx])[:, 1]))
    return float(np.mean(scores))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    polygons = load_polygons()
    tree = STRtree(polygons)

    # the study area: every cell the elevation raster actually covers
    with rasterio.open(RAW / "map_elev.tif") as raster:
        band = raster.read(1)
        mask = np.isfinite(band) & (band != raster.nodata) & (np.abs(band) < 1e30)
        rows, cols = np.nonzero(mask)
        step = max(1, len(rows) // 400_000)          # thin the candidate pool
        rows, cols = rows[::step], cols[::step]
        xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
    valid_xy = np.column_stack([np.asarray(xs), np.asarray(ys)])
    print(f"study-area candidate cells for negative sampling: {len(valid_xy):,}")

    records = []
    for k in STRATEGIES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            SLIVER_FALLBACKS["count"] = 0
            positive_xy = np.array(
                [p for polygon in polygons for p in points_in_polygon(polygon, k, rng)]
            )
            negative_xy = sample_negatives(len(positive_xy), valid_xy, tree, rng)

            # The polygon each positive came from is attached BEFORE the dropna, not
            # reconstructed from row position afterwards. Reconstructing it as i // k
            # is wrong the moment a single row is dropped for missing factor values:
            # every positive after the gap shifts into a neighbouring polygon's group,
            # which splits same-polygon points across folds and puts back the very
            # leakage the grouping exists to remove.
            positives = sample_rasters(positive_xy).assign(
                landslide=1, polygon_id=np.arange(len(positive_xy)) // k)
            negatives = sample_rasters(negative_xy).assign(
                landslide=0, polygon_id=-1 - np.arange(len(negative_xy)))
            frame = pd.concat([positives, negatives], ignore_index=True) \
                .dropna(subset=ALL_FACTORS).reset_index(drop=True)

            x = frame[ALL_FACTORS]
            y = frame["landslide"].astype(int)
            coords = frame[["x_utm49s", "y_utm49s"]].to_numpy(dtype=float)

            def involved_in_duplicate(subset: pd.DataFrame) -> float:
                """Fraction of rows sharing their factor vector with another row.

                `keep="first"` would exclude the first member of each group, which
                is not what "points that duplicate another point" means.
                """
                return float(subset.duplicated(subset=ALL_FACTORS, keep=False).mean())

            positives = frame[frame.landslide == 1]
            negatives = frame[frame.landslide == 0]
            duplicate_fraction = involved_in_duplicate(positives)
            negative_duplicate_fraction = involved_in_duplicate(negatives)

            random_folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                                random_state=seed).split(x, y))

            # Design A: leave-one-block-out over 10 k-means blocks — the scheme the
            # source study uses, and the one Table 3 shows gives the largest gaps.
            blocks = KMeans(n_clusters=N_FOLDS, random_state=seed, n_init=10).fit_predict(coords)
            spatial_folds = list(GroupKFold(n_splits=N_FOLDS).split(
                x, y, groups=pd.Series(blocks)))

            # Design C: leave-one-landslide-out. Folds are grouped by the polygon a
            # point came from, not by region. This is the remedy the literature names
            # for pseudo-replication among points of one mapped failure, and it is the
            # experiment that separates that effect from regional extrapolation, which
            # spatial blocking confounds. Positive points are generated polygon by
            # polygon, k at a time, so the polygon index is i // k; negatives are given
            # their own singleton groups so they can fall anywhere.
            lolo_folds = list(GroupKFold(n_splits=N_FOLDS).split(
                x, y, groups=frame["polygon_id"]))

            # Design B: 12 blocks assigned round robin to 4 folds — a milder scheme
            # from the other end of Table 3, so the flat line is not one design's artefact.
            blocks_b = KMeans(n_clusters=12, random_state=seed, n_init=10).fit_predict(coords)
            mapping = {b: i % 4 for i, b in enumerate(pd.unique(blocks_b))}
            spatial_folds_b = list(GroupKFold(n_splits=4).split(
                x, y, groups=pd.Series([mapping[b] for b in blocks_b])))

            record = {
                "points_per_polygon": k,
                "seed": seed,
                "n_records": int(len(frame)),
                "n_positive": int(y.sum()),
                "positive_duplicate_fraction": round(duplicate_fraction, 4),
                "positive_duplicate_fraction_keep_first": round(float(
                    positives.duplicated(subset=ALL_FACTORS).mean()), 4),
                "negative_duplicate_fraction": round(negative_duplicate_fraction, 4),
                "sliver_fallbacks": SLIVER_FALLBACKS["count"],
                "random_cv_auroc": round(evaluate(x, y, random_folds, seed), 4),
                "spatial_cv_auroc": round(evaluate(x, y, spatial_folds, seed), 4),
                "spatial_cv_auroc_round_robin_12": round(evaluate(x, y, spatial_folds_b, seed), 4),
                "lolo_cv_auroc": round(evaluate(x, y, lolo_folds, seed), 4),
            }
            record["gap"] = round(record["random_cv_auroc"] - record["spatial_cv_auroc"], 4)
            records.append(record)
            print(f"  k={k:2d} seed={seed}: n={record['n_records']:5d} "
                  f"dup={duplicate_fraction:.3f} "
                  f"random={record['random_cv_auroc']:.4f} "
                  f"spatial={record['spatial_cv_auroc']:.4f}")

    per_run = pd.DataFrame(records)
    per_run.to_csv(OUT / "sampling_strategy_per_run.csv", index=False)

    summary = (
        per_run.groupby("points_per_polygon", as_index=False)
        .agg(n_records=("n_records", "mean"),
             positive_duplicate_fraction=("positive_duplicate_fraction", "mean"),
             negative_duplicate_fraction=("negative_duplicate_fraction", "mean"),
             sliver_fallbacks=("sliver_fallbacks", "mean"),
             random_cv_auroc=("random_cv_auroc", "mean"),
             random_cv_sd=("random_cv_auroc", "std"),
             spatial_cv_auroc=("spatial_cv_auroc", "mean"),
             spatial_cv_sd=("spatial_cv_auroc", "std"),
             spatial_rr12_auroc=("spatial_cv_auroc_round_robin_12", "mean"),
             spatial_rr12_sd=("spatial_cv_auroc_round_robin_12", "std"),
             lolo_auroc=("lolo_cv_auroc", "mean"),
             lolo_sd=("lolo_cv_auroc", "std"),
             gap=("gap", "mean"))
        .round(4)
    )
    summary.to_csv(OUT / "sampling_strategy_summary.csv", index=False)

    design = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/pacitan_sampling_experiment.py",
        "strategies_points_per_polygon": STRATEGIES,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "blocking_designs": {
            "primary": "leave-one-block-out over 10 k-means blocks (the source study's scheme)",
            "secondary": "12 k-means blocks assigned round robin to 4 folds",
            "leave_one_landslide_out": (
                "folds grouped by the polygon each point came from, negatives in "
                "singleton groups; separates within-failure pseudo-replication from "
                "regional extrapolation, which spatial blocking confounds"
            ),
        },
        "duplicate_fraction_definition": (
            "fraction of points sharing their 11-factor vector with at least one "
            "other point of the same class (pandas keep=False)"
        ),
        "model": "RandomForestClassifier(n_estimators=300)",
        "negative_sampling": (
            "uniform over cells the elevation raster covers, rejecting any cell "
            "inside a landslide polygon, matched one-to-one to the positive count"
        ),
        "published_claim": {
            "source": "Samodra & Nugroho, doi:10.1016/j.aiig.2024.100093",
            "one_point_strategies_auroc": [0.70, 0.78],
            "multi_point_strategies_auroc": [0.86, 0.98],
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": __import__("sklearn").__version__,
            "rasterio": rasterio.__version__,
        },
    }
    (OUT / "sampling_strategy_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")

    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
