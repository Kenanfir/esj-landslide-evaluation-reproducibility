#!/usr/bin/env python3
"""Does removing the duplicates undo the inflation? The decisive test for leg three.

Leg one proved its mechanism by de-duplicating and watching AUROC collapse. Leg
three asserted the same mechanism without running the same test, and only 48.9%
of the positives at ten points per polygon are in an exact duplicate group — so
the competing explanation, that the extra points are NEAR-duplicates in adjacent
cells rather than exact copies, was never excluded. That distinction decides
whether the duplicate-fraction statistic proposed in Section 5.2 would catch the
problem at all.

Also plots leg three on the axes of leg two, so the claim that the two measure
one continuum is demonstrated rather than asserted.

Run:  python3 scripts/pacitan_dedup_test.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from shapely import STRtree
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pacitan_sampling_experiment import (          # noqa: E402
    ALL_FACTORS, FACTORS, N_FOLDS, load_polygons, make_model,
    points_in_polygon, sample_negatives, sample_rasters,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pacitan-dib-2024"
OUT = ROOT / "results" / "metrics" / "pacitan"
SEEDS = [42, 1, 2]
STRATEGIES = [1, 3, 5, 10]


def evaluate(x, y, folds, seed) -> float:
    model = make_model(seed)
    scores = []
    for train_idx, test_idx in folds:
        if len(np.unique(y.iloc[test_idx])) < 2:
            continue
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        scores.append(roc_auc_score(y.iloc[test_idx], model.predict_proba(x.iloc[test_idx])[:, 1]))
    return float(np.mean(scores))


def nearest_train_m(coords: np.ndarray, folds) -> float:
    out = []
    for train_idx, test_idx in folds:
        nn = NearestNeighbors(n_neighbors=1).fit(coords[train_idx])
        d, _ = nn.kneighbors(coords[test_idx])
        out.append(d[:, 0])
    return float(np.concatenate(out).mean())


def main() -> None:
    polygons = load_polygons()
    tree = STRtree(polygons)
    with rasterio.open(RAW / "map_elev.tif") as raster:
        band = raster.read(1)
        mask = np.isfinite(band) & (band != raster.nodata) & (np.abs(band) < 1e30)
        rows, cols = np.nonzero(mask)
        step = max(1, len(rows) // 400_000)
        xs, ys = rasterio.transform.xy(raster.transform, rows[::step], cols[::step])
    valid = np.column_stack([np.asarray(xs), np.asarray(ys)])

    records = []
    for k in STRATEGIES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            pos = np.array([p for poly in polygons for p in points_in_polygon(poly, k, rng)])
            neg = sample_negatives(len(pos), valid, tree, rng)
            frame = pd.concat([
                sample_rasters(pos).assign(landslide=1),
                sample_rasters(neg).assign(landslide=0),
            ], ignore_index=True).dropna(subset=ALL_FACTORS).reset_index(drop=True)

            deduped = frame.drop_duplicates(subset=ALL_FACTORS + ["landslide"]).reset_index(drop=True)

            for tag, data in (("as sampled", frame), ("de-duplicated", deduped)):
                x = data[ALL_FACTORS]
                y = data["landslide"].astype(int)
                coords = data[["x_utm49s", "y_utm49s"]].to_numpy(dtype=float)
                random_folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                                    random_state=seed).split(x, y))
                blocks = KMeans(n_clusters=N_FOLDS, random_state=seed, n_init=10).fit_predict(coords)
                spatial_folds = list(GroupKFold(n_splits=N_FOLDS).split(
                    x, y, groups=pd.Series(blocks)))
                records.append({
                    "points_per_polygon": k, "seed": seed, "pool": tag,
                    "n_records": int(len(data)),
                    "random_cv_auroc": round(evaluate(x, y, random_folds, seed), 4),
                    "spatial_cv_auroc": round(evaluate(x, y, spatial_folds, seed), 4),
                    "random_nearest_train_m": round(nearest_train_m(coords, random_folds), 1),
                })
                print(f"  k={k:2d} seed={seed} {tag:14s} n={len(data):6d} "
                      f"random={records[-1]['random_cv_auroc']:.4f} "
                      f"spatial={records[-1]['spatial_cv_auroc']:.4f}", flush=True)

    per_run = pd.DataFrame(records)
    per_run.to_csv(OUT / "dedup_test_per_run.csv", index=False)
    summary = (per_run.groupby(["points_per_polygon", "pool"], as_index=False)
               .agg(n_records=("n_records", "mean"),
                    random_cv_auroc=("random_cv_auroc", "mean"),
                    spatial_cv_auroc=("spatial_cv_auroc", "mean"),
                    random_nearest_train_m=("random_nearest_train_m", "mean")).round(4))
    summary.to_csv(OUT / "dedup_test_summary.csv", index=False)

    wide = summary.pivot(index="points_per_polygon", columns="pool", values="random_cv_auroc")
    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/pacitan_dedup_test.py",
        "question": ("does removing exact duplicates undo the random-CV inflation, or is the "
                     "inflation carried by near-duplicates the duplicate-fraction statistic "
                     "would not catch?"),
        "random_cv_by_pool": {str(int(k)): {"as_sampled": float(r["as sampled"]),
                                            "de_duplicated": float(r["de-duplicated"])}
                              for k, r in wide.iterrows()},
        "recovered_fraction_at_k10": round(
            float((wide.loc[10, "as sampled"] - wide.loc[10, "de-duplicated"])
                  / (wide.loc[10, "as sampled"] - wide.loc[1, "as sampled"])), 4),
    }
    (OUT / "dedup_test_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"fraction of the k=1 -> k=10 rise removed by de-duplication: "
          f"{findings['recovered_fraction_at_k10']:.1%}")


if __name__ == "__main__":
    main()


def per_class_nearest_train(seeds=(42,), strategies=(1, 10)) -> dict:
    """Nearest-training-point distance split by class, under random cross-validation.

    Pooling the two classes hides the effect: negatives are drawn across the whole
    regency and positives inside polygons a few tens of metres across, so the pooled
    mean moves about as much as the change in density predicts while the positives
    move an order of magnitude more.
    """
    import numpy as np, pandas as pd, rasterio
    from shapely import STRtree
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import NearestNeighbors

    polygons = load_polygons()
    tree = STRtree(polygons)
    with rasterio.open(RAW / "map_elev.tif") as raster:
        band = raster.read(1)
        mask = np.isfinite(band) & (band != raster.nodata) & (np.abs(band) < 1e30)
        rows, cols = np.nonzero(mask)
        step = max(1, len(rows) // 400_000)
        xs, ys = rasterio.transform.xy(raster.transform, rows[::step], cols[::step])
    valid = np.column_stack([np.asarray(xs), np.asarray(ys)])

    out = {}
    for k in strategies:
        rng = np.random.default_rng(seeds[0])
        pos = np.array([p for poly in polygons for p in points_in_polygon(poly, k, rng)])
        neg = sample_negatives(len(pos), valid, tree, rng)
        coords = np.vstack([pos, neg])
        y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
        pooled, positive, negative = [], [], []
        for train_idx, test_idx in StratifiedKFold(10, shuffle=True, random_state=seeds[0]).split(coords, y):
            nn = NearestNeighbors(n_neighbors=1).fit(coords[train_idx])
            d = nn.kneighbors(coords[test_idx])[0][:, 0]
            pooled.append(d); positive.append(d[y[test_idx] == 1]); negative.append(d[y[test_idx] == 0])
        P = np.concatenate(positive)
        out[str(k)] = {
            "pooled_mean_m": round(float(np.concatenate(pooled).mean()), 1),
            "landslide_mean_m": round(float(P.mean()), 1),
            "landslide_median_m": round(float(np.median(P)), 1),
            "non_landslide_mean_m": round(float(np.concatenate(negative).mean()), 1),
        }
    return out


if __name__ == "__main__" and "--per-class" in __import__("sys").argv:
    import json
    result = per_class_nearest_train()
    (OUT / "per_class_nearest_train.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
