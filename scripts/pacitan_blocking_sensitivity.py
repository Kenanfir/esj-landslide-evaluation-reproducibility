#!/usr/bin/env python3
"""How much of the random-vs-spatial AUROC gap depends on the blocking choice?

A single spatial-CV number is only meaningful next to the amount of spatial
separation the blocking actually achieved. This sweeps block count and reports
the gap against a physical x-axis: the mean distance from a test point to its
nearest training point. It also checks that each model's own pipeline is
independent, since a shared preprocessing object across pipelines is the kind
of thing that silently couples runs.

Run:  python3 scripts/pacitan_blocking_sensitivity.py
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pacitan_points.csv"
OUT = ROOT / "results" / "metrics" / "pacitan"

LABEL = "landslide"
CONTINUOUS = [
    "slope_deg", "elevation_m", "aspect_deg", "plan_curvature", "profile_curvature",
    "terrain_wetness_index", "stream_power_index", "distance_to_river_m", "distance_to_road_m",
]
CATEGORICAL = ["lithology_class", "landuse_class"]
COORDS = ["x_utm49s", "y_utm49s"]

N_FOLDS = 4
SEEDS = [42, 1, 2, 3, 4]
BLOCK_COUNTS = [4, 6, 8, 12, 20, 40, 80, 160]


def make_pre() -> ColumnTransformer:
    """A fresh transformer per pipeline — never share one across estimators."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), CONTINUOUS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_models(seed: int) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [("pre", make_pre()), ("model", LogisticRegression(max_iter=2000, random_state=seed))]
        ),
        "Random Forest": Pipeline(
            [("pre", make_pre()),
             ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))]
        ),
        "XGBoost": Pipeline(
            [("pre", make_pre()),
             ("model", XGBClassifier(eval_metric="logloss", n_estimators=300,
                                     random_state=seed, n_jobs=-1))]
        ),
    }


def fold_compactness(coords: np.ndarray, folds: np.ndarray) -> dict:
    """Mean member-to-centroid distance within a fold, against the whole area.

    A ratio near or above 1 means the fold is no more compact than the study
    area, i.e. the design is not contiguous whatever it is called.
    """
    spreads = []
    for fold in np.unique(folds):
        member = coords[folds == fold]
        spreads.append(float(np.linalg.norm(member - member.mean(axis=0), axis=1).mean() / 1000))
    whole = float(np.linalg.norm(coords - coords.mean(axis=0), axis=1).mean() / 1000)
    return {"mean_fold_spread_km": round(float(np.mean(spreads)), 3),
            "study_area_spread_km": round(whole, 3)}


def nearest_train_distance(coords: np.ndarray, folds: list) -> float:
    """Mean distance (m) from a test point to its nearest training point."""
    distances = []
    for train_idx, test_idx in folds:
        nn = NearestNeighbors(n_neighbors=1).fit(coords[train_idx])
        d, _ = nn.kneighbors(coords[test_idx])
        distances.append(d[:, 0])
    return float(np.concatenate(distances).mean())


def run_folds(x, y, folds, seed) -> dict[str, dict[str, float]]:
    """Return both aggregations. They answer different questions.

    The mean of per-fold AUROC asks how well the model ranks inside each held-out
    region; the pooled out-of-fold AUROC asks how well it ranks across the whole
    held-out set, which is the regional-transfer question susceptibility mapping
    is actually for. Reporting only the first hides that a design with very
    unequal fold prevalence is being scored on the easier question.
    """
    out: dict[str, dict[str, float]] = {}
    for name, model in build_models(seed).items():
        scores = []
        out_of_fold = np.full(len(y), np.nan)
        for train_idx, test_idx in folds:
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(x.iloc[test_idx])[:, 1]
            out_of_fold[test_idx] = proba
            if len(np.unique(y.iloc[test_idx])) > 1:
                scores.append(roc_auc_score(y.iloc[test_idx], proba))
        scored = ~np.isnan(out_of_fold)
        out[name] = {
            "per_fold": float(np.mean(scores)),
            "pooled": float(roc_auc_score(y[scored], out_of_fold[scored])),
        }
    return out


def fold_class_structure(y, folds) -> dict[str, float]:
    """How unequally a design spreads the classes across its test folds."""
    prevalence = [float(np.mean(y.iloc[test_idx] == 1)) for _, test_idx in folds]
    minority = [int(min((y.iloc[test_idx] == 1).sum(), (y.iloc[test_idx] == 0).sum()))
                for _, test_idx in folds]
    return {"prevalence_sd": float(np.std(prevalence)),
            "min_minority_count": int(min(minority)),
            "n_folds": len(folds)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    x = df[CONTINUOUS + CATEGORICAL].reset_index(drop=True)
    y = df[LABEL].astype(int).reset_index(drop=True)
    coords = df[COORDS].to_numpy(dtype=float)

    rows: list[dict] = []

    for seed in SEEDS:
        random_folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(x, y))
        random_scores = run_folds(x, y, random_folds, seed)
        random_nn = nearest_train_distance(coords, random_folds)
        random_structure = fold_class_structure(y, random_folds)
        for name, scores in random_scores.items():
            rows.append({
                "regime": "random", "n_blocks": None, "assignment": None,
                "seed": seed, "model": name,
                "auroc": scores["per_fold"], "auroc_pooled": scores["pooled"],
                "mean_nearest_train_m": random_nn, **random_structure,
            })

        for n_blocks in BLOCK_COUNTS:
            blocks = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(coords)
            # Order blocks by centroid position, not by the order their labels
            # happen to appear in the input file. The contiguous scheme chunks
            # this sequence, so with an arbitrary order its folds were only
            # accidentally compact — two of four were measurably less compact
            # than the whole study area.
            centroids = np.array([coords[blocks == b].mean(axis=0) for b in np.unique(blocks)])
            # sort along the study area's dominant axis, then by the other
            principal = centroids - centroids.mean(axis=0)
            axis = np.linalg.svd(principal, full_matrices=False)[2][0]
            order = np.unique(blocks)[np.argsort(principal @ axis)]

            for assignment in ("round_robin", "contiguous"):
                if assignment == "round_robin":
                    mapping = {b: i % N_FOLDS for i, b in enumerate(order)}
                else:
                    per = int(np.ceil(len(order) / N_FOLDS))
                    mapping = {b: min(i // per, N_FOLDS - 1) for i, b in enumerate(order)}
                groups = pd.Series([mapping[b] for b in blocks])
                if groups.nunique() < N_FOLDS:
                    continue
                folds = list(GroupKFold(n_splits=N_FOLDS).split(x, y, groups=groups))
                scores = run_folds(x, y, folds, seed)
                nn = nearest_train_distance(coords, folds)
                compactness = fold_compactness(coords, groups.to_numpy())
                structure = fold_class_structure(y, folds)
                for name, s in scores.items():
                    rows.append({
                        "regime": "spatial", "n_blocks": n_blocks, "assignment": assignment,
                        "seed": seed, "model": name,
                        "auroc": s["per_fold"], "auroc_pooled": s["pooled"],
                        "mean_nearest_train_m": nn,
                        "fold_spread_km": compactness["mean_fold_spread_km"],
                        "study_area_spread_km": compactness["study_area_spread_km"],
                        **structure,
                    })

        # leave-one-block-out at a coarse blocking: the strongest separation available
        for n_blocks in (4, 6, 8):
            blocks = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(coords)
            groups = pd.Series(blocks)
            folds = list(GroupKFold(n_splits=n_blocks).split(x, y, groups=groups))
            folds = [(tr, te) for tr, te in folds if len(np.unique(y.iloc[te])) > 1]
            scores = run_folds(x, y, folds, seed)
            nn = nearest_train_distance(coords, folds)
            compactness = fold_compactness(coords, groups.to_numpy())
            structure = fold_class_structure(y, folds)
            for name, s in scores.items():
                rows.append({
                    "regime": "spatial", "n_blocks": n_blocks, "assignment": "leave_one_block_out",
                    "seed": seed, "model": name,
                    "auroc": s["per_fold"], "auroc_pooled": s["pooled"],
                    "mean_nearest_train_m": nn,
                    "fold_spread_km": compactness["mean_fold_spread_km"],
                    "study_area_spread_km": compactness["study_area_spread_km"],
                    **structure,
                })

    per_run = pd.DataFrame(rows)
    per_run.round(6).to_csv(OUT / "blocking_sensitivity_per_run.csv", index=False)

    baseline = (
        per_run[per_run.regime == "random"].groupby("model", as_index=False)
        .agg(random_auroc=("auroc", "mean"), random_auroc_pooled=("auroc_pooled", "mean"),
             random_nn_m=("mean_nearest_train_m", "mean"))
    )
    spatial = (
        per_run[per_run.regime == "spatial"]
        .groupby(["assignment", "n_blocks", "model"], as_index=False)
        .agg(spatial_auroc=("auroc", "mean"), spatial_auroc_sd=("auroc", "std"),
             spatial_auroc_pooled=("auroc_pooled", "mean"),
             mean_nearest_train_m=("mean_nearest_train_m", "mean"),
             fold_spread_km=("fold_spread_km", "mean"),
             study_area_spread_km=("study_area_spread_km", "mean"),
             prevalence_sd=("prevalence_sd", "mean"),
             min_minority_count=("min_minority_count", "mean"))
    )
    summary = spatial.merge(baseline, on="model")
    summary["gap"] = (summary.random_auroc - summary.spatial_auroc).round(4)
    summary["gap_pooled"] = (summary.random_auroc_pooled - summary.spatial_auroc_pooled).round(4)
    summary = summary.round(4).sort_values(["assignment", "model", "n_blocks"])
    summary.to_csv(OUT / "blocking_sensitivity_summary.csv", index=False)

    design = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/pacitan_blocking_sensitivity.py",
        "n_records": int(len(df)),
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "block_counts": BLOCK_COUNTS,
        "assignments": ["round_robin", "contiguous", "leave_one_block_out"],
        "random_baseline_auroc": baseline.round(4).to_dict(orient="records"),
        "note": (
            "mean_nearest_train_m is the mean distance from a held-out point to "
            "its nearest training point. It is the physical measure of how much "
            "separation the blocking bought, and the gap should be read against "
            "it rather than against the block count."
        ),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
        },
    }
    (OUT / "blocking_sensitivity_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")

    print(baseline.round(4).to_string(index=False))
    print()
    for assignment in ("round_robin", "contiguous", "leave_one_block_out"):
        sub = summary[summary.assignment == assignment]
        if sub.empty:
            continue
        print(f"=== {assignment} ===")
        pivot = sub.pivot_table(index=["n_blocks", "mean_nearest_train_m"], columns="model", values="gap")
        print(pivot.round(4).to_string())
        print()


if __name__ == "__main__":
    main()
