#!/usr/bin/env python3
"""Random vs spatially-blocked cross-validation on the Pacitan benchmark.

This is the rebuilt version of the ESJ #2026-4228 experiment. The original ran
on a dataset that was 55% duplicate rows with district labels carrying no
spatial information; both defects are documented in
manuscript/esj/REVISION_R1_2026-08-29/FINDINGS_R4_BLOCKING.md.

Two design changes answer Reviewer #4 directly:

  * the spatial partition is k-means blocking on projected coordinates and is
    **re-drawn for every seed**, so the paired observations across seeds are
    genuinely different partitions rather than one partition repeated (R4.3);
  * the blocks are checked for actual spatial separation before use, and the
    check is written into the output (R4.6).

Run:  python3 scripts/pacitan_cv_experiment.py
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pacitan_points.csv"
OUT = ROOT / "results" / "metrics" / "pacitan"

LABEL = "landslide"
CONTINUOUS = [
    "slope_deg",
    "elevation_m",
    "aspect_deg",
    "plan_curvature",
    "profile_curvature",
    "terrain_wetness_index",
    "stream_power_index",
    "distance_to_river_m",
    "distance_to_road_m",
]
CATEGORICAL = ["lithology_class", "landuse_class"]
COORDS = ["x_utm49s", "y_utm49s"]

N_FOLDS = 4
N_BLOCKS = 12          # spatial k-means blocks, then grouped into N_FOLDS
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RANDOM_STATE_GRID = {"n_estimators": 300}


def make_pre() -> ColumnTransformer:
    """A fresh transformer per pipeline.

    Sharing one ColumnTransformer instance across several Pipelines is safe only
    as long as every fit/predict cycle finishes before the next one starts, since
    Pipeline fits its steps in place rather than cloning them. That is a property
    of the loop, not of the code, so it does not survive a refactor. Build a new
    one per estimator instead.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), CONTINUOUS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": Pipeline(
            [("pre", make_pre()), ("model", LogisticRegression(max_iter=2000, random_state=random_state))]
        ),
        "Random Forest": Pipeline(
            [
                ("pre", make_pre()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=RANDOM_STATE_GRID["n_estimators"],
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    from xgboost import XGBClassifier

    models["XGBoost"] = Pipeline(
        [
            ("pre", make_pre()),
            (
                "model",
                XGBClassifier(
                    eval_metric="logloss",
                    n_estimators=RANDOM_STATE_GRID["n_estimators"],
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return models


def spatial_blocks(coords: np.ndarray, seed: int) -> np.ndarray:
    """k-means blocks on projected coordinates, re-drawn per seed."""
    km = KMeans(n_clusters=N_BLOCKS, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def block_separation(coords: np.ndarray, blocks: np.ndarray) -> dict:
    n = len(coords)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)) / 1000.0
    same = blocks[:, None] == blocks[None, :]
    upper = np.triu_indices(n, 1)
    within = float(d[upper][same[upper]].mean())
    between = float(d[upper][~same[upper]].mean())
    work = d.copy()
    np.fill_diagonal(work, np.inf)
    nn = work.argmin(axis=1)
    chance = float((pd.Series(blocks).value_counts(normalize=True) ** 2).sum())
    return {
        "mean_pairwise_km_within_block": round(within, 3),
        "mean_pairwise_km_between_block": round(between, 3),
        "within_over_between": round(within / between, 4),
        "nn_same_block_rate": round(float((blocks == blocks[nn]).mean()), 4),
        "nn_same_block_chance": round(chance, 4),
    }


def score(model, x_tr, y_tr, x_te, y_te) -> dict:
    model.fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)[:, 1]
    pred = model.predict(x_te)
    out = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "n_test": int(len(y_te)),
        "test_pos_frac": float(np.mean(y_te == 1)),
    }
    out["auroc"] = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else float("nan")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    x = df[CONTINUOUS + CATEGORICAL].reset_index(drop=True)
    y = df[LABEL].astype(int).reset_index(drop=True)
    coords = df[COORDS].to_numpy(dtype=float)

    rows: list[dict] = []
    partitions: list[dict] = []
    seen_spatial: set[tuple] = set()

    for seed in SEEDS:
        blocks = spatial_blocks(coords, seed)
        # map the 12 blocks onto 4 folds, keeping whole blocks together
        block_to_fold = {b: i % N_FOLDS for i, b in enumerate(pd.unique(blocks))}
        groups = pd.Series([block_to_fold[b] for b in blocks])

        sep = block_separation(coords, blocks)
        spatial_folds = [
            tuple(sorted(te.tolist()))
            for _, te in GroupKFold(n_splits=N_FOLDS).split(x, y, groups=groups)
        ]
        seen_spatial.add(tuple(spatial_folds))
        partitions.append({"seed": seed, "n_blocks": N_BLOCKS, **sep})

        regimes = {
            "random": list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(x, y)),
            "spatial": list(GroupKFold(n_splits=N_FOLDS).split(x, y, groups=groups)),
        }

        for regime, folds in regimes.items():
            for name, model in build_models(seed).items():
                for fold_idx, (tr, te) in enumerate(folds):
                    metrics = score(model, x.iloc[tr], y.iloc[tr], x.iloc[te], y.iloc[te])
                    rows.append(
                        {
                            "model": name,
                            "regime": regime,
                            "seed": seed,
                            "fold": fold_idx,
                            "n_train": int(len(tr)),
                            **metrics,
                        }
                    )

    per_fold = pd.DataFrame(rows)
    per_fold.round(6).to_csv(OUT / "cv_per_fold.csv", index=False)

    per_seed = (
        per_fold.groupby(["model", "regime", "seed"], as_index=False)
        .agg(auroc=("auroc", "mean"), f1=("f1", "mean"), recall=("recall", "mean"))
    )
    per_seed.round(6).to_csv(OUT / "cv_per_seed.csv", index=False)

    headline = (
        per_seed.groupby(["model", "regime"], as_index=False)
        .agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), f1=("f1", "mean"))
        .round(4)
    )
    headline.to_csv(OUT / "cv_headline.csv", index=False)

    gap_rows = []
    for name in per_seed["model"].unique():
        rnd = per_seed[(per_seed.model == name) & (per_seed.regime == "random")].sort_values("seed")["auroc"].to_numpy()
        spt = per_seed[(per_seed.model == name) & (per_seed.regime == "spatial")].sort_values("seed")["auroc"].to_numpy()
        diff = rnd - spt
        stat, p = stats.wilcoxon(rnd, spt, alternative="greater")
        gap_rows.append(
            {
                "model": name,
                "n_partitions": len(diff),
                "random_auroc_mean": round(float(rnd.mean()), 4),
                "spatial_auroc_mean": round(float(spt.mean()), 4),
                "gap_mean": round(float(diff.mean()), 4),
                "gap_sd": round(float(diff.std(ddof=1)), 4),
                "gap_min": round(float(diff.min()), 4),
                "gap_max": round(float(diff.max()), 4),
                "gap_median": round(float(np.median(diff)), 4),
                "wilcoxon_statistic": float(stat),
                "wilcoxon_p_one_sided": float(p),
            }
        )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(OUT / "cv_regime_gap.csv", index=False)

    design = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/pacitan_cv_experiment.py",
        "input": "data/processed/pacitan_points.csv",
        "n_records": int(len(df)),
        "n_positive": int(y.sum()),
        "n_folds": N_FOLDS,
        "n_spatial_blocks": N_BLOCKS,
        "seeds": SEEDS,
        "distinct_spatial_partitions_across_seeds": len(seen_spatial),
        "spatial_partitions_vary_across_seeds": len(seen_spatial) == len(SEEDS),
        "note_on_the_wilcoxon_test": (
            "the p-value here tests whether the random-minus-spatial gap is "
            "consistently positive across independently drawn spatial "
            "partitions of the same study area. It is not a population "
            "inference: one region, one trigger event, and the partitions "
            "share their underlying data. The gap distribution (mean, SD, "
            "range) carries the result; the p-value only reports sign "
            "consistency across partitions."
        ),
        "block_separation_per_seed": partitions,
        "features_continuous": CONTINUOUS,
        "features_categorical": CATEGORICAL,
        "categorical_encoding": "one-hot, unknown categories ignored",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    (OUT / "experiment_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")

    print(f"records {len(df)}  positives {int(y.sum())}")
    print(f"distinct spatial partitions across {len(SEEDS)} seeds: {len(seen_spatial)}")
    print(f"block separation (seed 42): within/between = {partitions[0]['within_over_between']}")
    print()
    print(headline.to_string(index=False))
    print()
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
