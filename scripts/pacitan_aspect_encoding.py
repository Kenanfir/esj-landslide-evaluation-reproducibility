#!/usr/bin/env python3
"""Does treating aspect as a linear variable change any conclusion?

Aspect is circular: 1 degree and 359 degrees are adjacent, but a linear encoding
puts them at opposite ends of the range. The source study treats it linearly and
we followed, which the manuscript's limitations section waved off as something
that "disadvantages all models equally". That is testable rather than assertable,
and it matters here for a specific reason: aspect is the second-strongest single
predictor in this dataset, and a tree ensemble can split around the discontinuity
while a linear model cannot — so the claim disadvantages LOGISTIC REGRESSION
most, and the paper's Random-Forest-over-logistic-regression margin is one of the
numbers it uses to argue that model choice matters less than evaluation design.

Compares linear aspect against a sin/cos encoding under both CV regimes.

Run:  python3 scripts/pacitan_aspect_encoding.py
"""

from __future__ import annotations

import json
import platform
import warnings
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pacitan_points.csv"
OUT = ROOT / "results" / "metrics" / "pacitan"

BASE_CONTINUOUS = [
    "slope_deg", "elevation_m", "plan_curvature", "profile_curvature",
    "terrain_wetness_index", "stream_power_index", "distance_to_river_m",
    "distance_to_road_m",
]
CATEGORICAL = ["lithology_class", "landuse_class"]
N_FOLDS = 4
N_BLOCKS = 12
SEEDS = [42, 1, 2, 3, 4]


def build(continuous: list[str], seed: int) -> dict[str, Pipeline]:
    def pre() -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler(), continuous),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
    return {
        "Logistic Regression": Pipeline([("pre", pre()),
            ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
        "Random Forest": Pipeline([("pre", pre()),
            ("m", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))]),
        "XGBoost": Pipeline([("pre", pre()),
            ("m", XGBClassifier(eval_metric="logloss", n_estimators=300,
                                random_state=seed, n_jobs=-1))]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    radians = np.radians(df["aspect_deg"].to_numpy(dtype=float))
    df["aspect_sin"] = np.sin(radians)
    df["aspect_cos"] = np.cos(radians)
    y = df["landslide"].astype(int).reset_index(drop=True)
    coords = df[["x_utm49s", "y_utm49s"]].to_numpy(dtype=float)

    encodings = {
        "linear": BASE_CONTINUOUS + ["aspect_deg"],
        "sin_cos": BASE_CONTINUOUS + ["aspect_sin", "aspect_cos"],
    }

    rows: list[dict] = []
    for name, continuous in encodings.items():
        x = df[continuous + CATEGORICAL].reset_index(drop=True)
        for seed in SEEDS:
            blocks = KMeans(n_clusters=N_BLOCKS, random_state=seed, n_init=10).fit_predict(coords)
            mapping = {b: i % N_FOLDS for i, b in enumerate(pd.unique(blocks))}
            groups = pd.Series([mapping[b] for b in blocks])
            regimes = {
                "random": list(StratifiedKFold(N_FOLDS, shuffle=True,
                                               random_state=seed).split(x, y)),
                "spatial": list(GroupKFold(n_splits=N_FOLDS).split(x, y, groups=groups)),
            }
            for regime, folds in regimes.items():
                for model_name, model in build(continuous, seed).items():
                    scores = []
                    for train_idx, test_idx in folds:
                        model.fit(x.iloc[train_idx], y.iloc[train_idx])
                        proba = model.predict_proba(x.iloc[test_idx])[:, 1]
                        scores.append(roc_auc_score(y.iloc[test_idx], proba))
                    rows.append({"encoding": name, "regime": regime, "seed": seed,
                                 "model": model_name, "auroc": float(np.mean(scores))})

    per_run = pd.DataFrame(rows)
    per_run.round(6).to_csv(OUT / "aspect_encoding_per_run.csv", index=False)

    summary = (per_run.groupby(["encoding", "regime", "model"], as_index=False)
               .agg(auroc=("auroc", "mean"), sd=("auroc", "std")).round(4))
    summary.to_csv(OUT / "aspect_encoding_summary.csv", index=False)

    def cell(encoding: str, regime: str, model: str) -> float:
        row = summary[(summary.encoding == encoding) & (summary.regime == regime)
                      & (summary.model == model)]
        return float(row.auroc.iloc[0])

    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/pacitan_aspect_encoding.py",
        "question": "does a circular encoding of aspect change any conclusion?",
        "design": f"{N_BLOCKS} k-means blocks round robin into {N_FOLDS} folds, {len(SEEDS)} seeds",
        "rf_over_lr_margin_random": {
            "linear": round(cell("linear", "random", "Random Forest")
                            - cell("linear", "random", "Logistic Regression"), 4),
            "sin_cos": round(cell("sin_cos", "random", "Random Forest")
                             - cell("sin_cos", "random", "Logistic Regression"), 4),
        },
        "logistic_regression_change": {
            regime: round(cell("sin_cos", regime, "Logistic Regression")
                          - cell("linear", regime, "Logistic Regression"), 4)
            for regime in ("random", "spatial")
        },
        "gap_by_encoding": {
            encoding: {
                model: round(cell(encoding, "random", model) - cell(encoding, "spatial", model), 4)
                for model in ("Logistic Regression", "Random Forest", "XGBoost")
            } for encoding in encodings
        },
        "environment": {"python": platform.python_version(),
                        "scikit_learn": __import__("sklearn").__version__},
    }
    (OUT / "aspect_encoding_findings.json").write_text(json.dumps(findings, indent=2),
                                                       encoding="utf-8")
    print(summary.to_string(index=False))
    print()
    print(json.dumps({k: v for k, v in findings.items()
                      if k in ("rf_over_lr_margin_random", "logistic_regression_change",
                               "gap_by_encoding")}, indent=2))


if __name__ == "__main__":
    main()
