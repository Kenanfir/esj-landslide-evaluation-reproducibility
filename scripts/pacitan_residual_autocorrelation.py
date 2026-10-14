#!/usr/bin/env python3
"""Is the label's spatial autocorrelation a property of the process or the sampling?

The manuscript leans on Moran's I of the landslide label (0.279, z = 22.8) to
argue that spatial blocking is warranted. A reviewer can fairly object that this
is close to guaranteed by construction: the positives are inventory centroids
clustered in the dissected uplands, the negatives are drawn across the whole
regency, so any such design yields a clustered label whatever the underlying
process does.

The test that answers the objection is Moran's I of the model RESIDUALS, which is
also what Roberts et al. recommend for choosing a block size. If the covariates
explain the clustering, the residuals are unstructured and blocking buys little.
If structure survives in the residuals, there is spatial dependence the model has
not captured, and blocking is measuring something real.

Run:  python3 scripts/pacitan_residual_autocorrelation.py
"""

from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pacitan_points.csv"
OUT = ROOT / "results" / "metrics" / "pacitan"

CONTINUOUS = [
    "slope_deg", "elevation_m", "aspect_deg", "plan_curvature", "profile_curvature",
    "terrain_wetness_index", "stream_power_index", "distance_to_river_m",
    "distance_to_road_m",
]
CATEGORICAL = ["lithology_class", "landuse_class"]
K = 8
PERMUTATIONS = 999
SEED = 42


def morans_i(coords: np.ndarray, values: np.ndarray, k: int = K) -> dict:
    """Moran's I with row-standardised k-NN weights on projected coordinates."""
    n = len(values)
    v = values.astype(float)
    deviation = v - v.mean()

    diff = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)

    weights = np.zeros((n, n))
    for i in range(n):
        weights[i, np.argsort(distance[i])[:k]] = 1.0
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

    s0 = weights.sum()
    denominator = float((deviation ** 2).sum())
    observed = (n / s0) * float((weights * np.outer(deviation, deviation)).sum()) / denominator

    rng = np.random.default_rng(0)
    null = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        permuted = rng.permutation(v)
        d = permuted - permuted.mean()
        null[i] = (n / s0) * float((weights * np.outer(d, d)).sum()) / float((d ** 2).sum())
    z = float((observed - null.mean()) / (null.std() + 1e-12))
    p = float((np.abs(null - null.mean()) >= abs(observed - null.mean())).mean())
    return {"morans_i": round(float(observed), 6), "z": round(z, 4),
            "p_two_sided": round(p, 4), "k": k, "permutations": PERMUTATIONS, "n": int(n)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    x = df[CONTINUOUS + CATEGORICAL]
    y = df["landslide"].astype(int).to_numpy()
    coords = df[["x_utm49s", "y_utm49s"]].to_numpy(dtype=float)

    model = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), CONTINUOUS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])),
        ("m", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)),
    ])

    # out-of-fold predictions, so residuals are not in-sample artefacts
    out_of_fold = np.full(len(y), np.nan)
    for train_idx, test_idx in StratifiedKFold(10, shuffle=True, random_state=SEED).split(x, y):
        model.fit(x.iloc[train_idx], y[train_idx])
        out_of_fold[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
    residual = y - out_of_fold

    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/pacitan_residual_autocorrelation.py",
        "question": ("does spatial structure survive in the residuals, or is the label's "
                     "autocorrelation an artefact of presence/pseudo-absence sampling?"),
        "label": morans_i(coords, y.astype(float)),
        "out_of_fold_residual": morans_i(coords, residual),
        "predicted_probability": morans_i(coords, out_of_fold),
        "note": (
            "the label statistic is partly guaranteed by the sampling design; the "
            "residual statistic is not, because it asks what spatial structure the "
            "covariates failed to explain"
        ),
        "environment": {"python": platform.python_version(),
                        "scikit_learn": __import__("sklearn").__version__},
    }
    (OUT / "residual_autocorrelation.json").write_text(json.dumps(findings, indent=2),
                                                       encoding="utf-8")
    for key in ("label", "predicted_probability", "out_of_fold_residual"):
        r = findings[key]
        print(f"  {key:24s} I = {r['morans_i']:+.4f}   z = {r['z']:7.2f}   p = {r['p_two_sided']:.4f}")


if __name__ == "__main__":
    main()
