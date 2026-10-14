#!/usr/bin/env python3
"""Dataset integrity gate — run this before a dataset is modelled on.

Written after the ESJ #2026-4228 round-1 audit, which found that the pool the
submitted paper was built on was 55% duplicate rows, carried five predictors
indistinguishable from uniform noise, and used group labels with no spatial
meaning. Each of those is a check here. A dataset that fails any FAIL-level
check does not get modelled on.

Usage:
    python3 scripts/validate_dataset.py <csv> --label <col> \
        [--features a,b,c] [--x <col> --y <col>] [--group <col>] \
        [--coord-crs utm|lonlat] [--json <out.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EARTH_RADIUS_KM = 6371.0

# A check fails when it crosses these. They are deliberately loose: each one
# is set where the answer stops being arguable.
THRESHOLDS = {
    "duplicate_row_fraction": 0.02,      # >2% identical rows is a pipeline bug
    "leakage_univariate_auroc": 0.98,    # a single column that nearly is the label
    "uniform_ks_p": 0.20,                # indistinguishable from a uniform draw
    "nn_zero_distance_fraction": 0.10,   # near-duplicates in feature space
    "group_separation_ratio": 0.90,      # within/between mean pairwise distance
}


def _round_key(frame: pd.DataFrame, columns: list[str], dp: int) -> pd.DataFrame:
    return frame[columns].round(dp)


def check_duplicates(df: pd.DataFrame, key: list[str], dp: int) -> dict:
    exact = int(df.duplicated(subset=key).sum())
    rounded = int(_round_key(df, key, dp).duplicated().sum())
    fraction = rounded / len(df) if len(df) else 0.0
    return {
        "check": "duplicate_rows",
        "n_rows": int(len(df)),
        "exact_duplicate_rows": exact,
        "duplicate_rows_at_%ddp" % dp: rounded,
        "duplicate_row_fraction": round(float(fraction), 4),
        "distinct_rows": int(len(df) - rounded),
        "threshold": THRESHOLDS["duplicate_row_fraction"],
        "status": "FAIL" if fraction > THRESHOLDS["duplicate_row_fraction"] else "PASS",
    }


def check_leakage(df: pd.DataFrame, features: list[str], label: str) -> dict:
    y = df[label].astype(int)
    rows = []
    worst = 0.0
    for column in features:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().all():
            continue
        auroc = float(roc_auc_score(y, values.fillna(values.median())))
        separation = max(auroc, 1.0 - auroc)
        worst = max(worst, separation)
        rows.append(
            {
                "feature": column,
                "univariate_auroc": round(auroc, 4),
                "separation": round(separation, 4),
                "pearson_r": round(float(np.corrcoef(values.fillna(values.median()), y)[0, 1]), 4),
            }
        )
    return {
        "check": "single_feature_label_leakage",
        "per_feature": rows,
        "max_separation": round(worst, 4),
        "threshold": THRESHOLDS["leakage_univariate_auroc"],
        "status": "FAIL" if worst >= THRESHOLDS["leakage_univariate_auroc"] else "PASS",
        "note": (
            "separation is max(AUROC, 1-AUROC): a perfectly inverted predictor "
            "leaks exactly as much as a perfectly aligned one"
        ),
    }


def check_synthetic(df: pd.DataFrame, features: list[str]) -> dict:
    rows = []
    uniform_like = 0
    for column in features:
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 20 or values.max() == values.min():
            continue
        scaled = (values - values.min()) / (values.max() - values.min())
        ks_uniform = stats.kstest(scaled, "uniform")
        is_uniform = ks_uniform.pvalue > THRESHOLDS["uniform_ks_p"]
        uniform_like += int(is_uniform)
        rows.append(
            {
                "feature": column,
                "observed_min": round(float(values.min()), 4),
                "observed_max": round(float(values.max()), 4),
                "distinct_values": int(len(np.unique(values))),
                "ks_p_vs_uniform": round(float(ks_uniform.pvalue), 4),
                "indistinguishable_from_uniform": bool(is_uniform),
            }
        )
    return {
        "check": "features_are_measurements_not_noise",
        "per_feature": rows,
        "features_indistinguishable_from_uniform": uniform_like,
        "features_tested": len(rows),
        "threshold": f"more than half of the features uniform at p > {THRESHOLDS['uniform_ks_p']}",
        "status": "FAIL" if rows and uniform_like > len(rows) / 2 else "PASS",
        "note": (
            "real terrain and climate covariates are skewed; a full set of "
            "predictors that each match a uniform draw over a round interval "
            "is a generated file, not a measured one"
        ),
    }


def check_feature_space_neighbours(df: pd.DataFrame, features: list[str], label: str) -> dict:
    from sklearn.neighbors import NearestNeighbors

    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median())
    x = StandardScaler().fit_transform(numeric)
    y = df[label].astype(int).to_numpy()
    neighbours = NearestNeighbors(n_neighbors=2).fit(x)
    distances, indices = neighbours.kneighbors(x)
    nn_distance = distances[:, 1]
    zero_fraction = float((nn_distance < 1e-6).mean())
    return {
        "check": "near_duplicate_neighbours_in_feature_space",
        "min_nn_distance": round(float(nn_distance.min()), 6),
        "median_nn_distance": round(float(np.median(nn_distance)), 6),
        "fraction_nn_distance_below_1e-6": round(zero_fraction, 4),
        "nn_label_agreement": round(float((y == y[indices[:, 1]]).mean()), 4),
        "nn_label_agreement_chance": round(
            float((pd.Series(y).value_counts(normalize=True) ** 2).sum()), 4
        ),
        "threshold": THRESHOLDS["nn_zero_distance_fraction"],
        "status": "FAIL" if zero_fraction > THRESHOLDS["nn_zero_distance_fraction"] else "PASS",
        "note": (
            "a high rate of coincident neighbours means random K-fold will put "
            "the same record on both sides of the split"
        ),
    }


def _pairwise_km(df: pd.DataFrame, xcol: str, ycol: str, crs: str) -> np.ndarray:
    x = np.asarray(df[xcol], dtype=float)
    y = np.asarray(df[ycol], dtype=float)
    if crs == "lonlat":
        lat = np.radians(y)
        lon = np.radians(x)
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
        )
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    # projected metres
    return np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2) / 1000.0


def check_label_spatial_structure(distances: np.ndarray, label: np.ndarray, k: int = 8) -> dict:
    """Moran's I on the label with row-standardised k-NN weights, permutation z."""
    n = len(label)
    y = label.astype(float)
    deviation = y - y.mean()

    weights = np.zeros((n, n))
    for i in range(n):
        # Exclude the point by index, not by sort position. With coincident
        # duplicate coordinates the zero-distance tie-break does not reliably
        # place the point first, so slicing from 1 leaves it a self-weight and
        # inflates the statistic by counting each label as its own neighbour.
        order = np.argsort(distances[i])
        nearest = order[order != i][:k]
        weights[i, nearest] = 1.0
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

    s0 = weights.sum()
    denominator = float((deviation ** 2).sum())
    observed = (
        (n / s0) * float((weights * np.outer(deviation, deviation)).sum()) / denominator
        if denominator > 0
        else float("nan")
    )

    rng = np.random.default_rng(0)
    null = np.empty(999)
    for i in range(999):
        permuted = rng.permutation(y)
        pdev = permuted - permuted.mean()
        null[i] = (n / s0) * float((weights * np.outer(pdev, pdev)).sum()) / float((pdev ** 2).sum())
    z = float((observed - null.mean()) / (null.std() + 1e-12))
    p = float((np.abs(null - null.mean()) >= abs(observed - null.mean())).mean())

    return {
        "check": "label_spatial_autocorrelation",
        "morans_i": round(float(observed), 6),
        "expected_under_null": round(-1.0 / (n - 1), 6),
        "z_score_permutation": round(z, 4),
        "p_two_sided_permutation": round(p, 4),
        "permutations": 999,
        "k_nearest_neighbours": k,
        "status": "INFO",
        "note": (
            "informational, not pass/fail: this is the condition that makes "
            "spatial cross-validation necessary, so a near-zero I means the "
            "spatial-CV argument does not apply to this data"
        ),
    }


def check_group_is_spatial(distances: np.ndarray, groups: np.ndarray) -> dict:
    n = len(groups)
    same = groups[:, None] == groups[None, :]
    upper = np.triu_indices(n, 1)
    within = float(distances[upper][same[upper]].mean())
    between = float(distances[upper][~same[upper]].mean())
    ratio = within / between if between else float("nan")

    working = distances.copy()
    np.fill_diagonal(working, np.inf)
    nearest = working.argmin(axis=1)
    chance = float((pd.Series(groups).value_counts(normalize=True) ** 2).sum())
    return {
        "check": "grouping_variable_is_spatial",
        "mean_pairwise_km_within_group": round(within, 3),
        "mean_pairwise_km_between_group": round(between, 3),
        "within_over_between": round(ratio, 4),
        "nn_same_group_rate": round(float((groups == groups[nearest]).mean()), 4),
        "nn_same_group_chance": round(chance, 4),
        "n_groups": int(pd.Series(groups).nunique()),
        "threshold": THRESHOLDS["group_separation_ratio"],
        "status": "FAIL" if ratio > THRESHOLDS["group_separation_ratio"] else "PASS",
        "note": (
            "grouped K-fold only blocks on space when the groups are spatially "
            "separated; a ratio near 1.0 means the groups are interleaved and "
            "the fold structure carries no spatial meaning"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--features", default=None, help="comma-separated; default = all other numeric columns")
    parser.add_argument("--x", default=None)
    parser.add_argument("--y", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--coord-crs", default="lonlat", choices=["lonlat", "utm"])
    parser.add_argument("--dedup-dp", type=int, default=6)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.features:
        features = [c.strip() for c in args.features.split(",")]
    else:
        skip = {args.label, args.x, args.y, args.group}
        features = [
            c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]

    key = [*features] + ([args.x, args.y] if args.x and args.y else [])
    checks = [
        check_duplicates(df, key, args.dedup_dp),
        check_leakage(df, features, args.label),
        check_synthetic(df, features),
        check_feature_space_neighbours(df, features, args.label),
    ]

    if args.x and args.y:
        distances = _pairwise_km(df, args.x, args.y, args.coord_crs)
        checks.append(
            check_label_spatial_structure(distances, df[args.label].astype(int).to_numpy())
        )
        if args.group:
            checks.append(
                check_group_is_spatial(distances, np.asarray(df[args.group], dtype=object))
            )

    failed = [c["check"] for c in checks if c["status"] == "FAIL"]
    report = {
        "dataset": str(args.csv),
        "n_rows": int(len(df)),
        "label": args.label,
        "features": features,
        "verdict": "FAIL" if failed else "PASS",
        "failed_checks": failed,
        "checks": checks,
    }

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {report['verdict']}" + (f"  ({', '.join(failed)})" if failed else ""), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
