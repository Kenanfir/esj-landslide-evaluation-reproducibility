#!/usr/bin/env python3
"""Reviewer-driven audit for ESJ #2026-4228 round 1.

Answers, from the data and the committed pipeline only, the six points
Reviewer #4 raised. Every number the response document cites comes from
this script and lands under results/metrics/revision_r1/.

Run:  python3 scripts/revision_r1_audit.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import FEATURE_COLUMNS, RAW_FILES  # noqa: E402
from spatial_cv import N_FOLDS, SEEDS, morans_i  # noqa: E402
from train_model import TARGET_COLUMN, get_model_specs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT = ROOT / "results" / "metrics" / "revision_r1"

# The four district labels the manuscript calls the "Indonesian Java pool".
JAVA_DISTRICTS = ["Bogor", "Cianjur", "Semarang", "Yogyakarta"]

# Identity key for a physical observation: position plus the five predictors.
# Rounded to 6 dp — ~0.1 m in coordinates, far below any measurement precision
# these variables could carry.
DEDUP_PRECISION = 6
IDENTITY_KEY = ["longitude", "latitude", *FEATURE_COLUMNS]

EARTH_RADIUS_KM = 6371.0


def load_retired_points(processed_dir: Path) -> pd.DataFrame:
    """Read the retired dataset, preferring the CSV and falling back to its JSON twin.

    The repository ignores `*.csv`, so a fresh clone carries
    `landslide_points.json` and not `landslide_points.csv`. Both are written by
    scripts/prepare_dataset.py from the same frame, so either reproduces this
    analysis; without the fallback the audit only ran on a machine that had
    already generated the CSV locally.
    """
    csv = processed_dir / "landslide_points.csv"
    if csv.exists():
        return pd.read_csv(csv)
    twin = processed_dir / "landslide_points.json"
    if twin.exists():
        return pd.read_json(twin)
    raise FileNotFoundError(
        f"Neither {csv.name} nor {twin.name} found in {processed_dir}. "
        "Run scripts/prepare_dataset.py first."
    )


def load_pools() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (all records, the 1,011 manuscript pool, its de-duplicated form)."""
    every = load_retired_points(PROCESSED_DIR)
    pool = every[every["district"].isin(JAVA_DISTRICTS)].copy().reset_index(drop=True)
    rounded = {c: pool[c].round(DEDUP_PRECISION) for c in IDENTITY_KEY}
    dedup = (
        pool.assign(**rounded)
        .drop_duplicates(subset=[*IDENTITY_KEY, "district", TARGET_COLUMN])
        .reset_index(drop=True)
    )
    return every, pool, dedup


# --- R4.1 -----------------------------------------------------------------

def r4_1_pool_construction(every: pd.DataFrame, pool: pd.DataFrame, dedup: pd.DataFrame) -> dict:
    """Reconcile every source record to the 1,011 figure, then to the unique count."""
    per_source = []
    for filename in RAW_FILES:
        rows = every[every["source_file"] == filename]
        kept = rows[rows["district"].isin(JAVA_DISTRICTS)]
        per_source.append(
            {
                "source_file": filename,
                "records_in_file": int(len(rows)),
                "retained_in_java_pool": int(len(kept)),
                "dropped_non_java_district": int(len(rows) - len(kept)),
                "distinct_after_dedup": int(
                    kept.assign(**{c: kept[c].round(DEDUP_PRECISION) for c in IDENTITY_KEY})
                    .drop_duplicates(subset=[*IDENTITY_KEY, "district", TARGET_COLUMN])
                    .shape[0]
                ),
            }
        )

    # How much does each file contribute that no earlier file already had?
    seen: set[tuple] = set()
    unique_contribution = {}
    for filename in RAW_FILES:
        rows = every[(every["source_file"] == filename) & every["district"].isin(JAVA_DISTRICTS)]
        keys = {
            tuple(v)
            for v in rows.assign(**{c: rows[c].round(DEDUP_PRECISION) for c in IDENTITY_KEY})[
                [*IDENTITY_KEY, "district", TARGET_COLUMN]
            ].values.tolist()
        }
        unique_contribution[filename] = int(len(keys - seen))
        seen |= keys

    # The duplicate count depends on the precision at which identity is judged,
    # and the manuscript quotes all three. Persist them rather than leaving the
    # paper to cite a number computed once in a shell.
    key = [*IDENTITY_KEY, "district", TARGET_COLUMN]
    # Count text-precision duplicates over the SAME pool the paper describes: the
    # Java-filtered rows, not every row in the files. Counting the unfiltered
    # files gave 8.2% against the pool's true 21.7%.
    raw_rows: list[str] = []
    for name in ("Landslide_Dataset_450.csv", "Landslide_Dataset_766.csv"):
        path = ROOT / "data" / "raw" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            fields = line.split(",")
            if fields[2].strip() in set(JAVA_DISTRICTS):
                raw_rows.append(",".join(fields[:8]))
    text_distinct = len(set(raw_rows)) if raw_rows else None
    text_total = len(raw_rows) if raw_rows else None

    return {
        "duplicate_fraction_by_precision": {
            "raw_source_text": round(1 - text_distinct / text_total, 4) if text_distinct else None,
            "float_exact": round(1 - pool.drop_duplicates(subset=[c for c in key]).shape[0] / len(pool), 4),
            "six_decimal_places": round(1 - dedup.shape[0] / len(pool), 4),
        },
        "total_records_all_sources": int(len(every)),
        "manuscript_pool_records": int(len(pool)),
        "filtering_rule": (
            "keep rows whose district label is one of "
            f"{JAVA_DISTRICTS}; no de-duplication is applied by "
            "scripts/prepare_dataset.py, which concatenates the three source CSVs"
        ),
        "per_source": per_source,
        "unique_contribution_in_load_order": unique_contribution,
        "duplicate_rows_in_pool": int(len(pool) - len(dedup)),
        "distinct_records_in_pool": int(len(dedup)),
    }


# --- R4.2 -----------------------------------------------------------------

def r4_2_leakage(pool: pd.DataFrame, dedup: pd.DataFrame) -> dict:
    """Is `stability` a proxy for the label? Does `prediction` reach the features?"""
    rows = []
    for frame_name, frame in (("pool_1011", pool), ("dedup", dedup)):
        y = frame[TARGET_COLUMN].astype(int)
        for column in [*FEATURE_COLUMNS, "prediction"]:
            rows.append(
                {
                    "pool": frame_name,
                    "column": column,
                    "in_feature_set": column in FEATURE_COLUMNS,
                    "univariate_auroc": round(float(roc_auc_score(y, frame[column])), 4),
                    "pearson_r_with_label": round(float(np.corrcoef(frame[column], y)[0, 1]), 4),
                    "min": round(float(frame[column].min()), 4),
                    "max": round(float(frame[column].max()), 4),
                }
            )
    table = pd.DataFrame(rows)

    y = dedup[TARGET_COLUMN].astype(int)
    uniformity = []
    for column in FEATURE_COLUMNS:
        v = dedup[column].to_numpy(dtype=float)
        lo, hi = v.min(), v.max()
        ks = stats.kstest((v - lo) / (hi - lo), "uniform")
        uniformity.append(
            {
                "feature": column,
                "observed_min": round(float(lo), 4),
                "observed_max": round(float(hi), 4),
                "ks_D_vs_uniform": round(float(ks.statistic), 4),
                "ks_p_value": round(float(ks.pvalue), 4),
            }
        )

    agreement = float((pool["prediction"] == pool[TARGET_COLUMN]).mean())
    return {
        "feature_columns_used_by_pipeline": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "prediction_column_in_feature_set": "prediction" in FEATURE_COLUMNS,
        "prediction_label_agreement": round(agreement, 4),
        "per_column": table.to_dict(orient="records"),
        "uniformity_test_dedup": uniformity,
    }, table


# --- R4.3 -----------------------------------------------------------------

def r4_3_fold_identity(pool: pd.DataFrame) -> dict:
    """Are the district-grouped folds identical across the five seeds?"""
    x = pool[FEATURE_COLUMNS].reset_index(drop=True)
    y = pool[TARGET_COLUMN].astype(int).reset_index(drop=True)
    groups = pool["district"].reset_index(drop=True)

    signatures = {}
    for seed in SEEDS:
        spatial = [
            tuple(sorted(test.tolist()))
            for _, test in GroupKFold(n_splits=N_FOLDS).split(x, y, groups=groups)
        ]
        random_ = [
            tuple(sorted(test.tolist()))
            for _, test in StratifiedKFold(
                n_splits=N_FOLDS, shuffle=True, random_state=seed
            ).split(x, y)
        ]
        signatures[seed] = {"spatial": hash(tuple(spatial)), "random": hash(tuple(random_))}

    spatial_hashes = {s["spatial"] for s in signatures.values()}
    random_hashes = {s["random"] for s in signatures.values()}
    return {
        "n_seeds": len(SEEDS),
        "distinct_spatial_partitions": len(spatial_hashes),
        "distinct_random_partitions": len(random_hashes),
        "spatial_folds_identical_across_seeds": len(spatial_hashes) == 1,
        "mechanism": (
            "scripts/spatial_cv.py calls GroupKFold(n_splits=4), which takes no "
            "random_state and is deterministic given the group vector; the seed "
            "reaches only get_model_specs(random_state=seed)"
        ),
        "n_districts_in_pool": int(pool["district"].nunique()),
    }


# --- R4.4 -----------------------------------------------------------------

def r4_4_model_identity() -> dict:
    specs = get_model_specs(random_state=42)
    try:
        import xgboost

        xgboost_version = xgboost.__version__
        xgboost_available = True
    except ModuleNotFoundError:
        xgboost_version = None
        xgboost_available = False

    booster = specs.get("XGBoost")
    return {
        "xgboost_importable_in_this_environment": xgboost_available,
        "xgboost_version": xgboost_version,
        "model_names_instantiated": list(specs),
        "gradient_boosting_fallback_active": "Gradient Boosting" in specs,
        "xgboost_params": (
            {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
             for k, v in booster.get_params().items()}
            if booster is not None
            else None
        ),
        "explicitly_set_params": {
            "eval_metric": "logloss",
            "n_estimators": 300,
            "random_state": "the CV seed (42, 1, 2, 3, 4); 42 for the holdout run",
        },
    }


# --- R4.5 -----------------------------------------------------------------

def r4_5_morans(pool: pd.DataFrame, dedup: pd.DataFrame, every: pd.DataFrame) -> dict:
    modelling_1216 = every[every["case_study_eligible"] == 0].reset_index(drop=True)
    runs = {}
    for tag, frame in (
        ("pool_1011", pool),
        ("dedup", dedup),
        ("unfiltered_1216", modelling_1216),
    ):
        result = morans_i(
            frame[["latitude", "longitude"]].to_numpy(dtype=float),
            frame[TARGET_COLUMN].to_numpy(dtype=int),
            k=8,
        )
        runs[tag] = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in result.items()}

    return {
        "method_as_implemented": {
            "weights": "k = 8 nearest neighbours, binary, self excluded",
            "distance": "haversine great-circle km on (latitude, longitude)",
            "symmetrized": False,
            "note_on_symmetry": (
                "the k-NN adjacency is built row-wise and never symmetrized; "
                "row standardisation is applied afterwards, so W is asymmetric"
            ),
            "row_standardised": True,
            "permutations": 999,
            "z_score_source": "permutation distribution, not the analytic variance",
            "p_value": "two-sided, from the permutation distribution",
            "rng_seed": 0,
        },
        "runs": runs,
        "manuscript_reports": {
            "morans_i": 0.216,
            "z_score": 12.4,
            "permutations_claimed": 999,
            "pool_claimed": 1011,
        },
    }


# --- R4.6 -----------------------------------------------------------------

def _haversine_matrix(frame: pd.DataFrame) -> np.ndarray:
    lat = np.radians(np.asarray(frame["latitude"], dtype=float))
    lon = np.radians(np.asarray(frame["longitude"], dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def r4_6_memorisation(pool: pd.DataFrame, dedup: pd.DataFrame) -> dict:
    """Duplicate structure, and whether the district groups are spatially separated."""
    keyed = pool.assign(**{c: pool[c].round(DEDUP_PRECISION) for c in IDENTITY_KEY})
    grouped = keyed.groupby(IDENTITY_KEY, sort=False)
    sizes = grouped.size()
    multi = grouped.filter(lambda g: len(g) > 1)
    per_group = multi.groupby(IDENTITY_KEY, sort=False).agg(
        districts=("district", "nunique"), labels=(TARGET_COLUMN, "nunique")
    )

    duplicates = {
        "rows_in_pool": int(len(pool)),
        "distinct_records": int(sizes.shape[0]),
        "duplicate_rows": int(len(pool) - sizes.shape[0]),
        "duplicate_row_fraction": round(float(1 - sizes.shape[0] / len(pool)), 4),
        "largest_duplicate_group": int(sizes.max()),
        "duplicate_groups": int(per_group.shape[0]),
        "duplicate_groups_spanning_multiple_districts": int((per_group["districts"] > 1).sum()),
        "duplicate_groups_with_mixed_labels": int((per_group["labels"] > 1).sum()),
    }

    distances = _haversine_matrix(dedup)
    district = np.asarray(dedup["district"], dtype=object)
    label = np.asarray(dedup[TARGET_COLUMN], dtype=int)
    same_district = district[:, None] == district[None, :]
    upper = np.triu_indices(len(dedup), 1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    district_share = float((pd.Series(district).value_counts(normalize=True) ** 2).sum())
    label_share = float((pd.Series(label).value_counts(normalize=True) ** 2).sum())
    np.fill_diagonal(distances, 0.0)

    separation = {
        "pool": "dedup",
        "n": int(len(dedup)),
        "mean_pairwise_km_within_district": round(
            float(distances[upper][same_district[upper]].mean()), 3
        ),
        "mean_pairwise_km_between_district": round(
            float(distances[upper][~same_district[upper]].mean()), 3
        ),
        "nn_same_district_rate": round(float((district == district[nearest]).mean()), 4),
        "nn_same_district_chance": round(district_share, 4),
        "nn_same_label_rate": round(float((label == label[nearest]).mean()), 4),
        "nn_same_label_chance": round(label_share, 4),
    }
    return {"duplicates": duplicates, "district_spatial_separation": separation}


# --- the decisive re-run ---------------------------------------------------

def cv_comparison(pool: pd.DataFrame, dedup: pd.DataFrame) -> pd.DataFrame:
    """Re-run both CV regimes on the pool as submitted and on its unique records."""
    rows: list[dict[str, object]] = []
    for tag, frame in (("pool_1011", pool), ("dedup", dedup)):
        x = frame[FEATURE_COLUMNS].reset_index(drop=True)
        y = frame[TARGET_COLUMN].astype(int).reset_index(drop=True)
        groups = frame["district"].reset_index(drop=True)
        for seed in SEEDS:
            for regime in ("random", "spatial"):
                if regime == "random":
                    folds = list(
                        StratifiedKFold(
                            n_splits=N_FOLDS, shuffle=True, random_state=seed
                        ).split(x, y)
                    )
                else:
                    folds = list(GroupKFold(n_splits=N_FOLDS).split(x, y, groups=groups))
                for name, model in get_model_specs(random_state=seed).items():
                    scores = []
                    for train_idx, test_idx in folds:
                        model.fit(x.iloc[train_idx], y.iloc[train_idx])
                        proba = model.predict_proba(x.iloc[test_idx])[:, 1]
                        if len(np.unique(y.iloc[test_idx])) > 1:
                            scores.append(float(roc_auc_score(y.iloc[test_idx], proba)))
                    rows.append(
                        {
                            "pool": tag,
                            "model": name,
                            "regime": regime,
                            "seed": seed,
                            "auroc_mean_over_folds": round(float(np.mean(scores)), 6),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    every, pool, dedup = load_pools()

    leakage, leakage_table = r4_2_leakage(pool, dedup)
    findings = {
        "r4_1_pool_construction": r4_1_pool_construction(every, pool, dedup),
        "r4_2_leakage": leakage,
        "r4_3_fold_identity": r4_3_fold_identity(pool),
        "r4_4_model_identity": r4_4_model_identity(),
        "r4_5_morans_i": r4_5_morans(pool, dedup, every),
        "r4_6_memorisation": r4_6_memorisation(pool, dedup),
    }

    cv = cv_comparison(pool, dedup)
    cv.to_csv(OUT / "cv_regime_by_pool.csv", index=False)

    summary = (
        cv.groupby(["pool", "model", "regime"], as_index=False)
        .agg(auroc=("auroc_mean_over_folds", "mean"), auroc_sd=("auroc_mean_over_folds", "std"))
        .round(4)
    )
    summary.to_csv(OUT / "cv_regime_by_pool_summary.csv", index=False)

    gaps = summary.pivot_table(index=["pool", "model"], columns="regime", values="auroc")
    gaps["random_minus_spatial"] = (gaps["random"] - gaps["spatial"]).round(4)
    gaps.reset_index().to_csv(OUT / "cv_regime_gap_by_pool.csv", index=False)

    leakage_table.to_csv(OUT / "univariate_signal.csv", index=False)
    (OUT / "reviewer4_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    import sklearn
    import scipy

    environment = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/revision_r1_audit.py",
        "input": "data/processed/landslide_points.csv (or its .json twin)",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": findings["r4_4_model_identity"]["xgboost_version"],
        "config": {
            "java_districts": JAVA_DISTRICTS,
            "dedup_precision_dp": DEDUP_PRECISION,
            "identity_key": IDENTITY_KEY,
            "n_folds": N_FOLDS,
            "seeds": SEEDS,
            "morans_k": 8,
        },
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    print(json.dumps(findings, indent=2))
    print()
    print(gaps.reset_index().to_string(index=False))
    print()
    print(f"Wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
