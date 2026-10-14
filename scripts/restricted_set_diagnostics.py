#!/usr/bin/env python3
"""Diagnostics for the restricted blocking set and the leg-three decomposition.

Section 4.3 reports a correlation as an absence rather than as an estimate, and
Section 4.5 reports a two-step decomposition. Both make claims about numbers
that are derived from the sweep rather than read off it, so they are computed
here and written to a metric file instead of being typed into the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACITAN = ROOT / "results" / "metrics" / "pacitan"
COMPACTNESS_LIMIT = 0.80


def main() -> None:
    sweep = pd.read_csv(PACITAN / "blocking_sensitivity_summary.csv")
    forest = sweep[sweep["model"] == "Random Forest"].copy()
    forest["fold_spread_ratio"] = forest["fold_spread_km"] / forest["study_area_spread_km"]
    distinct = forest.drop_duplicates(subset=["n_blocks", "mean_nearest_train_m", "gap"])
    restricted = distinct[distinct["fold_spread_ratio"] < COMPACTNESS_LIMIT]

    def correlation(frame: pd.DataFrame) -> float:
        return float(np.corrcoef(frame["mean_nearest_train_m"], frame["gap"])[0, 1])

    most_separated = restricted["mean_nearest_train_m"].idxmax()
    sampling = pd.read_csv(PACITAN / "sampling_strategy_summary.csv").set_index("points_per_polygon")
    low, high = sampling.loc[1], sampling.loc[10]

    payload = {
        "compactness_limit": COMPACTNESS_LIMIT,
        "restricted": {
            "n_partitions": int(len(restricted)),
            "correlation_separation_gap": round(correlation(restricted), 4),
            "correlation_dropping_most_separated": round(
                correlation(restricted.drop(index=most_separated)), 4),
            "most_separated_n_blocks": int(restricted.loc[most_separated, "n_blocks"]),
            "separation_sd_m": round(float(restricted["mean_nearest_train_m"].std()), 0),
            "gap_min": round(float(restricted["gap"].min()), 3),
            "gap_max": round(float(restricted["gap"].max()), 3),
            "n_labelled_contiguous": int((restricted["assignment"] == "contiguous").sum()),
        },
        "full_sweep": {
            "n_partitions": int(len(distinct)),
            "separation_sd_m": round(float(distinct["mean_nearest_train_m"].std()), 0),
        },
        "leg_three_decomposition_at_k10": {
            "random_cv": round(float(high["random_cv_auroc"]), 4),
            "leave_one_landslide_out": round(float(high["lolo_auroc"]), 4),
            "blocked_by_region": round(float(high["spatial_cv_auroc"]), 4),
            "pseudo_replication_step": round(
                float(high["random_cv_auroc"] - high["lolo_auroc"]), 3),
            "regional_extrapolation_step": round(
                float(high["lolo_auroc"] - high["spatial_cv_auroc"]), 3),
        },
        "grouping_removes_fraction_of_inflation": round(float(
            1 - (high["lolo_auroc"] - low["lolo_auroc"])
            / (high["random_cv_auroc"] - low["random_cv_auroc"])), 3),
    }
    out = PACITAN / "restricted_set_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
