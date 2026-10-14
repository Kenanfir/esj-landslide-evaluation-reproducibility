#!/usr/bin/env python3
"""Run the integrity gate on both datasets and write the side-by-side record.

The manuscript reports the gate as a table comparing the retired dataset with
the Pacitan benchmark. That table has to come from a file rather than from
prose, so this script produces it.

Run:  python3 scripts/gate_comparison.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metrics" / "gate_comparison.json"
SCRATCH = ROOT / "results" / "metrics" / "_gate_tmp"

JAVA = ["Bogor", "Cianjur", "Semarang", "Yogyakarta"]
OLD_FEATURES = ["stability", "rainfall", "vegetation", "river_distance", "slope"]
NEW_FEATURES = [
    "slope_deg", "elevation_m", "aspect_deg", "plan_curvature", "profile_curvature",
    "terrain_wetness_index", "stream_power_index", "distance_to_river_m",
    "distance_to_road_m", "lithology_class", "landuse_class",
]


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


def run_gate(csv: Path, name: str, label: str, features: list[str], x: str, y: str,
             crs: str, group: str | None) -> dict:
    target = SCRATCH / f"{name}.json"
    command = [
        sys.executable, str(ROOT / "scripts" / "validate_dataset.py"), str(csv),
        "--label", label,
        "--features", ",".join(features),
        "--x", x, "--y", y, "--coord-crs", crs, "--json", str(target),
    ]
    if group:
        command += ["--group", group]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if not target.exists():
        raise RuntimeError(f"gate produced no report for {name}: {completed.stderr.strip()[-400:]}")
    return json.loads(target.read_text())


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)

    old = load_retired_points(ROOT / "data" / "processed")
    java = old[old.district.isin(JAVA)].copy()
    key = ["longitude", "latitude", *OLD_FEATURES]
    dedup = (java.assign(**{c: java[c].round(6) for c in key})
             .drop_duplicates(subset=[*key, "district", "ground_truth"]))
    java.to_csv(SCRATCH / "retired_pool.csv", index=False)
    dedup.to_csv(SCRATCH / "retired_dedup.csv", index=False)

    results = {
        "retired_as_assembled": run_gate(
            SCRATCH / "retired_pool.csv", "retired_as_assembled", "ground_truth",
            OLD_FEATURES, "longitude", "latitude", "lonlat", "district"),
        "retired_deduplicated": run_gate(
            SCRATCH / "retired_dedup.csv", "retired_deduplicated", "ground_truth",
            OLD_FEATURES, "longitude", "latitude", "lonlat", "district"),
        "pacitan": run_gate(
            ROOT / "data" / "processed" / "pacitan_points.csv", "pacitan", "landslide",
            NEW_FEATURES, "x_utm49s", "y_utm49s", "utm", None),
    }

    # Validation reports store the input filename they were given. Keep the
    # checked-in aggregate portable by recording a repository-relative path.
    for report in results.values():
        dataset_path = Path(report["dataset"]).resolve()
        try:
            report["dataset"] = dataset_path.relative_to(ROOT).as_posix()
        except ValueError:
            report["dataset"] = dataset_path.name

    def summarise(report: dict) -> dict:
        by_name = {c["check"]: c for c in report["checks"]}
        duplicates = by_name["duplicate_rows"]
        leakage = by_name["single_feature_label_leakage"]
        synthetic = by_name["features_are_measurements_not_noise"]
        neighbours = by_name["near_duplicate_neighbours_in_feature_space"]
        moran = by_name.get("label_spatial_autocorrelation", {})
        grouping = by_name.get("grouping_variable_is_spatial")
        return {
            "n_rows": report["n_rows"],
            "verdict": report["verdict"],
            "failed_checks": report["failed_checks"],
            "duplicate_row_fraction": duplicates["duplicate_row_fraction"],
            "distinct_rows": duplicates["distinct_rows"],
            "max_single_feature_separation": leakage["max_separation"],
            "features_tested": synthetic["features_tested"],
            "features_indistinguishable_from_uniform": synthetic["features_indistinguishable_from_uniform"],
            "fraction_coincident_neighbours": neighbours["fraction_nn_distance_below_1e-6"],
            "nn_label_agreement": neighbours["nn_label_agreement"],
            "nn_label_agreement_chance": neighbours["nn_label_agreement_chance"],
            "morans_i": moran.get("morans_i"),
            "morans_z": moran.get("z_score_permutation"),
            "morans_p": moran.get("p_two_sided_permutation"),
            "group_within_over_between": grouping["within_over_between"] if grouping else None,
            "group_nn_same_rate": grouping["nn_same_group_rate"] if grouping else None,
            "group_nn_same_chance": grouping["nn_same_group_chance"] if grouping else None,
        }

    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/gate_comparison.py",
        "gate": "scripts/validate_dataset.py",
        "datasets": {name: summarise(report) for name, report in results.items()},
        "full_reports": results,
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    table = pd.DataFrame(record["datasets"]).T
    print(table[["n_rows", "verdict", "duplicate_row_fraction", "distinct_rows",
                 "max_single_feature_separation", "features_indistinguishable_from_uniform",
                 "fraction_coincident_neighbours", "morans_i", "morans_z",
                 "group_within_over_between"]].to_string())
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
