#!/usr/bin/env python3
"""Prepare the Bogor landslide dataset for modeling and paper evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_FILES = [
    "Landslide_District_Bogor_GroundTruth_50.csv",
    "Landslide_Dataset_450.csv",
    "Landslide_Dataset_766.csv",
]

FEATURE_COLUMNS = [
    "stability",
    "rainfall",
    "vegetation",
    "river_distance",
    "slope",
]

LABEL_COLUMNS = ["ground_truth", "prediction"]

NORMALIZED_COLUMNS = [
    "longitude",
    "latitude",
    "district",
    *FEATURE_COLUMNS,
    *LABEL_COLUMNS,
    "source_file",
    "case_study_eligible",
]

COLUMN_VARIANTS = {
    "longitude": ["Longitude"],
    "latitude": ["Latitude"],
    "district": ["District"],
    "stability": ["Stability"],
    "rainfall": ["Rainfall"],
    "vegetation": ["Vegetation"],
    "river_distance": ["River_Dist", "River_distance", "River_Distance"],
    "slope": ["Slope"],
    "ground_truth": ["Ground_Truth", "GroundTruth"],
    "prediction": ["Predicted", "Prediction"],
}

CASE_STUDY_FILE = "Landslide_District_Bogor_GroundTruth_50.csv"


def _find_source_column(frame: pd.DataFrame, variants: list[str], normalized_name: str) -> str:
    lookup = {column.strip().lower(): column for column in frame.columns}
    for variant in variants:
        match = lookup.get(variant.lower())
        if match is not None:
            return match
    raise ValueError(
        f"Missing required column for {normalized_name!r}; expected one of {variants}"
    )


def normalize_frame(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Return one source CSV using the normalized landslide schema."""

    normalized = pd.DataFrame()
    for normalized_name, variants in COLUMN_VARIANTS.items():
        source_column = _find_source_column(frame, variants, normalized_name)
        normalized[normalized_name] = frame[source_column]

    normalized["district"] = normalized["district"].astype(str).str.strip()
    normalized["source_file"] = source_file
    normalized["case_study_eligible"] = int(source_file == CASE_STUDY_FILE)

    numeric_columns = ["longitude", "latitude", *FEATURE_COLUMNS]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(float)

    for column in LABEL_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
        invalid = sorted(set(normalized[column].dropna()) - {0, 1})
        if invalid:
            raise ValueError(f"{column} labels must contain only 0 and 1; found {invalid}")

    return normalized[NORMALIZED_COLUMNS]


def load_raw_dataset(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for filename in RAW_FILES:
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw dataset file: {path}")
        frames.append(normalize_frame(pd.read_csv(path), filename))
    return pd.concat(frames, ignore_index=True)


def build_dataset_summary(points: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy summary CSV for source, district, labels, and features."""

    rows: list[dict[str, object]] = []

    for source_file, group in points.groupby("source_file", sort=True):
        rows.extend(
            [
                {
                    "summary_type": "source_file",
                    "source_file": source_file,
                    "district": "",
                    "metric": "records",
                    "value": int(len(group)),
                },
                {
                    "summary_type": "source_file",
                    "source_file": source_file,
                    "district": "",
                    "metric": "positive_labels",
                    "value": int(group["ground_truth"].sum()),
                },
                {
                    "summary_type": "source_file",
                    "source_file": source_file,
                    "district": "",
                    "metric": "negative_labels",
                    "value": int((group["ground_truth"] == 0).sum()),
                },
                {
                    "summary_type": "source_file",
                    "source_file": source_file,
                    "district": "",
                    "metric": "case_study_eligible_records",
                    "value": int(group["case_study_eligible"].sum()),
                },
            ]
        )

    district_counts = (
        points.groupby(["source_file", "district"], sort=True)
        .agg(records=("district", "size"), positive_labels=("ground_truth", "sum"))
        .reset_index()
    )
    for row in district_counts.itertuples(index=False):
        rows.append(
            {
                "summary_type": "district",
                "source_file": row.source_file,
                "district": row.district,
                "metric": "records",
                "value": int(row.records),
            }
        )
        rows.append(
            {
                "summary_type": "district",
                "source_file": row.source_file,
                "district": row.district,
                "metric": "positive_labels",
                "value": int(row.positive_labels),
            }
        )

    for source_file, group in points.groupby("source_file", sort=True):
        for feature in FEATURE_COLUMNS:
            stats = group[feature].agg(["min", "max", "mean", "std"])
            for metric, value in stats.items():
                rows.append(
                    {
                        "summary_type": "feature",
                        "source_file": source_file,
                        "district": "",
                        "metric": f"{feature}_{metric}",
                        "value": round(float(value), 6),
                    }
                )

    return pd.DataFrame(rows, columns=["summary_type", "source_file", "district", "metric", "value"])


def write_dataset_summary_table(points: pd.DataFrame, output_path: Path) -> None:
    source_summary = (
        points.groupby("source_file", sort=True)
        .agg(
            records=("source_file", "size"),
            positive_labels=("ground_truth", "sum"),
            negative_labels=("ground_truth", lambda values: int((values == 0).sum())),
            case_study_eligible_records=("case_study_eligible", "sum"),
            unique_districts=("district", "nunique"),
        )
        .reset_index()
    )

    source_summary["purpose"] = source_summary["source_file"].map(
        {
            CASE_STUDY_FILE: "Bogor district case studies and missions",
            "Landslide_Dataset_450.csv": "Mixed-region baseline modeling",
            "Landslide_Dataset_766.csv": "Broader mixed-region baseline modeling",
        }
    )

    table = source_summary[
        [
            "source_file",
            "records",
            "unique_districts",
            "positive_labels",
            "negative_labels",
            "case_study_eligible_records",
            "purpose",
        ]
    ]

    output_path.write_text(
        "# Dataset Summary\n\n" + table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def prepare_dataset(project_root: Path) -> pd.DataFrame:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    metrics_dir = project_root / "results" / "metrics"
    tables_dir = project_root / "manuscript" / "tables"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    points = load_raw_dataset(raw_dir)
    points.to_csv(processed_dir / "landslide_points.csv", index=False)
    points.to_json(
        processed_dir / "landslide_points.json",
        orient="records",
        indent=2,
    )

    summary = build_dataset_summary(points)
    summary.to_csv(metrics_dir / "dataset_summary.csv", index=False)
    write_dataset_summary_table(points, tables_dir / "table_dataset_summary.md")

    return points


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    points = prepare_dataset(project_root)
    print(f"Prepared {len(points)} normalized records.")
    print("Wrote data/processed/landslide_points.csv")
    print("Wrote data/processed/landslide_points.json")
    print("Wrote results/metrics/dataset_summary.csv")
    print("Wrote manuscript/tables/table_dataset_summary.md")


if __name__ == "__main__":
    main()
