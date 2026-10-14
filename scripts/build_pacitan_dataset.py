#!/usr/bin/env python3
"""Build the Pacitan modelling table from the Data in Brief benchmark deposit.

Source: Samodra, Malawani, Suhendro & Mardiatno (2024), "Spatial datasets for
benchmarking machine learning-based landslide susceptibility models",
Data in Brief 57, 111155. Data: Mendeley Data 10.17632/vrtx3w6mjd.1 (CC BY 4.0).

743 landslide centroids and 743 random non-landslide points, each sampled
against 11 controlling-factor rasters at 10 m in EPSG:32749 (WGS 84 / UTM 49S).

Run:  python3 scripts/build_pacitan_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pacitan-dib-2024"
OUT_DIR = ROOT / "data" / "processed"
OUT_CSV = OUT_DIR / "pacitan_points.csv"
PROVENANCE = OUT_DIR / "pacitan_points.provenance.json"

POINT_LAYERS = {"ls_center.gpkg": ("ls_center", 1), "non_ls.gpkg": ("non_ls", 0)}

# raster stem -> column name in the modelling table
FACTORS = {
    "map_slope": "slope_deg",
    "map_elev": "elevation_m",
    "map_aspect": "aspect_deg",
    "map_planc": "plan_curvature",
    "map_profc": "profile_curvature",
    "map_twi": "terrain_wetness_index",
    "map_spi": "stream_power_index",
    "map_distriv": "distance_to_river_m",
    "map_distroad": "distance_to_road_m",
    "map_geologi": "lithology_class",
    "map_landuse": "landuse_class",
}
CATEGORICAL = {"lithology_class", "landuse_class"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_points() -> pd.DataFrame:
    """Read the two point layers straight out of the GeoPackage SQLite tables."""
    frames = []
    for filename, (layer, label) in POINT_LAYERS.items():
        path = RAW / filename
        with sqlite3.connect(path) as connection:
            frame = pd.read_sql_query(f'SELECT fid, X, Y FROM "{layer}"', connection)
        frame["landslide"] = label
        frame["source_layer"] = layer
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)
    points = points.rename(columns={"X": "x_utm49s", "Y": "y_utm49s"})
    points["point_id"] = [f"{r.source_layer}-{int(r.fid)}" for r in points.itertuples()]
    return points[["point_id", "source_layer", "x_utm49s", "y_utm49s", "landslide"]]


def sample_rasters(points: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    coords = list(zip(points["x_utm49s"], points["y_utm49s"]))
    sampled = points.copy()
    raster_meta = {}

    for stem, column in FACTORS.items():
        path = RAW / f"{stem}.tif"
        with rasterio.open(path) as raster:
            values = np.array([v[0] for v in raster.sample(coords)], dtype="float64")
            nodata = raster.nodata
            raster_meta[column] = {
                "file": path.name,
                "crs": str(raster.crs),
                "resolution_m": [abs(raster.res[0]), abs(raster.res[1])],
                "dtype": str(raster.dtypes[0]),
                "nodata": None if nodata is None else float(nodata),
                "sha256": sha256(path),
            }
        if nodata is not None:
            values = np.where(np.isclose(values, nodata), np.nan, values)
        # float32 sentinels can survive as huge magnitudes
        values = np.where(np.abs(values) > 1e30, np.nan, values)
        sampled[column] = values
        raster_meta[column]["missing_at_points"] = int(np.isnan(values).sum())

    return sampled, raster_meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = read_points()
    table, raster_meta = sample_rasters(points)

    for column in CATEGORICAL:
        table[column] = table[column].astype("Int64")

    complete = table.dropna(subset=list(FACTORS.values()))
    dropped = len(table) - len(complete)

    complete.to_csv(OUT_CSV, index=False)

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_pacitan_dataset.py",
        "dataset": {
            "title": "Spatial datasets for benchmarking machine learning-based landslide susceptibility models",
            "authors": "Samodra, G., Malawani, M.N., Suhendro, I., Mardiatno, D.",
            "article_doi": "10.1016/j.dib.2024.111155",
            "data_doi": "10.17632/vrtx3w6mjd.1",
            "licence": "CC BY 4.0",
            "region": "Pacitan Regency, East Java, Indonesia",
            "trigger_event": "Tropical Cyclone Cempaka, 27-29 November 2017",
            "inventory_method": (
                "743 landslide polygons manually delineated from pre- and "
                "post-event high-resolution satellite imagery and verified by "
                "field survey; centroids used as positive points; 743 random "
                "non-landslide points supplied by the depositors"
            ),
            "factor_source": "Indonesian national DEM (DEMNAS) and topographic maps",
        },
        "vector_sources": {
            filename: {"layer": layer, "label": label, "sha256": sha256(RAW / filename)}
            for filename, (layer, label) in POINT_LAYERS.items()
        },
        "rasters": raster_meta,
        "rows_sampled": int(len(table)),
        "rows_written": int(len(complete)),
        "rows_dropped_incomplete": int(dropped),
        "label_column": "landslide",
        "categorical_columns": sorted(CATEGORICAL),
        "continuous_columns": sorted(set(FACTORS.values()) - CATEGORICAL),
        "coordinate_reference_system": "EPSG:32749 (WGS 84 / UTM zone 49S), metres",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"points read       : {len(table)}")
    print(f"written           : {len(complete)}  ({dropped} dropped for missing factor values)")
    print(f"positives         : {int(complete['landslide'].sum())}")
    print(complete[sorted(set(FACTORS.values()) - CATEGORICAL)].describe().T.round(3).to_string())
    print()
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {PROVENANCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
