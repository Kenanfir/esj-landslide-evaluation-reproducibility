# Pacitan landslide benchmark dataset

**Dataset:** Samodra, G., Malawani, M.N., Suhendro, I., & Mardiatno, D. (2024).
*Spatial datasets for benchmarking machine learning-based landslide
susceptibility models.* Mendeley Data, V1. **DOI 10.17632/vrtx3w6mjd.1**

**Companion article:** *Data in Brief* 57, 111155 (November 2024).
**DOI 10.1016/j.dib.2024.111155**

**Licence:** CC BY 4.0 — share and adapt with attribution.
**Retrieved:** 2026-08-30. **Published:** 2024-09-10.

## Why this dataset

It replaces `github.com/giantfor/Landslide_Dataset_Bogor`, which was retired
from this project on 2026-08-30. That file was 55.5% duplicate rows, its
district labels carried no spatial information, and its five predictors were
each indistinguishable from a uniform draw over a round interval. The audit is
at `manuscript/esj/REVISION_R1_2026-08-29/FINDINGS_R4_BLOCKING.md`.

This dataset passes `scripts/validate_dataset.py` on every check — no duplicate
rows, no single-feature label leakage, no uniform-noise predictors, no
coincident neighbours in feature space — and the label carries genuine spatial
autocorrelation (Moran's I = 0.279, z = 22.8, permutation p < 0.001), which is
the condition that makes spatially-blocked cross-validation necessary.
See `results/metrics/pacitan/validation.json`.

## What it contains

- **Region:** Pacitan Regency, East Java, Indonesia.
- **Trigger:** Tropical Cyclone Cempaka, 27–29 November 2017.
- **Inventory:** 743 landslide polygons, manually delineated from pre- and
  post-event high-resolution satellite imagery and verified by field survey.
- **Points:** 743 landslide centroids (`ls_center.gpkg`) and 743 random
  non-landslide points (`non_ls.gpkg`), both `lslpts`-labelled.
- **Factors:** 11 controlling-factor rasters at 10 m in EPSG:32749
  (WGS 84 / UTM zone 49S), derived from Indonesia's national DEM (DEMNAS) and
  topographic maps — slope, elevation, aspect, plan curvature, profile
  curvature, terrain wetness index, stream power index, distance to river,
  distance to road, lithology, land use.

The `.vat.dbf` sidecars give the class codes and pixel counts for the two
categorical rasters. The deposit carries no legend mapping those codes to
named lithology or land-use classes.

## Files in this paper-only release

This repository does not mirror the original Mendeley deposit. Download the
source files with the command below before rebuilding the derived table. The
derived-table provenance file records the source file hashes used for the
published analysis, so the included table can be checked without storing the
large raster files here.

## Re-download

```bash
python3 - <<'PY'
import json, os, subprocess, urllib.request
DEST = "data/raw/pacitan-dib-2024"
os.makedirs(DEST, exist_ok=True)
meta = json.load(urllib.request.urlopen(
    "https://data.mendeley.com/public-api/datasets/vrtx3w6mjd"))
for f in meta["files"]:
    out = os.path.join(DEST, f["filename"])
    if os.path.exists(out) and os.path.getsize(out) == f["size"]:
        continue
    subprocess.run(["curl", "-sL", "-o", out,
                    f["content_details"]["download_url"]], check=True)
    print(f["filename"], os.path.getsize(out))
PY
python3 scripts/build_pacitan_dataset.py
```

Every downloaded file was verified against the sha256 the Mendeley API
publishes for it at retrieval time.
