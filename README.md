# Reproducibility materials for ESJ #2026-4228

This repository contains the analysis scripts, derived Pacitan modelling table,
and intermediate metric outputs for **“Inventory Sampling and Evaluation
Design Dominate Reported AUROC in Landslide Susceptibility: A Field-Verified
Benchmark.”** It supports the analysis reported in the manuscript; it is not
the wider landslide digital-twin project.

## Contents

- `scripts/`: dataset preparation, integrity checks, the experiments reported
  in the paper, the retired-dataset audit, and figure generation.
- `data/processed/pacitan_points.csv`: the derived 1,485-row modelling table
  generated from the public Pacitan benchmark, with its provenance record.
- `results/metrics/pacitan/`: experiment designs and per-run and summary
  outputs used in the paper.
- `results/metrics/revision_r1/`: aggregate outputs for the retired-dataset
  analysis. Row-level Bogor data are not included.
- `results/metrics/gate_comparison.json`: aggregate integrity-gate comparison.

## Input data

The Pacitan benchmark is Samodra et al. (2024), [Mendeley Data, Version 1,
DOI 10.17632/vrtx3w6mjd.1](https://data.mendeley.com/datasets/vrtx3w6mjd/1),
licensed CC BY 4.0. Download the deposit and place its files under
`data/raw/pacitan-dib-2024/`; see `data/raw/pacitan-dib-2024/SOURCE.md`.
The derived point table in this repository is provided for convenience and
retains attribution to that source.

The retired Bogor dataset is available from
[Djarot Hindarto’s public repository](https://github.com/giantfor/Landslide_Dataset_Bogor)
at commit `d19e5adf840047ee70979e180da228c5bb2f7e31`. Its raw files are not
redistributed here because the source repository does not state a licence.
To rerun the retired-dataset audit, obtain those source files from that commit
and place them in `data/raw/` before running `scripts/prepare_dataset.py` and
`scripts/revision_r1_audit.py`.

The locator map uses Natural Earth 1:110m data. Its source and terms are
recorded in `data/raw/naturalearth/SOURCE.md`.

## Reproduction

The manuscript records Python 3.13.11, NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1,
scikit-learn 1.8.0, XGBoost 3.2.0, and rasterio 1.5.1. The analysis and figure
scripts also require Matplotlib and Shapely; their exact versions were not
preserved in the experiment record.

After downloading the Pacitan deposit and installing the packages above:

```bash
python3 scripts/build_pacitan_dataset.py
python3 scripts/validate_dataset.py data/processed/pacitan_points.csv \
  --label landslide \
  --features slope_deg,elevation_m,aspect_deg,plan_curvature,profile_curvature,terrain_wetness_index,stream_power_index,distance_to_river_m,distance_to_road_m,lithology_class,landuse_class \
  --x x_utm49s --y y_utm49s --coord-crs utm
python3 scripts/pacitan_cv_experiment.py
python3 scripts/pacitan_replication_audit.py
python3 scripts/pacitan_blocking_sensitivity.py
python3 scripts/pacitan_sampling_experiment.py
python3 scripts/pacitan_dedup_test.py
python3 scripts/pacitan_dedup_test.py --per-class
python3 scripts/pacitan_residual_autocorrelation.py
python3 scripts/pacitan_aspect_encoding.py
python3 scripts/restricted_set_diagnostics.py
python3 scripts/generate_r1_figures.py
```

The precomputed metric files are included so the reported figures and tables
can be checked without rerunning the experiments. Random states and experiment
settings are recorded in the corresponding JSON files under
`results/metrics/pacitan/`.

## Software licensing

No separate software licence was present in the source project, so this release
does not add a software licence grant. The Pacitan source data retain their
CC BY 4.0 terms; the retired Bogor source data are linked but not redistributed.
