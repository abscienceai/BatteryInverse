# RWTH Dataset Audit

## Processed files
- `data/00_external_raw/aging/RWTH/original/Rawdata.zip`
- `data/03_processed/BatteryML/RWTH/*.pkl`
- `data/03_processed/RWTH/rwth_cell_summary.csv`
- `data/03_processed/RWTH/rwth_cell_summary.parquet`

## Dataset role in BatteryInverse
- NMC/graphite 18650 degradation dataset.
- Used as a cross-chemistry degradation-validation layer beyond LFP/graphite and LCO/graphite datasets.
- Useful for testing whether degradation-aware inverse-design constraints generalise across cathode chemistries.

## Basic metadata
- Number of processed cells: 48
- Form factor: cylindrical_18650
- Anode material: graphite
- Cathode material: NMC
- Nominal capacity values: [1.11] Ah
- Voltage window: 3.5-3.9 V
- Max current limit values: [4] A

## Cycle-life statistics
- count: 48.0
- mean: 2187.625
- std: 205.2068624168761
- min: 1486.0
- 25%: 2004.75
- 50%: 2247.5
- 75%: 2334.25
- max: 2440.0

## Capacity-retention statistics
- count: 48.0
- mean: 0.25664918035819867
- std: 0.06524481272098277
- min: 0.1844034895196722
- 25%: 0.2172887115966741
- 50%: 0.23129186831513232
- 75%: 0.2964384608806923
- max: 0.5390055822789316

## Notes for manuscript
- RWTH should be described as a cell-level NMC/graphite degradation dataset.
- It provides cross-chemistry evidence complementary to HUST, MATR, and CALCE.
- It is especially useful for robustness testing because cells are cycled to substantial degradation.
