# CALCE Dataset Audit

## Processed files
- `data/03_processed/BatteryML/CALCE/*.pkl`
- `data/03_processed/CALCE/calce_cell_summary.csv`
- `data/03_processed/CALCE/calce_cell_summary.parquet`

## Dataset role in BatteryInverse
- Cross-chemistry LCO/graphite aging dataset.
- Used as an external degradation-validation layer beyond LFP/graphite datasets.
- Useful for testing whether degradation-aware ranking remains chemically robust.

## Basic metadata
- Number of processed cells: 13
- Form factor: prismatic
- Anode material: graphite
- Cathode material: LCO
- Nominal capacity values: [1.1, 1.35] Ah
- Voltage window: 2.7-4.2 V

## Cycle-life statistics
- count: 13.0
- mean: 1099.8461538461538
- std: 337.72494877583597
- min: 632.0
- 25%: 832.0
- 50%: 1072.0
- 75%: 1363.0
- max: 1579.0

## Capacity-retention statistics
- count: 13.0
- mean: 0.35159982606507784
- std: 0.17079228989029738
- min: 0.0881734723295954
- 25%: 0.2420436970351758
- 50%: 0.4019516714717544
- 75%: 0.4334168231345951
- max: 0.6515334903408403

## Notes for manuscript
- CALCE should be described as a cell-level LCO/graphite degradation dataset.
- It provides cross-chemistry aging evidence complementary to HUST and MATR.
- It should be used carefully because some cells are cycled to very low remaining capacity.
