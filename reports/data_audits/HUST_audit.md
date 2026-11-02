# HUST Dataset Audit

## Processed files
- `data/03_processed/BatteryML/HUST/*.pkl`
- `data/03_processed/HUST/hust_cell_summary.csv`
- `data/03_processed/HUST/hust_cell_summary.parquet`

## Dataset role in BatteryInverse
- Protocol-sensitive LFP/graphite cycle-life and capacity-fade dataset.
- Used as a degradation-aware constraint layer, not as a direct electrode-composition discovery dataset.

## Basic metadata
- Number of processed cells: 77
- Form factor: cylindrical_18650
- Anode material: graphite
- Cathode material: LFP
- Nominal capacity: 1.1 Ah
- Voltage window: 2.0-3.6 V

## Cycle-life statistics
- count: 77.0
- mean: 1897.6623376623377
- std: 389.87692425031327
- min: 1142.0
- 25%: 1609.0
- 50%: 1873.0
- 75%: 2202.0
- max: 2689.0

## Capacity-retention statistics
- count: 77.0
- mean: 0.746689976878623
- std: 0.009367515770592126
- min: 0.7179822893275971
- 25%: 0.7428625462593307
- 50%: 0.748765866066781
- 75%: 0.7531294872923304
- max: 0.7636906756146304

## Notes for manuscript
- HUST should be described as a cell-level aging/degradation dataset.
- It supports degradation-aware ranking and operating-protocol sensitivity analysis.
- It should not be described as a dataset containing new electrode compositions.
