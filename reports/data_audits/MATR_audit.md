# MATR Dataset Audit

## Processed files
- `data/03_processed/BatteryML/MATR/*.pkl`
- `data/03_processed/MATR/matr_cell_summary.csv`
- `data/03_processed/MATR/matr_cell_summary.parquet`

## Dataset role in BatteryInverse
- High-impact fast-charging LFP/graphite cycle-life dataset.
- Used as a degradation-aware constraint layer for protocol-sensitive inverse design.
- Particularly useful for early-cycle aging signatures, capacity-fade modelling, and robust candidate ranking.

## Basic metadata
- Number of processed cells: 180
- Form factor: cylindrical_18650
- Anode material: graphite
- Cathode material: LFP
- Nominal capacity: 1.1 Ah
- Voltage window: 2.0-3.5 V

## Cycle-life statistics
- count: 180.0
- mean: 856.8388888888888
- std: 362.08411572325804
- min: 169.0
- 25%: 560.5
- 50%: 829.0
- 75%: 1001.25
- max: 2236.0

## Capacity-retention statistics
- count: 180.0
- mean: 0.7595498858864902
- std: 0.11320715403323398
- min: 0.2412621266365702
- 25%: 0.7658001444692736
- 50%: 0.8055499494724112
- 75%: 0.8239103635121205
- max: 0.907976244706486

## Protocol metadata
- Charge protocol length values: [2, 4]
- Discharge protocol length values: [1]

## Notes for manuscript
- MATR should be described as a cell-level fast-charging aging dataset.
- It supports degradation-aware inverse design by linking operating protocols with lifetime outcomes.
- It should not be described as a direct new-material synthesis dataset.
