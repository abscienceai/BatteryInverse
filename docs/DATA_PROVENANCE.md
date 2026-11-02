# Dataset provenance and data model

## Primary analytical table

Path: `data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv`

- 567 cell-level records; 38 columns.
- Source families: `BatteryML_main` (318 cells) and `BatteryLife_selected` (249 cells).
- 12 distinct dataset identifiers.
- `dataset_id` + `cell_id` is unique within this table.
- `final_capacity_retention` contains no missing values.
- Values in normalized retention can slightly exceed 1.0 in a few cells. Do not impose unvalidated clipping.

## Per-dataset composition

| Dataset | Processing-source family | Included cells |
|---|---|---:|
| CALB | BatteryLife_selected | 27 |
| HNEI | BatteryLife_selected | 14 |
| MICH | BatteryLife_selected | 40 |
| MICH_EXP | BatteryLife_selected | 18 |
| NA-ion | BatteryLife_selected | 64 |
| SNL | BatteryLife_selected | 61 |
| UL_PUR | BatteryLife_selected | 2 |
| XJTU | BatteryLife_selected | 23 |
| CALCE | BatteryML_main | 13 |
| HUST | BatteryML_main | 77 |
| MATR | BatteryML_main | 180 |
| RWTH | BatteryML_main | 48 |
| **Total** | **2** | **567** |

### Processing-source records versus original research datasets

BatteryML-derived grouping: MATR, HUST, CALCE, RWTH. BatteryLife-selected grouping: CALB, HNEI, MICH, MICH_EXP, NA-ion, SNL, UL_PUR, XJTU. These grouping names describe the processing path, not necessarily original laboratory ownership or common test conditions.

Primary dataset/platform papers and original dataset publications should be cited in a research manuscript as applicable. This package's `reports/data_audits/` provides partial background, but full upstream archive versions and raw-file manifests have not been conclusively established.

The processed BatteryLife `source_file` entries are relative provenance hints (for example `.pkl` cell records), not downloadable links and not proof of the original raw measurement format. Entries may be missing for the BatteryML-derived subset.

## Major provenance limitations

1. This snapshot supports repeat analysis *from the compact cell table*, not a guaranteed reconstruction from all original raw archives.
2. Upstream owners' rights and downstream redistribution terms require review before publishing derived data.
3. The table contains summarized early-cycle capacity features and metadata rather than complete original charge-discharge time series.
4. Cross-source and deconfounded comparisons constructed from combined target observations remain retrospective and must be interpreted accordingly.
5. The two-cell `UL_PUR` domain is small; avoid treating its held-out metrics as statistically robust.
