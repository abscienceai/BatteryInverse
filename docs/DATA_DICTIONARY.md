# Complete model-ready CSV dictionary

The CSV is **comma-delimited**, uses a period `.` as the decimal separator, and has a UTF-8 header row. Missing values are typically empty fields/NaN; do not convert them to zero without a feature-specific justification. A CSV parser such as `pandas.read_csv()` is recommended.

The dtype column below is the dtype detected when loading the archived CSV with pandas and is not a database schema guarantee. The missingness column reports null values in the saved 567-row table.

| Column | Detected pandas dtype | Missing cells | Interpretation |
|---|---|---:|---|
| `source_family` | `object` | 0 | Processing-source family (BatteryML_main or BatteryLife_selected). |
| `dataset_id` | `object` | 0 | Dataset/domain identifier used for group-aware evaluation. |
| `cell_id` | `object` | 0 | Cell identifier within the source; not necessarily unique across all original archives. |
| `source_file` | `object` | 318 | Relative provenance hint when present; not a download URL. |
| `final_capacity_retention` | `float64` | 0 | Outcome: final capacity estimate divided by an initial capacity estimate. |
| `cap_cycle_1` | `float64` | 0 | Early-cycle capacity descriptor near cycle 1, source-specific units. |
| `cap_cycle_10` | `float64` | 0 | Capacity descriptor near cycle 10. |
| `cap_cycle_50` | `float64` | 0 | Capacity descriptor near cycle 50. |
| `cap_cycle_100` | `float64` | 0 | Capacity descriptor near cycle 100. |
| `cap_mean_first_100` | `float64` | 0 | Summary mean of available early-cycle capacities. |
| `cap_std_first_100` | `float64` | 0 | Summary standard deviation of available early-cycle capacities. |
| `cap_slope_first_100` | `float64` | 0 | Fitted/summary early-cycle capacity slope. |
| `cap_fade_1_to_100` | `float64` | 0 | Difference between capacity descriptors at cycles 1 and 100. |
| `nominal_capacity_Ah` | `float64` | 0 | Reported nominal capacity in ampere-hours (where available). |
| `min_voltage_V` | `float64` | 0 | Reported minimum voltage window value in volts. |
| `max_voltage_V` | `float64` | 0 | Reported maximum voltage window value in volts. |
| `chemistry` | `object` | 0 | Chemistry category or source label; not necessarily standardized across sources. |
| `form_factor` | `object` | 0 | Cell form-factor category. |
| `anode_material` | `object` | 0 | Anode material label. |
| `cathode_material` | `object` | 0 | Cathode material label. |
| `ambient_temperature_C` | `float64` | 382 | Ambient temperature in degrees Celsius when reported. |
| `quality_flags` | `object` | 318 | Quality/audit flags; inspect source processing before interpreting. |
| `long_life_label` | `float64` | 567 | Derived binary/indicator lifetime label; use source code for exact threshold. |
| `high_retention_label` | `float64` | 567 | Derived high-retention label; use source code for exact threshold. |
| `long_life_label_within_dataset` | `float64` | 249 | Dataset-relative derived lifetime label. |
| `high_retention_label_within_dataset` | `float64` | 249 | Dataset-relative derived retention label. |
| `aging_quality_score_proxy` | `float64` | 249 | Derived quality score proxy; not a calibrated physical measurement. |
| `cap_cycle_10_norm` | `float64` | 0 | Normalized capacity feature near cycle 10. |
| `cap_cycle_50_norm` | `float64` | 0 | Normalized capacity feature near cycle 50. |
| `cap_cycle_100_norm` | `float64` | 0 | Normalized capacity feature near cycle 100. |
| `cap_mean_first_100_norm` | `float64` | 0 | Normalized early-cycle capacity mean. |
| `cap_std_first_100_norm` | `float64` | 0 | Normalized early-cycle capacity variability. |
| `cap_slope_first_100_norm` | `float64` | 0 | Normalized early-cycle capacity slope. |
| `cap_fade_1_to_100_norm` | `float64` | 0 | Normalized early-cycle capacity fade. |
| `cap_ratio_100_to_1` | `float64` | 0 | Capacity ratio near cycle 100 to cycle 1. |
| `cap_ratio_50_to_1` | `float64` | 0 | Capacity ratio near cycle 50 to cycle 1. |
| `cap_ratio_10_to_1` | `float64` | 0 | Capacity ratio near cycle 10 to cycle 1. |
| `cap_fade_fraction_1_to_100` | `float64` | 0 | Early-cycle capacity fade expressed as fraction. |

## Cautions

- Do not put `final_capacity_retention`, derived target labels, dataset/source IDs, source filenames, or evaluation outcomes into a predictor input unintentionally.
- Some early-cycle descriptor values are source-dependent; do not treat absolute capacities from different protocols as directly comparable physical quantities without checking units and definitions.
- Different datasets may have incomplete metadata. Fit imputers and encoders only on training folds when estimating generalization.
- The table is cell-level, not a complete longitudinal record. The word `first_100` in feature names does not by itself prove that precisely 100 cycles were measured in each source.
