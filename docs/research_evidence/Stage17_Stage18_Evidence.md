# BatteryInverse: Stage 17–18 Reproducibility Evidence

## Stage 17AK–17AL: Feature sensitivity

12-dataset LODO macro MAE, 567 cells.

| Representation | RF | ET |
|---|---:|---:|
| Original 7 | 0.209085 | 0.208147 |
| Harmonized 7 | 0.278837 | 0.276973 |
| Fade excluded 6 | 0.279025 | 0.275570 |

## Stage 17AM–17AN: Aggregation sensitivity

| Aggregation | Median prior | RF | ET |
|---|---:|---:|---:|
| Equal-domain | 0.232742 | 0.209085 | 0.208147 |
| Cell-weighted | 0.177778 | 0.200754 | 0.191152 |

Model superiority relative to the median reference changes
with aggregation. RF and ET do not exchange ranks.
Exploratory domain-resampling intervals include zero.

## Stage 18: Source identification

565 cells, 11 datasets, 30 matched stratified splits.

| Normalized features | Mean balanced accuracy |
|---|---:|
| Original 7 | 0.808799 |
| Harmonized 7 | 0.755964 |
| Fade excluded 6 | 0.764451 |

The original classification results were reproduced.

Intervals across splits are descriptive rather than formal
population confidence intervals.

The absolute-capacity classifier does not include the
modified normalized fade feature.

## Reproduction

Run:

python external_validation/scripts/stage18_harmonization_separability_audit.py

Original Stage 17 CSV/JSON evidence and Stage 18 results
are in external_validation/reports/.

These exploratory findings do not establish causal
shortcut use or prospective deployment validity.
