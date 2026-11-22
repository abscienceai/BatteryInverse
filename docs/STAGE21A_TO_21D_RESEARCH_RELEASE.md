# BatteryInverse: Stage 21 Controlled Feature-Encoding Audit

This release supplements the previously published Stage 20
evidence. The canonical 567-cell benchmark and historical
results remain unchanged.

## Verified scientific results

The historical `cap_fade_fraction_1_to_100` feature follows:

- BatteryML_main: Q100/Q1 - 1
- BatteryLife_selected: 1 - Q100/Q1

Processing family is algebraically distinguishable in
560 of 560 nonambiguous cells. Seven near-unity-ratio
cells are ambiguous at the specified tolerance.

The earlier Stage21A `FADE_IN_MODEL False` line was
caused by checking a different column name.
Stage21B verified `FADE_IN_MODEL True` for the actual
historical model feature.

All 24 historical model-domain MAEs were reproduced.
Stage21C verified zero historical cell-prediction drift.

Five representations, two models and 12 held-out
datasets produced 5,670 cell-level predictions.

## Full benchmark: 567 cells

| Representation | RF macro-MAE | ET macro-MAE |
|---|---:|---:|
| Original seven | 0.209121 | 0.208147 |
| Harmonized seven | 0.278848 | 0.276973 |
| Excluding fade | 0.278982 | 0.275570 |
| Harmonized + family | 0.217691 | 0.196448 |
| Family only | 0.261284 | 0.261102 |
| Global train-median prior | 0.232742 | 0.232742 |
| Source-family train-median prior | 0.223572 | 0.223572 |

## Within-domain cell alignment

Across 100 seeded within-domain permutation draws,
harmonized RF/ET macro-MAE increased by
0.013138/0.012547 on average.

The family-only negative control showed zero change.

Complete domain-level results, including MAE, bias,
and Spearman correlation, are provided in:

external_validation/reports/stage21_full_12_domain_table.csv

## Scientific boundaries

Source-family membership is known for held-out datasets.

Both median priors use training labels only.

The analysis does not evaluate an entirely new
processing family.

The 205-cell quality-filtered BatteryLife evaluation
does not constitute an endpoint-standardized cohort.

Filtering changes scoring membership, not model fitting.

The 318 BatteryML_main cells lack equivalent
processed-source target reconstruction.

Within-domain prediction alignment does not establish
a causal degradation mechanism or prospective transfer.

Repeated permutations are not independent experiments.

## Reproducibility

Stage21A-D scripts are under:

external_validation/scripts/

Reports and predictions are under:

external_validation/reports/

Run Stage21A, B, C and D in that order using the
canonical benchmark and existing Stage20 evidence.

The historical replay requires scikit-learn==1.8.0.

Software versions are recorded in:

docs/STAGE21A_TO_21D_ENVIRONMENT.txt

File checksums are recorded in:

docs/STAGE21A_TO_21D_SHA256SUMS.txt

This release does not claim a Zenodo DOI.
