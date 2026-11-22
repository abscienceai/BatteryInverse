# BatteryInverse — Stage 20 Scientific Audit Release

This milestone adds selected Stage 20 scripts, validation
reports, source-target audits, and fixed-prediction robustness
results to the existing public BatteryInverse repository.

## Scientific evidence

- Canonical benchmark: 567 cells from 12 datasets.
- BatteryLife processed-source target reconstruction: 249/249.
- Endpoint-quality flags: 29 no-positive-post-100 cases,
  one sparse post-100 case, and 14 MICH target-construction
  sensitivity cases.
- Union of three flags: 44 cells.
- BatteryLife cells without these flags: 205.
- Historical Stage 19 RF/ET LODO replay: 24/24 archived
  fold-level results reproduced exactly with scikit-learn 1.8.0.
- Reconstructed cell-model predictions: 1,134.
- Overall fixed-prediction evaluation: 567 versus 523 cells.
- BatteryLife fixed-prediction evaluation: 249 versus 205 cells.
- Small-domain sensitivity: 520 and 202 cells.
- Evaluation-domain-deletion robustness: Stage 20S.

## Reproducibility interpretation

Historical models were refitted to recover previously unavailable
cell-level LODO predictions. No models were retrained or
recalibrated after applying the endpoint-quality filters.

Filtering changed only the scored evaluation membership.
Original training-fold median priors remained fixed.

The overall evaluation retained the macro-versus-cell-weighted
model-prior direction, but that direction was sensitive to some
individual evaluation-domain deletions.

In the BatteryLife-only scoring subset, the training-median
prior retained lower MAE for the evaluated RF/ET models
under both aggregation approaches.

The BatteryLife-only results do not represent BatteryLife-only
model training. Evaluation-domain deletion is not model
retraining.

The 318 BatteryML_main cells have not received equivalent
processed-source quality verification. The 205 BatteryLife
cells without the tested flags are not certified to share
a physically comparable endpoint.

These are retrospective benchmark audits, not causal
or prospective validation results.

## Dataset integrity

Canonical dataset:

data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv

SHA-256:

8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c

## Published files

Selected Stage 20 scripts and audit reports are located in:

- external_validation/scripts/
- external_validation/reports/

The file-specific SHA-256 manifest is:

docs/STAGE20O_TO_20S_SHA256SUMS.txt

Original third-party battery archives, instrument-level raw
trajectories, and unselected intermediate research files
are not redistributed.

Processed-source reconstruction may require downloading
the cited upstream archives separately.

This release supplements the manuscript and its
Supplementary Information without establishing common
physical EOL equivalence or deployment safety.
