# Independent measured-data stress tests

This material supports the research manuscript
*BatteryInverse exposes dataset-prior shortcuts in
scientific machine learning for battery degradation*.

The original 567-cell benchmark is located at:

`data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv`

## Evaluation roles

### ISU-ILCC

242 physically distinct external cells were evaluated
with an isolated replay of the original training recipe.

Last-recorded C/5 capacity and the first post-100 C/5
diagnostic are different endpoints. Equivalence with the
heterogeneous training target has not been established.

### Tongji/KIT NCA

The public source archive contains 66 physical cells.

A strict first-100-cycle screening produced the
30-cell pre-outcome-selected evaluation cohort.

Four RF/ET prediction branches were locked before
terminal capacity outcomes were inspected.

The frozen prediction file is:

`external_validation/reports/stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv`

Frozen prediction SHA-256:

`cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757`

### Subsequent exploratory analyses

Stages 17AK–17AN evaluated feature-sign harmonization,
redundant-feature removal, aggregation sensitivity,
and dataset-level influence.

They are post-outcome exploratory analyses, not
prospectively prespecified independent validation.

## Verification

The original repository verification command is:

`python scripts/verify_repository.py`

The additional SHA-256 manifest is:

`external_validation/SHA256SUMS_submission_20261120.txt`

From the repository root on GNU/Linux, verify with:

`sha256sum -c external_validation/SHA256SUMS_submission_20261120.txt`

This supplementary manifest does not replace
the original root SHA256SUMS.txt.

## Scope and limitations

This curated release contains selected scripts,
derived records, metrics and diagnostic figures.

It does not redistribute all third-party raw cycling
archives, every intermediate CSV, all fitted models
or all internal research logs.

Re-executing certain scripts may require public
upstream archives, project-specific paths and
optional Python dependencies.

Do not run scripts directly against the archived
results directory when preserving recorded outputs.

Last-recorded capacity is not necessarily physical EOL.

The 2500 mAh NCA reference is a post-outcome
observation-window diagnostic, not a prospective
forecast or a verified per-cell stopping command.

The main manuscript and Supplementary Table S1
remain the authoritative study descriptions.

## Included evidence files

Curated files: 65

Scripts: 21

Reports and figures: 44

See the SHA-256 manifest for exact filenames
and byte-level integrity.
