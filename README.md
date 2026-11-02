# BatteryInverse

**Language:** Python | **Focus:** Dataset-prior shortcut diagnostics for battery degradation machine learning.

**Repository:** https://github.com/abscienceai/BatteryInverse

BatteryInverse is a **diagnostic and reproducibility project**, not a universally accurate lifetime predictor. It examines whether apparent predictive accuracy across heterogeneous battery datasets reflects transferable cell-level degradation information or dataset-specific fingerprints and target priors. This repository provides code, a compact model-ready analysis table, selected saved analysis outputs and figures.

## What is included

- **567 cell-level rows, 38 columns, 12 datasets and two source families**, in `data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv`.
- Python source for dataset separability, random-split versus leave-one-dataset-out (LODO), target-prior baselines, shuffling controls, deconfounded and cross-source evaluation, feature-distance/error diagnostics, fold-safe confound injection, residual coverage, active acquisition stress testing, learned mitigation, canonical ERM/IRM/DANN comparison and ESOL molecular stress testing.
- Saved CSV/JSON summaries, reports and non-duplicated figure panels.
- A file-integrity check, a data dictionary, result interpretation and reproducibility guidance.

**Main research question:** Can benchmark performance be explained by identifiable data sources and target-distribution priors, without demonstrating reliable transfer to unseen domains?

## Quick start (Windows, macOS or Linux)

1. Clone the repository or download its ZIP archive from GitHub and extract it.
2. Open a terminal in the folder containing this README and the `scripts/`, `data/`, and `reports/` directories.
3. Use **Python 3.10–3.12** if available. Create a virtual environment:

    python -m venv .venv

4. Activate it:

   - **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
   - **Windows Command Prompt:** `.venv\Scripts\activate.bat`
   - **macOS/Linux:** `source .venv/bin/activate`

5. Install core packages:

    python -m pip install -r requirements.txt

6. Check the shipped files and data structure:

    python scripts/verify_repository.py

7. Inspect the model-ready table:

    python -c "import pandas as pd; x=pd.read_csv('data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'); print(x.groupby(['source_family','dataset_id']).size()); print(x.shape)"

**Start with the included data and summaries rather than downloading raw datasets.** Only the optional upstream conversion scripts require additional provider archives. Some model refits can be computationally expensive and may differ slightly across versions, hardware and random seeds.

## Folders and navigation

| Path | Purpose |
|---|---|
| `data/05_model_ready/aging/` | Compact canonical battery cell-level table (CSV) |
| `scripts/` | Original audit scripts, optional source-data conversion and integrity check |
| `reports/benchmark_audits/` | Main battery benchmark results, including prior, LODO, shuffling and transfer diagnostics |
| `reports/gtci_v2_fold_safe/` | Fold-safe synthetic confound positive-control results |
| `reports/gtci_conformal_active/` | Retrospective residual diagnostics and acquisition stress-test outputs |
| `reports/batteryinverse_net_confoundadam_v1/` | Learned mitigation and representation-audit results |
| `reports/canonical_domain_generalization_baselines_v2_stable/` | ERM, IRM, DANN baselines and closure checks |
| `reports/cross_domain_moleculenet_esol_audit/` | MoleculeNet ESOL scaffold-domain stress test |
| `reports/consistency_checks/` | Domain-wise residual and mitigation consistency results |
| `reports/data_audits/` | Dataset description and preprocessing notes |
| `figures/` | Non-duplicated figure panels supporting the results |
| `docs/` | Detailed reproducibility, provenance, data dictionary and upload guidance |
| `SHA256SUMS.txt` | SHA256 for each file included at packaging time |

## Source-family / dataset composition

| Dataset | Processing-source family | Cells |
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
| **Total** | **2 source families** | **567** |

The model target is `final_capacity_retention`; the domain label is `dataset_id`. The table is **one row per cell**, not a raw cycle-by-cycle time series. Some retention values can slightly exceed 1.0; do not unconditionally clip them.

## Running analyses

The core scripts use project-root-relative paths. They will write output under `reports/` and may overwrite included saved result files. **Make a working copy before rerunning experiments**, then use the following order as appropriate:

1. `python scripts/expanded_dataset_separability_analysis.py`
2. `python scripts/expanded_model_family_robustness.py`
3. `python scripts/expanded_random_split_vs_lodo_inflation.py`
4. `python scripts/expanded_explicit_prior_baseline_analysis.py`
5. `python scripts/expanded_negative_control_shuffle_analysis.py`
6. `python scripts/expanded_deconfounded_transfer_analysis.py`
7. `python scripts/expanded_cross_source_transfer_analysis.py`
8. `python scripts/expanded_transfer_gap_feature_family_analysis.py`
9. `python scripts/expanded_deconfounded_bootstrap_summary.py` (requires an intermediate prediction table; consult reproducibility guide)
10. `python scripts/batteryinverse_gtci_v2_fold_safe.py`
11. `python scripts/batteryinverse_gtci_conformal_active_v0.py`

Optional deep-learning runs require `python -m pip install -r requirements-optional.txt`:

- `python scripts/canonical_domain_generalization_baselines_v2_stable.py`
- `python scripts/batteryinverse_net_confoundadam_v1_stability.py`

The ESOL audit additionally retrieves its public source CSV when run and requires internet access and RDKit:

- `python scripts/cross_domain_moleculenet_esol_audit_v1.py`

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for prerequisites, dependencies, output interpretation, and known limits. See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for all 38 columns and missingness information.

## Important interpretation limits

- Random splits may share source-level information between train and test data; LODO is a different, more demanding evaluation protocol.
- Dataset-centered or ranked targets constructed using held-out target statistics are **retrospective explanatory diagnostics**, not zero-shot outcomes available beforehand.
- The **full TGI** includes held-out prediction error and therefore needs labels. Its risk bands are descriptive and not certified decision thresholds.
- Pooled nominal conformal residual width and 0.903 empirical pooled coverage were computed using the **same 567 residuals**, not an independent calibration/test evaluation; they do **not** give a prospective coverage guarantee.
- The ESOL evaluation is a limited second-domain stress test, not proof of general applicability across scientific fields.

## Project documentation

- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Data provenance and data rights](docs/DATA_PROVENANCE.md)
- [Reproducibility workflow and script map](docs/REPRODUCIBILITY.md)
- [Results interpretation guide](docs/RESULTS_GUIDE.md)
- [Figure index](docs/FIGURE_INDEX.md)
- [Data notice](DATA_NOTICE.md)

## Citing and licensing

Citation metadata are in `CITATION.cff`. The MIT license in `LICENSE` applies to **original project software only**; it does not grant rights to upstream datasets.
