# Reproducibility and code navigation

## Scope and prerequisites

The package preserves Python scripts and saved outputs. Re-running every model is **not required** to inspect findings. This repository is designed to reproduce many analyses from the packaged 567-row cell table. It does not include full upstream battery archives or the older 318-cell intermediate source tables needed to rebuild the harmonized table from raw inputs.

Environment: Python 3.10–3.12 recommended; NumPy, pandas, scikit-learn, SciPy, matplotlib and joblib. Install `requirements-optional.txt` for PyTorch models and RDKit molecular experiments. GPU is optional for most scripts but can accelerate deep models. If using Windows, PowerShell activation or `python -m pip` may be easier than Unix shell scripts.

At the extracted repository root:

    python -m venv .venv
    python -m pip install -r requirements.txt
    python scripts/verify_repository.py

The integrity script checks files and cell-table structure; it **does not reproduce model estimates**.

## Recommended run order and outputs

| Step | Script | Output folder | Input or special dependency |
|---|---|---|---|
| 1 | `expanded_dataset_separability_analysis.py` | `reports/benchmark_audits/` | 567-row CSV; skips tiny domains for some stratified tests |
| 2 | `expanded_model_family_robustness.py` | `reports/benchmark_audits/` | 567-row CSV; produces LODO comparison |
| 3 | `expanded_random_split_vs_lodo_inflation.py` | `reports/benchmark_audits/` | Step 2 saved model family results |
| 4 | `expanded_explicit_prior_baseline_analysis.py` | `reports/benchmark_audits/` | 567-row CSV |
| 5 | `expanded_negative_control_shuffle_analysis.py` | `reports/benchmark_audits/` | 567-row CSV |
| 6 | `expanded_deconfounded_transfer_analysis.py` | `reports/benchmark_audits/` | 567-row CSV; retrospective transformed targets |
| 7 | `expanded_cross_source_transfer_analysis.py` | `reports/benchmark_audits/` | 567-row CSV; retrospective source transformations |
| 8 | `expanded_transfer_gap_feature_family_analysis.py` | `reports/benchmark_audits/` | 567-row CSV; LODO distances and errors |
| 9 | `expanded_deconfounded_bootstrap_summary.py` | `reports/benchmark_audits/` | Step 6 results **plus step 8 intermediate prediction CSV**, which is produced by rerunning step 8 |
| 10 | `batteryinverse_gtci_v2_fold_safe.py` | `reports/gtci_v2_fold_safe/` | Fold-safe injection results |
| 11 | `batteryinverse_gtci_conformal_active_v0.py` | `reports/gtci_conformal_active/` | Retrospective residual and acquisition outputs |
| 12 | `batteryinverse_net_confoundadam_v1_stability.py` | `reports/batteryinverse_net_confoundadam_v1/` | PyTorch, model training may take time |
| 13 | `canonical_domain_generalization_baselines_v2_stable.py` | `reports/canonical_domain_generalization_baselines_v2_stable/` | PyTorch, ERM/IRM/DANN comparison |
| 14 | `cross_domain_moleculenet_esol_audit_v1.py` | `reports/cross_domain_moleculenet_esol_audit/` | RDKit and reachable ESOL download source |

For most scripts the standard command is `python scripts/<name>.py` from the repository root. Deep-learning scripts may have options: inspect `--help` before execution when implemented. Full model runs can be CPU-intensive; output changes from package versions or random states are possible.

Run experiments in a **copy of the repository** to preserve checksums and archived outputs. After a run, file-integrity verification is expected to detect modified reports.

### Optional upstream build scripts

- `build_batterylife_selected_summary.py` requires downloaded provider processed archives under `data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original/`.
- `build_expanded_aging_features_with_batterylife.py` requires both the upstream BatteryLife selected summary and an upstream 318-cell `early_cycle_features_with_labels.csv` (not included here).

These scripts document feature-building logic but are **not runnable end-to-end solely with this compact repository**. Do not present the included model-ready CSV as proof of complete provider-archive-level reconstruction.

### Normalized Transfer-Gap Index

The scaled index uses two global medians over 48 domain × feature-family combinations: nearest-train distance `0.4721379483825` and MAE `0.165756844094`. Full index uses observed held-out error; it is a **retrospective** diagnostic. The thresholds 2 and 4 are descriptive, not certified decision gates.

### Centered and ranked target analyses

The original audited target transformations compute dataset/source target statistics from the combined labeled table before the domain split, whereas preprocessing and predictors are trained on the training rows only. Held-out labels enter the retrospective definition of the evaluated endpoint, not the predictive model training. These are **explanatory tests**, not prospective zero-shot predictions.

### Residual coverage

The pooled 90% residual half-width of approximately 0.444 and empirical coverage near 0.903 come from the **same pool of 567 LODO residuals**. They are not an independent train/calibration/test conformal coverage validation. For small domains, clipped order-statistic quantiles are descriptive and should not be interpreted as formal finite-sample guarantees.

### Internal audit detail

Some scripts or reports use the 565-row, 11-domain analysis subset that excludes the two-cell `UL_PUR` dataset for stratification or stable domain-generalization evaluation. This does not change the fact that the packaged canonical dataset contains 567 rows and 12 dataset IDs.
