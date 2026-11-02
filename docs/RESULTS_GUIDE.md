# Reading the saved result files

## How to interpret the reports

Each experiment has an associated CSV/JSON output and, in many cases, a Markdown summary. The `reports/benchmark_audits/` folder includes detailed fold/domain/feature-family outputs. The other report folders contain independent evidence components.

| Evidence | Representative file | What it means |
|---|---|---|
| Dataset fingerprints | `reports/benchmark_audits/expanded_dataset_separability_summary.csv` | Recoverability of dataset labels from features; not a degradation prediction metric |
| Random versus held-out | `reports/benchmark_audits/expanded_random_split_vs_lodo_inflation.csv` | Difference between in-domain testing and domain holdout for matched settings |
| Target priors | `reports/benchmark_audits/expanded_explicit_prior_baseline_summary.csv` | Baseline accuracy achievable from grouping-level target information |
| Target shuffling | `reports/benchmark_audits/expanded_negative_control_shuffle_summary.csv` | Preservation of apparent accuracy after cell-level target alignment is disrupted |
| Deconfounded targets | `reports/benchmark_audits/expanded_deconfounded_bootstrap_summary.csv` | Retrospective effect after reducing dataset target offsets |
| Cross-source | `reports/benchmark_audits/expanded_cross_source_transfer_summary.csv` | Two transfer directions between processing-source families |
| TGI | `reports/benchmark_audits/expanded_transfer_gap_feature_family_metrics.csv` | Retrospective normalized distance and held-out error diagnostic |
| GTCI | `reports/gtci_v2_fold_safe/GTCI_V2_FOLD_SAFE_SUMMARY.csv` | Behavior of known synthetic non-causal feature under two validation modes |
| Residual diagnostics | `reports/gtci_conformal_active/CONFORMAL_SUMMARY.csv` | Descriptive pooled residual widths and empirical coverage |
| Learned mitigation | `reports/batteryinverse_net_confoundadam_v1/batteryinverse_net_v1_closure_separability.csv` | Remaining dataset identity in learned representations |
| ERM/IRM/DANN | `reports/canonical_domain_generalization_baselines_v2_stable/canonical_dg_summary_v2_stable.csv` | Held-out error and representation-level dataset classification |
| ESOL stress test | `reports/cross_domain_moleculenet_esol_audit/ESOL_CROSS_DOMAIN_AUDIT_SUMMARY.json` | Scaffold-domain audit in a second ML application |

## Selected headline observations (not new experiments)

- Total cell-level rows: **567** across **12 datasets** and **2 processing-source families**.
- Dataset separability balanced accuracy: about **0.962** using absolute capacity and **0.809** using normalized capacity.
- Dataset-median prior MAE about **0.065**, versus train-global median MAE about **0.169**, under random splits.
- Within-dataset target shuffle can retain random-split `R²` of approximately **0.761**, illustrating dependence on group-level target structure.
- The reported maximum LODO/random error ratio is sensitive to whether one uses the raw point estimate or a bootstrap summary. Consult the specific results table and evaluation definition rather than interchange estimates.
- The complete normalized TGI is label-dependent, retrospective and not a prospective operating threshold.
- The observed pooled 0.903 residual coverage does not establish an independently tested prospective guarantee.
- ESOL scaffold-domain separability BA is about **0.951**; LODO scaffold-domain error is larger than random-split error. The second-domain analysis is limited in scope.

## Direction and sign conventions

`MAE` is **lower-is-better**. Where `improvement = train_median_baseline_mae - model_mae`, **positive** means improvement over baseline and **negative** means the model is worse. Do not interpret all interval types as confidence intervals: some are empirical percentile ranges across held-out domains or configurations. See each report's statistical basis.
