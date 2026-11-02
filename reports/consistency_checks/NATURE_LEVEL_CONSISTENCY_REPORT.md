# Nature-level consistency assets v17

## Learned mitigation closure

No-metadata LODO mean MAE changed from vanilla Adam 0.191 to Stable ConfoundAdam 0.187. Mean improvement versus the train-median baseline changed from 0.042 to 0.046.

With metadata, vanilla Adam mean MAE was 0.232, whereas ConfoundAdam plus adversarial invariance was 0.266; this does not support a claim that the learned mitigation solves dataset shortcuts.

Closure testing showed residual dataset identity in the learned representation: learned-z separability was 0.766 without metadata and 0.902 with metadata.

## Mondrian conformal diagnostic

Global LODO conformal q90 is 0.444 retention. The worst global-coverage domains are: MICH, CALCE, MICH_EXP. Retrospective dataset-Mondrian q90 widths are largest for: MICH, CALCE, MICH_EXP. This is a retrospective calibration diagnostic, not a prospective guarantee for a completely unseen domain without domain-specific calibration labels.

## TGI-conformal convergent evidence

CALCE is flagged as high transfer-risk in Table 4 with max scaled TGI 10.605 and also shows poor global conformal coverage 0.385. This provides convergent evidence that TGI and conformal coverage failure can identify overlapping risk domains.

MICH shows poor global conformal coverage 0.125 even though it is not among the current Table 4 high-TGI examples. This is useful because it shows that TGI and conformal diagnostics are complementary rather than redundant.

## Output files

- reports/nature_level_consistency_assets/fig_conformal_global_vs_mondrian.png
- reports/nature_level_consistency_assets/fig_learned_mitigation_lodo_mae.png
- reports/nature_level_consistency_assets/fig_tgi_conformal_overlap.png
- reports/nature_level_consistency_assets/learned_mitigation_closure.csv
- reports/nature_level_consistency_assets/learned_mitigation_summary.csv
- reports/nature_level_consistency_assets/learned_mitigation_weights.csv
- reports/nature_level_consistency_assets/mondrian_conformal_by_dataset_retrospective.csv
- reports/nature_level_consistency_assets/mondrian_conformal_by_source_family.csv
- reports/nature_level_consistency_assets/tgi_conformal_overlap_table.csv
