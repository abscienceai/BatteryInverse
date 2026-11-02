# Cross-domain BatteryInverse audit: MoleculeNet ESOL

This pilot applies the BatteryInverse audit chain to a second scientific ML domain: molecular property prediction.

## Domain definition

- Dataset: MoleculeNet ESOL / Delaney solubility
- Features: 1024-bit Morgan fingerprints
- Target: measured log solubility
- Domain label: Bemis-Murcko scaffold domain
- Eligible molecules: 686
- Scaffold domains with n >= 15: 7

## Headline diagnostics

- Scaffold-domain separability balanced accuracy: 0.951 [0.941 to 0.961]
- Random-split model MAE: 0.894 [0.876 to 0.911]
- LODO scaffold-domain model MAE: 1.692 [0.627 to 3.366] empirical domain interval
- Random-to-LODO MAE ratio: 1.892
- Random-split global-median prior MAE: 1.537 [1.501 to 1.571]
- Random-split scaffold-median prior MAE: 1.228 [1.208 to 1.248]
- Scaffold-prior gain over global median: 0.309 [0.281 to 0.338]
- LODO improvement over train-global-median baseline: 0.114 [-0.466 to 0.859] empirical domain interval

## Interpretation

The ESOL pilot is a cross-domain generalization check, not a new molecular-property benchmark. It asks whether the same audit chain can expose scaffold-domain structure in a second scientific ML setting. If scaffold identity is recoverable from fingerprints and random-split performance differs from leave-one-scaffold-domain-out behavior, then the BatteryInverse audit logic is not battery-specific, even though the main case study remains battery degradation.

## Output files

- esol_scaffold_separability_splits.csv
- esol_random_split_regression_and_prior.csv
- esol_leave_one_scaffold_domain_out.csv
- ESOL_CROSS_DOMAIN_AUDIT_SUMMARY.json
- figures/esol_scaffold_domain_separability.png
- figures/esol_random_vs_lodo_mae.png
- figures/esol_tgi_distance_vs_error.png
