# Expanded Deconfounded Bootstrap Summary

## Purpose

This report bootstraps mean MAE improvement over train-median baseline across held-out datasets. Positive improvement means the model beats a simple train-median predictor. Results are shown both across all datasets and excluding UL_PUR, which has only two cells.

## Scope: all_datasets

### Target: dataset_mean_centered_retention

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  12         -0.019479           -0.035711            -0.004660                                 0.0050                        2                               10           -0.011327
absolute_plus_normalized_capacity                  12         -0.021643           -0.035059            -0.009894                                 0.0000                        1                               11           -0.011690
                    metadata_only                  12         -0.010705           -0.025777            -0.000527                                 0.0148                        3                                9           -0.003394
         normalized_capacity_only                  12         -0.009311           -0.015816            -0.002973                                 0.0005                        2                               10           -0.005475

### Target: dataset_median_centered_retention

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  12         -0.025864           -0.046351            -0.008031                                 0.0002                        1                               11           -0.010706
absolute_plus_normalized_capacity                  12         -0.031797           -0.047333            -0.017376                                 0.0000                        1                               11           -0.026669
                    metadata_only                  12         -0.016439           -0.035469            -0.004720                                 0.0000                        1                               11           -0.007335
         normalized_capacity_only                  12         -0.020582           -0.031604            -0.011979                                 0.0000                        1                               11           -0.015142

### Target: raw_retention

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  12         -0.013778           -0.100627             0.081452                                 0.3697                        3                                9           -0.034017
absolute_plus_normalized_capacity                  12          0.013917           -0.071610             0.109051                                 0.6001                        6                                6           -0.002818
                    metadata_only                  12          0.019203           -0.084886             0.129944                                 0.6269                        5                                7           -0.011419
         normalized_capacity_only                  12          0.024557           -0.054303             0.121854                                 0.6852                        6                                6           -0.002404

### Target: within_dataset_percentile

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  12         -0.034113           -0.076756             0.001027                                 0.0293                        4                                8           -0.018870
absolute_plus_normalized_capacity                  12         -0.019622           -0.058162             0.013934                                 0.1493                        5                                7           -0.008486
                    metadata_only                  12         -0.007051           -0.025250             0.012607                                 0.2256                        4                                8           -0.002837
         normalized_capacity_only                  12         -0.011895           -0.035858             0.019273                                 0.1995                        3                                9           -0.016166

## Scope: excluding_UL_PUR

### Target: dataset_mean_centered_retention

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  11         -0.021509           -0.038319            -0.006225                                 0.0034                        1                               10           -0.011330
absolute_plus_normalized_capacity                  11         -0.024234           -0.037906            -0.012102                                 0.0000                        0                               11           -0.013682
                    metadata_only                  11         -0.010502           -0.027024             0.000439                                 0.0375                        3                                8           -0.001982
         normalized_capacity_only                  11         -0.009809           -0.016824            -0.003008                                 0.0018                        2                                9           -0.007120

### Target: dataset_median_centered_retention

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  11         -0.027513           -0.049323            -0.008177                                 0.0006                        1                               10           -0.013647
absolute_plus_normalized_capacity                  11         -0.033564           -0.049920            -0.017907                                 0.0000                        1                               10           -0.037264
                    metadata_only                  11         -0.018078           -0.038366            -0.005511                                 0.0000                        0                               11           -0.008837
         normalized_capacity_only                  11         -0.021490           -0.033141            -0.012428                                 0.0000                        1                               10           -0.016509

### Target: raw_retention

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  11         -0.013200           -0.106398             0.087149                                 0.3825                        3                                8           -0.047900
absolute_plus_normalized_capacity                  11          0.017940           -0.072318             0.120814                                 0.6202                        6                                5            0.002029
                    metadata_only                  11          0.038602           -0.069218             0.150582                                 0.7514                        5                                6           -0.003521
         normalized_capacity_only                  11          0.026646           -0.061035             0.133234                                 0.6837                        5                                6           -0.006389

### Target: within_dataset_percentile

                      feature_set  n_heldout_datasets  mean_improvement  bootstrap_ci95_low  bootstrap_ci95_high  bootstrap_p_mean_improvement_positive  n_positive_improvements  n_negative_or_zero_improvements  median_improvement
           absolute_capacity_only                  11         -0.032573           -0.079057             0.004424                                 0.0515                        4                                7           -0.015720
absolute_plus_normalized_capacity                  11         -0.015074           -0.057862             0.019269                                 0.2347                        5                                6           -0.000739
                    metadata_only                  11         -0.007886           -0.027547             0.013653                                 0.2169                        3                                8           -0.003305
         normalized_capacity_only                  11         -0.009147           -0.035458             0.023336                                 0.2571                        3                                8           -0.010732

## Interpretation guide

- A positive mean improvement with CI above zero indicates robust transfer benefit across held-out datasets.
- Raw-retention gains that vanish or become negative for centered/percentile targets indicate dataset-level prior learning rather than transferable cell-level degradation prediction.
- Metadata-only raw-retention gains should be interpreted cautiously if centered/percentile gains are near zero or negative.
