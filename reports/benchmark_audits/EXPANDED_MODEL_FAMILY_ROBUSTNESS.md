# Expanded Model-Family Robustness

## Purpose

This analysis tests whether the expanded transfer-gap conclusions depend on one model family. Multiple regressors are evaluated under leave-one-dataset-out transfer.

## Summary

             feature_set                  model  n_evaluations  mean_mae  median_mae  mean_improvement  median_improvement  n_positive  n_nonpositive
  absolute_capacity_only             elasticnet             12  0.946951    0.251372         -0.714210           -0.026819           5              7
  absolute_capacity_only            extra_trees             12  0.244382    0.201256         -0.011640           -0.047411           3              9
  absolute_capacity_only      gradient_boosting             12  0.263720    0.224355         -0.030979           -0.042264           3              9
  absolute_capacity_only hist_gradient_boosting             12  0.263212    0.272405         -0.030471           -0.051157           2             10
  absolute_capacity_only                    knn             12  0.268240    0.256024         -0.035499           -0.066494           3              9
  absolute_capacity_only          random_forest             12  0.246258    0.189523         -0.013517           -0.033756           3              9
  absolute_capacity_only                  ridge             12  0.942784    0.251165         -0.710043           -0.027549           5              7
all_non_dataset_features             elasticnet             12  1.180897    0.233225         -0.948155           -0.078311           5              7
all_non_dataset_features            extra_trees             12  0.262887    0.251406         -0.030146           -0.003844           6              6
all_non_dataset_features      gradient_boosting             12  0.222236    0.138604          0.010505           -0.019447           5              7
all_non_dataset_features hist_gradient_boosting             12  0.202759    0.140318          0.029983            0.019925           8              4
all_non_dataset_features                    knn             12  0.256688    0.194113         -0.023947           -0.045641           3              9
all_non_dataset_features          random_forest             12  0.225185    0.109580          0.007556           -0.007980           6              6
all_non_dataset_features                  ridge             12  1.194192    0.257316         -0.961450           -0.083267           5              7
           metadata_only             elasticnet             12  1.069565    0.417066         -0.836824           -0.064654           5              7
           metadata_only            extra_trees             12  0.252049    0.235540         -0.019307           -0.002631           5              7
           metadata_only      gradient_boosting             12  0.215339    0.129690          0.017402            0.022436           7              5
           metadata_only hist_gradient_boosting              4  0.226124    0.167936         -0.006199           -0.005984           2              2
           metadata_only                    knn             12  0.249177    0.206257         -0.016436           -0.016502           5              7
           metadata_only          random_forest             12  0.214197    0.126404          0.018545           -0.011119           5              7
           metadata_only                  ridge             12  1.083849    0.409728         -0.851107           -0.076440           4              8
normalized_capacity_only             elasticnet             12  0.216227    0.188481          0.016515           -0.002490           4              8
normalized_capacity_only            extra_trees             12  0.208147    0.158588          0.024594           -0.013097           5              7
normalized_capacity_only      gradient_boosting             12  0.226187    0.194211          0.006554           -0.029553           5              7
normalized_capacity_only hist_gradient_boosting             12  0.224017    0.204874          0.008725           -0.008726           5              7
normalized_capacity_only                    knn             12  0.200135    0.141793          0.032606           -0.001274           6              6
normalized_capacity_only          random_forest             12  0.209121    0.166713          0.023620           -0.004428           5              7
normalized_capacity_only                  ridge             12  0.214716    0.188826          0.018025           -0.002715           4              8

## Interpretation

- If no model family robustly beats the train-median baseline across datasets, the transfer-gap conclusion is model-family robust.
- If metadata-only raw retention remains strong only for some datasets/models, this supports dataset-prior rather than universal degradation learning.
- The strongest manuscript claim should focus on patterns that persist across model families.
