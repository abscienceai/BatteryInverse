# Expanded Dataset Separability Analysis

## Purpose

This analysis tests whether the held-out dataset identity can be predicted from early-cycle and metadata-derived features in the expanded 567-cell table. High separability indicates dataset fingerprints in the feature space.

## Dataset counts

dataset_id
CALB         27
CALCE        13
HNEI         14
HUST         77
MATR        180
MICH         40
MICH_EXP     18
NA-ion       64
RWTH         48
SNL          61
UL_PUR        2
XJTU         23

Datasets with fewer than 5 cells are excluded from stratified separability scoring but retained in descriptive counts.

## Summary metrics

                      feature_set  n_splits  mean_accuracy  std_accuracy  mean_balanced_accuracy  std_balanced_accuracy  mean_macro_f1  std_macro_f1  mean_weighted_f1  majority_class_accuracy  n_classes  n_total
           absolute_capacity_only        30       0.974314      0.011297                0.962409               0.019150       0.966401      0.016110          0.973637                 0.318584       11.0    565.0
absolute_plus_normalized_capacity        30       0.980784      0.008999                0.968715               0.016189       0.972273      0.013741          0.980382                 0.318584       11.0    565.0
         all_non_dataset_features        30       1.000000      0.000000                1.000000               0.000000       1.000000      0.000000          1.000000                 0.318584       11.0    565.0
                    metadata_only        30       1.000000      0.000000                1.000000               0.000000       1.000000      0.000000          1.000000                 0.318584       11.0    565.0
         normalized_capacity_only        30       0.880588      0.017073                0.808799               0.038382       0.815623      0.034119          0.876910                 0.318584       11.0    565.0

## Interpretation guide

- Accuracy far above majority-class accuracy indicates strong dataset separability.
- High separability using normalized capacity features means normalization does not remove dataset/protocol fingerprints.
- High separability using metadata is expected but still important because metadata-only raw-retention success can reflect target-prior transfer rather than cell-level degradation physics.
- This result should be interpreted together with the expanded deconfounded transfer analysis.
