# Expanded Random Split vs Leave-One-Dataset-Out Inflation

## Purpose

This analysis quantifies how much performance is inflated when cells are randomly split while preserving dataset mixture, compared with leave-one-dataset-out transfer.

## Mean MAE by validation mode

        validation_mode              feature_set             model  n_evaluations  mean_mae  std_mae  median_mae    mean_r2  median_r2  mean_baseline_mae  mean_improvement
random_stratified_split   absolute_capacity_only       extra_trees             50  0.051909 0.003812    0.051760   0.842553   0.845081           0.168842          0.116932
random_stratified_split   absolute_capacity_only gradient_boosting             50  0.061072 0.004066    0.061834   0.810651   0.806256           0.168842          0.107769
random_stratified_split   absolute_capacity_only     random_forest             50  0.057163 0.004276    0.055980   0.819924   0.824195           0.168842          0.111679
random_stratified_split all_non_dataset_features       extra_trees             50  0.049384 0.003714    0.049454   0.853429   0.853744           0.168842          0.119458
random_stratified_split all_non_dataset_features gradient_boosting             50  0.055158 0.003497    0.055101   0.837808   0.838862           0.168842          0.113684
random_stratified_split all_non_dataset_features     random_forest             50  0.052629 0.003760    0.052426   0.846072   0.849184           0.168842          0.116212
random_stratified_split            metadata_only       extra_trees             50  0.064781 0.003829    0.064478   0.797758   0.797031           0.168842          0.104060
random_stratified_split            metadata_only gradient_boosting             50  0.064435 0.003983    0.063599   0.790494   0.790654           0.168842          0.104406
random_stratified_split            metadata_only     random_forest             50  0.063904 0.003835    0.063677   0.802144   0.801909           0.168842          0.104937
random_stratified_split normalized_capacity_only       extra_trees             50  0.076410 0.005588    0.076320   0.719985   0.721705           0.168842          0.092431
random_stratified_split normalized_capacity_only gradient_boosting             50  0.088305 0.004743    0.088427   0.660635   0.660793           0.168842          0.080536
random_stratified_split normalized_capacity_only     random_forest             50  0.079724 0.005583    0.079651   0.705909   0.706436           0.168842          0.089118
  leave_one_dataset_out   absolute_capacity_only       extra_trees             12  0.244382 0.146278    0.201256 -48.372421 -15.411229           0.232742         -0.011640
  leave_one_dataset_out   absolute_capacity_only gradient_boosting             12  0.263720 0.146990    0.224355 -75.010060 -13.267270           0.232742         -0.030979
  leave_one_dataset_out   absolute_capacity_only     random_forest             12  0.246258 0.135677    0.189523 -65.621208  -8.642476           0.232742         -0.013517
  leave_one_dataset_out all_non_dataset_features       extra_trees             12  0.262887 0.193711    0.251406 -52.784100 -19.638049           0.232742         -0.030146
  leave_one_dataset_out all_non_dataset_features gradient_boosting             12  0.222236 0.198272    0.138604 -20.591450 -11.682862           0.232742          0.010505
  leave_one_dataset_out all_non_dataset_features     random_forest             12  0.225185 0.211304    0.109580 -25.055721  -8.239213           0.232742          0.007556
  leave_one_dataset_out            metadata_only       extra_trees             12  0.252049 0.183677    0.235540 -49.995698  -9.907288           0.232742         -0.019307
  leave_one_dataset_out            metadata_only gradient_boosting             12  0.215339 0.179576    0.129690 -51.229749  -6.040309           0.232742          0.017402
  leave_one_dataset_out            metadata_only     random_forest             12  0.214197 0.200447    0.126404 -30.163848  -4.739700           0.232742          0.018545
  leave_one_dataset_out normalized_capacity_only       extra_trees             12  0.208147 0.162086    0.158588 -37.434005 -13.805653           0.232742          0.024594
  leave_one_dataset_out normalized_capacity_only gradient_boosting             12  0.226187 0.148449    0.194211 -96.277084 -17.440244           0.232742          0.006554
  leave_one_dataset_out normalized_capacity_only     random_forest             12  0.209121 0.159130    0.166713 -41.653933 -18.102009           0.232742          0.023620

## LODO-to-random inflation

             feature_set             model  leave_one_dataset_out  random_stratified_split  lodo_to_random_mae_ratio  lodo_minus_random_mae
  absolute_capacity_only       extra_trees               0.244382                 0.051909                  4.707875               0.192473
  absolute_capacity_only gradient_boosting               0.263720                 0.061072                  4.318170               0.202648
  absolute_capacity_only     random_forest               0.246258                 0.057163                  4.308041               0.189096
all_non_dataset_features       extra_trees               0.262887                 0.049384                  5.323354               0.213503
all_non_dataset_features gradient_boosting               0.222236                 0.055158                  4.029097               0.167078
all_non_dataset_features     random_forest               0.225185                 0.052629                  4.278712               0.172556
           metadata_only       extra_trees               0.252049                 0.064781                  3.890754               0.187267
           metadata_only gradient_boosting               0.215339                 0.064435                  3.341936               0.150904
           metadata_only     random_forest               0.214197                 0.063904                  3.351843               0.150293
normalized_capacity_only       extra_trees               0.208147                 0.076410                  2.724077               0.131737
normalized_capacity_only gradient_boosting               0.226187                 0.088305                  2.561425               0.137882
normalized_capacity_only     random_forest               0.209121                 0.079724                  2.623075               0.129398

## Interpretation

- Random splits can share dataset/protocol fingerprints between train and test.
- LODO evaluation removes entire dataset domains and is therefore a stronger transfer test.
- Large LODO-to-random MAE ratios indicate benchmark leakage or domain-shift inflation.
