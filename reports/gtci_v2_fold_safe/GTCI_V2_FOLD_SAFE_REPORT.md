# GTCI v2 fold-safe target-prior confound report

This version is split-aware. Random-split xi is computed from train-split dataset medians. LODO test xi does not use the held-out dataset median; it receives only the train-global unknown-domain prior. Therefore LODO cannot access held-out target-prior information through xi.

## Summary

       feature_set            validation  mean_mae  ci_low_mae  ci_high_mae  mean_improvement  ci_low_improvement  ci_high_improvement
           xi_only          random_split  0.071636    0.061242     0.080079          0.097315            0.086669             0.104916
           xi_only leave_one_dataset_out  0.229073    0.045197     0.510108          0.003668           -0.020295             0.031326
    absolute_clean          random_split  0.051974    0.045282     0.061362          0.116977            0.106067             0.128877
    absolute_clean leave_one_dataset_out  0.241213    0.077219     0.508128         -0.008472           -0.217797             0.363940
  absolute_plus_xi          random_split  0.050337    0.044059     0.058190          0.118614            0.109645             0.130484
  absolute_plus_xi leave_one_dataset_out  0.253852    0.067900     0.556354         -0.021110           -0.071501             0.041928
  normalized_clean          random_split  0.076128    0.065270     0.086159          0.092823            0.080489             0.104512
  normalized_clean leave_one_dataset_out  0.207342    0.054514     0.566889          0.025399           -0.136386             0.358138
normalized_plus_xi          random_split  0.054516    0.048077     0.061700          0.114435            0.103653             0.124624
normalized_plus_xi leave_one_dataset_out  0.227311    0.072034     0.517724          0.005430           -0.070526             0.099387
         all_clean          random_split  0.049129    0.042263     0.058001          0.119823            0.109570             0.130408
         all_clean leave_one_dataset_out  0.263079    0.045631     0.601349         -0.030338           -0.259491             0.196921
       all_plus_xi          random_split  0.048208    0.041678     0.056236          0.120743            0.112251             0.131520
       all_plus_xi leave_one_dataset_out  0.242815    0.028156     0.603452         -0.010074           -0.138668             0.068572

## Delta table

    family            validation  delta_mae_plus_xi_minus_clean    ci_low   ci_high                          interpretation
  absolute          random_split                      -0.001637 -0.006056  0.002389 negative means xi improves apparent MAE
  absolute leave_one_dataset_out                       0.012639 -0.162844  0.360261 negative means xi improves apparent MAE
normalized          random_split                      -0.021612 -0.029328 -0.014104 negative means xi improves apparent MAE
normalized leave_one_dataset_out                       0.019969 -0.111214  0.291045 negative means xi improves apparent MAE
       all          random_split                      -0.000920 -0.002526  0.000801 negative means xi improves apparent MAE
       all leave_one_dataset_out                      -0.020264 -0.234984  0.173013 negative means xi improves apparent MAE

## Xi permutation importance under random split

                                                                                                                                                                                                                                                                                                                                                                                                                                        feature_set  xi_permutation_importance_mean   ci_low  ci_high   feature_set_name
                                                                                                                                                                                                                                                                                                                                                                                                                     gtci_xi_fold_safe_target_prior                        0.158484 0.130006 0.174913            xi_only
                                                                                                                                                                                                                                                                                      cap_cycle_1,cap_cycle_10,cap_cycle_50,cap_cycle_100,cap_mean_first_100,cap_std_first_100,cap_slope_first_100,cap_fade_1_to_100,gtci_xi_fold_safe_target_prior                        0.176407 0.147605 0.195537   absolute_plus_xi
                                                                                                                                                                                                                                                           cap_ratio_10_to_1,cap_ratio_50_to_1,cap_ratio_100_to_1,cap_mean_first_100_norm,cap_std_first_100_norm,cap_slope_first_100_norm,cap_fade_fraction_1_to_100,gtci_xi_fold_safe_target_prior                        0.172374 0.146366 0.191372 normalized_plus_xi
cap_cycle_1,cap_cycle_10,cap_cycle_50,cap_cycle_100,cap_mean_first_100,cap_std_first_100,cap_slope_first_100,cap_fade_1_to_100,cap_ratio_10_to_1,cap_ratio_50_to_1,cap_ratio_100_to_1,cap_mean_first_100_norm,cap_std_first_100_norm,cap_slope_first_100_norm,cap_fade_fraction_1_to_100,nominal_capacity_Ah,min_voltage_V,max_voltage_V,chemistry,form_factor,anode_material,cathode_material,ambient_temperature_C,gtci_xi_fold_safe_target_prior                        0.130032 0.110336 0.148268        all_plus_xi

## Interpretation guide

- Strong positive-control behaviour: xi improves random-split MAE but not LODO MAE.
- xi_only should look useful under random split but collapse to global-prior behaviour under LODO.
- If LODO improves strongly after xi, check for leakage.
