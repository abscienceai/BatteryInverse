# Expanded Cross-Source Transfer Analysis

## Purpose

This analysis evaluates transfer between the original BatteryML-derived source family and the selected BatteryLife source family. It tests whether models trained on one source family generalize to the other.

## Source family counts

source_family
BatteryML_main          318
BatteryLife_selected    249

## Summary

        train_source          test_source                            target              feature_set  n_models  n_train  n_test  mean_mae  median_mae  mean_baseline_mae  mean_improvement  median_improvement  n_positive  n_nonpositive
BatteryLife_selected       BatteryML_main   dataset_mean_centered_retention   absolute_capacity_only         3      249     318  0.118307    0.117493           0.057376         -0.060931           -0.060116           0              3
BatteryLife_selected       BatteryML_main   dataset_mean_centered_retention all_non_dataset_features         3      249     318  0.100714    0.101316           0.057376         -0.043337           -0.043939           0              3
BatteryLife_selected       BatteryML_main   dataset_mean_centered_retention            metadata_only         3      249     318  0.061192    0.059170           0.057376         -0.003815           -0.001793           0              3
BatteryLife_selected       BatteryML_main   dataset_mean_centered_retention normalized_capacity_only         3      249     318  0.073975    0.071485           0.057376         -0.016599           -0.014108           0              3
BatteryLife_selected       BatteryML_main dataset_median_centered_retention   absolute_capacity_only         3      249     318  0.123181    0.118366           0.052276         -0.070906           -0.066091           0              3
BatteryLife_selected       BatteryML_main dataset_median_centered_retention all_non_dataset_features         3      249     318  0.105931    0.105942           0.052276         -0.053656           -0.053667           0              3
BatteryLife_selected       BatteryML_main dataset_median_centered_retention            metadata_only         3      249     318  0.056905    0.053235           0.052276         -0.004629           -0.000959           0              3
BatteryLife_selected       BatteryML_main dataset_median_centered_retention normalized_capacity_only         3      249     318  0.087165    0.081892           0.052276         -0.034889           -0.029616           0              3
BatteryLife_selected       BatteryML_main                     raw_retention   absolute_capacity_only         3      249     318  0.219004    0.232522           0.180009         -0.038995           -0.052513           0              3
BatteryLife_selected       BatteryML_main                     raw_retention all_non_dataset_features         3      249     318  0.179189    0.187555           0.180009          0.000820           -0.007546           1              2
BatteryLife_selected       BatteryML_main                     raw_retention            metadata_only         3      249     318  0.142740    0.142349           0.180009          0.037269            0.037660           3              0
BatteryLife_selected       BatteryML_main                     raw_retention normalized_capacity_only         3      249     318  0.248764    0.251836           0.180009         -0.068755           -0.071827           0              3
BatteryLife_selected       BatteryML_main    source_mean_centered_retention   absolute_capacity_only         3      249     318  0.154514    0.159708           0.139575         -0.014939           -0.020132           0              3
BatteryLife_selected       BatteryML_main    source_mean_centered_retention all_non_dataset_features         3      249     318  0.113281    0.118165           0.139575          0.026294            0.021410           3              0
BatteryLife_selected       BatteryML_main    source_mean_centered_retention            metadata_only         3      249     318  0.093154    0.084473           0.139575          0.046421            0.055102           3              0
BatteryLife_selected       BatteryML_main    source_mean_centered_retention normalized_capacity_only         3      249     318  0.197519    0.205055           0.139575         -0.057944           -0.065479           0              3
BatteryLife_selected       BatteryML_main  source_median_centered_retention   absolute_capacity_only         3      249     318  0.144376    0.148512           0.138133         -0.006243           -0.010379           1              2
BatteryLife_selected       BatteryML_main  source_median_centered_retention all_non_dataset_features         3      249     318  0.101069    0.107148           0.138133          0.037064            0.030985           3              0
BatteryLife_selected       BatteryML_main  source_median_centered_retention            metadata_only         3      249     318  0.088337    0.079480           0.138133          0.049796            0.058653           3              0
BatteryLife_selected       BatteryML_main  source_median_centered_retention normalized_capacity_only         3      249     318  0.189223    0.190494           0.138133         -0.051090           -0.052361           0              3
      BatteryML_main BatteryLife_selected   dataset_mean_centered_retention   absolute_capacity_only         3      318     249  0.129333    0.125710           0.076618         -0.052714           -0.049091           0              3
      BatteryML_main BatteryLife_selected   dataset_mean_centered_retention all_non_dataset_features         3      318     249  0.127487    0.124405           0.076618         -0.050869           -0.047787           0              3
      BatteryML_main BatteryLife_selected   dataset_mean_centered_retention            metadata_only         3      318     249  0.108461    0.108918           0.076618         -0.031843           -0.032300           0              3
      BatteryML_main BatteryLife_selected   dataset_mean_centered_retention normalized_capacity_only         3      318     249  0.099984    0.090935           0.076618         -0.023365           -0.014317           0              3
      BatteryML_main BatteryLife_selected dataset_median_centered_retention   absolute_capacity_only         3      318     249  0.122943    0.125364           0.074619         -0.048323           -0.050744           0              3
      BatteryML_main BatteryLife_selected dataset_median_centered_retention all_non_dataset_features         3      318     249  0.119936    0.113785           0.074619         -0.045316           -0.039165           0              3
      BatteryML_main BatteryLife_selected dataset_median_centered_retention            metadata_only         3      318     249  0.090475    0.089395           0.074619         -0.015856           -0.014775           0              3
      BatteryML_main BatteryLife_selected dataset_median_centered_retention normalized_capacity_only         3      318     249  0.083074    0.083688           0.074619         -0.008455           -0.009069           0              3
      BatteryML_main BatteryLife_selected                     raw_retention   absolute_capacity_only         3      318     249  0.293515    0.297985           0.208639         -0.084876           -0.089347           0              3
      BatteryML_main BatteryLife_selected                     raw_retention all_non_dataset_features         3      318     249  0.374219    0.360575           0.208639         -0.165580           -0.151937           0              3
      BatteryML_main BatteryLife_selected                     raw_retention            metadata_only         3      318     249  0.389186    0.379774           0.208639         -0.180548           -0.171135           0              3
      BatteryML_main BatteryLife_selected                     raw_retention normalized_capacity_only         3      318     249  0.326315    0.325501           0.208639         -0.117676           -0.116862           0              3
      BatteryML_main BatteryLife_selected    source_mean_centered_retention   absolute_capacity_only         3      318     249  0.253546    0.258411           0.188473         -0.065073           -0.069938           0              3
      BatteryML_main BatteryLife_selected    source_mean_centered_retention all_non_dataset_features         3      318     249  0.327590    0.322693           0.188473         -0.139117           -0.134221           0              3
      BatteryML_main BatteryLife_selected    source_mean_centered_retention            metadata_only         3      318     249  0.320512    0.321820           0.188473         -0.132040           -0.133347           0              3
      BatteryML_main BatteryLife_selected    source_mean_centered_retention normalized_capacity_only         3      318     249  0.285687    0.280157           0.188473         -0.097215           -0.091684           0              3
      BatteryML_main BatteryLife_selected  source_median_centered_retention   absolute_capacity_only         3      318     249  0.250392    0.252914           0.187409         -0.062983           -0.065505           0              3
      BatteryML_main BatteryLife_selected  source_median_centered_retention all_non_dataset_features         3      318     249  0.328293    0.335185           0.187409         -0.140884           -0.147776           0              3
      BatteryML_main BatteryLife_selected  source_median_centered_retention            metadata_only         3      318     249  0.314576    0.313645           0.187409         -0.127167           -0.126237           0              3
      BatteryML_main BatteryLife_selected  source_median_centered_retention normalized_capacity_only         3      318     249  0.283549    0.285914           0.187409         -0.096140           -0.098505           0              3

## Interpretation guide

- Raw-retention success can reflect source-family target-distribution priors.
- Source-centered and dataset-centered targets test whether cross-source transfer captures residual degradation behavior.
- Strong asymmetry between BatteryML-to-BatteryLife and BatteryLife-to-BatteryML indicates source-family domain shift.
