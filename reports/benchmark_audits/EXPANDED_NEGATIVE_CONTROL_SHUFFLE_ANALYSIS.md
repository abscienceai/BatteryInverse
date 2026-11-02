# Expanded Negative-Control Shuffle Analysis

## Purpose

This analysis tests whether the pipeline produces apparent performance when the target signal is destroyed globally or within each dataset. Within-dataset shuffling preserves dataset-level target distributions but destroys cell-level feature-target alignment.

## Summary

        validation_mode                target_variant              feature_set         model  n_evaluations  mean_mae  median_mae    mean_r2  median_r2  mean_baseline_mae  mean_improvement  median_improvement
  leave_one_dataset_out         target_global_shuffle   absolute_capacity_only   extra_trees             12  0.209064    0.213916  -0.947794  -0.173351           0.162227         -0.046837           -0.037355
  leave_one_dataset_out         target_global_shuffle   absolute_capacity_only random_forest             12  0.213081    0.195247  -1.296086  -0.103381           0.162227         -0.050853           -0.035222
  leave_one_dataset_out         target_global_shuffle all_non_dataset_features   extra_trees             12  0.197403    0.199426  -0.456793  -0.086049           0.162227         -0.035175           -0.010109
  leave_one_dataset_out         target_global_shuffle all_non_dataset_features random_forest             12  0.219031    0.195084  -0.848896  -0.194130           0.162227         -0.056804           -0.043397
  leave_one_dataset_out         target_global_shuffle            metadata_only   extra_trees             12  0.182146    0.191470  -0.295617  -0.124987           0.162227         -0.019919           -0.012211
  leave_one_dataset_out         target_global_shuffle            metadata_only random_forest             12  0.188115    0.192787  -0.783703  -0.039425           0.162227         -0.025887           -0.014005
  leave_one_dataset_out         target_global_shuffle normalized_capacity_only   extra_trees             12  0.216504    0.213299  -1.664605  -0.203955           0.162227         -0.054277           -0.034240
  leave_one_dataset_out         target_global_shuffle normalized_capacity_only random_forest             12  0.215799    0.208614  -1.472767  -0.135939           0.162227         -0.053571           -0.036881
  leave_one_dataset_out               target_original   absolute_capacity_only   extra_trees             12  0.244448    0.202086 -47.635663 -15.524005           0.232742         -0.011706           -0.048319
  leave_one_dataset_out               target_original   absolute_capacity_only random_forest             12  0.245366    0.188029 -64.945600  -8.485153           0.232742         -0.012624           -0.033400
  leave_one_dataset_out               target_original all_non_dataset_features   extra_trees             12  0.261993    0.251238 -52.521349 -19.489288           0.232742         -0.029252           -0.003645
  leave_one_dataset_out               target_original all_non_dataset_features random_forest             12  0.224534    0.109193 -25.360631  -8.056304           0.232742          0.008207           -0.007041
  leave_one_dataset_out               target_original            metadata_only   extra_trees             12  0.251659    0.236329 -50.919279  -9.768680           0.232742         -0.018918           -0.001822
  leave_one_dataset_out               target_original            metadata_only random_forest             12  0.214390    0.126394 -30.285101  -4.913346           0.232742          0.018351           -0.011402
  leave_one_dataset_out               target_original normalized_capacity_only   extra_trees             12  0.208312    0.159218 -37.793768 -14.668365           0.232742          0.024430           -0.012621
  leave_one_dataset_out               target_original normalized_capacity_only random_forest             12  0.208318    0.166253 -41.026860 -17.887870           0.232742          0.024424           -0.002466
  leave_one_dataset_out target_within_dataset_shuffle   absolute_capacity_only   extra_trees             12  0.228513    0.218456 -46.984660 -17.970332           0.232742          0.004229           -0.033825
  leave_one_dataset_out target_within_dataset_shuffle   absolute_capacity_only random_forest             12  0.222454    0.215006 -37.292982 -13.872696           0.232742          0.010287           -0.019764
  leave_one_dataset_out target_within_dataset_shuffle all_non_dataset_features   extra_trees             12  0.250708    0.219519 -49.060928 -11.542928           0.232742         -0.017967            0.009966
  leave_one_dataset_out target_within_dataset_shuffle all_non_dataset_features random_forest             12  0.217328    0.135908 -27.803028  -8.308111           0.232742          0.015413            0.008469
  leave_one_dataset_out target_within_dataset_shuffle            metadata_only   extra_trees             12  0.251277    0.242930 -54.678804  -7.686065           0.232742         -0.018535            0.013426
  leave_one_dataset_out target_within_dataset_shuffle            metadata_only random_forest             12  0.207776    0.109999 -28.465838  -6.630111           0.232742          0.024965           -0.001519
  leave_one_dataset_out target_within_dataset_shuffle normalized_capacity_only   extra_trees             12  0.223207    0.187184 -49.960637 -21.649419           0.232742          0.009534           -0.025086
  leave_one_dataset_out target_within_dataset_shuffle normalized_capacity_only random_forest             12  0.211803    0.178322 -52.681695 -15.784461           0.232742          0.020938           -0.022474
random_stratified_split         target_global_shuffle   absolute_capacity_only   extra_trees             30  0.207158    0.208736  -0.247520  -0.235081           0.166154         -0.041003           -0.040512
random_stratified_split         target_global_shuffle   absolute_capacity_only random_forest             30  0.203562    0.203930  -0.192118  -0.177496           0.166154         -0.037407           -0.035804
random_stratified_split         target_global_shuffle all_non_dataset_features   extra_trees             30  0.207599    0.210188  -0.266323  -0.271110           0.166154         -0.041445           -0.040126
random_stratified_split         target_global_shuffle all_non_dataset_features random_forest             30  0.204017    0.203819  -0.199079  -0.201798           0.166154         -0.037863           -0.038234
random_stratified_split         target_global_shuffle            metadata_only   extra_trees             30  0.194686    0.194314  -0.085667  -0.083827           0.166154         -0.028532           -0.030049
random_stratified_split         target_global_shuffle            metadata_only random_forest             30  0.194542    0.193293  -0.086534  -0.083764           0.166154         -0.028388           -0.028152
random_stratified_split         target_global_shuffle normalized_capacity_only   extra_trees             30  0.207213    0.207861  -0.245857  -0.232506           0.166154         -0.041059           -0.041009
random_stratified_split         target_global_shuffle normalized_capacity_only random_forest             30  0.207403    0.207358  -0.232594  -0.241812           0.166154         -0.041249           -0.041529
random_stratified_split               target_original   absolute_capacity_only   extra_trees             30  0.052374    0.051995   0.840044   0.842415           0.168407          0.116033            0.116650
random_stratified_split               target_original   absolute_capacity_only random_forest             30  0.057918    0.056785   0.816266   0.822885           0.168407          0.110489            0.111012
random_stratified_split               target_original all_non_dataset_features   extra_trees             30  0.049352    0.049508   0.852912   0.853922           0.168407          0.119055            0.118666
random_stratified_split               target_original all_non_dataset_features random_forest             30  0.052773    0.052447   0.844249   0.851265           0.168407          0.115634            0.116119
random_stratified_split               target_original            metadata_only   extra_trees             30  0.064497    0.063814   0.800036   0.804742           0.168407          0.103910            0.103641
random_stratified_split               target_original            metadata_only random_forest             30  0.063510    0.062310   0.805186   0.813882           0.168407          0.104897            0.104839
random_stratified_split               target_original normalized_capacity_only   extra_trees             30  0.077015    0.076521   0.715180   0.722972           0.168407          0.091392            0.090832
random_stratified_split               target_original normalized_capacity_only random_forest             30  0.080148    0.079706   0.700839   0.697769           0.168407          0.088259            0.088357
random_stratified_split target_within_dataset_shuffle   absolute_capacity_only   extra_trees             30  0.080916    0.081543   0.710746   0.709433           0.169727          0.088811            0.089532
random_stratified_split target_within_dataset_shuffle   absolute_capacity_only random_forest             30  0.081651    0.081835   0.706023   0.698457           0.169727          0.088076            0.087602
random_stratified_split target_within_dataset_shuffle all_non_dataset_features   extra_trees             30  0.080937    0.081299   0.711439   0.712393           0.169727          0.088790            0.089226
random_stratified_split target_within_dataset_shuffle all_non_dataset_features random_forest             30  0.077558    0.078014   0.736392   0.724827           0.169727          0.092169            0.091996
random_stratified_split target_within_dataset_shuffle            metadata_only   extra_trees             30  0.073217    0.074281   0.757220   0.755234           0.169727          0.096510            0.097717
random_stratified_split target_within_dataset_shuffle            metadata_only random_forest             30  0.072941    0.073542   0.760585   0.759750           0.169727          0.096786            0.097745
random_stratified_split target_within_dataset_shuffle normalized_capacity_only   extra_trees             30  0.097634    0.096987   0.604411   0.605245           0.169727          0.072093            0.071851
random_stratified_split target_within_dataset_shuffle normalized_capacity_only random_forest             30  0.099552    0.099228   0.591589   0.597699           0.169727          0.070176            0.070793

## Interpretation guide

- Global shuffling should remove meaningful signal.
- If random splits retain non-trivial performance under within-dataset shuffling, that indicates dataset-level priors can drive apparent performance.
- LODO performance under shuffled targets should generally fail to beat the baseline.
