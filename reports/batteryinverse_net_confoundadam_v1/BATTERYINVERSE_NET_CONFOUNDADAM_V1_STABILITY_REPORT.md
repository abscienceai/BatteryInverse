# BatteryInverse-Net + Stable ConfoundAdam v1 report

## Purpose

This v1 experiment fixes target scaling, prediction clipping, unstable negative-unfamiliarity weighting, and weight-collapse reporting from v0. It should still not be used as a manuscript claim unless the closure diagnostics support it.

## Settings

- epochs: 100
- batch_size: 64
- lambda_recon: 0.5
- gamma_adv: 1.0
- grl_alpha: 1.0
- lr: 0.0005
- weight_decay: 0.0001
- grad_clip: 5.0
- hidden_dim: 64
- z_dim: 32
- min_weight: 0.2
- device: auto

## LODO summary

 use_metadata             mode  n_folds  mean_mae  median_mae  mean_improvement  median_improvement
        False         adv_only       12  0.240502    0.147899         -0.007760           -0.012008
        False     confoundadam       12  0.186658    0.134474          0.046083            0.002203
        False confoundadam_adv       12  0.236396    0.183444         -0.003654           -0.019566
        False     vanilla_adam       12  0.190548    0.140609          0.042193           -0.009396
         True         adv_only       12  0.226658    0.158534          0.006083           -0.022076
         True     confoundadam       12  0.223494    0.108588          0.009248           -0.002185
         True confoundadam_adv       12  0.265525    0.203762         -0.032783           -0.009421
         True     vanilla_adam       12  0.231692    0.130725          0.001050           -0.007878

## Closure separability

 use_metadata       feature_source  dataset_balanced_accuracy_mean   ci_low  ci_high
        False            learned_z                        0.766040 0.683221 0.837539
        False confound_weight_norm                        0.201955 0.138642 0.256059
        False      unfamiliarity_U                        0.335159 0.278320 0.396508
         True            learned_z                        0.902361 0.862674 0.977257
         True confound_weight_norm                        0.195332 0.154576 0.239030
         True      unfamiliarity_U                        0.318505 0.253100 0.385289

## Dataset weight summary

dataset_id   n  mean_weight_norm  median_weight_norm  mean_weight_raw  mean_unfamiliarity  mean_risk  full_model_mae  use_metadata
      CALB  27          1.138592            1.147577         0.437123           -0.421074   2.672445        0.017189         False
     CALCE  13          0.770847            0.750676         0.293593           -1.907724   1.302958        0.133242         False
      HNEI  14          0.802287            0.797250         0.309483           -2.099219   1.046362        0.423471         False
      HUST  77          0.915286            0.902196         0.353320           -2.519801   0.809127        0.028942         False
      MATR 180          1.206048            1.180053         0.467128           -3.253248   0.503656        0.073666         False
      MICH  40          1.139265            1.096888         0.439339           -3.383048   0.321103        0.492852         False
  MICH_EXP  18          0.995520            0.885463         0.384221           -1.879321   1.296902        0.182895         False
    NA-ion  64          0.897352            0.834184         0.342959           -2.473114   0.846062        0.108520         False
      RWTH  48          0.725441            0.716495         0.279503           -2.579433   1.107917        0.118860         False
       SNL  61          0.872717            0.822597         0.337587           -1.932377   1.251437        0.085814         False
    UL_PUR   2          0.863224            0.863224         0.331227           -2.188186   0.784139        0.071138         False
      XJTU  23          0.727631            0.722976         0.280355           -0.261761   2.799760        0.038959         False
      CALB  27          1.074300            1.122576         0.423482           -1.018978   1.585170        0.030691          True
     CALCE  13          0.697820            0.642709         0.268810           -0.684841   1.609690        0.724871          True
      HNEI  14          0.863008            0.848246         0.333758           -1.648808   0.905995        0.079847          True
      HUST  77          1.061233            1.043527         0.417599           -2.878067   0.533883        0.011249          True
      MATR 180          1.238440            1.239538         0.490872           -4.020998   0.382428        0.077256          True
      MICH  40          0.967402            0.961310         0.383328           -2.083879   0.863145        0.421769          True
  MICH_EXP  18          1.015942            1.040443         0.398272           -2.066262   0.876013        0.151084          True
    NA-ion  64          0.879506            0.842611         0.346516           -2.068547   0.926700        0.091845          True
      RWTH  48          0.630002            0.631333         0.245234           -0.761127   1.749357        0.860320          True
       SNL  61          0.882819            0.829338         0.348991           -1.526117   1.106619        0.084143          True
    UL_PUR   2          0.886316            0.886316         0.368132           -1.639033   1.153359        0.416688          True
      XJTU  23          0.568256            0.570276         0.219878           -0.304692   1.850449        0.017348          True

## Decision rules

- If learned_z separability remains high, do not claim adversarial invariance solved dataset fingerprints.
- If weights collapse to a narrow range or encode dataset identity, do not claim ConfoundAdam is successful.
- If LODO improvement is not positive relative to train-median, report the method as an attempted mitigation rather than a successful mitigation.
