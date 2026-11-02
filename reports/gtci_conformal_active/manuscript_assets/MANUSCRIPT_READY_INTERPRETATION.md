# Manuscript-ready interpretation: GTCI, conformal abstention and active acquisition

## Ground-truth confound injection

The first GTCI implementation did not create a dominant shortcut. Random-split MAE was 0.049 [0.043--0.056] in the clean setting and 0.049 [0.043--0.057] after adding the injected dataset confound. LODO MAE changed from 0.262 [0.046--0.601] to 0.268 [0.045--0.607]. The injected feature had a small but non-zero permutation importance of 0.0040 [0.0027--0.0054]. This result should be interpreted as a weak positive-control rather than definitive evidence that the method suppresses a known shortcut. A stronger target-prior-aligned GTCI variant is needed before making a main-text mitigation claim.

## Dataset-Mondrian / LODO conformal abstention

The conformal analysis provides a strong rank-safety result. The global LODO residual calibration gives a 90% half-width of 0.444 retention and an 80% half-width of 0.301 retention. Both exceed the ±0.10 retention threshold used for rank-safe candidate ordering. Therefore, under this rule the model should abstain from inverse-design ranking without additional domain-specific validation data. Coverage is also highly domain-dependent: the hardest domains include MICH, CALCE, MICH_EXP, SNL, MATR. This supports the manuscript's claim that exchangeability cannot be assumed under domain shift.

## Retrospective core-set acquisition

Core-set acquisition partially closes the transfer gap, but it is not uniformly superior to random sampling. Aggregated across datasets, the fraction of cases where core-set selection beats random selection is:
 k  fraction_coreset_better  mean_advantage_vs_random  median_advantage_vs_random
 3                 0.272727                 -0.017968                   -0.007500
 5                 0.363636                 -0.028874                   -0.003452
10                 0.363636                 -0.014992                   -0.002610

This means the current raw-feature k-center strategy should be treated as a prospective sampling aid, not as a guaranteed optimal acquisition method. The strongest gains appear in:
heldout_dataset  k  coreset_advantage_vs_random
          CALCE  5                     0.023012
           MATR  3                     0.021157
           MICH 10                     0.019110
           CALB 10                     0.009980
           RWTH  3                     0.009015

The weakest cases are:
heldout_dataset  k  coreset_advantage_vs_random
       MICH_EXP  5                    -0.197415
       MICH_EXP  3                    -0.114539
           MICH  5                    -0.076966
       MICH_EXP 10                    -0.068758
          CALCE 10                    -0.061376

## Recommended manuscript framing

These experiments should be added as a cautious risk-control layer. The conformal abstention result is strong enough for the main text. GTCI v0 and core-set v0 should be reported as preliminary controls or supplementary analyses unless improved by a stronger GTCI variant and a debiased-z acquisition strategy.
