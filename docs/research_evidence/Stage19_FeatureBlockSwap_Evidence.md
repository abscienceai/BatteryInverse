# BatteryInverse — Stage 19 Feature-Block Swap

## Purpose

Exploratory test of cell-specific early-cycle feature-target
association under the original seven-feature RF/ET LODO recipe.

## Experimental design

- 567 cells, 12 datasets
- 100 seeded permutations per held-out dataset
- Entire seven-feature test blocks reassigned within dataset
- Training data and fitted models unchanged during permutations
- Test targets unchanged
- Dataset membership unchanged
- Train-median prior unchanged

## Main results

| Model | Weighting | Original MAE | Swapped MAE | Difference |
|---|---|---:|---:|---:|
| RF | Equal domain | 0.209121 | 0.221790 | +0.012669 |
| ET | Equal domain | 0.208147 | 0.222040 | +0.013893 |
| RF | Cell weighted | 0.200779 | 0.206843 | +0.006063 |
| ET | Cell weighted | 0.191152 | 0.197450 | +0.006299 |

For both models and both weighting conventions, all
100 sampled aggregate permutations increased MAE.

## Sensitivity excluding UL_PUR

UL_PUR contains only two cells and permits only one
non-identity block permutation.

Excluding UL_PUR:

| Model | Equal-domain difference | Cell-weighted difference |
|---|---:|---:|
| RF | +0.015852 | +0.006164 |
| ET | +0.017703 | +0.006420 |

## Reproduction qualification

Extra Trees replay agrees with the archived original
fold results to numerical precision.

The Random Forest replay has a maximum fold-level
MAE discrepancy of approximately 0.000211.

The cause of this minor discrepancy is not established.

## Interpretation

The observed error increases are consistent with
the models using some cell-specific feature-target
association within represented held-out datasets.

The analysis does not establish causal physical
degradation learning, prospective generalization,
or independent population-level significance.

Repeated permutations within fixed evaluation
datasets are not independent experimental cohorts.

The effects of individual datasets should be
considered using the recorded domain-influence
analysis.

## Reproduction

Run from the project root:

python external_validation/scripts/stage19_featureblock_swap.py

The original results are stored under
external_validation/reports/stage19_featureblock_swap_*.
