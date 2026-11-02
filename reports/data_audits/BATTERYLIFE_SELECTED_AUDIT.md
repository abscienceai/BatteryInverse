# BatteryLife Selected Dataset Audit

## Scope

This audit summarizes selected BatteryLife processed archives downloaded from Zenodo record 14934405. The selected subset includes CALB, HNEI, MICH, MICH_EXP, NA-ion, SNL, UL_PUR, and XJTU.

## Summary

- Parsed cells: 249
- Capacity trajectory rows: 160852
- Parse errors: 0

## Cells by dataset

dataset_id
CALB        27
HNEI        14
MICH        40
MICH_EXP    18
NA-ion      64
SNL         61
UL_PUR       2
XJTU        23

## Chemistry counts

cathode_material
Unknown    64
NMC111     58
NMC        44
NCM        27
NCA        22
LFP        18
LCO        16

## Form factor counts

form_factor
pouch                81
cylindrical_18650    77
18650                64
Prismatic            27

## Quality flags

quality_flags
ok    249

## Numeric summary

       n_cycles_total  n_valid_capacity_cycles  first_capacity_robust_Ah  last_capacity_robust_Ah  capacity_retention_robust  nominal_capacity_Ah  ambient_temperature_C
count      249.000000               249.000000                249.000000               249.000000                 249.000000           249.000000             185.000000
mean       645.991968               645.991968                  7.889018                 7.157638                   0.739235             8.261526              25.432886
std        884.781529               884.781529                 17.032483                16.949646                   0.265417            17.415250              13.501972
min         91.000000                91.000000                  0.706572                 0.067000                   0.021849             1.000000               0.000000
25%        205.000000               205.000000                  0.964036                 0.818073                   0.627590             1.000000              24.048000
50%        401.000000               401.000000                  2.218000                 1.041000                   0.842165             2.360000              25.000000
75%        589.000000               589.000000                  2.870000                 2.344000                   0.926667             3.200000              35.000000
max       4569.000000              4569.000000                 60.480890                60.978710                   1.008231            58.000000              45.000000

