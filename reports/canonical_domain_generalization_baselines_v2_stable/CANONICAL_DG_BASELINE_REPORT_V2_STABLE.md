# Stable canonical domain-generalization baseline audit

This experiment compares small neural ERM, IRM and DANN-style baselines under leave-one-dataset-out battery validation using target standardization, Huber loss and prediction clipping to the training target range. The goal is to test whether canonical mitigation objectives remove dataset-prior shortcut risk under a fairer stable neural setup.

## Detected setup

- Rows: 565
- Domain column: dataset_id
- Target column: final_capacity_retention
- Feature matrix shape: (565, 50)
- Held-out datasets: 11

## Summary

- dann: mean LODO MAE = 0.261; median LODO MAE = 0.205; mean improvement over train median = -0.018; learned-representation dataset BA = 0.956
- erm: mean LODO MAE = 0.259; median LODO MAE = 0.183; mean improvement over train median = -0.016; learned-representation dataset BA = 0.968
- irm: mean LODO MAE = 0.244; median LODO MAE = 0.166; mean improvement over train median = -0.002; learned-representation dataset BA = 0.964

## Interpretation

If ERM, IRM and DANN remain close in LODO error and their learned representations remain highly dataset-identifiable, the result supports a conservative conclusion: canonical domain-generalization objectives do not by themselves certify transferable scientific learning. They require the same closure audits used elsewhere in BatteryInverse.
