#!/usr/bin/env python3
"""Stage21A: sign encoding and training-only family-prior audit."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path.cwd()
data_file = root / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)
flag_file = root / (
    "external_validation/reports/"
    "stage20o_unified_endpoint_cells.csv"
)
out_dir = root / "external_validation/reports"
out_csv = out_dir / "stage21a_family_prior_by_domain.csv"
out_json = out_dir / "stage21a_feature_sign_preflight.json"

expected_hash = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(
    all(not p.exists() for p in (out_csv, out_json)),
    "Stage21A outputs already exist; refusing overwrite"
)

require(
    hashlib.sha256(data_file.read_bytes()).hexdigest()
    == expected_hash,
    "Benchmark SHA256 mismatch"
)

d = pd.read_csv(data_file)

require(
    len(d) == 567 and d.dataset_id.nunique() == 12,
    "Unexpected benchmark dimensions"
)
require(
    not d.duplicated(["dataset_id", "cell_id"]).any(),
    "Duplicate benchmark cell key"
)
require(
    set(d.source_family)
    == {"BatteryML_main", "BatteryLife_selected"},
    "Unexpected source families"
)

fcol = next(
    (
        c for c in (
            "cap_fade_1_to_100_norm",
            "cap_fade_fraction_1_to_100",
        )
        if c in d.columns
    ),
    None
)
require(fcol is not None, "Fade feature not found")

spec = importlib.util.spec_from_file_location(
    "historical_stage19",
    root / "external_validation/scripts/"
           "stage19_featureblock_swap.py"
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
model_features = list(original.F)

r = d.cap_ratio_100_to_1.to_numpy(float)
f = d[fcol].to_numpy(float)
family = d.source_family.to_numpy(str)

expected_f = np.where(
    family == "BatteryML_main",
    r - 1,
    1 - r
)

require(
    np.isfinite(r).all() and np.isfinite(f).all(),
    "Nonfinite ratio or fade"
)

max_error = float(np.max(np.abs(f - expected_f)))
require(
    max_error < 1e-8,
    "Source-family fade formula mismatch"
)

ratio_checks = {}
for col in ("cap_cycle_100_norm", "cap_ratio_100_to_1"):
    if col in d:
        ratio_checks[col] = float(
            np.max(np.abs(d[col].to_numpy(float) - r))
        )

ambiguous = np.isclose(
    r, 1, rtol=0, atol=1e-9
)
error_ml = np.abs(f - (r - 1))
error_bl = np.abs(f - (1 - r))

inferred = np.where(
    error_ml < error_bl,
    "BatteryML_main",
    "BatteryLife_selected"
)

nonambiguous = int((~ambiguous).sum())
matched = int(
    ((inferred == family) & ~ambiguous).sum()
)

require(
    matched == nonambiguous,
    "Source-family sign recovery mismatch"
)

flags = pd.read_csv(flag_file)

require(
    len(flags) == 249,
    "Unexpected Stage20O cell count"
)

require(
    not flags.duplicated(["dataset", "cell_id"]).any(),
    "Duplicate quality-audit key"
)

excluded = {
    (str(row.dataset), str(row.cell_id))
    for row in flags.itertuples(index=False)
    if (
        int(row.positive_post100_records) < 3
        or int(row.mich_nonpositive_endpoint_flag) == 1
    )
}
require(len(excluded) == 44, "Flag count mismatch")

rows = []

for domain in sorted(d.dataset_id.unique()):
    test = d.dataset_id == domain
    held = d.loc[test]
    train = d.loc[~test]

    require(
        held.source_family.nunique() == 1,
        "Mixed source families in one dataset"
    )

    held_family = str(held.source_family.iloc[0])

    family_targets = train.loc[
        train.source_family == held_family,
        "final_capacity_retention"
    ].to_numpy(float)

    require(
        len(family_targets) > 0,
        "No training cells from held-out source family"
    )

    global_prior = float(
        np.median(train.final_capacity_retention)
    )
    family_prior = float(
        np.median(family_targets)
    )

    y = held.final_capacity_retention.to_numpy(float)

    retain = np.array([
        (str(row.dataset_id), str(row.cell_id))
        not in excluded
        for row in held.itertuples(index=False)
    ])

    policies = [
        ("all", np.ones(len(held), dtype=bool)),
        ("exclude_44_BatteryLife_flags", retain),
    ]

    for policy, mask in policies:
        if not mask.any():
            continue

        yy = y[mask]
        global_mae = float(
            np.abs(yy - global_prior).mean()
        )
        family_mae = float(
            np.abs(yy - family_prior).mean()
        )

        rows.append({
            "dataset": str(domain),
            "source_family": held_family,
            "policy": policy,
            "cells": len(yy),
            "training_family_cells": len(family_targets),
            "global_train_median": global_prior,
            "family_train_median": family_prior,
            "global_prior_mae": global_mae,
            "family_prior_mae": family_mae,
            "global_minus_family_mae": (
                global_mae - family_mae
            ),
        })

print("=== STAGE21A FEATURE SIGN AUDIT ===")
print("STAGE19_MODEL_FEATURES", model_features)
print("FADE_FEATURE", fcol)
print("FADE_IN_MODEL", fcol in model_features)
print(
    "NORMALIZED_Q100_IN_MODEL",
    "cap_cycle_100_norm" in model_features
)
print("RATIO_REDUNDANCY", ratio_checks)
print("MAX_SIGN_FORMULA_ERROR", max_error)
print("FAMILY_RECOVERED", matched, "OF", nonambiguous)
print("AMBIGUOUS_RATIO_NEAR_ONE", int(ambiguous.sum()))

print("=== STAGE21A SOURCE FAMILY PRIOR ===")

for policy in (
    "all",
    "exclude_44_BatteryLife_flags",
):
    selected = [
        x for x in rows if x["policy"] == policy
    ]
    print(
        "POLICY", policy,
        "CELLS", sum(x["cells"] for x in selected),
        "DOMAINS", len(selected)
    )

    for metric in (
        "global_prior_mae",
        "family_prior_mae",
    ):
        macro = float(np.mean([
            x[metric] for x in selected
        ]))
        weighted = float(np.average(
            [x[metric] for x in selected],
            weights=[x["cells"] for x in selected]
        ))

        print(
            metric,
            "MACRO", round(macro, 9),
            "WEIGHTED", round(weighted, 9)
        )

print("=== DOMAIN RESULTS ===")

for row in rows:
    if row["policy"] == "all":
        print(
            row["dataset"],
            "N", row["cells"],
            "GLOBAL_MAE",
            round(row["global_prior_mae"], 9),
            "FAMILY_MAE",
            round(row["family_prior_mae"], 9)
        )

with out_csv.open(
    "x", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

with out_json.open("x", encoding="utf-8") as f:
    json.dump({
        "stage": "21A",
        "benchmark_sha256": expected_hash,
        "model_features": model_features,
        "fade_column": fcol,
        "max_formula_error": max_error,
        "ratio_checks": ratio_checks,
        "nonambiguous_cells": nonambiguous,
        "family_recovered": matched,
        "ratio_near_one_cells": int(ambiguous.sum()),
        "interpretation": (
            "Feature algebra and training-only "
            "source-family prior; no model trained. "
            "Known-family assumption; not unseen-family "
            "generalization or causal evidence."
        )
    }, f, indent=2)

print("STAGE21A_COMPLETE=YES")
print("REPORT", out_csv)
print("AUDIT", out_json)
