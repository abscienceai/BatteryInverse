#!/usr/bin/env python3
"""Stage18 matched-split source identification and harmonization controls."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score
)

root = Path(__file__).resolve().parents[2]
out = root / "external_validation/reports"

dataset = root / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
original_code = root / "scripts/expanded_dataset_separability_analysis.py"
old_metrics = root / "reports/benchmark_audits/expanded_dataset_separability_split_metrics.csv"

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

assert sha(dataset) == "8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c"

spec = importlib.util.spec_from_file_location("original", original_code)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

raw = pd.read_csv(dataset).replace([np.inf, -np.inf], np.nan)

assert raw.shape == (567, 38)
assert raw.source_family.value_counts().to_dict() == {
    "BatteryML_main": 318,
    "BatteryLife_selected": 249
}

counts = raw.dataset_id.value_counts()
df = raw.loc[
    raw.dataset_id.isin(counts[counts >= 5].index)
].copy().reset_index(drop=True)

assert len(df) == 565
assert df.dataset_id.nunique() == 11

fade = "cap_fade_fraction_1_to_100"

features = [
    "cap_ratio_10_to_1",
    "cap_ratio_50_to_1",
    "cap_ratio_100_to_1",
    "cap_mean_first_100_norm",
    "cap_std_first_100_norm",
    "cap_slope_first_100_norm",
    fade
]

harm = df.copy()
mask = harm.source_family.eq("BatteryML_main")

assert int(mask.sum()) == 318

harm.loc[mask, fade] = 1 - harm.loc[mask, "cap_ratio_100_to_1"]

splits = list(
    StratifiedShuffleSplit(
        n_splits=30,
        test_size=0.30,
        random_state=42
    ).split(df, df.dataset_id.astype(str))
)

rows = []

for condition in ("original", "harmonized", "no_fade"):
    source = harm if condition == "harmonized" else df

    cols = [
        f for f in features
        if condition != "no_fade" or f != fade
    ]

    for sid, (tr, te) in enumerate(splits, 1):
        train = source.iloc[tr]
        test = source.iloc[te]

        clf = RandomForestClassifier(
            n_estimators=700,
            random_state=42 + sid,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1
        )

        model = Pipeline([
            ("preprocess", original.make_preprocessor(cols, train)),
            ("model", clf)
        ])

        model.fit(train[cols], train.dataset_id.astype(str))

        y = test.dataset_id.astype(str)
        pred = model.predict(test[cols])

        rows.append({
            "feature_family": "normalized_capacity_only",
            "condition": condition,
            "split_id": sid,
            "n_total": 565,
            "n_train": len(tr),
            "n_test": len(te),
            "n_classes": 11,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro"))
        })

    print("CONDITION_FINISHED=" + condition, flush=True)

with old_metrics.open() as f:
    prior = list(csv.DictReader(f))

old = {
    int(x["split_id"]): float(x["balanced_accuracy"])
    for x in prior
    if x["feature_set"] == "normalized_capacity_only"
}

new = {
    x["split_id"]: x["balanced_accuracy"]
    for x in rows
    if x["condition"] == "original"
}

assert len(old) == len(new) == 30

maxdelta = max(abs(old[k] - new[k]) for k in old)

assert maxdelta < 1e-10, f"Baseline reproduction failed: {maxdelta}"

ref = np.array([new[k] for k in sorted(new)])

expected = {
    "original": 0.8087987957743867,
    "harmonized": 0.7559639950410354,
    "no_fade": 0.76445091557029
}

summary = []

for condition in expected:
    a = np.array([
        x["balanced_accuracy"]
        for x in rows
        if x["condition"] == condition
    ])

    diff = a - ref

    assert abs(a.mean() - expected[condition]) < 2e-8, (
        "Unexpected result", condition, a.mean()
    )

    summary.append({
        "feature_family": "normalized_capacity_only",
        "condition": condition,
        "n_splits": 30,
        "n_cells": 565,
        "n_datasets": 11,
        "mean_balanced_accuracy": float(a.mean()),
        "sd_balanced_accuracy": float(a.std(ddof=1)),
        "split_quantile_025": float(np.quantile(a, .025)),
        "split_quantile_975": float(np.quantile(a, .975)),
        "mean_paired_delta_vs_original": float(diff.mean()),
        "paired_delta_quantile_025": float(np.quantile(diff, .025)),
        "paired_delta_quantile_975": float(np.quantile(diff, .975))
    })

def save(name, data):
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

save("stage18_source_identity_matched_split_metrics.csv", rows)
save("stage18_source_identity_summary.csv", summary)

report = {
    "analysis": "Matched-split source identification",
    "source_dataset_sha256": sha(dataset),
    "original_classifier_code_sha256": sha(original_code),
    "n_original": 567,
    "n_scored": 565,
    "n_datasets": 11,
    "harmonized_source_rows": 318,
    "original_baseline_max_abs_difference": maxdelta,
    "results": summary,
    "protocol": {
        "split_count": 30,
        "test_size": 0.30,
        "split_seed": 42,
        "trees": 700,
        "model_seed": "42+split_id",
        "class_weight": "balanced_subsample"
    },
    "limitation": "Descriptive split percentiles, not prospective or causal inference"
}

(out / "stage18_source_identity_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n"
)

for s in summary:
    print(
        "VERIFIED",
        s["condition"],
        round(s["mean_balanced_accuracy"], 9)
    )

print("STAGE18_VERIFIED=YES")
