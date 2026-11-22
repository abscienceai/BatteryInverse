#!/usr/bin/env python3
"""Stage20R: verified Stage19 recipe replay and fixed-prediction quality sensitivity."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

ROOT = Path.cwd()
REP = ROOT / "external_validation/reports"
SCRIPTS = ROOT / "external_validation/scripts"

DATA = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
FLAGS = REP / "stage20o_unified_endpoint_cells.csv"
STAGE19 = SCRIPTS / "stage19_featureblock_swap.py"
OLD_FOLDS = REP / "stage19_featureblock_swap_folds.csv"
OLD_REPORT = REP / "stage19_featureblock_swap_report.json"

OUT_PRED = REP / "stage20r_replayed_lodo_cell_predictions.csv"
OUT_SUMMARY = REP / "stage20r_fixed_prediction_quality_sensitivity.csv"
OUT_DOMAINS = REP / "stage20r_quality_domain_details.csv"
OUT_REPORT = REP / "stage20r_replay_quality_report.json"

EXPECTED_SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)

OUTPUTS = [OUT_PRED, OUT_SUMMARY, OUT_DOMAINS, OUT_REPORT]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def load_stage19():
    spec = importlib.util.spec_from_file_location(
        "stage19_original_replay", STAGE19
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    require(
        all(not p.exists() for p in OUTPUTS),
        "Stage20R output exists; refusing overwrite"
    )

    require(sha(DATA) == EXPECTED_SHA, "Benchmark hash mismatch")

    original = load_stage19()

    require(
        sha(original.RECIPE) == original.SHA_RECIPE,
        "Original Stage16A recipe hash mismatch"
    )
    require(
        original.DATA.resolve() == DATA.resolve(),
        "Stage19 dataset path mismatch"
    )

    stage19_report = json.loads(
        OLD_REPORT.read_text(encoding="utf-8")
    )

    require(
        stage19_report["input_data_sha256"] == EXPECTED_SHA,
        "Archived Stage19 input mismatch"
    )
    require(
        stage19_report["stage17ak_fold_sha256"] == sha(original.OLD),
        "Archived Stage17AK fold mismatch"
    )
    require(
        stage19_report["stage16a_recipe_sha256"] == sha(original.RECIPE),
        "Archived original recipe mismatch"
    )

    data = pd.read_csv(DATA)
    flags = pd.read_csv(FLAGS)
    historical = pd.read_csv(OLD_FOLDS)

    require(data.shape == (567, 38), "Benchmark shape mismatch")
    require(data.cell_id.is_unique, "Duplicate benchmark cell ID")
    require(data.dataset_id.nunique() == 12, "Dataset count mismatch")
    require(len(flags) == 249, "Stage20O population mismatch")
    require(len(historical) == 24, "Stage19 fold count mismatch")

    feature_names = list(original.F)
    require(len(feature_names) == 7, "Feature count changed")

    flag_lookup = {}
    flag_sets = {
        "no_post100": set(),
        "sparse_post100": set(),
        "mich_nonpositive": set(),
    }

    for _, row in flags.iterrows():
        key = (str(row.dataset), str(row.cell_id))
        require(key not in flag_lookup, "Duplicate Stage20O cell")

        post = int(row.positive_post100_records)
        nonpositive = int(row.mich_nonpositive_endpoint_flag)

        require(
            float(row.source_target_absdiff) < 1e-6,
            "Unverified Stage20O source target"
        )

        flag_lookup[key] = (post, nonpositive)

        if post == 0:
            flag_sets["no_post100"].add(key)
        elif post < 3:
            flag_sets["sparse_post100"].add(key)

        if nonpositive:
            flag_sets["mich_nonpositive"].add(key)

    no_future = flag_sets["no_post100"]
    sparse = flag_sets["sparse_post100"]
    nonpositive = flag_sets["mich_nonpositive"]

    two_flags = no_future | nonpositive
    three_flags = two_flags | sparse

    require(
        (
            len(no_future), len(sparse),
            len(nonpositive), len(two_flags),
            len(three_flags)
        ) == (29, 1, 14, 43, 44),
        "Stage20O flag counts mismatch"
    )

    benchmark_keys = set(
        zip(data.dataset_id.astype(str), data.cell_id.astype(str))
    )

    require(
        set(flag_lookup).issubset(benchmark_keys),
        "Stage20O key not in canonical benchmark"
    )

    require(
        int((data.source_family == "BatteryLife_selected").sum()) == 249,
        "BatteryLife population mismatch"
    )

    require(
        int((data.source_family == "BatteryML_main").sum()) == 318,
        "BatteryML population mismatch"
    )

    y = data.final_capacity_retention.to_numpy(float)
    datasets = data.dataset_id.astype(str).to_numpy()

    require(np.isfinite(y).all(), "Nonfinite benchmark target")

    records = []
    replay_checks = []

    print("=== STAGE20R ORIGINAL RECIPE REPLAY ===", flush=True)

    for domain in sorted(data.dataset_id.unique()):
        test = datasets == domain
        train = ~test

        Xtr = data.loc[train, feature_names]
        Xte = data.loc[test, feature_names]

        ytr = y[train]
        yte = y[test]

        prior = float(np.median(ytr))
        prior_mae = float(np.mean(np.abs(yte - prior)))

        for model_name, klass in [
            ("RF", RandomForestRegressor),
            ("ET", ExtraTreesRegressor),
        ]:
            # Identical Stage19 preprocessing and model recipe.
            model = original.pipeline(
                klass(
                    n_estimators=700,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                )
            )

            model.fit(Xtr, ytr)

            pred = np.asarray(model.predict(Xte), dtype=float)

            require(
                len(pred) == len(yte) and np.isfinite(pred).all(),
                "Invalid replay predictions"
            )

            mae = float(np.mean(np.abs(pred - yte)))

            matching = historical[
                (historical.domain == domain)
                & (historical.model == model_name)
            ]

            require(
                len(matching) == 1,
                "Missing historical Stage19 fold"
            )

            old = matching.iloc[0]

            mae_delta = abs(mae - float(old.original_mae))
            prior_delta = abs(prior_mae - float(old.prior_mae))

            replay_checks.append({
                "domain": domain,
                "model": model_name,
                "test_n": int(test.sum()),
                "replayed_mae": mae,
                "archived_stage19_mae": float(old.original_mae),
                "absolute_mae_delta": mae_delta,
                "absolute_prior_delta": prior_delta,
            })

            print(
                "REPLAY",
                domain,
                model_name,
                "n", int(test.sum()),
                "MAE", round(mae, 9),
                "DRIFT", round(mae_delta, 12),
                flush=True
            )

            # Do not publish sensitivity results if historical
            # fold-level predictions cannot be reproduced.
            require(
                mae_delta < 1e-8 and prior_delta < 1e-10,
                "Historical Stage19 replay gate failed: "
                + domain + " " + model_name
                + " MAE_delta=" + str(mae_delta)
            )

            held = data.loc[test, [
                "dataset_id", "cell_id",
                "source_family",
                "final_capacity_retention",
            ]]

            for (_, cell), prediction in zip(
                held.iterrows(), pred
            ):
                dataset = str(cell.dataset_id)
                cell_id = str(cell.cell_id)
                key = (dataset, cell_id)

                family = str(cell.source_family)
                target = float(cell.final_capacity_retention)

                if family == "BatteryLife_selected":
                    require(
                        key in flag_lookup,
                        "Missing BatteryLife quality audit"
                    )
                    quality_status = "AUDITED_BATTERYLIFE"
                else:
                    require(
                        key not in flag_lookup,
                        "Unexpected BatteryML quality flag"
                    )
                    quality_status = "NOT_AUDITED_BATTERYML"

                records.append({
                    "dataset_id": dataset,
                    "cell_id": cell_id,
                    "source_family": family,
                    "quality_audit_status": quality_status,
                    "model": model_name,
                    "heldout_dataset": domain,
                    "y_true": target,
                    "y_pred": float(prediction),
                    "train_median_prior": prior,
                    "model_abs_error": abs(prediction - target),
                    "prior_abs_error": abs(prior - target),
                    "flag_no_post100": int(key in no_future),
                    "flag_sparse_post100": int(key in sparse),
                    "flag_mich_nonpositive": int(key in nonpositive),
                    "flag_any_three": int(key in three_flags),
                })

        print("COMPLETED_DOMAIN", domain, flush=True)

    require(len(records) == 1134, "Expected 567 x 2 predictions")

    pred_df = pd.DataFrame(records)

    require(
        not pred_df.duplicated(
            ["dataset_id", "cell_id", "model"]
        ).any(),
        "Duplicate LODO cell-model predictions"
    )

    require(
        set(pred_df.model) == {"RF", "ET"},
        "Unexpected model labels"
    )

    require(
        len(replay_checks) == 24,
        "Incomplete historical fold replay"
    )

    # All filtering below changes TEST SCORING ONLY.
    # No model is refitted or recalibrated after filtering.
    policies = [
        ("all_567", False, set()),
        ("all_exclude_no_post100_538", False, no_future),
        ("all_exclude_mich_nonpositive_553", False, nonpositive),
        ("all_exclude_two_flags_524", False, two_flags),
        ("all_exclude_three_flags_523", False, three_flags),
        ("batterylife_249", True, set()),
        ("batterylife_exclude_no_post100_220", True, no_future),
        ("batterylife_exclude_mich_nonpositive_235", True, nonpositive),
        ("batterylife_exclude_two_flags_206", True, two_flags),
        ("batterylife_exclude_three_flags_205", True, three_flags),
    ]

    expected_sizes = [
        567, 538, 553, 524, 523,
        249, 220, 235, 206, 205,
    ]

    summaries = []
    domain_details = []

    for (policy, battery_life_only, excluded), expected_n in zip(
        policies, expected_sizes
    ):
        for model_name in ("RF", "ET"):
            subset = pred_df[pred_df.model == model_name].copy()

            if battery_life_only:
                subset = subset[
                    subset.source_family == "BatteryLife_selected"
                ]

            keys = list(zip(
                subset.dataset_id.astype(str),
                subset.cell_id.astype(str)
            ))

            keep = np.array([
                key not in excluded for key in keys
            ], dtype=bool)

            subset = subset.loc[keep].copy()

            require(
                len(subset) == expected_n,
                "Unexpected filtered count: " + policy
            )

            grouped = subset.groupby(
                "dataset_id", sort=True
            ).agg(
                cells=("cell_id", "size"),
                model_mae=("model_abs_error", "mean"),
                prior_mae=("prior_abs_error", "mean"),
            )

            for domain, row in grouped.iterrows():
                domain_details.append({
                    "policy": policy,
                    "model": model_name,
                    "domain": domain,
                    "cells": int(row.cells),
                    "model_mae": float(row.model_mae),
                    "prior_mae": float(row.prior_mae),
                    "prior_minus_model": float(
                        row.prior_mae - row.model_mae
                    ),
                })

            macro_model = float(grouped.model_mae.mean())
            macro_prior = float(grouped.prior_mae.mean())
            weighted_model = float(subset.model_abs_error.mean())
            weighted_prior = float(subset.prior_abs_error.mean())

            summaries.append({
                "policy": policy,
                "model": model_name,
                "cells": len(subset),
                "domains": len(grouped),
                "smallest_domain_n": int(grouped.cells.min()),
                "equal_domain_model_mae": macro_model,
                "equal_domain_prior_mae": macro_prior,
                "equal_domain_prior_minus_model": (
                    macro_prior - macro_model
                ),
                "cell_weighted_model_mae": weighted_model,
                "cell_weighted_prior_mae": weighted_prior,
                "cell_weighted_prior_minus_model": (
                    weighted_prior - weighted_model
                ),
                "training_changed_after_filter": False,
                "analysis_type": "REPLAY_FIXED_PREDICTION_RESCORE",
            })

    summary_df = pd.DataFrame(summaries)
    domain_df = pd.DataFrame(domain_details)

    # Confirm the replayed full benchmark also reproduces
    # the original Stage19 aggregate summaries.
    old_summary = pd.read_csv(
        REP / "stage19_featureblock_swap_summary.csv"
    )

    for model_name in ("RF", "ET"):
        previous = old_summary[
            (old_summary.model == model_name)
            & (old_summary.cohort == "all12")
        ]

        current = summary_df[
            (summary_df.policy == "all_567")
            & (summary_df.model == model_name)
        ].iloc[0]

        for weighting, column in [
            ("equal_domain", "equal_domain_model_mae"),
            ("cell_weighted", "cell_weighted_model_mae"),
        ]:
            matching = previous[
                previous.weighting == weighting
            ]

            require(len(matching) == 1, "Missing Stage19 aggregate")

            require(
                abs(
                    float(matching.iloc[0].orig_mae)
                    - float(current[column])
                ) < 1e-8,
                "Stage19 aggregate replay mismatch"
            )

    report = {
        "stage": "20R",
        "status": "FOLD_AND_AGGREGATE_REPLAY_VERIFIED",
        "benchmark_sha256": sha(DATA),
        "stage20o_flags_sha256": sha(FLAGS),
        "stage19_script_sha256": sha(STAGE19),
        "stage19_folds_sha256": sha(OLD_FOLDS),
        "stage16a_recipe_sha256": sha(original.RECIPE),
        "sklearn_version": sklearn.__version__,
        "model_recipe": (
            "Stage19 original seven-feature 12-dataset LODO RF/ET "
            "pipeline, 700 trees, min_samples_leaf=2, "
            "random_state=42, n_jobs=1"
        ),
        "historical_fold_checks": replay_checks,
        "max_model_fold_mae_replay_delta": max(
            x["absolute_mae_delta"] for x in replay_checks
        ),
        "max_prior_fold_mae_replay_delta": max(
            x["absolute_prior_delta"] for x in replay_checks
        ),
        "cell_model_predictions": len(records),
        "quality_flags": {
            "no_post100": len(no_future),
            "sparse_post100": len(sparse),
            "mich_nonpositive": len(nonpositive),
            "union_two_flags": len(two_flags),
            "union_three_flags": len(three_flags),
            "batterylife_without_three_flags": 205,
            "batteryml_not_source_audited": 318,
        },
        "limitations": [
            "Models were refitted from the historical recipe because "
            "archived per-cell predictions were unavailable.",
            "All historical Stage19 fold and aggregate MAEs were "
            "verified before filtered scoring.",
            "Training folds retain original flagged cells.",
            "Filtering changes evaluation membership only, not "
            "the fitted models or train-median priors.",
            "BatteryML_main cells were not source-quality-audited.",
            "Filtered cells are not certified physically comparable.",
            "CALB may contain only one evaluated cell after filtering.",
            "All differences are descriptive; no causal or "
            "prospective claims are made.",
        ],
    }

    # Save only after all replay gates have passed.
    pred_df.to_csv(OUT_PRED, index=False, mode="x")
    summary_df.to_csv(OUT_SUMMARY, index=False, mode="x")
    domain_df.to_csv(OUT_DOMAINS, index=False, mode="x")

    with OUT_REPORT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("=== STAGE20R VERIFIED RESULTS ===")
    print("REPLAYED_CELL_MODEL_PREDICTIONS", len(pred_df))
    print(
        "MAX_FOLD_REPLAY_DELTA",
        report["max_model_fold_mae_replay_delta"]
    )
    print(
        "MAX_PRIOR_REPLAY_DELTA",
        report["max_prior_fold_mae_replay_delta"]
    )

    for _, row in summary_df.iterrows():
        print(
            "POLICY", row.policy,
            "MODEL", row.model,
            "N", row.cells,
            "DOMAINS", row.domains,
            "MACRO_GAIN",
            round(row.equal_domain_prior_minus_model, 9),
            "WEIGHTED_GAIN",
            round(row.cell_weighted_prior_minus_model, 9),
            flush=True
        )

    print("STAGE20R_REPLAY_VERIFIED=YES")
    print("REPORT", OUT_REPORT)


if __name__ == "__main__":
    main()
