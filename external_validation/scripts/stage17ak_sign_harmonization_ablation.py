#!/usr/bin/env python3
"""Stage17AK: exploratory controlled single-feature sign harmonization ablation.

Train/evaluate original 567 ONLY. Source family holdouts + 12-dataset LODO.
No NCA/ISU outcome rescoring, no frozen model changes, no manuscript edits.
"""
import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path.cwd().resolve()
R = ROOT / "external_validation/reports"

TRAIN = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)

ORIGINAL = ROOT / (
    "external_validation/scripts/"
    "stage16a_isu_exploratory_transfer.py"
)

AD = R / "stage17ad_preinference_locked_recipe_and_fade.json"
AE = R / "stage17ae_nca_preoutcome_model_lock_report.json"

PRED = R / (
    "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"
)

OUT_FOLDS = R / (
    "stage17ak_harmonization_source_and_lodo_folds.csv"
)

OUT_PAIRS = R / (
    "stage17ak_harmonization_paired_method_folds.csv"
)

OUT_REPORT = R / (
    "stage17ak_harmonization_ablation_report.json"
)

OUTS = (OUT_FOLDS, OUT_PAIRS, OUT_REPORT)

PRED_SHA = (
    "cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757"
)

FEATURES = [
    "cap_ratio_10_to_1",
    "cap_ratio_50_to_1",
    "cap_ratio_100_to_1",
    "cap_mean_first_100_norm",
    "cap_std_first_100_norm",
    "cap_slope_first_100_norm",
    "cap_fade_fraction_1_to_100",
]

FADE = "cap_fade_fraction_1_to_100"

FAMILIES = {
    "BatteryML_main": 318,
    "BatteryLife_selected": 249,
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for piece in iter(
            lambda: stream.read(1048576),
            b"",
        ):
            h.update(piece)

    return h.hexdigest()


def validate_source():
    inputs = (TRAIN, ORIGINAL, AD, AE, PRED)

    for path in inputs:
        need(
            path.is_file(),
            f"Missing source: {path}",
        )

    for path in OUTS:
        need(
            not path.exists(),
            f"Refusing overwrite: {path}",
        )

    ad = json.loads(
        AD.read_text(encoding="utf-8")
    )

    ae = json.loads(
        AE.read_text(encoding="utf-8")
    )

    need(
        ad.get("training_n") == 567,
        "Stage17AD training count mismatch",
    )

    need(
        ae.get("training_cells") == 567
        and ae.get(
            "NCA_predictions_locked_before_terminal_outcome_review"
        ) is True,
        "Stage17AE provenance mismatch",
    )

    need(
        sha(PRED) == PRED_SHA,
        "Frozen NCA predictions changed",
    )

    for path in (TRAIN, ORIGINAL):
        relative = path.relative_to(ROOT).as_posix()

        need(
            ad.get("source_sha256", {}).get(relative)
            == sha(path),
            f"Stage17AD original input changed: {relative}",
        )

        need(
            ae.get("input_sha256", {}).get(relative)
            == sha(path),
            f"Stage17AE original input changed: {relative}",
        )

    definitions = [
        (
            "BatteryML_main",
            "Q100_divided_by_Q1_minus_one",
        ),
        (
            "BatteryLife_selected",
            "one_minus_Q100_divided_by_Q1",
        ),
    ]

    for family, formula in definitions:
        confirmations = [
            item
            for item in ad.get(
                "fade_equation_comparisons", []
            )
            if item.get("source_family") == family
            and item.get("candidate_equation") == formula
        ]

        need(
            len(confirmations) == 1
            and confirmations[0]["matching_at_1e_8"]
            == FAMILIES[family],
            f"Unverified fade convention: {family}",
        )

    return ad, ae


def load_recipe():
    spec = importlib.util.spec_from_file_location(
        "stage17ak_unmodified_stage16a",
        ORIGINAL,
    )

    need(
        spec is not None
        and spec.loader is not None,
        "Cannot load original Stage16A recipe",
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    need(
        list(module.FEATURES) == FEATURES
        and callable(module.make_pipeline),
        "Original feature order or model factory changed",
    )

    return module.make_pipeline


def score(y, prediction):
    return {
        "MAE": float(
            mean_absolute_error(y, prediction)
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(y, prediction)
            )
        ),
        "R2": (
            float(r2_score(y, prediction))
            if np.ptp(y) > 1e-12
            else None
        ),
        "bias_pred_minus_true": float(
            np.mean(prediction - y)
        ),
    }


def save_csv(path, records):
    need(
        len(records) > 0,
        f"Empty output: {path}",
    )

    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0]),
        )

        writer.writeheader()
        writer.writerows(records)


def main():
    validate_source()

    make_pipeline = load_recipe()

    columns = [
        "cell_id",
        "dataset_id",
        "source_family",
        "final_capacity_retention",
    ] + FEATURES

    data = pd.read_csv(
        TRAIN,
        dtype={
            "cell_id": str,
            "dataset_id": str,
            "source_family": str,
        },
        usecols=columns,
    )

    need(
        len(data) == 567
        and data["cell_id"].is_unique
        and data["cell_id"].notna().all(),
        "Invalid training roster",
    )

    need(
        data["dataset_id"].notna().all()
        and data["source_family"].notna().all(),
        "Missing source/domain identifiers",
    )

    need(
        data["source_family"].value_counts().to_dict()
        == FAMILIES,
        "Source family cell counts changed",
    )

    domains = sorted(
        data["dataset_id"].unique().tolist()
    )

    need(
        len(domains) == 12,
        f"Expected 12 datasets, found {domains}",
    )

    for column in FEATURES + [
        "final_capacity_retention"
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

        need(
            np.isfinite(
                data[column].to_numpy(dtype=float)
            ).all(),
            f"Nonfinite training column: {column}",
        )

    ratio = data[
        "cap_ratio_100_to_1"
    ].to_numpy(dtype=float)

    original_fade = data[
        FADE
    ].to_numpy(dtype=float)

    batteryml = data[
        "source_family"
    ].eq("BatteryML_main").to_numpy(dtype=bool)

    # Canonical convention:
    # positive loss = 1 - Q100/Q1.
    # Only BatteryML rows require sign reversal.
    positive_fade = original_fade.copy()
    positive_fade[batteryml] *= -1.0

    identity_delta = float(
        np.max(
            np.abs(
                positive_fade - (1.0 - ratio)
            )
        )
    )

    need(
        identity_delta < 1e-8,
        "Feature harmonization identity failed: "
        + str(identity_delta),
    )

    changed_rows = int(
        np.sum(
            np.abs(
                positive_fade - original_fade
            ) > 1e-8
        )
    )

    need(
        changed_rows > 0,
        "No feature values changed",
    )

    original_X = data[
        FEATURES
    ].copy()

    harmonized_X = original_X.copy()

    harmonized_X[
        FADE
    ] = positive_fade

    y = data[
        "final_capacity_retention"
    ].to_numpy(dtype=float)

    source = data[
        "source_family"
    ].to_numpy(dtype=str)

    dataset = data[
        "dataset_id"
    ].to_numpy(dtype=str)

    splits = []

    # Main diagnostic: hold out one full source family.
    for heldout in sorted(FAMILIES):
        splits.append((
            "SOURCE_FAMILY_HOLDOUT",
            heldout,
            source == heldout,
        ))

    # Supporting diagnostic: leave out each dataset.
    for heldout in domains:
        splits.append((
            "DATASET_LODO",
            heldout,
            dataset == heldout,
        ))

    need(
        len(splits) == 14,
        "Unexpected validation fold grid",
    )

    folds = []
    paired = []

    for split_type, heldout, testmask in splits:
        trainmask = ~testmask

        test_n = int(testmask.sum())
        train_n = int(trainmask.sum())

        need(
            test_n >= 2 and train_n >= 50,
            f"Invalid domain fold: {heldout}",
        )

        train_y = y[trainmask]
        test_y = y[testmask]

        prior = float(
            np.median(train_y)
        )

        baseline = score(
            test_y,
            np.full(test_n, prior),
        )

        estimators = [
            (
                "random_forest",
                RandomForestRegressor,
            ),
            (
                "extra_trees",
                ExtraTreesRegressor,
            ),
        ]

        for model_name, estimator_type in estimators:
            trials = {}

            conditions = [
                (
                    "ORIGINAL_MIXED_SIGN",
                    original_X,
                ),
                (
                    "HARMONIZED_POSITIVE_LOSS",
                    harmonized_X,
                ),
            ]

            for condition, matrix in conditions:
                estimator = estimator_type(
                    n_estimators=700,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )

                model = make_pipeline(
                    estimator
                )

                preprocessor = model.named_steps[
                    "preprocessor"
                ]

                need(
                    list(model.named_steps)
                    == ["preprocessor", "model"]
                    and len(
                        preprocessor.transformers
                    ) == 1
                    and list(
                        preprocessor.transformers[0][2]
                    ) == FEATURES,
                    "Original model structure changed",
                )

                # Fit only on the current training fold.
                model.fit(
                    matrix.loc[
                        trainmask, FEATURES
                    ],
                    train_y,
                )

                prediction = np.asarray(
                    model.predict(
                        matrix.loc[
                            testmask, FEATURES
                        ]
                    ),
                    dtype=float,
                )

                need(
                    len(prediction) == test_n
                    and np.isfinite(
                        prediction
                    ).all(),
                    "Invalid fold predictions",
                )

                stats = score(
                    test_y,
                    prediction,
                )

                trials[condition] = (
                    stats,
                    prediction,
                )

                folds.append({
                    "split_type": split_type,
                    "heldout": heldout,
                    "model": model_name,
                    "feature_condition": condition,
                    "train_n": train_n,
                    "test_n": test_n,
                    "baseline_trainmedian": prior,
                    "baseline_MAE": baseline["MAE"],
                    **stats,
                    "improvement_vs_baseline":
                        baseline["MAE"] - stats["MAE"],
                    "new_experiment_type":
                        "POSTOUTCOME_EXPLORATORY_ABLATION",
                })

            original_stats, original_pred = trials[
                "ORIGINAL_MIXED_SIGN"
            ]

            harmonized_stats, harmonized_pred = trials[
                "HARMONIZED_POSITIVE_LOSS"
            ]

            paired_gain = (
                np.abs(original_pred - test_y)
                - np.abs(harmonized_pred - test_y)
            )

            paired.append({
                "split_type": split_type,
                "heldout": heldout,
                "model": model_name,
                "test_n": test_n,
                "MAE_original_mixed":
                    original_stats["MAE"],
                "MAE_harmonized":
                    harmonized_stats["MAE"],
                "MAE_gain_original_minus_harmonized":
                    float(np.mean(paired_gain)),
                "RMSE_original_mixed":
                    original_stats["RMSE"],
                "RMSE_harmonized":
                    harmonized_stats["RMSE"],
                "prior_MAE_same_holdout":
                    baseline["MAE"],
                "cells_lower_absolute_error_after_harmonization":
                    int(np.sum(paired_gain > 1e-12)),
                "cells_higher_absolute_error_after_harmonization":
                    int(np.sum(paired_gain < -1e-12)),
                "positive_gain_means_harmonization_better":
                    True,
            })

        print(
            f"COMPLETED_FOLD={split_type}:{heldout} "
            f"HOLDOUT_CELLS={test_n}",
            flush=True,
        )

    primary_results = [
        row for row in paired
        if row["split_type"]
        == "SOURCE_FAMILY_HOLDOUT"
    ]

    lodo_results = [
        row for row in paired
        if row["split_type"] == "DATASET_LODO"
    ]

    lodo_macro = {}

    for model_name in (
        "random_forest",
        "extra_trees",
    ):
        selected = [
            row for row in lodo_results
            if row["model"] == model_name
        ]

        need(
            len(selected) == 12,
            "Incomplete dataset LODO analysis",
        )

        gains = [
            row[
                "MAE_gain_original_minus_harmonized"
            ]
            for row in selected
        ]

        lodo_macro[model_name] = {
            "datasets": len(selected),
            "unweighted_mean_MAE_original":
                float(np.mean([
                    r["MAE_original_mixed"]
                    for r in selected
                ])),
            "unweighted_mean_MAE_harmonized":
                float(np.mean([
                    r["MAE_harmonized"]
                    for r in selected
                ])),
            "unweighted_mean_MAE_gain":
                float(np.mean(gains)),
            "datasets_harmonization_better":
                sum(g > 1e-12 for g in gains),
            "datasets_harmonization_worse":
                sum(g < -1e-12 for g in gains),
            "two_cell_UL_PUR_included": True,
        }

    report = {
        "stage":
            "17AK_ORIGINAL567_FEATURE_SIGN_HARMONIZATION_CONTROLLED_ABLATION",

        "status":
            "POST_EXTERNAL_OUTCOME_EXPLORATORY_NOT_FROZEN_VALIDATION",

        "hypothesis":
            "Source-family-dependent feature sign can affect transfer",

        "intervention":
            "Flip only BatteryML_main cap_fade_fraction_1_to_100 "
            "to positive loss; leave all other inputs and target unchanged",

        "positive_loss_formula":
            "1 - cap_ratio_100_to_1",

        "feature_identity_max_abs_discrepancy":
            identity_delta,

        "sign_changed_training_rows":
            changed_rows,

        "original_training_n": 567,

        "source_family_n": FAMILIES,

        "dataset_ids": domains,

        "split_grid": {
            "source_family_holdouts": 2,
            "dataset_LODO_holdouts": 12,
            "estimators": 2,
            "feature_conditions": 2,
            "total_fits": 56,
        },

        "source_family_holdout_results":
            primary_results,

        "dataset_LODO_macro_descriptive":
            lodo_macro,

        "heldout_labels_used_only_for_final_scoring":
            True,

        "same_training_targets_and_split_membership":
            True,

        "not_a_causal_identification_of_external_failure":
            True,

        "not_comparable_as_prospective_external_improvement":
            True,

        "source_target_endpoint_equivalence_certified":
            False,

        "models_fit_only_to_original_training_fold":
            True,

        "original_training_table_modified":
            False,

        "prior_frozen_NCA_models_and_predictions_unchanged":
            True,

        "frozen_NCA_prediction_sha256":
            sha(PRED),

        "input_sha256": {
            path.relative_to(ROOT).as_posix():
                sha(path)
            for path in (
                TRAIN,
                ORIGINAL,
                AD,
                AE,
                PRED,
            )
        },

        "prism_updated": False,
    }

    R.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="stage17ak_",
        dir=R,
    ) as folder:
        tmp = Path(folder)

        save_csv(
            tmp / "folds.csv",
            folds,
        )

        save_csv(
            tmp / "pairs.csv",
            paired,
        )

        (
            tmp / "report.json"
        ).write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        need(
            not any(
                path.exists()
                for path in OUTS
            ),
            "Output appeared: refusing overwrite",
        )

        os.replace(
            tmp / "folds.csv",
            OUT_FOLDS,
        )

        os.replace(
            tmp / "pairs.csv",
            OUT_PAIRS,
        )

        os.replace(
            tmp / "report.json",
            OUT_REPORT,
        )

    print(
        "=== STAGE 17AK FEATURE SIGN HARMONIZATION ABLATION ==="
    )

    print(
        "FROZEN_NCA_PREDICTIONS_UNCHANGED=YES"
    )

    print(
        "ORIGINAL_TRAINING_ROWS=567 "
        "DATASETS=12 SOURCE_FAMILIES=2"
    )

    print(
        "SIGN_HARMONIZATION_IDENTITY_MAX_ABS_ERROR=",
        identity_delta,
    )

    print(
        "SIGN_CHANGED_ORIGINAL_ROWS=",
        changed_rows,
    )

    print(
        "SOURCE_FAMILY_HOLDOUT_PAIRED_COMPARISONS=",
        json.dumps(
            primary_results,
            ensure_ascii=False,
        ),
    )

    print(
        "DATASET_LODO_MACRO_SUMMARY=",
        json.dumps(
            lodo_macro,
            ensure_ascii=False,
        ),
    )

    print(
        "DATASET_LODO_FOLD_GAINS=",
        json.dumps([
            {
                "dataset": row["heldout"],
                "model": row["model"],
                "n": row["test_n"],
                "gain": row[
                    "MAE_gain_original_minus_harmonized"
                ],
            }
            for row in lodo_results
        ]),
    )

    print("FOLD_CSV=", OUT_FOLDS)
    print("PAIRED_COMPARISONS_CSV=", OUT_PAIRS)
    print("REPORT_JSON=", OUT_REPORT)

    print(
        "POSTOUTCOME_EXPLORATORY=YES "
        "NEW_EXTERNAL_VALIDATION=NO"
    )

    print(
        "ORIGINAL_MODELS_CHANGED=NO "
        "PRISM_UPDATED=NO"
    )

    print(
        "=== END STAGE 17AK ==="
    )


if __name__ == "__main__":
    main()
