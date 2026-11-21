#!/usr/bin/env python3
"""Stage17AL: bounded exploratory six-feature redundancy control.

Only the original 567 training cells are used.
Existing Stage17AK seven-feature results are reused.
No ISU/NCA outcome analysis, frozen-model changes, or manuscript edits.
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.cwd().resolve()
R = ROOT / "external_validation/reports"

TRAIN = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)

RECIPE = ROOT / (
    "external_validation/scripts/"
    "stage16a_isu_exploratory_transfer.py"
)

AE = R / "stage17ae_nca_preoutcome_model_lock_report.json"
AK = R / "stage17ak_harmonization_ablation_report.json"
AK_FOLDS = R / "stage17ak_harmonization_source_and_lodo_folds.csv"

PRED = R / (
    "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"
)

OUT_FOLDS = R / (
    "stage17al_sixfeature_deletion_fold_comparison.csv"
)

OUT_REPORT = R / (
    "stage17al_sixfeature_redundancy_control_report.json"
)

LOCK_SHA = (
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

SIX = [
    f for f in FEATURES
    if f != "cap_fade_fraction_1_to_100"
]

MODELS = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
}

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
        for block in iter(
            lambda: stream.read(1048576),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_recipe():
    spec = importlib.util.spec_from_file_location(
        "stage17al_original_recipe",
        RECIPE,
    )

    need(
        spec is not None
        and spec.loader is not None,
        "Original Stage16A recipe unavailable",
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    need(
        list(module.FEATURES) == FEATURES
        and callable(module.make_pipeline),
        "Original recipe changed",
    )

    return module.make_pipeline


def score(y, p):
    return {
        "MAE": float(
            mean_absolute_error(y, p)
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(y, p)
            )
        ),
        "R2": (
            float(r2_score(y, p))
            if np.ptp(y) > 1e-12 else None
        ),
        "bias": float(np.mean(p - y)),
    }


def load_stage17ak_scores():
    with AK_FOLDS.open(
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        reader = csv.DictReader(stream)

        required = {
            "split_type",
            "heldout",
            "model",
            "feature_condition",
            "train_n",
            "test_n",
            "MAE",
            "RMSE",
            "baseline_MAE",
        }

        need(
            required <= set(reader.fieldnames or []),
            "Stage17AK score header mismatch",
        )

        rows = list(reader)

    need(
        len(rows) == 56,
        "Expected 56 existing Stage17AK scores",
    )

    keyed = {}

    for row in rows:
        key = (
            row["split_type"],
            row["heldout"],
            row["model"],
            row["feature_condition"],
        )

        need(
            key not in keyed,
            "Duplicate Stage17AK comparison",
        )

        keyed[key] = row

    return keyed


def verify_inputs():
    sources = (
        TRAIN,
        RECIPE,
        AE,
        AK,
        AK_FOLDS,
        PRED,
    )

    for path in sources:
        need(
            path.is_file(),
            f"Missing source: {path}",
        )

    for path in (OUT_FOLDS, OUT_REPORT):
        need(
            not path.exists(),
            f"Refusing overwrite: {path}",
        )

    ae = json.loads(
        AE.read_text(encoding="utf-8")
    )

    ak = json.loads(
        AK.read_text(encoding="utf-8")
    )

    need(
        ae.get("training_cells") == 567
        and ak.get("original_training_n") == 567,
        "Original 567-cell cohort changed",
    )

    need(
        ak.get("split_grid", {}).get("total_fits")
        == 56,
        "Stage17AK experiment grid changed",
    )

    need(
        ak.get("prism_updated") is False
        and ak.get("original_training_table_modified")
        is False
        and ak.get(
            "prior_frozen_NCA_models_and_predictions_unchanged"
        ) is True,
        "Stage17AK provenance mismatch",
    )

    need(
        sha(PRED) == LOCK_SHA
        == ak.get("frozen_NCA_prediction_sha256"),
        "Frozen NCA predictions changed",
    )

    for path in (
        TRAIN, RECIPE, AE, PRED
    ):
        rel = path.relative_to(ROOT).as_posix()

        need(
            ak.get("input_sha256", {}).get(rel)
            == sha(path),
            f"Stage17AK input changed: {rel}",
        )

    for path in (TRAIN, RECIPE):
        rel = path.relative_to(ROOT).as_posix()

        need(
            ae.get("input_sha256", {}).get(rel)
            == sha(path),
            f"Stage17AE original input changed: {rel}",
        )

    return ak


def main():
    verify_inputs()

    previous = load_stage17ak_scores()
    make_pipeline = load_recipe()

    columns = [
        "cell_id",
        "source_family",
        "dataset_id",
        "final_capacity_retention",
    ] + FEATURES

    data = pd.read_csv(
        TRAIN,
        usecols=columns,
        dtype={
            "cell_id": str,
            "source_family": str,
            "dataset_id": str,
        },
    )

    need(
        len(data) == 567
        and data["cell_id"].notna().all()
        and data["cell_id"].is_unique,
        "Training roster mismatch",
    )

    need(
        data["source_family"].value_counts().to_dict()
        == FAMILIES,
        "Source family counts changed",
    )

    need(
        data["dataset_id"].notna().all()
        and data["dataset_id"].nunique() == 12,
        "Dataset roster changed",
    )

    for col in FEATURES + [
        "final_capacity_retention"
    ]:
        data[col] = pd.to_numeric(
            data[col],
            errors="raise",
        )

        need(
            np.isfinite(
                data[col].to_numpy(dtype=float)
            ).all(),
            f"Invalid training column: {col}",
        )

    original_fade = data[
        "cap_fade_fraction_1_to_100"
    ].to_numpy(dtype=float)

    ratio = data[
        "cap_ratio_100_to_1"
    ].to_numpy(dtype=float)

    is_ml = data[
        "source_family"
    ].eq("BatteryML_main").to_numpy(dtype=bool)

    positive_fade = original_fade.copy()
    positive_fade[is_ml] *= -1.0

    duplicate_gap = float(
        np.max(
            np.abs(
                positive_fade - (1 - ratio)
            )
        )
    )

    need(
        duplicate_gap < 1e-8,
        "Harmonized fade is not redundant with Q100/Q1",
    )

    X = data.loc[:, SIX]

    y = data[
        "final_capacity_retention"
    ].to_numpy(dtype=float)

    families = data[
        "source_family"
    ].to_numpy(dtype=str)

    datasets = data[
        "dataset_id"
    ].to_numpy(dtype=str)

    splits = [
        (
            "SOURCE_FAMILY_HOLDOUT",
            family,
            families == family,
        )
        for family in sorted(FAMILIES)
    ]

    splits += [
        (
            "DATASET_LODO",
            domain,
            datasets == domain,
        )
        for domain in sorted(
            data["dataset_id"].unique()
        )
    ]

    need(
        len(splits) == 14,
        "Unexpected holdout grid",
    )

    comparisons = []

    for split_type, heldout, testmask in splits:
        trainmask = ~testmask

        test_n = int(testmask.sum())
        train_n = int(trainmask.sum())

        need(
            test_n >= 2 and train_n >= 50,
            "Invalid holdout size",
        )

        ytrain = y[trainmask]
        ytest = y[testmask]

        prior = float(np.median(ytrain))

        prior_mae = float(
            np.mean(np.abs(ytest - prior))
        )

        for model_name, cls in MODELS.items():
            original = previous.get((
                split_type,
                heldout,
                model_name,
                "ORIGINAL_MIXED_SIGN",
            ))

            harmonized = previous.get((
                split_type,
                heldout,
                model_name,
                "HARMONIZED_POSITIVE_LOSS",
            ))

            need(
                original is not None
                and harmonized is not None,
                f"Stage17AK score missing: {heldout}",
            )

            for reference in (original, harmonized):
                need(
                    int(reference["train_n"]) == train_n
                    and int(reference["test_n"]) == test_n,
                    "Holdout membership/count mismatch",
                )

                need(
                    abs(
                        float(reference["baseline_MAE"])
                        - prior_mae
                    ) < 1e-10,
                    "Changed target or train-median baseline",
                )

            estimator = cls(
                n_estimators=700,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            model = make_pipeline(estimator)

            need(
                list(model.named_steps)
                == ["preprocessor", "model"],
                "Unexpected pipeline structure",
            )

            transformer = model.named_steps[
                "preprocessor"
            ]

            need(
                len(transformer.transformers) == 1,
                "Unexpected preprocessing branches",
            )

            name, num_pipe, original_cols = (
                transformer.transformers[0]
            )

            need(
                name == "num"
                and list(original_cols) == FEATURES,
                "Original feature recipe changed",
            )

            need(
                list(num_pipe.named_steps)
                == ["imputer", "scaler"]
                and num_pipe.named_steps[
                    "imputer"
                ].strategy == "median"
                and type(
                    num_pipe.named_steps["scaler"]
                ).__name__ == "StandardScaler",
                "Original numerical preprocessing changed",
            )

            # Modify only this new, temporary model instance.
            # The original Stage16A script is untouched.
            transformer.transformers = [
                (name, num_pipe, SIX)
            ]

            model.fit(
                X.loc[trainmask, SIX],
                ytrain,
            )

            predicted = np.asarray(
                model.predict(
                    X.loc[testmask, SIX]
                ),
                dtype=float,
            )

            need(
                len(predicted) == test_n
                and np.isfinite(predicted).all(),
                "Invalid six-feature predictions",
            )

            result = score(ytest, predicted)

            original_mae = float(
                original["MAE"]
            )

            harmonized_mae = float(
                harmonized["MAE"]
            )

            comparisons.append({
                "split_type": split_type,
                "heldout": heldout,
                "model": model_name,
                "train_n": train_n,
                "test_n": test_n,
                "prior_MAE": prior_mae,
                "original_mixed_7_MAE_from17AK":
                    original_mae,
                "harmonized_7_MAE_from17AK":
                    harmonized_mae,
                "no_fade_6_MAE": result["MAE"],
                "no_fade_6_RMSE": result["RMSE"],
                "no_fade_6_R2": result["R2"],
                "gain_orig7_minus_no_fade6":
                    original_mae - result["MAE"],
                "gain_harmonized7_minus_no_fade6":
                    harmonized_mae - result["MAE"],
                "gain_prior_minus_no_fade6":
                    prior_mae - result["MAE"],
                "analysis_status":
                    "POSTOUTCOME_EXPLORATORY_REPRESENTATION_CONTROL",
            })

        print(
            f"COMPLETED_SIX_FEATURE_FOLD="
            f"{split_type}:{heldout} "
            f"N={test_n}",
            flush=True,
        )

    need(
        len(comparisons) == 28,
        "Expected exactly 28 six-feature fits",
    )

    summary = {}

    for split_type in (
        "SOURCE_FAMILY_HOLDOUT",
        "DATASET_LODO",
    ):
        for model_name in MODELS:
            items = [
                row for row in comparisons
                if row["split_type"] == split_type
                and row["model"] == model_name
            ]

            subsets = [
                ("ALL", items),
                (
                    "EXCLUDE_UL_PUR_2CELL",
                    [
                        row for row in items
                        if row["heldout"] != "UL_PUR"
                    ],
                ),
            ]

            for label, subset in subsets:
                if not subset:
                    continue

                summary[
                    f"{split_type}:{model_name}:{label}"
                ] = {
                    "folds": len(subset),
                    "MAE_macro_original7": float(
                        np.mean([
                            r["original_mixed_7_MAE_from17AK"]
                            for r in subset
                        ])
                    ),
                    "MAE_macro_harmonized7": float(
                        np.mean([
                            r["harmonized_7_MAE_from17AK"]
                            for r in subset
                        ])
                    ),
                    "MAE_macro_no_fade6": float(
                        np.mean([
                            r["no_fade_6_MAE"]
                            for r in subset
                        ])
                    ),
                    "MAE_macro_prior": float(
                        np.mean([
                            r["prior_MAE"]
                            for r in subset
                        ])
                    ),
                    "folds_6_better_than_orig7": sum(
                        r["gain_orig7_minus_no_fade6"] > 1e-12
                        for r in subset
                    ),
                    "folds_6_better_than_harmonized7": sum(
                        r[
                            "gain_harmonized7_minus_no_fade6"
                        ] > 1e-12
                        for r in subset
                    ),
                    "folds_6_better_than_prior": sum(
                        r["gain_prior_minus_no_fade6"] > 1e-12
                        for r in subset
                    ),
                }

    report = {
        "stage":
            "17AL_EXPLORATORY_SIXFEATURE_REDUNDANCY_CONTROL",

        "scientific_question":
            "Does dropping the signed fade feature change transfer?",

        "status":
            "POST_EXTERNAL_OUTCOME_EXPLORATORY",

        "six_features": SIX,

        "dropped_feature":
            "cap_fade_fraction_1_to_100",

        "harmonized7_duplicate_identity_max_abs_difference":
            duplicate_gap,

        "training_rows": 567,
        "folds": 14,
        "new_models_fit": 28,

        "existing_17AK_scores_reused_not_retrained":
            True,

        "summary": summary,

        "no_new_ISU_or_NCA_inference": True,

        "does_not_identify_causal_reason_for_external_failure":
            True,

        "frozen_NCA_prediction_sha256":
            sha(PRED),

        "original_training_table_modified":
            False,

        "prism_updated": False,

        "source_sha256": {
            path.relative_to(ROOT).as_posix():
                sha(path)
            for path in (
                TRAIN, RECIPE, AE,
                AK, AK_FOLDS, PRED
            )
        },
    }

    with tempfile.TemporaryDirectory(
        prefix="stage17al_",
        dir=R,
    ) as folder:
        tmp = Path(folder)

        with (
            tmp / "comparison.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(comparisons[0]),
            )
            writer.writeheader()
            writer.writerows(comparisons)

        (
            tmp / "report.json"
        ).write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        need(
            not any(
                path.exists()
                for path in (OUT_FOLDS, OUT_REPORT)
            ),
            "Output appeared; refusing overwrite",
        )

        os.replace(
            tmp / "comparison.csv",
            OUT_FOLDS,
        )

        os.replace(
            tmp / "report.json",
            OUT_REPORT,
        )

    print(
        "=== STAGE 17AL SIX-FEATURE REDUNDANCY CONTROL ==="
    )

    print(
        "NCA_PREDICTION_SHA256_UNCHANGED=YES"
    )

    print(
        "SIX_FEATURE_FITS_COMPLETED=28"
    )

    print(
        "HARMONIZED_SEVEN_FEATURE_REDUNDANCY_MAX_DIFFERENCE=",
        duplicate_gap,
    )

    print(
        "CONTROLLED_SOURCE_FAMILY_RESULTS=",
        json.dumps({
            key: val
            for key, val in summary.items()
            if key.startswith("SOURCE_FAMILY_HOLDOUT")
        }),
    )

    print(
        "CONTROLLED_LODO_RESULTS=",
        json.dumps({
            key: val
            for key, val in summary.items()
            if key.startswith("DATASET_LODO")
        }),
    )

    print("FOLD_CSV=", OUT_FOLDS)
    print("REPORT_JSON=", OUT_REPORT)

    print(
        "NEW_EXTERNAL_INFERENCE=NO "
        "PRISM_UPDATED=NO"
    )

    print(
        "=== END STAGE 17AL ==="
    )


if __name__ == "__main__":
    main()
