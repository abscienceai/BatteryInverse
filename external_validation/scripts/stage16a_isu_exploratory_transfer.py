#!/usr/bin/env python3
"""Stage 16A: ISU exploratory transfer stress test.
Not finalized external validation. No external training.
"""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
P = ROOT / "external_validation/processed"
R = ROOT / "external_validation/reports"

TRAIN = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)
MODEL_SOURCE = ROOT / (
    "scripts/expanded_model_family_robustness.py"
)
FEATURES_FILE = P / (
    "stage15c6_isu_242_capacity15_provisional.csv"
)
TARGET_FILE = P / (
    "stage15c6_isu_c5_c2_terminal_rpt_sensitivity.csv"
)
GATE_FILE = P / (
    "stage15c7_isu_paired_protocol_diagnostics.csv"
)

PRED_OUT = P / "stage16a_isu_exploratory_predictions.csv"
SCORES_OUT = R / "stage16a_isu_exploratory_scores.csv"
REPORT_OUT = R / "stage16a_isu_exploratory_report.json"

EXPECTED_TRAIN_SHA = (
    "8bc5296425fbb0735faceb56fe1cf52"
    "c6d0292bec2e361be8870fc1626aa5c7c"
)
EXPECTED_MODEL_SHA = (
    "1929b61d2be3609aa121bc3bc4f8774"
    "d86bb3cb9184d300b929a80034198e446"
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

C5 = "terminal_c5_to_initial_ratio_CANDIDATE"
C2 = "terminal_c2_to_initial_ratio_CANDIDATE"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def unique(frame, label):
    require(
        "cell_id" in frame.columns,
        label + ": missing cell_id"
    )
    ids = frame["cell_id"].astype(str).str.strip()
    require(
        ids.ne("").all() and ids.is_unique,
        label + ": duplicate/empty cell_id"
    )
    result = frame.copy()
    result["cell_id"] = ids
    return result.set_index(
        "cell_id", verify_integrity=True
    )


def yes(value):
    return str(value).strip().lower() in (
        "true", "yes", "1"
    )


def score(y, p):
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(
            math.sqrt(mean_squared_error(y, p))
        ),
        "r2": float(r2_score(y, p)),
        "signed_bias_pred_minus_true":
            float(np.mean(p - y)),
        "pred_median": float(np.median(p)),
    }


def make_pipeline(estimator):
    preprocessing = ColumnTransformer([
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                ("scaler", StandardScaler()),
            ]),
            FEATURES,
        )
    ], remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessing),
        ("model", estimator),
    ])


def main():
    inputs = (
        TRAIN,
        MODEL_SOURCE,
        FEATURES_FILE,
        TARGET_FILE,
        GATE_FILE,
    )

    for path in inputs:
        require(
            path.is_file(),
            f"Missing input: {path}"
        )

    require(
        sha(TRAIN) == EXPECTED_TRAIN_SHA,
        "Training table SHA changed. STOP."
    )

    require(
        sha(MODEL_SOURCE) == EXPECTED_MODEL_SHA,
        "Original model script SHA changed. STOP."
    )

    for path in (
        PRED_OUT, SCORES_OUT, REPORT_OUT
    ):
        require(
            not path.exists(),
            f"Output exists. No overwrite: {path}"
        )

    tree = ast.parse(
        MODEL_SOURCE.read_text(encoding="utf-8")
    )

    feature_sets = None

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id == "FEATURE_SETS"
                for t in node.targets
            )
        ):
            feature_sets = ast.literal_eval(
                node.value
            )
            break

    require(
        feature_sets is not None
        and feature_sets.get(
            "normalized_capacity_only"
        ) == FEATURES,
        "Original normalized feature contract changed."
    )

    train = pd.read_csv(TRAIN)

    external = unique(
        pd.read_csv(FEATURES_FILE),
        "ISU features"
    )

    targets = unique(
        pd.read_csv(TARGET_FILE),
        "ISU targets"
    )

    gate = unique(
        pd.read_csv(GATE_FILE),
        "ISU chronology"
    )

    require(
        len(train) == 567
        and train["dataset_id"].nunique() == 12,
        "Unexpected training cohort"
    )

    require(
        len(external) == 242
        and set(external.index)
        == set(targets.index)
        == set(gate.index),
        "242-cell ISU identity mismatch"
    )

    flags = [
        "first_post100_rpt_chronological",
        "baseline_rpt_before_input",
        "terminal_rpt_after_final_block",
    ]

    require(
        all(
            yes(gate.loc[cell, flag])
            for cell in external.index
            for flag in flags
        ),
        "ISU chronology gate failed"
    )

    require(
        all(
            col in train.columns
            and col in external.columns
            for col in FEATURES
        ),
        "Feature columns missing"
    )

    Xtrain = (
        train[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    Xtest = (
        external[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    require(
        Xtrain.notna().any().all(),
        "Entire training feature missing"
    )

    require(
        Xtest.notna().all().all(),
        "Invalid ISU normalized features"
    )

    ytrain = pd.to_numeric(
        train["final_capacity_retention"],
        errors="coerce"
    ).to_numpy(float)

    require(
        np.isfinite(ytrain).all()
        and (ytrain > 0).all(),
        "Invalid training targets"
    )

    labels = (
        targets[[C5, C2]]
        .apply(pd.to_numeric, errors="coerce")
        .loc[Xtest.index]
    )

    y_external = labels.to_numpy(float)

    require(
        np.isfinite(y_external).all()
        and (y_external > 0).all(),
        "Invalid external target candidates"
    )

    # Never include C5 or C2 in model input.
    predictions = pd.DataFrame(
        index=Xtest.index
    )

    predictions[
        "c5_reference_candidate_NOT_FINAL"
    ] = labels[C5]

    predictions[
        "c2_reference_candidate_NOT_FINAL"
    ] = labels[C2]

    prior = float(np.median(ytrain))

    predictions[
        "training_median_prior_prediction"
    ] = prior

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, estimator in models.items():
        model = make_pipeline(estimator)

        # Fit exclusively on the original 567 cells.
        model.fit(Xtrain, ytrain)

        p = model.predict(Xtest)

        require(
            np.isfinite(p).all(),
            f"Invalid predictions: {name}"
        )

        predictions[
            name + "_prediction"
        ] = p

    scores = []

    for target_name, label_col in (
        ("C5_PRIMARY_CANDIDATE", C5),
        ("C2_SENSITIVITY", C2),
    ):
        observed = labels[
            label_col
        ].to_numpy(float)

        for name in (
            "training_median_prior",
            "random_forest",
            "extra_trees",
        ):
            pred_col = (
                "training_median_prior_prediction"
                if name == "training_median_prior"
                else name + "_prediction"
            )

            predicted = predictions[
                pred_col
            ].to_numpy(float)

            scores.append({
                "target": target_name,
                "model": name,
                **score(observed, predicted),
            })

    # Training-vs-external feature distribution audit.
    ood = {}
    outside_any = np.zeros(
        len(Xtest), dtype=bool
    )

    for col in FEATURES:
        x = Xtrain[col].dropna().to_numpy(float)
        v = Xtest[col].to_numpy(float)

        lower, upper = np.quantile(
            x, [0.05, 0.95]
        )

        outside_90 = (
            (v < lower) | (v > upper)
        )

        outside_full = (
            (v < x.min()) | (v > x.max())
        )

        outside_any |= outside_full

        ood[col] = {
            "training_median":
                float(np.median(x)),
            "ISU_median":
                float(np.median(v)),
            "ISU_outside_training_5th_95th":
                int(outside_90.sum()),
            "ISU_outside_training_min_max":
                int(outside_full.sum()),
        }

    report = {
        "experiment_type":
            "EXPLORATORY_TRANSFER_STRESS_TEST",
        "not_final_external_validation": True,
        "training_cells": 567,
        "ISU_cells": 242,
        "feature_family":
            "normalized_capacity_only",
        "features": FEATURES,
        "training_sha256": sha(TRAIN),
        "original_model_source_sha256":
            sha(MODEL_SOURCE),
        "ISU_features_sha256":
            sha(FEATURES_FILE),
        "ISU_candidate_targets_sha256":
            sha(TARGET_FILE),
        "chronology_gate_sha256":
            sha(GATE_FILE),
        "scikit_learn_version":
            sklearn.__version__,
        "training_only_on_567_cells": True,
        "external_training_or_tuning": False,
        "newly_retrained_models_not_saved_checkpoints":
            True,
        "RPT_protocol_approved": False,
        "capacity_unit_and_DoD_compatibility_verified":
            False,
        "endpoint_horizon_equivalence_verified":
            False,
        "canonical_ERM_IRM_DANN_not_used":
            True,
        "training_target": {
            "median": prior,
            "min": float(ytrain.min()),
            "max": float(ytrain.max()),
        },
        "ISU_candidate_target_medians": {
            "C5": float(np.median(labels[C5])),
            "C2": float(np.median(labels[C2])),
        },
        "ISU_outside_training_full_feature_range_any":
            int(outside_any.sum()),
        "feature_distribution_diagnostics": ood,
        "scores": scores,
        "interpretation":
            "Exploratory external transfer under "
            "unconfirmed endpoint equivalence. "
            "Do not report as finalized validation.",
    }

    P.mkdir(parents=True, exist_ok=True)
    R.mkdir(parents=True, exist_ok=True)

    predictions.reset_index().to_csv(
        PRED_OUT,
        index=False,
        mode="x",
    )

    with SCORES_OUT.open(
        "x", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(scores[0])
        )
        writer.writeheader()
        writer.writerows(scores)

    with REPORT_OUT.open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(
            report, f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "=== STAGE 16A EXPLORATORY "
        "ISU TRANSFER ==="
    )

    print(
        "TRAIN_CELLS=567 ISU_CELLS=242 FEATURES=7"
    )

    print("TRAIN_MEDIAN=", prior)

    print(
        "TARGET_CANDIDATE_MEDIANS=",
        json.dumps(
            report["ISU_candidate_target_medians"]
        )
    )

    print(
        "ISU_OUTSIDE_TRAIN_FEATURE_RANGE_ANY=",
        int(outside_any.sum())
    )

    print(
        "OOD_BY_FEATURE=",
        json.dumps(ood)
    )

    print(
        "METRICS=",
        json.dumps(scores)
    )

    print(
        "PREDICTIONS_CSV=",
        PRED_OUT
    )

    print(
        "SCORES_CSV=",
        SCORES_OUT
    )

    print(
        "REPORT_JSON=",
        REPORT_OUT
    )

    print("ISU_USED_FOR_MODEL_TRAINING=NO")
    print("FINAL_EXTERNAL_VALIDATION=NO")
    print("=== END STAGE 16A ===")


if __name__ == "__main__":
    main()
