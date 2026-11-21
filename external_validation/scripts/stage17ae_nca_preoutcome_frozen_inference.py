#!/usr/bin/env python3
"""Stage17AE: reproduce original RF/ET and freeze NCA predictions.

Both NCA fade-sign conventions are sensitivity branches.
No NCA terminal outcomes are read.
Original training data and scripts remain unchanged.
"""
import csv
import hashlib
import importlib.util
import json
import os
import platform
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

ROOT = Path.cwd().resolve()
EV = ROOT / "external_validation"
R = EV / "reports"
P = EV / "processed"
M = EV / "models"

TRAIN = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)

ORIGINAL = EV / (
    "scripts/stage16a_isu_exploratory_transfer.py"
)

ISU = P / (
    "stage15c6_isu_242_capacity15_provisional.csv"
)

ISU_REPLAY = P / (
    "stage17s_stage16a_isolated_replay_predictions.csv"
)

NCA = R / (
    "stage17ac_nca_30cell_PROVISIONAL_features.csv"
)

AD = R / (
    "stage17ad_preinference_locked_recipe_and_fade.json"
)

AC = R / (
    "stage17ac_feature_lineage_and_provisional_matrix.json"
)

YLOCK = R / (
    "stage17y_nca_pre_extraction_protocol.json"
)

OUT_PRED = R / (
    "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"
)

OUT_REPORT = R / (
    "stage17ae_nca_preoutcome_model_lock_report.json"
)

OUT_MANIFEST = R / (
    "stage17ae_nca_preoutcome_sha256_manifest.csv"
)

OUT_RF = M / (
    "stage17ae_original567_random_forest_fitted.joblib"
)

OUT_ET = M / (
    "stage17ae_original567_extra_trees_fitted.joblib"
)

OUTPUTS = (
    OUT_PRED,
    OUT_REPORT,
    OUT_MANIFEST,
    OUT_RF,
    OUT_ET,
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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_numeric(path, columns):
    require(
        path.is_file(),
        f"Missing input: {path}",
    )

    df = pd.read_csv(
        path,
        usecols=columns,
        dtype={"cell_id": str},
    )

    require(
        len(df) > 0,
        f"Empty input: {path}",
    )

    require(
        df["cell_id"].notna().all()
        and df["cell_id"].is_unique,
        f"Missing or duplicated cell ID: {path.name}",
    )

    require(
        df["cell_id"].astype(str).str.len().gt(0).all(),
        f"Blank cell ID: {path.name}",
    )

    for col in columns:
        if col != "cell_id" and col in df:
            df[col] = pd.to_numeric(
                df[col],
                errors="raise",
            )

    return df


def numeric_matrix(df, columns, name):
    x = df.loc[:, columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    require(
        np.isfinite(
            x.to_numpy(dtype=float)
        ).all(),
        f"Nonfinite {name} features",
    )

    return x


def load_locked_module():
    spec = importlib.util.spec_from_file_location(
        "stage17ae_original_stage16a",
        ORIGINAL,
    )

    require(
        spec is not None
        and spec.loader is not None,
        "Cannot import original Stage16A code",
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    require(
        list(getattr(module, "FEATURES", []))
        == FEATURES,
        "Original seven-feature order differs",
    )

    require(
        callable(
            getattr(module, "make_pipeline", None)
        ),
        "Original make_pipeline unavailable",
    )

    return module


def assert_recipe(model, estimator_type):
    require(
        list(model.named_steps)
        == ["preprocessor", "model"],
        "Original pipeline structure changed",
    )

    estimator = model.named_steps["model"]

    require(
        isinstance(estimator, estimator_type),
        "Wrong estimator family",
    )

    params = estimator.get_params()

    expected = {
        "n_estimators": 700,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    for key, value in expected.items():
        require(
            params.get(key) == value,
            f"Changed model parameter: "
            f"{key}={params.get(key)}",
        )

    transformer = model.named_steps[
        "preprocessor"
    ]

    require(
        len(transformer.transformers) == 1,
        "Unexpected preprocessing branches",
    )

    name, pipe, cols = transformer.transformers[0]

    require(
        name == "num"
        and list(cols) == FEATURES,
        "Unexpected numeric feature contract",
    )

    require(
        list(pipe.named_steps)
        == ["imputer", "scaler"],
        "Unexpected preprocessing steps",
    )

    require(
        pipe.named_steps["imputer"].strategy
        == "median",
        "Unexpected imputation strategy",
    )

    require(
        type(
            pipe.named_steps["scaler"]
        ).__name__ == "StandardScaler",
        "Unexpected scaler",
    )

    return params


def validate_source_locks():
    sources = (
        TRAIN,
        ORIGINAL,
        ISU,
        ISU_REPLAY,
        NCA,
        AD,
        AC,
        YLOCK,
    )

    for path in sources:
        require(
            path.is_file(),
            f"Missing locked source: {path}",
        )

    for output in OUTPUTS:
        require(
            not output.exists(),
            f"Refusing overwrite: {output}",
        )

    ad = json.loads(
        AD.read_text(encoding="utf-8")
    )

    ac = json.loads(
        AC.read_text(encoding="utf-8")
    )

    y = json.loads(
        YLOCK.read_text(encoding="utf-8")
    )

    require(
        ad.get("training_n") == 567
        and ad.get("nca_candidates_n") == 30,
        "Stage17AD cohort mismatch",
    )

    require(
        ad.get("label_free_input_matrix_locked")
        is True
        and ad.get("terminal_targets_read") is False
        and ad.get("external_predictions_produced")
        is False,
        "Stage17AD was not a pre-outcome lock",
    )

    require(
        ac.get("NCA_candidate_cells") == 30
        and ac.get(
            "no_external_terminal_targets_read"
        ) is True
        and ac.get(
            "no_external_predictions"
        ) is True,
        "Stage17AC pre-outcome status mismatch",
    )

    require(
        y.get("training_567_sha256")
        == sha(TRAIN)
        and y.get(
            "original_stage16a_script_sha256"
        ) == sha(ORIGINAL),
        "Original training/model source changed",
    )

    for rel, expected in ad.get(
        "source_sha256", {}
    ).items():
        p = Path(rel)

        require(
            not p.is_absolute()
            and ".." not in p.parts,
            f"Unsafe source: {rel}",
        )

        full = (ROOT / p).resolve()

        require(
            full.is_relative_to(ROOT)
            and full.is_file()
            and sha(full) == expected,
            f"Changed Stage17AD source: {rel}",
        )

    nca_rel = NCA.relative_to(ROOT).as_posix()

    require(
        ad["source_sha256"].get(nca_rel)
        == sha(NCA),
        "NCA features differ from locked source",
    )

    comparisons = ad.get(
        "fade_equation_comparisons", []
    )

    definitions = (
        (
            "BatteryML_main",
            "Q100_divided_by_Q1_minus_one",
            318,
        ),
        (
            "BatteryLife_selected",
            "one_minus_Q100_divided_by_Q1",
            249,
        ),
    )

    for family, formula, n in definitions:
        matches = [
            row
            for row in comparisons
            if row["source_family"] == family
            and row["candidate_equation"] == formula
        ]

        require(
            len(matches) == 1
            and matches[0]["matched_rows"] == n
            and matches[0][
                "matching_at_1e_8"
            ] == n,
            f"Fade convention unverified: {family}",
        )

    return ad


def main():
    validate_source_locks()

    R.mkdir(parents=True, exist_ok=True)
    M.mkdir(parents=True, exist_ok=True)

    source_module = load_locked_module()

    train_columns = [
        "cell_id",
        "final_capacity_retention",
    ] + FEATURES

    train = read_numeric(
        TRAIN,
        train_columns,
    )

    isu = read_numeric(
        ISU,
        ["cell_id"] + FEATURES,
    )

    replay = read_numeric(
        ISU_REPLAY,
        [
            "cell_id",
            "random_forest_prediction",
            "extra_trees_prediction",
        ],
    )

    nca_df = pd.read_csv(
        NCA,
        dtype={"cell_id": str},
    )

    require(
        len(train) == 567
        and len(isu) == 242
        and len(replay) == 242
        and len(nca_df) == 30,
        "Training, ISU or NCA size mismatch",
    )

    require(
        nca_df["cell_id"].is_unique
        and nca_df["cell_id"].notna().all(),
        "Duplicate NCA physical cell ID",
    )

    require(
        set(isu.cell_id)
        == set(replay.cell_id),
        "ISU archived prediction cohort mismatch",
    )

    require(
        not (
            set(nca_df.cell_id)
            & set(train.cell_id)
        ),
        "Training/NCA identifier overlap",
    )

    require(
        not (
            set(nca_df.cell_id)
            & set(isu.cell_id)
        ),
        "ISU/NCA identifier overlap",
    )

    forbidden = {
        "final_capacity_retention",
        "target",
        "label",
        "terminal_capacity",
        "terminal_retention",
        "c5_reference_candidate_NOT_FINAL",
    }

    for name in nca_df.columns:
        require(
            name not in forbidden,
            f"Target column in NCA input: {name}",
        )

    require(
        nca_df["feature_status"].eq(
            "PROVISIONAL_NOT_CERTIFIED"
        ).all(),
        "NCA provisional feature status changed",
    )

    Xtrain = numeric_matrix(
        train,
        FEATURES,
        "training",
    )

    ytrain = pd.to_numeric(
        train["final_capacity_retention"],
        errors="raise",
    ).to_numpy(dtype=float)

    require(
        np.isfinite(ytrain).all(),
        "Nonfinite training targets",
    )

    Xisu = numeric_matrix(
        isu,
        FEATURES,
        "ISU replay",
    )

    Xnca_positive = numeric_matrix(
        nca_df,
        FEATURES,
        "NCA positive loss",
    )

    # NCA outcomes are not accessed.
    positive_fade = Xnca_positive[
        "cap_fade_fraction_1_to_100"
    ].to_numpy(dtype=float)

    cycle100_ratio = Xnca_positive[
        "cap_ratio_100_to_1"
    ].to_numpy(dtype=float)

    require(
        np.max(
            np.abs(
                positive_fade
                - (1.0 - cycle100_ratio)
            )
        ) < 1e-10,
        "Positive-loss feature does not match "
        "the NCA Q100/Q1 ratio",
    )

    Xnca_signed = Xnca_positive.copy()

    Xnca_signed[
        "cap_fade_fraction_1_to_100"
    ] = -positive_fade

    require(
        np.max(
            np.abs(
                Xnca_signed[
                    "cap_fade_fraction_1_to_100"
                ].to_numpy()
                - (cycle100_ratio - 1)
            )
        ) < 1e-10,
        "Signed-change branch is inconsistent",
    )

    reference = (
        replay.set_index("cell_id")
        .loc[isu["cell_id"]]
    )

    methods = [
        (
            "random_forest",
            RandomForestRegressor,
            "random_forest_prediction",
            OUT_RF,
        ),
        (
            "extra_trees",
            ExtraTreesRegressor,
            "extra_trees_prediction",
            OUT_ET,
        ),
    ]

    predicted = {
        "cell_id":
            nca_df["cell_id"].astype(str).tolist(),

        "program":
            nca_df["program"].astype(str).tolist(),

        "source_feature_status":
            ["PROVISIONAL_NOT_CERTIFIED"]
            * len(nca_df),

        "training_median_prior_prediction":
            [float(np.median(ytrain))]
            * len(nca_df),

        "fade_positive_loss_BatteryLife_definition":
            positive_fade.tolist(),

        "fade_negative_signed_BatteryML_definition":
            (-positive_fade).tolist(),
    }

    fitted = {}
    isu_checks = {}
    recipes = {}

    for method, cls, field, model_out in methods:
        estimator = cls(
            n_estimators=700,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model = source_module.make_pipeline(
            estimator
        )

        params = assert_recipe(
            model,
            cls,
        )

        recipes[method] = {
            k: v
            for k, v in params.items()
            if k in (
                "n_estimators",
                "min_samples_leaf",
                "random_state",
                "n_jobs",
            )
        }

        # Fit exclusively on 567 original cells.
        model.fit(Xtrain, ytrain)

        calculated_isu = np.asarray(
            model.predict(Xisu),
            dtype=float,
        )

        saved_isu = reference[
            field
        ].to_numpy(dtype=float)

        require(
            len(calculated_isu) == 242
            and np.isfinite(calculated_isu).all(),
            f"Bad ISU replay: {method}",
        )

        max_diff = float(
            np.max(
                np.abs(
                    calculated_isu - saved_isu
                )
            )
        )

        require(
            max_diff < 1e-10,
            f"Stage16A replay mismatch: "
            f"{method}, max difference={max_diff}",
        )

        isu_checks[method] = {
            "cells": 242,
            "max_abs_difference": max_diff,
            "within_tolerance_1e_10": True,
        }

        p_positive = np.asarray(
            model.predict(Xnca_positive),
            dtype=float,
        )

        p_signed = np.asarray(
            model.predict(Xnca_signed),
            dtype=float,
        )

        require(
            np.isfinite(p_positive).all()
            and np.isfinite(p_signed).all(),
            f"Nonfinite NCA prediction: {method}",
        )

        predicted[
            f"{method}_positive_loss_branch"
        ] = p_positive.tolist()

        predicted[
            f"{method}_negative_signed_branch"
        ] = p_signed.tolist()

        fitted[method] = model

        print(
            f"STAGE17AE_MODEL={method} "
            f"ISU_242_MAX_ABS_REPLAY_DIFF="
            f"{max_diff:.12g} "
            f"NCA_30_DUAL_SIGN_PREDICTIONS=GENERATED",
            flush=True,
        )

    # No NCA outcome scoring here.
    out_df = pd.DataFrame(predicted)

    require(
        len(out_df) == 30
        and out_df["cell_id"].is_unique,
        "Unexpected NCA output shape",
    )

    input_sha = {
        str(path.relative_to(ROOT)): sha(path)
        for path in (
            TRAIN,
            ORIGINAL,
            ISU,
            ISU_REPLAY,
            NCA,
            AD,
            AC,
            YLOCK,
        )
    }

    with tempfile.TemporaryDirectory(
        prefix="stage17ae_",
        dir=R,
    ) as folder:
        temp = Path(folder)

        tmp_pred = temp / "predictions.csv"
        tmp_rf = temp / "random_forest.joblib"
        tmp_et = temp / "extra_trees.joblib"
        tmp_report = temp / "report.json"
        tmp_manifest = temp / "manifest.csv"

        out_df.to_csv(
            tmp_pred,
            index=False,
            float_format="%.17g",
        )

        joblib.dump(
            fitted["random_forest"],
            tmp_rf,
            compress=3,
        )

        joblib.dump(
            fitted["extra_trees"],
            tmp_et,
            compress=3,
        )

        # Verify saved models by loading them.
        reloaded_rf = joblib.load(tmp_rf)
        reloaded_et = joblib.load(tmp_et)

        for method, loaded in (
            ("random_forest", reloaded_rf),
            ("extra_trees", reloaded_et),
        ):
            for branch, matrix in (
                ("positive_loss", Xnca_positive),
                ("negative_signed", Xnca_signed),
            ):
                actual = np.asarray(
                    loaded.predict(matrix),
                    dtype=float,
                )

                expected = out_df[
                    f"{method}_{branch}_branch"
                ].to_numpy(dtype=float)

                require(
                    np.max(
                        np.abs(
                            actual - expected
                        )
                    ) < 1e-12,
                    "Model reload prediction mismatch: "
                    f"{method}/{branch}",
                )

        created_sha = {
            str(OUT_PRED.relative_to(ROOT)):
                sha(tmp_pred),

            str(OUT_RF.relative_to(ROOT)):
                sha(tmp_rf),

            str(OUT_ET.relative_to(ROOT)):
                sha(tmp_et),
        }

        report = {
            "stage":
                "17AE_NCA_PREOUTCOME_DUAL_FADE_SIGN_INFERENCE_LOCK",

            "training_cells": 567,
            "ISU_replay_anchor_cells": 242,
            "NCA_provisional_cells": 30,

            "NCA_source":
                "Tongji_KIT_NCA_Zenodo_10.5281/zenodo.6405084",

            "original_Stage16A_make_pipeline_used":
                True,

            "ISU_replay_anchor":
                isu_checks,

            "original_recipe_parameters":
                recipes,

            "training_target_median_prior":
                float(np.median(ytrain)),

            "features_in_original_order":
                FEATURES,

            "two_prespecified_NCA_fade_semantic_branches": {
                "positive_loss_BatteryLife":
                    "1 - Q100/Q1",

                "negative_signed_BatteryML":
                    "Q100/Q1 - 1",
            },

            "interpretation": (
                "Both branches are pre-outcome "
                "sensitivity analyses. Neither "
                "may be selected after examining "
                "NCA outcomes. Feature and "
                "terminal-target equivalence "
                "remain uncertified."
            ),

            "input_sha256": input_sha,

            "frozen_prediction_and_model_sha256":
                created_sha,

            "python_version":
                platform.python_version(),

            "numpy_version":
                np.__version__,

            "pandas_version":
                pd.__version__,

            "scikit_learn_version":
                sklearn.__version__,

            "joblib_version":
                joblib.__version__,

            "NCA_terminal_labels_accessed":
                False,

            "NCA_predictions_locked_before_terminal_outcome_review":
                True,

            "NCA_external_success_claimed":
                False,

            "original_training_table_modified":
                False,

            "prism_updated":
                False,
        }

        tmp_report.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest_entries = {
            **input_sha,
            **created_sha,
            str(OUT_REPORT.relative_to(ROOT)):
                sha(tmp_report),
        }

        manifest = [
            {
                "file": rel,
                "sha256": hash_value,
            }
            for rel, hash_value
            in sorted(manifest_entries.items())
        ]

        with tmp_manifest.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "sha256"],
            )

            writer.writeheader()
            writer.writerows(manifest)

        require(
            not any(
                path.exists()
                for path in OUTPUTS
            ),
            "Output appeared during computation; "
            "refusing overwrite",
        )

        os.replace(tmp_rf, OUT_RF)
        os.replace(tmp_et, OUT_ET)
        os.replace(tmp_pred, OUT_PRED)
        os.replace(tmp_report, OUT_REPORT)
        os.replace(tmp_manifest, OUT_MANIFEST)

    require(
        sha(OUT_PRED)
        == created_sha[
            str(OUT_PRED.relative_to(ROOT))
        ],
        "Committed prediction file hash mismatch",
    )

    print(
        "=== STAGE 17AE NCA PRE-OUTCOME FROZEN INFERENCE ==="
    )

    print(
        "ORIGINAL_TRAIN_567=YES "
        "ISU_242_REPLAY_ANCHOR=PASSED"
    )

    print(
        "ISU_REPLAY_MAX_ABS_DIFFERENCES=",
        json.dumps(isu_checks),
    )

    print(
        "NCA_30_PROVISIONAL_INPUTS_SHA256=",
        sha(NCA),
    )

    print(
        "TWO_NCA_FADE_SIGN_BRANCHES=FROZEN_TOGETHER"
    )

    print(
        "PREOUTCOME_PREDICTIONS_SHA256=",
        sha(OUT_PRED),
    )

    print(
        "RF_MODEL_SHA256=",
        sha(OUT_RF),
    )

    print(
        "ET_MODEL_SHA256=",
        sha(OUT_ET),
    )

    print("PREDICTIONS_CSV=", OUT_PRED)
    print("MODEL_RF=", OUT_RF)
    print("MODEL_ET=", OUT_ET)
    print("LOCK_REPORT=", OUT_REPORT)
    print("HASH_MANIFEST=", OUT_MANIFEST)

    print(
        "NCA_TARGET_READ=NO "
        "NCA_OUTCOME_SCORING=NO"
    )

    print(
        "FEATURE_EQUIVALENCE_CERTIFIED=NO "
        "TARGET_EQUIVALENCE_CERTIFIED=NO"
    )

    print("PRISM_UPDATED=NO")

    print(
        "=== END STAGE 17AE ==="
    )


if __name__ == "__main__":
    main()
