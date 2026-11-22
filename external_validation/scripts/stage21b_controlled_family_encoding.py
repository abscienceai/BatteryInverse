#!/usr/bin/env python3
"""Stage21B: matched LODO representation and explicit-family controls."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from scipy.stats import spearmanr
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

R = Path.cwd()
D = R / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
F = R / "external_validation/reports/stage20o_unified_endpoint_cells.csv"
H = R / "external_validation/reports/stage19_featureblock_swap_folds.csv"
S = R / "external_validation/scripts/stage19_featureblock_swap.py"
O = R / "external_validation/reports"

PRED = O / "stage21b_cell_predictions.csv"
FOLDS = O / "stage21b_domain_results.csv"
SUMMARY = O / "stage21b_representation_summary.csv"
REPORT = O / "stage21b_controlled_family_report.json"

OUTPUTS = [PRED, FOLDS, SUMMARY, REPORT]

SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def filehash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def spearman_safe(y, pred):
    if (
        len(y) < 5
        or len(np.unique(y)) < 2
        or len(np.unique(pred)) < 2
    ):
        return None
    result = float(spearmanr(y, pred).statistic)
    return result if np.isfinite(result) else None


def variant_pipeline(
    original,
    model_class,
    variant,
    used_features,
    original_features,
    fade,
):
    """Adapt the original preprocessing column selectors safely."""

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    model = original.pipeline(
        model_class(
            n_estimators=700,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
    )

    # Historical pipeline remains completely unchanged.
    if variant in ("original7", "harmonized7"):
        return model

    matches = []

    def visit(estimator):
        if isinstance(estimator, ColumnTransformer):
            matches.append(estimator)
        elif isinstance(estimator, Pipeline):
            for _, child in estimator.steps:
                visit(child)

    visit(model)

    check(
        len(matches) == 1,
        "Expected exactly one original ColumnTransformer; "
        "inspect pipeline before adapting"
    )

    preprocessor = matches[0]
    branches = []
    original_selected = []

    for name, transformer, selector in preprocessor.transformers:

        check(
            isinstance(
                selector,
                (list, tuple, np.ndarray, pd.Index)
            ),
            "Unsupported original feature selector: "
            + str(name)
        )

        cols = list(selector)

        check(
            all(isinstance(c, str) for c in cols),
            "Non-string feature selector"
        )

        is_dropped = (
            isinstance(transformer, str)
            and transformer == "drop"
        )

        if not is_dropped:
            original_selected.extend(cols)

        branches.append(
            (name, transformer, cols)
        )

    check(
        len(original_selected) == len(original_features)
        and set(original_selected) == set(original_features),
        "Original ColumnTransformer feature selection "
        "does not match the historical seven features: "
        + str(original_selected)
    )

    family_column = "source_family_binary_EXPLICIT"

    fade_branches = sum(
        fade in cols
        and not (
            isinstance(transformer, str)
            and transformer == "drop"
        )
        for _, transformer, cols in branches
    )

    check(
        fade_branches == 1,
        "The original fade feature must belong "
        "to exactly one active preprocessing branch"
    )

    adapted = []

    for name, transformer, cols in branches:

        if variant == "no_fade6":
            new_cols = [
                c for c in cols
                if c != fade
            ]

        elif variant == "harmonized7_plus_family":
            new_cols = list(cols)

            if fade in cols:
                new_cols.append(family_column)

        elif variant == "family_only":
            new_cols = (
                [family_column]
                if fade in cols
                else []
            )

        else:
            raise RuntimeError(
                "Unknown representation: " + variant
            )

        adapted.append(
            (name, transformer, new_cols)
        )

    actual_selected = []

    for name, transformer, cols in adapted:
        is_dropped = (
            isinstance(transformer, str)
            and transformer == "drop"
        )

        if not is_dropped:
            actual_selected.extend(cols)

    check(
        len(actual_selected) == len(used_features)
        and set(actual_selected) == set(used_features),
        "Adapted preprocessing feature mismatch. "
        "Expected: " + str(used_features)
        + " Actual: " + str(actual_selected)
    )

    # Preserve the original imputation/scaling transformers
    # and model parameters. Modify only column selections.
    preprocessor.transformers = adapted

    return model


def main():
    check(
        all(not p.exists() for p in OUTPUTS),
        "Stage21B output exists; refusing overwrite"
    )

    check(
        sklearn.__version__ == "1.8.0",
        "Use the historical scikit-learn 1.8.0 environment"
    )
    check(filehash(D) == SHA, "Benchmark hash mismatch")

    spec = importlib.util.spec_from_file_location(
        "historical_stage19", S
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    check(
        filehash(original.RECIPE) == original.SHA_RECIPE,
        "Historical recipe changed"
    )

    d = pd.read_csv(D)
    historical = pd.read_csv(H)
    flags = pd.read_csv(F)

    check(d.shape == (567, 38), "Unexpected dataset shape")
    check(len(historical) == 24, "Historical fold count mismatch")
    check(len(flags) == 249, "Quality audit count mismatch")
    check(
        not d.duplicated(["dataset_id", "cell_id"]).any(),
        "Duplicate benchmark cell"
    )

    features = list(original.F)
    fade = "cap_fade_fraction_1_to_100"
    ratio = "cap_ratio_100_to_1"

    check(
        fade in features,
        "Actual Stage19 model fade feature not found"
    )
    check(
        ratio in features,
        "Stage19 model capacity-ratio feature missing"
    )
    check(len(features) == 7, "Unexpected model feature count")

    fam = d.source_family.astype(str).to_numpy()
    rr = d[ratio].to_numpy(float)
    observed_fade = d[fade].to_numpy(float)

    expected_original_fade = np.where(
        fam == "BatteryML_main",
        rr - 1.0,
        1.0 - rr
    )

    formula_error = float(
        np.max(np.abs(observed_fade - expected_original_fade))
    )

    check(
        np.isfinite(observed_fade).all(),
        "Nonfinite original fade"
    )
    check(
        formula_error < 1e-8,
        "ACTUAL model fade formula mismatch"
    )

    binary_family = (
        fam == "BatteryLife_selected"
    ).astype(float)

    Xoriginal = d[features].copy()
    Xharm = Xoriginal.copy()
    Xharm[fade] = 1.0 - rr

    Xdrop = Xoriginal.drop(columns=[fade])

    Xplus = Xharm.copy()
    Xplus["source_family_binary_EXPLICIT"] = binary_family

    Xfamily = pd.DataFrame({
        "source_family_binary_EXPLICIT": binary_family
    })

    variants = {
        "original7": Xoriginal,
        "harmonized7": Xharm,
        "no_fade6": Xdrop,
        "harmonized7_plus_family": Xplus,
        "family_only": Xfamily,
    }

    for name, X in variants.items():
        check(
            len(X) == 567,
            "Feature count mismatch: " + name
        )
        check(
            np.isfinite(X.to_numpy(float)).all(),
            "Nonfinite model feature: " + name
        )

    excluded = {
        (str(x.dataset), str(x.cell_id))
        for x in flags.itertuples(index=False)
        if (
            int(x.positive_post100_records) < 3
            or int(x.mich_nonpositive_endpoint_flag) == 1
        )
    }

    check(len(excluded) == 44, "Quality flag mismatch")

    y = d.final_capacity_retention.to_numpy(float)
    domains = d.dataset_id.astype(str).to_numpy()
    ids = d.cell_id.astype(str).to_numpy()

    prediction_rows = []
    historical_checks = []

    print("=== STAGE21B ORIGINAL MODEL FEATURE AUDIT ===", flush=True)
    print("ACTUAL_FADE_COLUMN", fade, flush=True)
    print("FADE_IN_MODEL", fade in features, flush=True)
    print("MAX_ACTUAL_FADE_FORMULA_ERROR", formula_error, flush=True)
    print("SKLEARN_VERSION", sklearn.__version__, flush=True)

    for domain in sorted(np.unique(domains)):
        train = domains != domain
        test = domains == domain

        ytr = y[train]
        yte = y[test]

        train_family = fam[train]
        held_family = np.unique(fam[test])

        check(
            len(held_family) == 1,
            "Mixed held-out processing families"
        )

        family_targets = ytr[
            train_family == held_family[0]
        ]

        check(
            len(family_targets) > 0,
            "No same-family training cells"
        )

        prior_global = float(np.median(ytr))
        prior_family = float(np.median(family_targets))

        for model_name, model_class in [
            ("RF", RandomForestRegressor),
            ("ET", ExtraTreesRegressor),
        ]:
            for variant, X in variants.items():

                model = variant_pipeline(
                    original=original,
                    model_class=model_class,
                    variant=variant,
                    used_features=list(X.columns),
                    original_features=features,
                    fade=fade,
                )

                model.fit(X.loc[train], ytr)

                pred = np.asarray(
                    model.predict(X.loc[test]),
                    dtype=float
                )

                check(
                    len(pred) == len(yte)
                    and np.isfinite(pred).all(),
                    "Invalid prediction"
                )

                mae = float(np.abs(pred - yte).mean())
                prior_mae = float(
                    np.abs(prior_global - yte).mean()
                )

                if variant == "original7":
                    old = historical[
                        (historical.domain == domain)
                        & (historical.model == model_name)
                    ]

                    check(
                        len(old) == 1,
                        "Missing historical fold"
                    )

                    old = old.iloc[0]

                    model_delta = abs(
                        mae - float(old.original_mae)
                    )
                    prior_delta = abs(
                        prior_mae - float(old.prior_mae)
                    )

                    check(
                        model_delta < 1e-10
                        and prior_delta < 1e-10,
                        "Historical replay failed: "
                        + domain + " " + model_name
                    )

                    historical_checks.append({
                        "domain": domain,
                        "model": model_name,
                        "model_mae_difference": model_delta,
                        "prior_mae_difference": prior_delta,
                    })

                test_indices = np.flatnonzero(test)

                for idx, p in zip(test_indices, pred):
                    key = (domains[idx], ids[idx])

                    prediction_rows.append({
                        "dataset_id": domains[idx],
                        "cell_id": ids[idx],
                        "source_family": fam[idx],
                        "model": model_name,
                        "variant": variant,
                        "y_true": float(y[idx]),
                        "y_pred": float(p),
                        "global_train_median_prior": prior_global,
                        "family_train_median_prior": prior_family,
                        "model_abs_error": abs(float(p) - y[idx]),
                        "global_prior_abs_error": abs(
                            prior_global - y[idx]
                        ),
                        "family_prior_abs_error": abs(
                            prior_family - y[idx]
                        ),
                        "quality_flag": int(key in excluded),
                    })

        print(
            "COMPLETED_DOMAIN",
            domain,
            "N", int(test.sum()),
            flush=True
        )

    check(
        len(historical_checks) == 24,
        "Historical replay incomplete"
    )

    check(
        len(prediction_rows) == 567 * 2 * len(variants),
        "Prediction population incomplete"
    )

    pred = pd.DataFrame(prediction_rows)

    check(
        not pred.duplicated([
            "dataset_id", "cell_id", "model", "variant"
        ]).any(),
        "Duplicate model predictions"
    )

    policies = [
        ("all_567", False, False, 567, 12),
        ("all_exclude44_523", False, True, 523, 12),
        ("batterylife_249", True, False, 249, 8),
        ("batterylife_exclude44_205", True, True, 205, 8),
    ]

    summaries = []
    domain_rows = []

    for policy, bl_only, filter_flags, expected_n, expected_domains in policies:
        for model_name in ("RF", "ET"):
            for variant in variants:
                sub = pred[
                    (pred.model == model_name)
                    & (pred.variant == variant)
                ].copy()

                if bl_only:
                    sub = sub[
                        sub.source_family == "BatteryLife_selected"
                    ]

                if filter_flags:
                    sub = sub[sub.quality_flag == 0]

                check(
                    len(sub) == expected_n,
                    "Policy size mismatch: " + policy
                )

                grouped = sub.groupby("dataset_id", sort=True)

                check(
                    len(grouped) == expected_domains,
                    "Policy domain mismatch"
                )

                domain_metrics = []

                for domain, g in grouped:
                    yy = g.y_true.to_numpy(float)
                    pp = g.y_pred.to_numpy(float)

                    metrics = {
                        "policy": policy,
                        "model": model_name,
                        "variant": variant,
                        "dataset_id": domain,
                        "cells": len(g),
                        "model_mae": float(g.model_abs_error.mean()),
                        "global_prior_mae": float(
                            g.global_prior_abs_error.mean()
                        ),
                        "family_prior_mae": float(
                            g.family_prior_abs_error.mean()
                        ),
                        "signed_bias_pred_minus_target": float(
                            np.mean(pp - yy)
                        ),
                        "within_domain_spearman": spearman_safe(
                            yy, pp
                        ),
                        "centered_mae_DIAGNOSTIC": float(
                            np.mean(np.abs(
                                (pp - pp.mean()) - (yy - yy.mean())
                            ))
                        ),
                    }

                    domain_metrics.append(metrics)
                    domain_rows.append(metrics)

                model_macro = float(np.mean([
                    x["model_mae"] for x in domain_metrics
                ]))
                global_macro = float(np.mean([
                    x["global_prior_mae"] for x in domain_metrics
                ]))
                family_macro = float(np.mean([
                    x["family_prior_mae"] for x in domain_metrics
                ]))

                model_weighted = float(sub.model_abs_error.mean())
                global_weighted = float(
                    sub.global_prior_abs_error.mean()
                )
                family_weighted = float(
                    sub.family_prior_abs_error.mean()
                )

                summaries.append({
                    "policy": policy,
                    "model": model_name,
                    "variant": variant,
                    "cells": len(sub),
                    "domains": len(domain_metrics),
                    "model_macro_mae": model_macro,
                    "global_prior_macro_mae": global_macro,
                    "family_prior_macro_mae": family_macro,
                    "macro_gain_vs_global": global_macro - model_macro,
                    "macro_gain_vs_family": family_macro - model_macro,
                    "model_weighted_mae": model_weighted,
                    "global_prior_weighted_mae": global_weighted,
                    "family_prior_weighted_mae": family_weighted,
                    "weighted_gain_vs_global": (
                        global_weighted - model_weighted
                    ),
                    "weighted_gain_vs_family": (
                        family_weighted - model_weighted
                    ),
                })

    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(domain_rows)

    report = {
        "stage": "21B",
        "status": "CONTROLLED_REPRESENTATION_LODO_COMPLETE",
        "benchmark_sha256": filehash(D),
        "quality_flags_sha256": filehash(F),
        "historical_script_sha256": filehash(S),
        "sklearn_version": sklearn.__version__,
        "actual_fade_feature": fade,
        "actual_fade_formula_max_abs_error": formula_error,
        "feature_variants": {
            name: list(X.columns)
            for name, X in variants.items()
        },
        "historical_replay_checks": historical_checks,
        "max_historical_model_mae_difference": max(
            x["model_mae_difference"]
            for x in historical_checks
        ),
        "max_historical_prior_mae_difference": max(
            x["prior_mae_difference"]
            for x in historical_checks
        ),
        "prediction_rows": len(pred),
        "limitations": [
            "Five representation controls were evaluated using matched 12-dataset LODO folds.",
            "The actual original model fade column was audited.",
            "Source-family labels were known for the held-out datasets.",
            "No held-out target was used to construct the two training-only median priors.",
            "All models were fitted under the historical recipe.",
            "Quality filtering changed scoring membership, not model training.",
            "Centered MAE uses held-out outcome means and is retrospective only.",
            "Spearman is undefined or omitted for very small or constant-prediction domains.",
            "The observed changes do not alone prove causal reliance on a processing artifact.",
        ],
    }

    # Write only after all historical replay checks pass.
    pred.to_csv(PRED, index=False, mode="x")
    detail.to_csv(FOLDS, index=False, mode="x")
    summary.to_csv(SUMMARY, index=False, mode="x")

    with REPORT.open("x", encoding="utf-8") as f:
        json.dump(
            report, f, indent=2, ensure_ascii=False
        )
        f.write("\n")

    print("=== STAGE21B CONTROLLED RESULTS ===")
    print("PREDICTION_ROWS", len(pred))
    print(
        "MAX_HISTORICAL_REPLAY_DIFFERENCE",
        report["max_historical_model_mae_difference"]
    )

    for policy in (
        "all_567",
        "all_exclude44_523",
        "batterylife_exclude44_205",
    ):
        for _, row in summary[
            summary.policy == policy
        ].iterrows():
            print(
                "POLICY", row.policy,
                "MODEL", row.model,
                "VARIANT", row.variant,
                "N", row.cells,
                "MACRO_MAE", round(row.model_macro_mae, 9),
                "GAIN_GLOBAL", round(row.macro_gain_vs_global, 9),
                "GAIN_FAMILY", round(row.macro_gain_vs_family, 9),
                "WEIGHTED_GAIN_GLOBAL",
                round(row.weighted_gain_vs_global, 9),
                flush=True
            )

    print("STAGE21B_COMPLETE=YES")
    print("SUMMARY", SUMMARY)
    print("DOMAIN_RESULTS", FOLDS)
    print("PREDICTIONS", PRED)
    print("REPORT", REPORT)

if __name__ == "__main__":
    main()
