#!/usr/bin/env python3
"""Stage21C: paired domain-level representation and ranking audit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path.cwd()
REP = R / "external_validation/reports"

PRED = REP / "stage21b_cell_predictions.csv"
OLD = REP / "stage20r_replayed_lodo_cell_predictions.csv"
SUMMARY = REP / "stage21b_representation_summary.csv"
SOURCE = REP / "stage21b_controlled_family_report.json"

OUT_DOMAIN = REP / "stage21c_paired_domain_diagnostics.csv"
OUT_SUMMARY = REP / "stage21c_mechanism_summary.csv"
OUT_REPORT = REP / "stage21c_mechanism_report.json"

OUTPUTS = [OUT_DOMAIN, OUT_SUMMARY, OUT_REPORT]

VARIANTS = [
    "original7",
    "harmonized7",
    "no_fade6",
    "harmonized7_plus_family",
    "family_only",
]

POLICIES = {
    "all_567": (None, False, 567, 12),
    "all_exclude44_523": (None, True, 523, 12),
    "batterylife_249": (
        "BatteryLife_selected", False, 249, 8
    ),
    "batterylife_exclude44_205": (
        "BatteryLife_selected", True, 205, 8
    ),
}

def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def rank_correlation(y, p):
    if len(y) < 5:
        return np.nan

    if (
        np.ptp(y) <= 1e-12
        or np.ptp(p) <= 1e-12
    ):
        return np.nan

    value = float(
        spearmanr(y, p).statistic
    )

    return (
        value if np.isfinite(value)
        else np.nan
    )

def main():
    require(
        all(not x.exists() for x in OUTPUTS),
        "Stage21C output already exists"
    )

    source = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )

    require(
        source["status"]
        == "CONTROLLED_REPRESENTATION_LODO_COMPLETE",
        "Stage21B is not verified"
    )

    require(
        source["prediction_rows"] == 5670,
        "Stage21B count mismatch"
    )

    require(
        source["max_historical_model_mae_difference"]
        < 1e-10,
        "Historical fold replay mismatch"
    )

    df = pd.read_csv(PRED)
    old = pd.read_csv(OLD)
    historical = pd.read_csv(SUMMARY)

    require(len(df) == 5670, "Incomplete Stage21B predictions")
    require(len(old) == 1134, "Incomplete Stage20R predictions")

    require(
        not df.duplicated([
            "dataset_id", "cell_id", "model", "variant"
        ]).any(),
        "Duplicate Stage21B predictions"
    )

    require(
        set(df.variant) == set(VARIANTS),
        "Unexpected feature representations"
    )

    require(
        set(df.model) == {"RF", "ET"},
        "Unexpected model labels"
    )

    require(
        np.isfinite(df.y_pred).all()
        and np.isfinite(df.y_true).all(),
        "Nonfinite predictions or targets"
    )

    # Verify that new original7 predictions agree with the
    # saved cell-level historical Stage20R reconstruction.
    legacy = df[
        df.variant == "original7"
    ][[
        "dataset_id", "cell_id", "model",
        "y_true", "y_pred"
    ]]

    ref = old[[
        "dataset_id", "cell_id", "model",
        "y_true", "y_pred"
    ]]

    merged = legacy.merge(
        ref,
        on=["dataset_id", "cell_id", "model"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )

    require(
        len(merged) == 1134,
        "Historical cell alignment mismatch"
    )

    model_drift = float(np.max(np.abs(
        merged.y_pred_new - merged.y_pred_old
    )))

    target_drift = float(np.max(np.abs(
        merged.y_true_new - merged.y_true_old
    )))

    require(
        model_drift < 1e-10
        and target_drift < 1e-12,
        "Original7 per-cell replay mismatch"
    )

    detail_rows = []
    summary_rows = []

    for policy, (
        family_filter, exclude_flags,
        expected_cells, expected_domains
    ) in POLICIES.items():

        for model in ("RF", "ET"):

            cohort = df[df.model == model].copy()

            if family_filter:
                cohort = cohort[
                    cohort.source_family
                    == family_filter
                ]

            if exclude_flags:
                cohort = cohort[
                    cohort.quality_flag == 0
                ]

            require(
                len(cohort) == expected_cells * 5,
                "Filtered variant count mismatch"
            )

            policy_rows = []

            for domain, group in cohort.groupby(
                "dataset_id", sort=True
            ):
                wide = group.pivot(
                    index="cell_id",
                    columns="variant",
                    values="y_pred",
                )

                require(
                    set(wide.columns) == set(VARIANTS),
                    "Missing model variant"
                )

                base = group[
                    group.variant == "original7"
                ].set_index("cell_id")

                require(
                    len(wide) == len(base),
                    "Domain cell mismatch"
                )

                y = base.loc[
                    wide.index, "y_true"
                ].to_numpy(float)

                global_prior = base.loc[
                    wide.index,
                    "global_train_median_prior"
                ].to_numpy(float)

                family_prior = base.loc[
                    wide.index,
                    "family_train_median_prior"
                ].to_numpy(float)

                result = {
                    "policy": policy,
                    "model": model,
                    "dataset": domain,
                    "cells": len(y),
                    "global_prior_mae": float(
                        np.mean(np.abs(
                            y - global_prior
                        ))
                    ),
                    "family_prior_mae": float(
                        np.mean(np.abs(
                            y - family_prior
                        ))
                    ),
                }

                for variant in VARIANTS:
                    p = wide[
                        variant
                    ].to_numpy(float)

                    result[
                        variant + "_mae"
                    ] = float(
                        np.mean(np.abs(y - p))
                    )

                    result[
                        variant + "_bias"
                    ] = float(
                        np.mean(p - y)
                    )

                    result[
                        variant + "_spearman"
                    ] = rank_correlation(y, p)

                    # Retrospective diagnostic only:
                    # uses held-out target mean.
                    result[
                        variant + "_centered_mae"
                    ] = float(
                        np.mean(np.abs(
                            (p - p.mean())
                            - (y - y.mean())
                        ))
                    )

                result["family_recovery_mae"] = (
                    result["harmonized7_mae"]
                    - result[
                        "harmonized7_plus_family_mae"
                    ]
                )

                result["original_vs_harmonized_mae"] = (
                    result["harmonized7_mae"]
                    - result["original7_mae"]
                )

                result["family_added_better"] = int(
                    result[
                        "harmonized7_plus_family_mae"
                    ]
                    < result["harmonized7_mae"]
                )

                result["family_added_beats_original"] = int(
                    result[
                        "harmonized7_plus_family_mae"
                    ]
                    < result["original7_mae"]
                )

                policy_rows.append(result)
                detail_rows.append(result)

            require(
                len(policy_rows) == expected_domains,
                "Domain count mismatch"
            )

            pframe = pd.DataFrame(policy_rows)

            output = {
                "policy": policy,
                "model": model,
                "cells": expected_cells,
                "domains": expected_domains,
                "family_added_better_domains": int(
                    pframe.family_added_better.sum()
                ),
                "family_added_beats_original_domains": int(
                    pframe.family_added_beats_original.sum()
                ),
                "macro_global_prior_mae": float(
                    pframe.global_prior_mae.mean()
                ),
                "macro_family_prior_mae": float(
                    pframe.family_prior_mae.mean()
                ),
                "macro_family_recovery": float(
                    pframe.family_recovery_mae.mean()
                ),
            }

            for variant in VARIANTS:
                value = float(
                    pframe[variant + "_mae"].mean()
                )

                output[
                    variant + "_macro_mae"
                ] = value

                ranks = pframe[
                    variant + "_spearman"
                ].dropna()

                output[
                    variant + "_valid_rank_domains"
                ] = int(len(ranks))

                output[
                    variant + "_median_spearman"
                ] = (
                    float(ranks.median())
                    if len(ranks)
                    else None
                )

                existing = historical[
                    (historical.policy == policy)
                    & (historical.model == model)
                    & (historical.variant == variant)
                ]

                require(
                    len(existing) == 1,
                    "Stage21B summary row missing"
                )

                require(
                    abs(
                        value - float(
                            existing.iloc[0].model_macro_mae
                        )
                    ) < 1e-10,
                    "Stage21B macro MAE mismatch"
                )

            summary_rows.append(output)

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    report = {
        "stage": "21C",
        "status": "PAIRED_DOMAIN_AUDIT_VERIFIED",
        "stage21b_prediction_sha256": sha(PRED),
        "stage20r_prediction_sha256": sha(OLD),
        "historical_cell_replay_max_prediction_drift": model_drift,
        "historical_cell_replay_max_target_drift": target_drift,
        "domain_diagnostic_rows": len(detail),
        "summary_rows": len(summary),
        "methodological_limits": [
            "Source-family label is known at evaluation time.",
            "Models are not refitted in this Stage21C audit.",
            "Domain-wise rank correlation requires adequate variation and at least five cells.",
            "Centered MAE is retrospective because it uses held-out target means.",
            "Any restoration of accuracy is a controlled performance association, not an identified causal effect.",
            "The audited 205-cell BatteryLife cohort is not a physically standardized benchmark.",
        ],
    }

    detail.to_csv(
        OUT_DOMAIN, index=False, mode="x"
    )

    summary.to_csv(
        OUT_SUMMARY, index=False, mode="x"
    )

    with OUT_REPORT.open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(
            report, f,
            ensure_ascii=False, indent=2
        )
        f.write("\n")

    print("=== STAGE21C VERIFICATION ===")
    print("ORIGINAL7_CELL_PREDICTION_DRIFT", model_drift)
    print("ORIGINAL7_CELL_TARGET_DRIFT", target_drift)
    print("DOMAIN_ROWS", len(detail))

    print("=== STAGE21C DOMAIN RESULTS ===")
    for _, row in detail[
        detail.policy == "all_567"
    ].iterrows():
        print(
            "DOMAIN", row.dataset,
            "MODEL", row.model,
            "N", row.cells,
            "ORIGINAL", round(row.original7_mae, 6),
            "HARM", round(row.harmonized7_mae, 6),
            "HARM_PLUS_FAMILY",
            round(
                row.harmonized7_plus_family_mae, 6
            ),
            "RESTORATION",
            round(row.family_recovery_mae, 6),
            "SPEARMAN_ORIGINAL",
            round(row.original7_spearman, 4),
            "SPEARMAN_HARM",
            round(row.harmonized7_spearman, 4),
            "SPEARMAN_PLUS",
            round(
                row.harmonized7_plus_family_spearman, 4
            ),
        )

    print("=== STAGE21C SUMMARY ===")
    for _, row in summary.iterrows():
        print(
            "POLICY", row.policy,
            "MODEL", row.model,
            "DOMAINS_FAMILY_HELPS",
            row.family_added_better_domains,
            "OF", row.domains,
            "MACRO_RECOVERY",
            round(row.macro_family_recovery, 9),
            "HARM_RANK_VALID",
            row.harmonized7_valid_rank_domains,
            "HARM_MEDIAN_SPEARMAN",
            row.harmonized7_median_spearman,
        )

    print("STAGE21C_COMPLETE=YES")
    print("REPORT", OUT_REPORT)
    print("DOMAIN_CSV", OUT_DOMAIN)
    print("SUMMARY_CSV", OUT_SUMMARY)

if __name__ == "__main__":
    main()
