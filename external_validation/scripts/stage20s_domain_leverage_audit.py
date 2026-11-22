#!/usr/bin/env python3
"""Stage20S: fixed-prediction domain-leverage sensitivity."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
REP = ROOT / "external_validation/reports"

PRED = REP / "stage20r_replayed_lodo_cell_predictions.csv"
PREVIOUS = REP / "stage20r_fixed_prediction_quality_sensitivity.csv"
REPLAY = REP / "stage20r_replay_quality_report.json"
FLAGS = REP / "stage20o_unified_endpoint_cells.csv"

OUT_MAIN = REP / "stage20s_domain_robustness_summary.csv"
OUT_FOLDS = REP / "stage20s_domain_leverage.csv"
OUT_REPORT = REP / "stage20s_domain_robustness_report.json"

OUTPUTS = [OUT_MAIN, OUT_FOLDS, OUT_REPORT]

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def filehash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def metrics(frame):
    grouped = frame.groupby("dataset_id").agg(
        n=("cell_id", "size"),
        model_mae=("model_abs_error", "mean"),
        prior_mae=("prior_abs_error", "mean"),
    )

    require(len(grouped) > 0, "Empty evaluation cohort")

    macro_model = float(grouped.model_mae.mean())
    macro_prior = float(grouped.prior_mae.mean())
    weighted_model = float(frame.model_abs_error.mean())
    weighted_prior = float(frame.prior_abs_error.mean())

    return {
        "cells": int(len(frame)),
        "domains": int(len(grouped)),
        "smallest_domain_n": int(grouped.n.min()),
        "macro_model_mae": macro_model,
        "macro_prior_mae": macro_prior,
        "macro_gain": macro_prior - macro_model,
        "weighted_model_mae": weighted_model,
        "weighted_prior_mae": weighted_prior,
        "weighted_gain": weighted_prior - weighted_model,
    }

def main():

    require(
        all(not p.exists() for p in OUTPUTS),
        "Stage20S output already exists; refusing overwrite"
    )

    report20r = json.loads(
        REPLAY.read_text(encoding="utf-8")
    )

    require(
        report20r["status"] == "FOLD_AND_AGGREGATE_REPLAY_VERIFIED",
        "Stage20R historical replay is not verified"
    )

    require(
        report20r["max_model_fold_mae_replay_delta"] < 1e-8,
        "Stage20R replay drift"
    )

    require(
        report20r["max_prior_fold_mae_replay_delta"] < 1e-10,
        "Stage20R prior replay drift"
    )

    require(
        filehash(FLAGS) == report20r["stage20o_flags_sha256"],
        "Stage20O quality file changed"
    )

    pred = pd.read_csv(PRED)
    previous = pd.read_csv(PREVIOUS)

    require(len(pred) == 1134, "Unexpected prediction population")

    require(
        set(pred.model) == {"RF", "ET"},
        "Unexpected model labels"
    )

    require(
        not pred.duplicated(
            ["dataset_id", "cell_id", "model"]
        ).any(),
        "Duplicate cell-model prediction"
    )

    for model in ("RF", "ET"):
        p = pred[pred.model == model]

        require(len(p) == 567, "Incomplete model prediction set")
        require(p.dataset_id.nunique() == 12, "Incomplete domains")

        require(
            int(p.flag_any_three.sum()) == 44,
            "Quality flag population mismatch"
        )

        require(
            int(p.flag_no_post100.sum()) == 29,
            "No-post100 flag mismatch"
        )

        require(
            int(p.flag_sparse_post100.sum()) == 1,
            "Sparse-horizon flag mismatch"
        )

        require(
            int(p.flag_mich_nonpositive.sum()) == 14,
            "MICH quality flag mismatch"
        )

        require(
            np.allclose(
                (p.y_true - p.y_pred).abs(),
                p.model_abs_error,
                rtol=0,
                atol=1e-12
            ),
            "Model error column mismatch"
        )

        require(
            np.allclose(
                (p.y_true - p.train_median_prior).abs(),
                p.prior_abs_error,
                rtol=0,
                atol=1e-12
            ),
            "Prior error column mismatch"
        )

    policies = {
        "all_567": lambda d: d,
        "all_523": lambda d: d[d.flag_any_three == 0],
        "batterylife_249": lambda d: d[
            d.source_family == "BatteryLife_selected"
        ],
        "batterylife_205": lambda d: d[
            (d.source_family == "BatteryLife_selected")
            & (d.flag_any_three == 0)
        ],
        "batterylife_205_min5": lambda d: d[
            (d.source_family == "BatteryLife_selected")
            & (d.flag_any_three == 0)
        ],
        "all_523_min5": lambda d: d[
            d.flag_any_three == 0
        ],
    }

    expected_n = {
        "all_567": 567,
        "all_523": 523,
        "batterylife_249": 249,
        "batterylife_205": 205,
        "batterylife_205_min5": 202,
        "all_523_min5": 520,
    }

    previous_policy = {
        "all_567": "all_567",
        "all_523": "all_exclude_three_flags_523",
        "batterylife_249": "batterylife_249",
        "batterylife_205": "batterylife_exclude_three_flags_205",
    }

    summaries = []
    leave_out = []

    for name, select in policies.items():
        for model in ("RF", "ET"):
            cohort = select(
                pred[pred.model == model]
            ).copy()

            if name.endswith("_min5"):
                domain_counts = cohort.groupby(
                    "dataset_id"
                ).size()

                eligible = domain_counts[
                    domain_counts >= 5
                ].index

                cohort = cohort[
                    cohort.dataset_id.isin(eligible)
                ]

            require(
                len(cohort) == expected_n[name],
                "Population mismatch: " + name
            )

            result = metrics(cohort)

            summaries.append({
                "policy": name,
                "model": model,
                **result,
            })

            if name in previous_policy:
                old = previous[
                    (previous.policy == previous_policy[name])
                    & (previous.model == model)
                ]

                require(
                    len(old) == 1,
                    "Missing Stage20R policy summary"
                )

                old = old.iloc[0]

                require(
                    abs(
                        result["macro_gain"]
                        - float(old.equal_domain_prior_minus_model)
                    ) < 1e-10,
                    "Stage20R macro gain replay mismatch"
                )

                require(
                    abs(
                        result["weighted_gain"]
                        - float(old.cell_weighted_prior_minus_model)
                    ) < 1e-10,
                    "Stage20R weighted gain replay mismatch"
                )

            if name not in (
                "all_523",
                "batterylife_205",
            ):
                continue

            for domain in sorted(cohort.dataset_id.unique()):
                reduced = cohort[
                    cohort.dataset_id != domain
                ]

                smaller = metrics(reduced)

                leave_out.append({
                    "policy": name,
                    "model": model,
                    "removed_domain": domain,
                    "removed_cells": int(
                        (cohort.dataset_id == domain).sum()
                    ),
                    "remaining_cells": smaller["cells"],
                    "remaining_domains": smaller["domains"],
                    "macro_gain_without_domain": smaller["macro_gain"],
                    "weighted_gain_without_domain": smaller["weighted_gain"],
                    "macro_change_from_full": (
                        smaller["macro_gain"]
                        - result["macro_gain"]
                    ),
                    "weighted_change_from_full": (
                        smaller["weighted_gain"]
                        - result["weighted_gain"]
                    ),
                })

    summary = pd.DataFrame(summaries)
    influence = pd.DataFrame(leave_out)

    domain_stability = []

    for (policy, model), group in influence.groupby(
        ["policy", "model"]
    ):
        domain_stability.append({
            "policy": policy,
            "model": model,
            "leave_one_domain_out_cases": len(group),
            "macro_gain_positive_cases": int(
                (group.macro_gain_without_domain > 0).sum()
            ),
            "macro_gain_negative_cases": int(
                (group.macro_gain_without_domain < 0).sum()
            ),
            "macro_gain_min": float(
                group.macro_gain_without_domain.min()
            ),
            "macro_gain_max": float(
                group.macro_gain_without_domain.max()
            ),
            "weighted_gain_positive_cases": int(
                (group.weighted_gain_without_domain > 0).sum()
            ),
            "weighted_gain_negative_cases": int(
                (group.weighted_gain_without_domain < 0).sum()
            ),
            "weighted_gain_min": float(
                group.weighted_gain_without_domain.min()
            ),
            "weighted_gain_max": float(
                group.weighted_gain_without_domain.max()
            ),
        })

    output = {
        "stage": "20S",
        "status": "FIXED_PREDICTION_DOMAIN_AUDIT",
        "source": "Stage20R 24-fold exactly replayed predictions",
        "stage20r_prediction_sha256": filehash(PRED),
        "stage20r_summary_sha256": filehash(PREVIOUS),
        "quality_flags_sha256": filehash(FLAGS),
        "results": summaries,
        "leave_one_domain_out_stability": domain_stability,
        "limitations": [
            "All calculations reuse the same historical fitted-model predictions.",
            "Removing test domains does not remove those domains from the training folds of other held-out datasets.",
            "The minimum-five-cell rule is an exploratory scoring sensitivity, not a validation standard.",
            "CALB and UL_PUR have one and two retained evaluation cells in the three-flag-filtered BatteryLife cohort.",
            "The BatteryML_main 318 cells were not source-quality-certified.",
            "No causal, statistical-significance or prospective claim is established."
        ],
    }

    summary.to_csv(
        OUT_MAIN, index=False, mode="x"
    )

    influence.to_csv(
        OUT_FOLDS, index=False, mode="x"
    )

    with OUT_REPORT.open("x", encoding="utf-8") as f:
        json.dump(
            output, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")

    print("=== STAGE20S ROBUSTNESS SUMMARY ===")

    for row in summaries:
        print(
            "POLICY", row["policy"],
            "MODEL", row["model"],
            "N", row["cells"],
            "DOMAINS", row["domains"],
            "MACRO_GAIN", round(row["macro_gain"], 9),
            "WEIGHTED_GAIN", round(row["weighted_gain"], 9)
        )

    print("=== STAGE20S DOMAIN LEVERAGE ===")

    for item in domain_stability:
        print(
            "POLICY", item["policy"],
            "MODEL", item["model"],
            "CASES", item["leave_one_domain_out_cases"],
            "MACRO_POSITIVE",
            item["macro_gain_positive_cases"],
            "MACRO_NEGATIVE",
            item["macro_gain_negative_cases"],
            "WEIGHTED_POSITIVE",
            item["weighted_gain_positive_cases"],
            "WEIGHTED_NEGATIVE",
            item["weighted_gain_negative_cases"],
            "MACRO_RANGE",
            (
                round(item["macro_gain_min"], 9),
                round(item["macro_gain_max"], 9)
            )
        )

    print("STAGE20S_DOMAIN_AUDIT_COMPLETE=YES")
    print("REPORT", OUT_REPORT)

if __name__ == "__main__":
    main()
