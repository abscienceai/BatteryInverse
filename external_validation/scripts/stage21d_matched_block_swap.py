#!/usr/bin/env python3
"""Stage21D: matched within-domain full-block permutation audit.

For a fixed row-wise estimator, permuting complete feature
vectors among test cells is equivalent to permuting its
already-computed predictions among those same test cells.

No model fitting and no held-out target modification.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
REP = ROOT / "external_validation/reports"

INPUT = REP / "stage21b_cell_predictions.csv"
SUMMARY_B = REP / "stage21b_representation_summary.csv"
REPORT_B = REP / "stage21b_controlled_family_report.json"
REPORT_C = REP / "stage21c_mechanism_report.json"

OUT_SUMMARY = REP / "stage21d_block_swap_summary.csv"
OUT_DOMAIN = REP / "stage21d_block_swap_domains.csv"
OUT_DRAWS = REP / "stage21d_block_swap_draws.csv"
OUT_REPORT = REP / "stage21d_block_swap_report.json"

OUTPUTS = [
    OUT_SUMMARY, OUT_DOMAIN,
    OUT_DRAWS, OUT_REPORT
]

VARIANTS = [
    "original7",
    "harmonized7",
    "no_fade6",
    "harmonized7_plus_family",
    "family_only",
]

POLICIES = [
    ("all_567", False, False, 567, 12),
    ("all_exclude44_523", False, True, 523, 12),
    ("batterylife_249", True, False, 249, 8),
    ("batterylife_exclude44_205", True, True, 205, 8),
]

N_PERM = 100
SEED = 20261122

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(
            lambda: stream.read(1048576), b""
        ):
            h.update(part)
    return h.hexdigest()

def main():
    require(
        all(not p.exists() for p in OUTPUTS),
        "Stage21D outputs already exist; refusing overwrite"
    )

    b = json.loads(REPORT_B.read_text())
    c = json.loads(REPORT_C.read_text())

    require(
        b["status"] == "CONTROLLED_REPRESENTATION_LODO_COMPLETE",
        "Stage21B status invalid"
    )
    require(
        b["max_historical_model_mae_difference"] < 1e-10,
        "Stage21B historical replay mismatch"
    )
    require(
        c["status"] == "PAIRED_DOMAIN_AUDIT_VERIFIED",
        "Stage21C not verified"
    )
    require(
        c["stage21b_prediction_sha256"] == sha(INPUT),
        "Stage21B prediction file changed"
    )
    require(
        c["historical_cell_replay_max_prediction_drift"] < 1e-10,
        "Stage21C cell replay mismatch"
    )

    df = pd.read_csv(
        INPUT,
        dtype={
            "dataset_id": str,
            "cell_id": str,
            "variant": str,
            "model": str,
        }
    )
    old_summary = pd.read_csv(SUMMARY_B)

    require(len(df) == 5670, "Incomplete prediction file")
    require(set(df.variant) == set(VARIANTS), "Variant mismatch")
    require(set(df.model) == {"RF", "ET"}, "Model mismatch")
    require(
        not df.duplicated([
            "dataset_id", "cell_id",
            "model", "variant"
        ]).any(),
        "Duplicate cell-model-variant"
    )
    require(
        np.isfinite(
            df[["y_true", "y_pred"]].to_numpy(float)
        ).all(),
        "Nonfinite predictions or targets"
    )

    summary_rows = []
    domain_rows = []
    draw_rows = []

    for policy_index, (
        policy, bl_only, filter_flags,
        expected_cells, expected_domains
    ) in enumerate(POLICIES):

        for model in ("RF", "ET"):
            cohort = df[df.model == model].copy()

            if bl_only:
                cohort = cohort[
                    cohort.source_family
                    == "BatteryLife_selected"
                ]

            if filter_flags:
                cohort = cohort[
                    cohort.quality_flag == 0
                ]

            require(
                len(cohort) == expected_cells * len(VARIANTS),
                "Policy count mismatch: " + policy
            )

            domain_data = {}
            domains = sorted(cohort.dataset_id.unique())

            require(
                len(domains) == expected_domains,
                "Domain count mismatch: " + policy
            )

            for domain_index, domain in enumerate(domains):
                group = cohort[
                    cohort.dataset_id == domain
                ]

                original = group[
                    group.variant == "original7"
                ].sort_values("cell_id")

                ids = original.cell_id.to_numpy(str)
                y = original.y_true.to_numpy(float)
                n = len(y)

                require(n >= 1, "Empty held-out domain")
                require(
                    len(np.unique(ids)) == n,
                    "Duplicate domain cell"
                )
                require(
                    original.source_family.nunique() == 1,
                    "Mixed source family in domain"
                )

                # Common permutations across models and
                # variants for each policy/domain.
                rng = np.random.default_rng(
                    SEED
                    + 10007 * policy_index
                    + 101 * domain_index
                )

                permutations = np.stack([
                    rng.permutation(n)
                    for _ in range(N_PERM)
                ])

                for variant in VARIANTS:
                    current = group[
                        group.variant == variant
                    ].sort_values("cell_id")

                    require(
                        len(current) == n
                        and np.array_equal(
                            current.cell_id.to_numpy(str), ids
                        ),
                        "Cell order mismatch"
                    )
                    require(
                        np.allclose(
                            current.y_true.to_numpy(float),
                            y,
                            rtol=0, atol=1e-12
                        ),
                        "Target mismatch across variants"
                    )

                    predictions = current.y_pred.to_numpy(float)

                    original_mae = float(
                        np.mean(np.abs(predictions - y))
                    )

                    # Exact full-feature-block-swap equivalent
                    # for a fitted row-wise prediction pipeline.
                    swapped_mae = np.mean(
                        np.abs(
                            predictions[permutations]
                            - y[None, :]
                        ),
                        axis=1
                    )

                    deltas = swapped_mae - original_mae

                    require(
                        np.isfinite(swapped_mae).all(),
                        "Nonfinite permuted MAE"
                    )

                    # A source-family-only model gives the same
                    # prediction to every cell of the held-out
                    # dataset: swapping assignments does nothing.
                    if variant == "family_only":
                        require(
                            np.max(np.abs(deltas)) < 1e-12,
                            "Family-only negative control failed"
                        )

                    domain_data[(domain, variant)] = {
                        "n": n,
                        "original": original_mae,
                        "swapped": swapped_mae,
                    }

                    domain_rows.append({
                        "policy": policy,
                        "model": model,
                        "variant": variant,
                        "dataset": domain,
                        "cells": n,
                        "original_mae": original_mae,
                        "swapped_mae_mean": float(
                            swapped_mae.mean()
                        ),
                        "mae_increase_mean": float(
                            deltas.mean()
                        ),
                        "positive_draws": int(
                            np.sum(deltas > 1e-12)
                        ),
                        "permutation_draws": N_PERM,
                    })

            for variant in VARIANTS:
                entries = [
                    domain_data[(domain, variant)]
                    for domain in domains
                ]

                weights = np.array(
                    [entry["n"] for entry in entries],
                    dtype=float
                )

                original_maes = np.array(
                    [entry["original"] for entry in entries]
                )

                swapped = np.stack([
                    entry["swapped"] for entry in entries
                ])

                original_macro = float(
                    original_maes.mean()
                )
                original_weighted = float(
                    np.average(
                        original_maes,
                        weights=weights
                    )
                )

                perm_macro = swapped.mean(axis=0)
                perm_weighted = np.average(
                    swapped,
                    weights=weights,
                    axis=0
                )

                macro_delta = perm_macro - original_macro
                weighted_delta = (
                    perm_weighted - original_weighted
                )

                historical = old_summary[
                    (old_summary.policy == policy)
                    & (old_summary.model == model)
                    & (old_summary.variant == variant)
                ]

                require(
                    len(historical) == 1,
                    "Missing Stage21B summary"
                )

                require(
                    abs(
                        original_macro
                        - float(
                            historical.iloc[0].model_macro_mae
                        )
                    ) < 1e-10,
                    "Original macro-MAE replay mismatch"
                )

                require(
                    abs(
                        original_weighted
                        - float(
                            historical.iloc[0].model_weighted_mae
                        )
                    ) < 1e-10,
                    "Original weighted MAE replay mismatch"
                )

                for k in range(N_PERM):
                    draw_rows.append({
                        "policy": policy,
                        "model": model,
                        "variant": variant,
                        "draw_id": k,
                        "macro_delta": float(macro_delta[k]),
                        "weighted_delta": float(weighted_delta[k]),
                    })

                summary_rows.append({
                    "policy": policy,
                    "model": model,
                    "variant": variant,
                    "cells": expected_cells,
                    "domains": expected_domains,
                    "original_macro_mae": original_macro,
                    "swapped_macro_mae_mean": float(
                        perm_macro.mean()
                    ),
                    "macro_mae_increase_mean": float(
                        macro_delta.mean()
                    ),
                    "macro_delta_p025": float(
                        np.percentile(macro_delta, 2.5)
                    ),
                    "macro_delta_p975": float(
                        np.percentile(macro_delta, 97.5)
                    ),
                    "positive_macro_draws": int(
                        np.sum(macro_delta > 1e-12)
                    ),
                    "original_weighted_mae": original_weighted,
                    "swapped_weighted_mae_mean": float(
                        perm_weighted.mean()
                    ),
                    "weighted_mae_increase_mean": float(
                        weighted_delta.mean()
                    ),
                    "positive_weighted_draws": int(
                        np.sum(weighted_delta > 1e-12)
                    ),
                })

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(domain_rows)
    draws = pd.DataFrame(draw_rows)

    require(len(summary) == 40, "Expected 40 summary rows")
    require(
        len(draws) == 40 * N_PERM,
        "Expected 4000 permutation aggregate records"
    )

    report = {
        "stage": "21D",
        "status": "MATCHED_BLOCK_SWAP_VERIFIED",
        "prediction_sha256": sha(INPUT),
        "stage21b_summary_sha256": sha(SUMMARY_B),
        "seed": SEED,
        "permutations_per_domain": N_PERM,
        "scope": (
            "Within-domain complete-feature-block swap "
            "via mathematically equivalent prediction "
            "reassignment for fixed row-wise estimators"
        ),
        "summary_rows": len(summary),
        "domain_rows": len(details),
        "draw_rows": len(draws),
        "limitations": [
            "No model was fitted or altered in Stage21D.",
            "Target outcomes remain fixed.",
            "Permutation indices are shared across variants and models within each domain and evaluation policy.",
            "Quality-filtered cohorts are permuted only among retained evaluation cells.",
            "Prediction reassignment is equivalent to full-block feature permutation for fixed row-wise inference.",
            "Family-only prediction invariance is a negative control.",
            "Repeated permutations are not independent physical experiments.",
            "Permutation percentiles are not population confidence intervals.",
            "The procedure tests within-domain cell-prediction alignment, not causal feature importance or prospective transfer.",
            "Small evaluation domains have limited permutation resolution.",
        ]
    }

    summary.to_csv(OUT_SUMMARY, index=False, mode="x")
    details.to_csv(OUT_DOMAIN, index=False, mode="x")
    draws.to_csv(OUT_DRAWS, index=False, mode="x")

    with OUT_REPORT.open("x", encoding="utf-8") as f:
        json.dump(
            report, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")

    print("=== STAGE21D MATCHED BLOCK-SWAP RESULTS ===")

    for _, row in summary.iterrows():
        print(
            "POLICY", row.policy,
            "MODEL", row.model,
            "VARIANT", row.variant,
            "N", row.cells,
            "MACRO_INCREASE",
            round(row.macro_mae_increase_mean, 9),
            "WEIGHTED_INCREASE",
            round(row.weighted_mae_increase_mean, 9),
            "MACRO_POSITIVE_DRAWS",
            row.positive_macro_draws,
            "WEIGHTED_POSITIVE_DRAWS",
            row.positive_weighted_draws,
            flush=True
        )

    print("STAGE21D_COMPLETE=YES")
    print("SUMMARY", OUT_SUMMARY)
    print("DOMAIN_DETAILS", OUT_DOMAIN)
    print("PERMUTATION_DRAWS", OUT_DRAWS)
    print("REPORT", OUT_REPORT)

if __name__ == "__main__":
    main()
