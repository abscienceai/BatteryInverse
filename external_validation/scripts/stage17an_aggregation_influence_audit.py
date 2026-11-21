#!/usr/bin/env python3
"""Stage17AN: no-fit LODO aggregation and domain influence audit.

Uses previously computed Stage17AM fold scores.
Does not train models or modify the manuscript.
"""
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
R = ROOT / "external_validation/reports"

FOLDS = R / "stage17am_three_condition_fold_score_evidence.csv"
AGG = R / "stage17am_macro_and_cell_weighted_results.csv"
AM = R / "stage17am_exploratory_ablation_consolidated_evidence.json"
PRED = R / "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"
TRAIN = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"

OUT_DOMAIN = R / "stage17an_lodo_domain_weight_contributions.csv"
OUT_BOOT = R / "stage17an_domain_resampling_sensitivity.csv"
OUT_DELETE = R / "stage17an_leave_one_domain_out_influence.csv"
OUT_JSON = R / "stage17an_aggregation_reversal_provenance.json"
OUT_LEDGER = R / "stage17an_deferred_prism_ledger_addendum.md"

OUTS = (OUT_DOMAIN, OUT_BOOT, OUT_DELETE, OUT_JSON, OUT_LEDGER)

LOCK = (
    "cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757"
)

B = 10000
MODELS = ("random_forest", "extra_trees")
CONDITIONS = ("original7", "harmonized7", "no_fade6")


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def hashfile(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)
    return h.hexdigest()


def readcsv(p, cols):
    need(p.is_file(), f"Missing CSV: {p}")
    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        need(
            set(cols).issubset(reader.fieldnames or []),
            f"Missing columns in {p.name}"
        )
        return list(reader)


def num(v, label):
    x = float(v)
    need(np.isfinite(x), f"Invalid {label}")
    return x


def intnum(v, label):
    x = num(v, label)
    need(
        x >= 0 and x.is_integer(),
        f"Invalid count {label}"
    )
    return int(x)


def writecsv(path, rows):
    need(bool(rows), f"No records: {path}")
    with path.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(vals, weights, draws):
    d = vals[draws]
    w = weights[draws]

    macro = np.mean(d, axis=1)
    weighted = np.sum(d * w, axis=1) / np.sum(w, axis=1)

    def interval(z):
        return {
            "p025": float(np.quantile(z, 0.025)),
            "p975": float(np.quantile(z, 0.975)),
            "fraction_positive": float(np.mean(z > 0))
        }

    return interval(macro), interval(weighted)


def main():
    sources = (FOLDS, AGG, AM, PRED, TRAIN)

    for p in sources:
        need(p.is_file(), f"Missing source: {p}")

    for p in OUTS:
        need(
            not p.exists(),
            f"Refusing overwrite: {p}"
        )

    report = json.loads(
        AM.read_text(encoding="utf-8")
    )

    need(
        report.get("stage")
        == "17AM_FROZEN_SENSITIVITY_SYNTHESIS"
        and report.get("original_training_cells") == 567
        and report.get("no_new_model_fitting") is True
        and report.get(
            "original_models_or_predictions_modified"
        ) is False,
        "Stage17AM provenance invalid"
    )

    need(
        hashfile(PRED) == LOCK
        == report.get("NCA_predictions_sha256"),
        "Frozen NCA prediction hash mismatch"
    )

    for rel, expected in report.get(
        "sources_sha256", {}
    ).items():
        p = Path(rel)

        need(
            not p.is_absolute()
            and ".." not in p.parts,
            "Unsafe Stage17AM source path"
        )

        resolved = (ROOT / p).resolve()

        need(
            resolved.is_relative_to(ROOT)
            and resolved.is_file()
            and hashfile(resolved) == expected,
            f"Stage17AM source changed: {rel}"
        )

    training = readcsv(
        TRAIN,
        ["cell_id", "dataset_id", "source_family"]
    )

    need(
        len(training) == 567
        and len({
            row["cell_id"] for row in training
        }) == 567,
        "Original training roster changed"
    )

    domaincounts = Counter(
        row["dataset_id"] for row in training
    )

    familycounts = Counter(
        row["source_family"] for row in training
    )

    need(
        len(domaincounts) == 12
        and familycounts == {
            "BatteryML_main": 318,
            "BatteryLife_selected": 249
        },
        "Original source roster differs"
    )

    rows = readcsv(
        FOLDS,
        [
            "split_type", "heldout", "model",
            "test_n", "train_n",
            "prior_MAE", "original7_MAE",
            "harmonized7_MAE", "no_fade6_MAE"
        ]
    )

    aggs = readcsv(
        AGG,
        [
            "split_type", "model", "subset",
            "folds", "test_cells_total"
        ] + [
            f"{a}_MAE_{v}"
            for a in ("macro", "cell_weighted")
            for v in ("prior",) + CONDITIONS
        ]
    )

    need(
        len(rows) == 28 and len(aggs) == 6,
        "Stage17AM result count mismatch"
    )

    bykey = {}

    for row in rows:
        key = (
            row["split_type"],
            row["heldout"],
            row["model"]
        )

        need(
            key not in bykey,
            "Duplicate Stage17AM fold"
        )

        n = intnum(row["test_n"], "test_n")
        nt = intnum(row["train_n"], "train_n")

        need(
            n + nt == 567,
            "Fold size mismatch"
        )

        expected = (
            domaincounts
            if row["split_type"] == "DATASET_LODO"
            else familycounts
        )

        need(
            row["split_type"] in (
                "DATASET_LODO",
                "SOURCE_FAMILY_HOLDOUT"
            )
            and row["heldout"] in expected
            and n == expected[row["heldout"]]
            and row["model"] in MODELS,
            "Fold membership mismatch"
        )

        values = {
            condition: num(
                row[condition + "_MAE"],
                condition
            )
            for condition in ("prior",) + CONDITIONS
        }

        need(
            all(x >= 0 for x in values.values()),
            "Negative MAE"
        )

        bykey[key] = dict(n=n, **values)

    expected_grid = {
        (kind, heldout, model)
        for kind, domains in (
            ("DATASET_LODO", domaincounts),
            ("SOURCE_FAMILY_HOLDOUT", familycounts)
        )
        for heldout in domains
        for model in MODELS
    }

    need(
        set(bykey) == expected_grid,
        "Fold coverage changed"
    )

    # Independently verify all published Stage17AM aggregates.
    for aggregate in aggs:
        kind = aggregate["split_type"]
        model = aggregate["model"]
        subset = aggregate["subset"]

        need(
            subset in ("ALL", "EXCLUDE_UL_PUR"),
            "Unexpected aggregation subset"
        )

        items = [
            value
            for (k, heldout, m), value in bykey.items()
            if k == kind
            and m == model
            and (
                subset == "ALL"
                or heldout != "UL_PUR"
            )
        ]

        need(
            len(items) == intnum(
                aggregate["folds"], "aggregate folds"
            )
            and sum(x["n"] for x in items)
            == intnum(
                aggregate["test_cells_total"],
                "aggregate cells"
            ),
            "Aggregate size mismatch"
        )

        for condition in ("prior",) + CONDITIONS:
            macro = np.mean([
                x[condition] for x in items
            ])

            weighted = np.average(
                [x[condition] for x in items],
                weights=[x["n"] for x in items]
            )

            need(
                abs(
                    macro - num(
                        aggregate[
                            "macro_MAE_" + condition
                        ],
                        "macro"
                    )
                ) < 1e-10
                and abs(
                    weighted - num(
                        aggregate[
                            "cell_weighted_MAE_" + condition
                        ],
                        "weighted"
                    )
                ) < 1e-10,
                "Stage17AM arithmetic mismatch"
            )

    contributions = []
    bootrows = []
    leaveouts = []
    overview = {}

    for subset in ("ALL", "EXCLUDE_UL_PUR"):
        domains = sorted(
            d for d in domaincounts
            if subset == "ALL" or d != "UL_PUR"
        )

        count = len(domains)

        rng = np.random.default_rng(
            1701142 + (
                0 if subset == "ALL" else 1
            )
        )

        # Resample entire datasets, not individual cells.
        draws = rng.integers(
            0, count,
            size=(B, count)
        )

        for model in MODELS:
            units = [
                bykey[("DATASET_LODO", d, model)]
                for d in domains
            ]

            weights = np.array(
                [v["n"] for v in units],
                dtype=float
            )

            prior = np.array(
                [v["prior"] for v in units],
                dtype=float
            )

            totals = {}

            for condition in CONDITIONS:
                predicted_error = np.array(
                    [v[condition] for v in units],
                    dtype=float
                )

                gain = prior - predicted_error

                macro = float(
                    np.mean(gain)
                )

                weighted = float(
                    np.average(
                        gain, weights=weights
                    )
                )

                ci_macro, ci_weighted = bootstrap(
                    gain, weights, draws
                )

                bootrows.append({
                    "subset": subset,
                    "model": model,
                    "condition": condition,
                    "domain_count": count,
                    "cell_count": int(weights.sum()),
                    "macro_prior_minus_model_MAE": macro,
                    "cell_weighted_prior_minus_model_MAE":
                        weighted,
                    "macro_domain_bootstrap_p025":
                        ci_macro["p025"],
                    "macro_domain_bootstrap_p975":
                        ci_macro["p975"],
                    "macro_bootstrap_fraction_positive":
                        ci_macro["fraction_positive"],
                    "weighted_domain_bootstrap_p025":
                        ci_weighted["p025"],
                    "weighted_domain_bootstrap_p975":
                        ci_weighted["p975"],
                    "weighted_bootstrap_fraction_positive":
                        ci_weighted["fraction_positive"],
                    "bootstrap_draws": B,
                    "descriptive_not_prospective_inference":
                        True
                })

                totals[condition] = {
                    "macro": macro,
                    "weighted": weighted,
                    "descriptive_macro_ci": [
                        ci_macro["p025"],
                        ci_macro["p975"]
                    ],
                    "descriptive_weighted_ci": [
                        ci_weighted["p025"],
                        ci_weighted["p975"]
                    ]
                }

                if condition == "original7":
                    for d, v, g, weight in zip(
                        domains, units, gain, weights
                    ):
                        contributions.append({
                            "subset": subset,
                            "model": model,
                            "dataset": d,
                            "test_cells": int(weight),
                            "prior_MAE": v["prior"],
                            "original7_MAE": v["original7"],
                            "prior_minus_original7_MAE":
                                float(g),
                            "contribution_to_cell_weighted_gain":
                                float(
                                    g * weight / weights.sum()
                                ),
                            "contribution_to_macro_gain":
                                float(g / count)
                        })

                    for j, d in enumerate(domains):
                        mask = np.arange(count) != j

                        mk = float(
                            np.mean(gain[mask])
                        )

                        wt = float(
                            np.average(
                                gain[mask],
                                weights=weights[mask]
                            )
                        )

                        leaveouts.append({
                            "subset": subset,
                            "model": model,
                            "omitted_domain": d,
                            "remaining_domains": count - 1,
                            "remaining_test_cells":
                                int(weights[mask].sum()),
                            "macro_prior_minus_original7_MAE":
                                mk,
                            "weighted_prior_minus_original7_MAE":
                                wt,
                            "macro_positive_weighted_negative":
                                mk > 0 and wt < 0
                        })

            overview[
                subset + ":" + model
            ] = totals

    reversal_counts = {
        key: sum(
            row["macro_positive_weighted_negative"]
            for row in leaveouts
            if row["subset"] + ":" + row["model"] == key
        )
        for key in overview
    }

    result = {
        "stage":
            "17AN_DOMAIN_INFLUENCE_AND_AGGREGATION_REVERSAL_AUDIT",
        "purpose":
            "Descriptive aggregation and domain influence audit",
        "original_training_cells": 567,
        "dataset_holdouts": 12,
        "previous_model_fits_reused": 84,
        "new_model_fits": 0,
        "NCA_prediction_sha256": hashfile(PRED),
        "domain_resampling_draws": B,
        "domain_resampling_caveat":
            "Resampling observed dataset domains is exploratory. "
            "Intervals are descriptive and conditional on the "
            "existing fitted fold models, dataset definitions "
            "and endpoints. They do not establish prospective "
            "performance or independent laboratory replication.",
        "results": overview,
        "leave_one_domain_out_reversal_counts":
            reversal_counts,
        "new_external_predictions": False,
        "manuscript_modified": False,
        "input_sha256": {
            p.relative_to(ROOT).as_posix():
                hashfile(p)
            for p in sources
        }
    }

    ledger = """# BatteryInverse — Stage17AN deferred Prism addendum

Do not edit the manuscript now.

- Audit macro versus physical-cell-weighted LODO MAE.
- Report the full domain counts and aggregation sensitivity.
- Check whether the reversal is driven by individual domains.
- Keep source-family holdout separate from dataset LODO.
- Stages 17AK–17AN are post-outcome exploratory analyses.
- Bootstrap intervals across observed domains are descriptive.
- No new models or independent external predictions were made.
- Preserve the previously established NCA endpoint limitations.

Integrate only relevant findings in one future consolidated
Prism revision, without duplicating existing Discussion text.
"""

    with tempfile.TemporaryDirectory(
        prefix="stage17an_", dir=R
    ) as folder:
        temp = Path(folder)

        writecsv(
            temp / "domain.csv",
            contributions
        )

        writecsv(
            temp / "bootstrap.csv",
            bootrows
        )

        writecsv(
            temp / "leaveout.csv",
            leaveouts
        )

        (
            temp / "report.json"
        ).write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        (
            temp / "ledger.md"
        ).write_text(
            ledger,
            encoding="utf-8"
        )

        need(
            not any(p.exists() for p in OUTS),
            "Output appeared; refusing overwrite"
        )

        for source, dest in (
            (temp / "domain.csv", OUT_DOMAIN),
            (temp / "bootstrap.csv", OUT_BOOT),
            (temp / "leaveout.csv", OUT_DELETE),
            (temp / "report.json", OUT_JSON),
            (temp / "ledger.md", OUT_LEDGER)
        ):
            os.replace(source, dest)

    print(
        "=== STAGE17AN DOMAIN INFLUENCE AUDIT ==="
    )

    print(
        "ORIGINAL567=YES "
        "LOCKED_NCA_HASH_UNCHANGED=YES"
    )

    print(
        "MAE_GAIN_PRIOR_MINUS_MODEL_POSITIVE_BETTER=YES"
    )

    print(
        "ALL12_ORIGINAL7_RECOMPUTED=",
        json.dumps({
            k: v["original7"]
            for k, v in overview.items()
            if k.startswith("ALL:")
        })
    )

    print(
        "EXCLUDE_UL_PUR_ORIGINAL7_RECOMPUTED=",
        json.dumps({
            k: v["original7"]
            for k, v in overview.items()
            if k.startswith("EXCLUDE_UL_PUR:")
        })
    )

    print(
        "LEAVE_ONE_DATASET_OUT_REVERSAL_COUNTS=",
        json.dumps(reversal_counts)
    )

    for model in MODELS:
        top = sorted(
            (
                r for r in contributions
                if r["subset"] == "ALL"
                and r["model"] == model
            ),
            key=lambda r: abs(
                r[
                    "contribution_to_cell_weighted_gain"
                ]
            ),
            reverse=True
        )[:5]

        print(
            "TOP5_CELL_WEIGHTED_DOMAIN_CONTRIBUTIONS_"
            + model.upper() + "=",
            json.dumps(top)
        )

    print("DOMAIN_CONTRIBUTION_CSV=", OUT_DOMAIN)
    print("DOMAIN_RESAMPLING_CSV=", OUT_BOOT)
    print("LEAVE_ONE_DOMAIN_OUT_CSV=", OUT_DELETE)
    print("REPORT_JSON=", OUT_JSON)
    print("DEFERRED_PRISM_ADDENDUM=", OUT_LEDGER)

    print(
        "NEW_FITS=NO "
        "NEW_EXTERNAL_INFERENCE=NO "
        "PRISM_UPDATED=NO"
    )

    print(
        "=== END STAGE17AN ==="
    )


if __name__ == "__main__":
    main()
