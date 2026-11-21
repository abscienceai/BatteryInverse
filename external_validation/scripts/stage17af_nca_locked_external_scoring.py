#!/usr/bin/env python3
"""Score locked NCA predictions against measured last-cycle capacity.

No model training, retuning, or new inference.
Measured last cycle is not automatically physical EOL.
"""
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
EV = ROOT / "external_validation"
R = EV / "reports"
M = EV / "models"

AE = R / "stage17ae_nca_preoutcome_model_lock_report.json"
MANIFEST = R / "stage17ae_nca_preoutcome_sha256_manifest.csv"
PRED = R / "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"
RF = M / "stage17ae_original567_random_forest_fitted.joblib"
ET = M / "stage17ae_original567_extra_trees_fitted.joblib"

LOCK = R / "stage17y_nca_pre_extraction_protocol.json"
ZREP = R / "stage17z_nca_raw_cycle_screen_report.json"
AREP = R / "stage17aa_nca_preinference_gate_report.json"
ZC = R / "stage17z_nca_observed_cycle_segments.csv"
AA = R / "stage17aa_nca_66cell_preinference_gate.csv"
AC = R / "stage17ac_nca_30cell_PROVISIONAL_features.csv"

TRAIN = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)

OUT_CELL = R / "stage17af_nca_30cell_last_cycle_diagnostic_and_locked_predictions.csv"
OUT_SCORE = R / "stage17af_nca_locked_external_metric_scores.csv"
OUT_BOOT = R / "stage17af_nca_paired_cell_bootstrap_descriptive.csv"
OUT_REPORT = R / "stage17af_nca_unsealed_outcome_evidence.json"

OUTPUTS = (OUT_CELL, OUT_SCORE, OUT_BOOT, OUT_REPORT)

EXPECTED = {
    PRED: "cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757",
    RF: "4a211926314c6bc1d665c76cafba42e8b974baf48b14a56ea635b53b8da10155",
    ET: "f1b59e19bcd86c5a9845ff955551dc8306b35870d7b4fdb1c934fccf6a509eae",
    AC: "38fdd44df61c889898f74267ee26ad5637c7f3662c873cabd3fe6c9382118b66",
}

METHODS = [
    ("training_median_prior",
     "training_median_prior_prediction"),
    ("random_forest_positive_loss",
     "random_forest_positive_loss_branch"),
    ("random_forest_negative_signed",
     "random_forest_negative_signed_branch"),
    ("extra_trees_positive_loss",
     "extra_trees_positive_loss_branch"),
    ("extra_trees_negative_signed",
     "extra_trees_negative_signed_branch"),
]


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def readcsv(path, required=()):
    need(path.is_file(), f"Missing source: {path}")

    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        need(
            set(required) <= set(columns),
            f"Missing columns: {path.name}"
        )

        return list(reader)


def value(x, label):
    try:
        n = float(x)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError(
            f"Invalid numeric value {label}: {x!r}"
        )
    need(math.isfinite(n), f"Nonfinite {label}")
    return n


def optional(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError, OverflowError):
        return None


def istrue(x):
    return str(x).strip().lower() == "true"


def indexed(rows, label):
    result = {}

    for row in rows:
        key = str(row.get("cell_id") or "").strip()

        need(
            key and key not in result,
            f"Duplicate or missing cell ID in {label}: {key}"
        )

        result[key] = row

    return result


def save(path, rows):
    need(bool(rows), f"Empty result: {path}")

    with path.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def ranks(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    result = np.empty(len(a), dtype=float)

    i = 0

    while i < len(a):
        j = i + 1

        while (
            j < len(a)
            and a[order[j]] == a[order[i]]
        ):
            j += 1

        result[order[i:j]] = (i + 1 + j) / 2
        i = j

    return result


def correlation(y, p):
    if (
        len(y) < 3
        or np.ptp(y) < 1e-13
        or np.ptp(p) < 1e-13
    ):
        return None

    return float(
        np.corrcoef(ranks(y), ranks(p))[0, 1]
    )


def metric(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    need(
        len(y) == len(p)
        and len(y) > 0
        and np.isfinite(y).all()
        and np.isfinite(p).all(),
        "Invalid metric arrays"
    )

    error = p - y
    sst = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": (
            float(1 - np.sum(error ** 2) / sst)
            if sst > 1e-24 else None
        ),
        "signed_bias_pred_minus_measured":
            float(np.mean(error)),
        "overprediction_cells": int(np.sum(error > 0)),
        "underprediction_cells": int(np.sum(error < 0)),
        "prediction_median": float(np.median(p)),
        "measured_target_median": float(np.median(y)),
        "spearman_DIAGNOSTIC": correlation(y, p),
    }


def verify_locked_inputs():
    required = (
        AE, MANIFEST, PRED, RF, ET,
        LOCK, ZREP, AREP, ZC, AA, AC, TRAIN
    )

    for path in required:
        need(path.is_file(), f"Missing input: {path}")

    for path in OUTPUTS:
        need(
            not path.exists(),
            f"Refusing overwrite: {path}"
        )

    for path, expected in EXPECTED.items():
        need(
            sha(path) == expected,
            f"Pre-outcome SHA256 mismatch: {path.name}"
        )

    ae = json.loads(AE.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    z = json.loads(ZREP.read_text(encoding="utf-8"))
    aa = json.loads(AREP.read_text(encoding="utf-8"))

    need(
        ae.get("training_cells") == 567
        and ae.get("ISU_replay_anchor_cells") == 242
        and ae.get("NCA_provisional_cells") == 30
        and ae.get("NCA_terminal_labels_accessed") is False
        and ae.get(
            "NCA_predictions_locked_before_terminal_outcome_review"
        ) is True
        and ae.get(
            "original_Stage16A_make_pipeline_used"
        ) is True,
        "Stage17AE provenance does not pass"
    )

    need(
        lock.get("source_doi") == "10.5281/zenodo.6405084"
        and lock.get("training_567_sha256") == sha(TRAIN),
        "Original prospective lock mismatch"
    )

    need(
        z.get("cells_in_archive") == 66
        and aa.get("physical_cells") == 66
        and aa.get(
            "quality_counts", {}
        ).get("strict_first100_gate_DIAGNOSTIC") == 30,
        "Original NCA cohort changed"
    )

    for rel, expected in ae.get("input_sha256", {}).items():
        part = Path(rel)

        need(
            not part.is_absolute()
            and ".." not in part.parts,
            f"Unsafe path: {rel}"
        )

        path = (ROOT / part).resolve()

        need(
            path.is_relative_to(ROOT)
            and path.is_file()
            and sha(path) == expected,
            f"Locked input changed: {rel}"
        )

    frozen = ae.get(
        "frozen_prediction_and_model_sha256", {}
    )

    for path in (PRED, RF, ET):
        rel = path.relative_to(ROOT).as_posix()

        need(
            frozen.get(rel) == sha(path),
            f"Model or prediction changed: {rel}"
        )

    manifest = readcsv(
        MANIFEST, ["file", "sha256"]
    )

    need(
        len(manifest) == len({
            r["file"] for r in manifest
        }),
        "Duplicate manifest entries"
    )

    for row in manifest:
        rel = Path(row["file"])

        need(
            not rel.is_absolute()
            and ".." not in rel.parts,
            "Unsafe manifest path"
        )

        path = (ROOT / rel).resolve()

        need(
            path.is_relative_to(ROOT)
            and path.is_file()
            and sha(path) == row["sha256"],
            f"Manifest verification failed: {rel}"
        )

    for rel, expected in aa.get(
        "source_sha256", {}
    ).items():
        part = Path(rel)

        need(
            not part.is_absolute()
            and ".." not in part.parts,
            f"Unsafe Stage17AA path: {rel}"
        )

        path = (ROOT / part).resolve()

        need(
            path.is_relative_to(ROOT)
            and path.is_file()
            and sha(path) == expected,
            f"Stage17AA source changed: {rel}"
        )

    return ae, lock, z


def main():
    ae, lock, z = verify_locked_inputs()

    predictions = indexed(
        readcsv(
            PRED,
            ["cell_id", "program"]
            + [col for _, col in METHODS]
            + ["source_feature_status"]
        ),
        "locked predictions"
    )

    gates = indexed(
        readcsv(
            AA,
            [
                "cell_id", "program",
                "strict_first100_gate_DIAGNOSTIC",
                "stage17z_recording_ambiguity"
            ]
        ),
        "pre-inference gates"
    )

    inputs = indexed(
        readcsv(
            AC,
            [
                "cell_id",
                "feature_status",
                "cycle1_capacity_mAh_DIAGNOSTIC",
                "cycle100_capacity_mAh_DIAGNOSTIC"
            ]
        ),
        "locked NCA early features"
    )

    segments = readcsv(
        ZC,
        [
            "cell_id",
            "instrument_cycle",
            "segment_index",
            "discharge_max_mAh_DIAGNOSTIC",
            "voltage_min_V",
            "voltage_max_V",
            "q_large_decreases",
            "time_backward_steps",
            "q_negative_rows",
            "rows"
        ]
    )

    need(
        len(predictions) == 30
        and len(inputs) == 30
        and len(gates) == 66
        and len(segments) == z["output_cycle_rows"],
        "Unexpected cohort sizes"
    )

    selected = {
        cid for cid, row in gates.items()
        if istrue(
            row["strict_first100_gate_DIAGNOSTIC"]
        )
    }

    need(
        set(predictions) == set(inputs) == selected,
        "Locked 30-cell roster mismatch"
    )

    for cid in selected:
        need(
            predictions[cid]["source_feature_status"]
            == "PROVISIONAL_NOT_CERTIFIED"
            and inputs[cid]["feature_status"]
            == "PROVISIONAL_NOT_CERTIFIED"
            and predictions[cid]["program"]
            == gates[cid]["program"]
            and not istrue(
                gates[cid]["stage17z_recording_ambiguity"]
            ),
            f"Pre-inference cell mismatch: {cid}"
        )

    bycell = defaultdict(list)

    for row in segments:
        bycell[row["cell_id"]].append(row)

    need(
        set(bycell) == set(gates),
        "Raw cycle cohort mismatch"
    )

    # Only now inspect the last recorded capacity.
    records = []

    for cid in sorted(predictions):
        blocks = sorted(
            bycell[cid],
            key=lambda r: int(
                value(r["segment_index"], "segment index")
            )
        )

        cycle_ids = [
            int(value(b["instrument_cycle"], "cycle"))
            for b in blocks
        ]

        need(
            len(set(cycle_ids)) == len(cycle_ids),
            f"Repeated instrument cycle: {cid}"
        )

        first = [
            b for b in blocks
            if int(value(b["instrument_cycle"], "cycle")) == 1
        ]

        hundred = [
            b for b in blocks
            if int(value(b["instrument_cycle"], "cycle")) == 100
        ]

        need(
            len(first) == len(hundred) == 1,
            f"Missing first/100th cycle: {cid}"
        )

        last = blocks[-1]

        last_cycle = int(
            value(last["instrument_cycle"], "last cycle")
        )

        q1 = value(
            first[0]["discharge_max_mAh_DIAGNOSTIC"],
            "Q1"
        )

        q100 = value(
            hundred[0]["discharge_max_mAh_DIAGNOSTIC"],
            "Q100"
        )

        need(
            q1 > 0 and q100 > 0,
            f"Invalid early capacities: {cid}"
        )

        need(
            abs(
                q1 - value(
                    inputs[cid][
                        "cycle1_capacity_mAh_DIAGNOSTIC"
                    ],
                    "locked Q1"
                )
            ) < 1e-8,
            f"Q1 changed: {cid}"
        )

        need(
            abs(
                q100 - value(
                    inputs[cid][
                        "cycle100_capacity_mAh_DIAGNOSTIC"
                    ],
                    "locked Q100"
                )
            ) < 1e-8,
            f"Q100 changed: {cid}"
        )

        qlast = optional(
            last["discharge_max_mAh_DIAGNOSTIC"]
        )

        vmin = optional(last["voltage_min_V"])
        vmax = optional(last["voltage_max_V"])

        nrows = int(value(last["rows"], "last rows"))

        anomaly = any(
            int(value(last[field], field)) > 0
            for field in [
                "q_large_decreases",
                "time_backward_steps",
                "q_negative_rows"
            ]
        )

        # Quality screen does not establish full
        # discharge completion or physical EOL.
        quality = bool(
            last_cycle > 100
            and qlast is not None
            and qlast > 0
            and vmin is not None
            and 2.55 <= vmin <= 2.75
            and vmax is not None
            and vmax > vmin
            and nrows >= 10
            and not anomaly
        )

        target = (
            qlast / q1
            if qlast is not None and qlast > 0
            else None
        )

        result = {
            "cell_id": cid,
            "program": predictions[cid]["program"],
            "first_cycle_capacity_mAh": q1,
            "cycle100_capacity_mAh": q100,
            "last_instrument_cycle": last_cycle,
            "last_cycle_capacity_mAh": qlast,
            "last_cycle_voltage_min_V": vmin,
            "last_cycle_voltage_max_V": vmax,
            "last_cycle_rows": nrows,
            "last_cycle_counter_clock_anomaly": anomaly,
            "last_cycle_quality_screen_pass": quality,
            "measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL":
                target,
            "physical_EOL_certified": False,
            "training_endpoint_equivalence_certified":
                False,
        }

        for method, column in METHODS:
            result[method] = value(
                predictions[cid][column],
                f"{cid}/{method}"
            )

        records.append(result)

    valid = [
        row for row in records
        if row[
            "measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL"
        ] is not None
    ]

    qualified = [
        row for row in valid
        if row["last_cycle_quality_screen_pass"]
    ]

    all30 = len(valid) == 30

    primary_name = (
        "ALL30_LAST_RECORDED_DIAGNOSTIC"
        if all30
        else "PRESELECTED_VALID_LAST_CAPACITY_INCOMPLETE30"
    )

    cohorts = [(primary_name, valid)]

    if len(qualified) >= 5 and len(qualified) < len(valid):
        cohorts.append(
            ("VOLTAGE_QUALITY_SCREEN_SECONDARY", qualified)
        )

    byprogram = defaultdict(list)

    for row in valid:
        byprogram[row["program"]].append(row)

    for program, group in sorted(byprogram.items()):
        if len(group) >= 3:
            cohorts.append(
                ("PROGRAM_" + program, group)
            )

    scores = []
    bootstrap = []

    for cohort_name, group in cohorts:
        y = np.array([
            r["measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL"]
            for r in group
        ])

        prior = np.array([
            r["training_median_prior"]
            for r in group
        ])

        baseline_mae = float(
            np.mean(np.abs(prior - y))
        )

        for method, _ in METHODS:
            p = np.array([
                r[method] for r in group
            ])

            stats = metric(y, p)

            gain = baseline_mae - stats["MAE"]

            scores.append({
                "endpoint":
                    "LAST_INSTRUMENT_CYCLE_QMAX_OVER_Q1_NOT_EOL",
                "evaluation_cohort": cohort_name,
                "physical_cells": len(group),
                "method": method,
                **stats,
                "training_median_baseline_MAE_same_cells":
                    baseline_mae,
                "MAE_gain_baseline_minus_model": gain,
                "relative_MAE_gain": (
                    gain / baseline_mae
                    if baseline_mae > 0 else None
                ),
                "status":
                    "EXPLORATORY_NONHARMONIZED_ENDPOINT",
            })

            if (
                method == "training_median_prior"
                or len(group) < 5
            ):
                continue

            paired = (
                np.abs(prior - y)
                - np.abs(p - y)
            )

            seed = (
                20261119
                + len(group) * 37
                + sum(
                    ord(ch)
                    for ch in cohort_name + method
                ) % 10000
            )

            rng = np.random.default_rng(seed)

            indexes = rng.integers(
                0, len(group),
                size=(5000, len(group))
            )

            means = paired[indexes].mean(axis=1)

            bootstrap.append({
                "evaluation_cohort": cohort_name,
                "physical_cells": len(group),
                "method": method,
                "paired_mean_MAE_gain":
                    float(paired.mean()),
                "bootstrap_CI_lower_2p5":
                    float(np.quantile(means, .025)),
                "bootstrap_CI_upper_97p5":
                    float(np.quantile(means, .975)),
                "resamples": 5000,
                "unit": "physical_cell",
                "interval_type":
                    "DESCRIPTIVE_NOT_CONFIRMATORY",
            })

    need(
        len(scores) >= 5,
        "No evaluable measured capacity outcomes"
    )

    report = {
        "stage":
            "17AF_NCA_PRELOCKED_MEASURED_EXTERNAL_TRANSFER",
        "source_doi": lock["source_doi"],
        "original_training_cells": 567,
        "NCA_archive_cells": 66,
        "NCA_locked_test_cells": 30,
        "NCA_positive_last_cycle_cells": len(valid),
        "NCA_all30_scored": all30,
        "NCA_last_cycle_quality_screen_pass":
            len(qualified),
        "NCA_last_cycle_quality_fail_ids": [
            r["cell_id"]
            for r in records
            if not r["last_cycle_quality_screen_pass"]
        ],
        "preoutcome_predictions_sha256": sha(PRED),
        "preoutcome_lock_report_sha256": sha(AE),
        "preoutcome_manifest_sha256": sha(MANIFEST),
        "raw_cycle_segments_sha256": sha(ZC),
        "first100_gate_sha256": sha(AA),
        "cohorts": [name for name, _ in cohorts],
        "scores": scores,
        "descriptive_bootstrap": bootstrap,
        "outcome_definition":
            "Last chronologically recorded instrument-cycle "
            "maximum discharge capacity divided by unique "
            "cycle-1 maximum discharge capacity.",
        "endpoint_equivalence_certified": False,
        "physical_EOL_certified": False,
        "complete_discharge_certified": False,
        "model_training": False,
        "new_model_inference": False,
        "prediction_branch_selected_posthoc": False,
        "prism_updated": False,
        "limitations": [
            "Training and NCA terminal endpoints are not harmonized.",
            "A last recorded instrument cycle is not necessarily physical EOL.",
            "Voltage-minimum screening does not prove a complete discharge.",
            "NCA features remain provisional.",
            "Both source-family fade-sign branches must be reported.",
            "The 30-cell quality-selected cohort is not representative of all 66 cells."
        ],
    }

    R.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="stage17af_", dir=R
    ) as folder:
        tmp = Path(folder)

        pcell = tmp / "cells.csv"
        pscore = tmp / "scores.csv"
        pboot = tmp / "bootstrap.csv"
        prep = tmp / "report.json"

        save(pcell, records)
        save(pscore, scores)
        save(pboot, bootstrap)

        prep.write_text(
            json.dumps(
                report, indent=2, ensure_ascii=False
            ),
            encoding="utf-8"
        )

        need(
            not any(path.exists() for path in OUTPUTS),
            "Output appeared during processing"
        )

        os.replace(pcell, OUT_CELL)
        os.replace(pscore, OUT_SCORE)
        os.replace(pboot, OUT_BOOT)
        os.replace(prep, OUT_REPORT)

    print(
        "=== STAGE 17AF NCA LOCKED EXTERNAL MEASUREMENTS ==="
    )
    print("TRAINING_CELLS=567 NCA_PRELOCKED_CELLS=30")

    print(
        "PREOUTCOME_PREDICTION_SHA256_VERIFIED=YES"
    )

    print(
        "ALL30_LAST_CYCLE_POSITIVE_OUTCOME_COUNT=",
        len(valid)
    )

    print(
        "LAST_CYCLE_QUALITY_SCREEN_PASS_COUNT=",
        len(qualified)
    )

    print(
        "LAST_CYCLE_QUALITY_SCREEN_FAILURES=",
        json.dumps(
            report["NCA_last_cycle_quality_fail_ids"]
        )
    )

    print(
        "LOCKED_MODEL_EXTERNAL_METRICS=",
        json.dumps(
            scores, ensure_ascii=False
        )
    )

    print(
        "PAIRED_BOOTSTRAP_DESCRIPTIVE=",
        json.dumps(
            bootstrap, ensure_ascii=False
        )
    )

    print("PER_CELL_CSV=", OUT_CELL)
    print("METRIC_SCORES_CSV=", OUT_SCORE)
    print("BOOTSTRAP_CSV=", OUT_BOOT)
    print("REPORT_JSON=", OUT_REPORT)

    print(
        "MODEL_TRAINING=NO "
        "PREDICTIONS_CHANGED=NO "
        "PRISM_UPDATED=NO"
    )

    print(
        "PHYSICAL_EOL_CERTIFIED=NO "
        "TARGET_EQUIVALENCE_CERTIFIED=NO"
    )

    print("=== END STAGE 17AF ===")


if __name__ == "__main__":
    main()
