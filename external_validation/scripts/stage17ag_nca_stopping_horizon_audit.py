#!/usr/bin/env python3
"""Stage17AG: descriptive NCA stopping-horizon audit.

Post-outcome diagnostic; not a prospective validation.
No model fitting, inference or changes to frozen predictions.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
R = ROOT / "external_validation/reports"

AF = R / "stage17af_nca_unsealed_outcome_evidence.json"
CELLS = R / "stage17af_nca_30cell_last_cycle_diagnostic_and_locked_predictions.csv"
CYCLES = R / "stage17z_nca_observed_cycle_segments.csv"
AE = R / "stage17ae_nca_preoutcome_model_lock_report.json"
PRED = R / "stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv"

OUT = R / "stage17ag_nca_30cell_stopping_horizon_diagnostics.csv"
REPORT = R / "stage17ag_nca_stopping_horizon_report.json"

THRESHOLDS = (0.70, 0.75, 0.80, 0.85)
EXPECTED_PRED_SHA = (
    "cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757"
)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path, required):
    need(path.is_file(), f"Missing input: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        need(
            set(required) <= set(reader.fieldnames or []),
            f"Missing columns: {path.name}"
        )
        return list(reader)


def num(value):
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError(f"Invalid numeric value: {value!r}")
    need(math.isfinite(x), "Nonfinite numeric value")
    return x


def optional_num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def describe(values):
    arr = np.asarray(values, dtype=float)
    need(len(arr) > 0 and np.isfinite(arr).all(), "Bad distribution")

    return {
        "n": int(len(arr)),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def first_crossings(capacities, threshold):
    candidates = sorted(
        c for c, ratio in capacities.items()
        if c >= 100 and ratio <= threshold
    )

    first = candidates[0] if candidates else None

    sustained = None
    for cycle in sorted(capacities):
        if cycle < 100:
            continue
        if all(
            (cycle + delta) in capacities
            and capacities[cycle + delta] <= threshold
            for delta in range(3)
        ):
            sustained = cycle
            break

    return first, sustained


def main():
    for path in (AF, CELLS, CYCLES, AE, PRED):
        need(path.is_file(), f"Missing: {path}")

    for path in (OUT, REPORT):
        need(not path.exists(), f"Refusing overwrite: {path}")

    af = json.loads(AF.read_text(encoding="utf-8"))
    ae = json.loads(AE.read_text(encoding="utf-8"))

    need(
        af.get("NCA_locked_test_cells") == 30
        and af.get("NCA_positive_last_cycle_cells") == 30
        and af.get("model_training") is False,
        "Stage17AF outcome status mismatch"
    )

    need(
        ae.get("NCA_terminal_labels_accessed") is False
        and ae.get("NCA_predictions_locked_before_terminal_outcome_review") is True,
        "Stage17AE pre-outcome status mismatch"
    )

    need(
        sha(PRED) == EXPECTED_PRED_SHA
        == af["preoutcome_predictions_sha256"],
        "Frozen predictions changed"
    )

    need(
        sha(CYCLES) == af["raw_cycle_segments_sha256"],
        "Raw cycle summary changed since Stage17AF"
    )

    rows = read_csv(
        CELLS,
        [
            "cell_id", "program",
            "first_cycle_capacity_mAh",
            "cycle100_capacity_mAh",
            "last_instrument_cycle",
            "last_cycle_capacity_mAh",
            "measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL",
            "last_cycle_quality_screen_pass"
        ]
    )

    predictions = read_csv(PRED, ["cell_id"])
    raw = read_csv(
        CYCLES,
        [
            "cell_id", "instrument_cycle",
            "segment_index",
            "discharge_max_mAh_DIAGNOSTIC"
        ]
    )

    need(len(rows) == 30, "Expected 30 external cells")

    ids = [r["cell_id"] for r in rows]
    pred_ids = [r["cell_id"] for r in predictions]

    need(
        len(set(ids)) == 30
        and len(set(pred_ids)) == 30
        and set(ids) == set(pred_ids),
        "Locked cohort identity mismatch"
    )

    by_cell = defaultdict(list)
    for row in raw:
        if row["cell_id"] in set(ids):
            by_cell[row["cell_id"]].append(row)

    need(
        set(by_cell) == set(ids),
        "Missing raw cycle summaries"
    )

    findings = []

    for row in sorted(rows, key=lambda x: x["cell_id"]):
        cid = row["cell_id"]
        q1 = num(row["first_cycle_capacity_mAh"])
        q100 = num(row["cycle100_capacity_mAh"])
        last_cycle = int(num(row["last_instrument_cycle"]))
        qlast = num(row["last_cycle_capacity_mAh"])
        last_ratio = num(
            row["measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL"]
        )

        need(q1 > 0, f"Invalid Q1: {cid}")
        need(
            abs(last_ratio - qlast / q1) < 1e-9,
            f"Last ratio mismatch: {cid}"
        )

        capacity_by_cycle = {}
        for segment in by_cell[cid]:
            cycle = int(num(segment["instrument_cycle"]))
            raw_q = optional_num(
                segment["discharge_max_mAh_DIAGNOSTIC"]
            )

            need(
                cycle not in capacity_by_cycle,
                f"Duplicate cycle in selected cell: {cid}"
            )

            if raw_q is not None and raw_q > 0:
                capacity_by_cycle[cycle] = raw_q / q1

        need(
            last_cycle in capacity_by_cycle
            and 1 in capacity_by_cycle
            and 100 in capacity_by_cycle,
            f"Missing required raw cycles: {cid}"
        )

        need(
            abs(capacity_by_cycle[100] - q100 / q1) < 1e-9
            and abs(capacity_by_cycle[last_cycle] - last_ratio) < 1e-9,
            f"Raw capacity mismatch: {cid}"
        )

        result = {
            "cell_id": cid,
            "program": row["program"],
            "first100_capacity_ratio": q100 / q1,
            "last_recorded_capacity_ratio": last_ratio,
            "last_instrument_cycle": last_cycle,
            "extra_cycles_after_100": last_cycle - 100,
            "last_cycle_quality_screen_pass":
                row["last_cycle_quality_screen_pass"],
        }

        for threshold in THRESHOLDS:
            tag = str(int(threshold * 100))
            first, sustained = first_crossings(
                capacity_by_cycle, threshold
            )

            result[f"last_below_{tag}pct"] = (
                last_ratio <= threshold
            )
            result[f"first_below_{tag}pct_cycle"] = first
            result[f"first_3consecutive_below_{tag}pct_cycle"] = sustained
            result[f"cycles_after_first_{tag}pct_crossing"] = (
                last_cycle - first
                if first is not None else None
            )

        recent = [
            capacity_by_cycle[c]
            for c in range(last_cycle - 5, last_cycle)
            if c in capacity_by_cycle
        ]

        result["prior_5cycle_capacity_median"] = (
            float(np.median(recent))
            if recent else None
        )
        result["last_minus_prior5_median"] = (
            last_ratio - float(np.median(recent))
            if recent else None
        )

        findings.append(result)

    programs = defaultdict(list)
    for row in findings:
        programs[row["program"]].append(row)

    all_last = [
        row["last_recorded_capacity_ratio"]
        for row in findings
    ]

    summary = {
        "last_recorded_retention": describe(all_last),
        "last_recorded_instrument_cycle": describe(
            [r["last_instrument_cycle"] for r in findings]
        ),
        "cycle100_retention": describe(
            [r["first100_capacity_ratio"] for r in findings]
        ),
        "count_last_retention_within_0p03_of_0p75": sum(
            abs(x - 0.75) <= 0.03 for x in all_last
        ),
        "count_last_retention_within_0p03_of_0p80": sum(
            abs(x - 0.80) <= 0.03 for x in all_last
        ),
        "count_last_retention_below_0p80": sum(
            x <= 0.80 for x in all_last
        ),
        "threshold_crossing_counts": {
            str(int(t * 100)): {
                "any_observed": sum(
                    row[f"first_below_{int(t * 100)}pct_cycle"]
                    is not None for row in findings
                ),
                "three_consecutive": sum(
                    row[f"first_3consecutive_below_{int(t * 100)}pct_cycle"]
                    is not None for row in findings
                )
            }
            for t in THRESHOLDS
        }
    }

    per_program = []
    for program, group in sorted(programs.items()):
        per_program.append({
            "program": program,
            "physical_cells": len(group),
            "last_retention": describe([
                r["last_recorded_capacity_ratio"]
                for r in group
            ]),
            "last_instrument_cycle": describe([
                r["last_instrument_cycle"]
                for r in group
            ]),
            "last_below_80pct": sum(
                r["last_below_80pct"] for r in group
            )
        })

    report = {
        "stage": "17AG_POSTOUTCOME_STOPPING_HORIZON_DIAGNOSTIC",
        "physical_cells": 30,
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (AF, CELLS, CYCLES, AE, PRED)
        },
        "summary": summary,
        "program_groups": per_program,
        "threshold_grid": list(THRESHOLDS),
        "threshold_grid_is_postoutcome_diagnostic": True,
        "first_crossing_requires_no_model_fitting": True,
        "first_three_consecutive_crossing_is_observational": True,
        "actual_experiment_stopping_rule_verified": False,
        "physical_EOL_certified": False,
        "training_endpoint_equivalence_certified": False,
        "model_training": False,
        "prediction_changes": False,
        "prism_updated": False,
    }

    with OUT.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(findings[0])
        )
        writer.writeheader()
        writer.writerows(findings)

    with REPORT.open("x", encoding="utf-8") as f:
        json.dump(
            report, f, ensure_ascii=False, indent=2
        )

    print("=== STAGE 17AG NCA STOPPING HORIZON AUDIT ===")
    print("FROZEN_PREDICTION_HASH_UNCHANGED=YES")
    print("PHYSICAL_CELLS=30")
    print(
        "STOPPING_HORIZON_SUMMARY=",
        json.dumps(summary)
    )
    print(
        "PROTOCOL_GROUPS=",
        json.dumps(per_program)
    )
    print("CELL_DIAGNOSTICS_CSV=", OUT)
    print("REPORT_JSON=", REPORT)
    print(
        "EXPERIMENT_STOPPING_RULE_VERIFIED=NO "
        "PHYSICAL_EOL_CERTIFIED=NO "
        "MODEL_TRAINING=NO PRISM_UPDATED=NO"
    )
    print("=== END STAGE 17AG ===")


if __name__ == "__main__":
    main()
