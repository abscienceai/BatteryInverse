#!/usr/bin/env python3
"""Stage17AB: NCA first-100 capacities and frozen feature contract.
No terminal labels, model fitting or external predictions.
"""
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
R = ROOT / "external_validation/reports"

T = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)
MODEL = ROOT / (
    "external_validation/scripts/"
    "stage16a_isu_exploratory_transfer.py"
)

LOCK = R / "stage17y_nca_pre_extraction_protocol.json"
Y = R / "stage17y_tongji_nca_archive_inventory.json"
Z = R / "stage17z_nca_raw_cycle_screen_report.json"
AA = R / "stage17aa_nca_preinference_gate_report.json"
ZC = R / "stage17z_nca_observed_cycle_segments.csv"
AC = R / "stage17aa_nca_66cell_preinference_gate.csv"

OUT = R / "stage17ab_nca_first100_capacity_support.csv"
EVIDENCE = R / "stage17ab_original_feature_formula_source_mentions.csv"
REPORT = R / "stage17ab_first100_feature_contract_report.json"

FEATURES = [
    "cap_ratio_10_to_1",
    "cap_ratio_50_to_1",
    "cap_ratio_100_to_1",
    "cap_mean_first_100_norm",
    "cap_std_first_100_norm",
    "cap_slope_first_100_norm",
    "cap_fade_fraction_1_to_100"
]

ANCHORS = [
    "cap_cycle_1",
    "cap_cycle_10",
    "cap_cycle_50",
    "cap_cycle_100"
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path, required):
    require(path.is_file(), f"Missing: {path}")

    with path.open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])

        require(
            set(required) <= columns,
            f"Missing columns: {path.name}"
        )

        return list(reader)


def num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def save_csv(path, rows, columns):
    with path.open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def training_formula_tests(training):
    attempts = {
        feature: defaultdict(list)
        for feature in FEATURES
    }

    for row in training:
        if not all(
            num(row.get(col)) is not None
            for col in ANCHORS + FEATURES
        ):
            continue

        capacities = np.asarray(
            [num(row[col]) for col in ANCHORS],
            dtype=float
        )

        if capacities[0] <= 0:
            continue

        normalized = capacities / capacities[0]

        candidates = {
            "cap_ratio_10_to_1": {
                "q10/q1": normalized[1]
            },
            "cap_ratio_50_to_1": {
                "q50/q1": normalized[2]
            },
            "cap_ratio_100_to_1": {
                "q100/q1": normalized[3]
            },
            "cap_mean_first_100_norm": {
                "mean_four_normalized_anchors":
                    np.mean(normalized)
            },
            "cap_std_first_100_norm": {
                "std_four_anchors_ddof0":
                    np.std(normalized, ddof=0),
                "std_four_anchors_ddof1":
                    np.std(normalized, ddof=1)
            },
            "cap_slope_first_100_norm": {
                "slope_cycles_1_10_50_100":
                    np.polyfit(
                        [1, 10, 50, 100],
                        normalized, 1
                    )[0],
                "slope_positions_1_2_3_4":
                    np.polyfit(
                        [1, 2, 3, 4],
                        normalized, 1
                    )[0]
            },
            "cap_fade_fraction_1_to_100": {
                "1_minus_q100_over_q1":
                    1 - normalized[3]
            }
        }

        for feature in FEATURES:
            actual = num(row[feature])

            for formula, predicted in candidates[feature].items():
                attempts[feature][formula].append(
                    abs(actual - float(predicted))
                )

    results = []

    for feature, formulas in attempts.items():
        for formula, errors in formulas.items():
            results.append({
                "feature": feature,
                "candidate_formula": formula,
                "tested_training_rows": len(errors),
                "max_abs_difference": max(errors),
                "p95_abs_difference":
                    float(np.quantile(errors, 0.95)),
                "matching_within_1e_8": sum(
                    delta < 1e-8 for delta in errors
                ),
                "all_tested_rows_match":
                    bool(errors)
                    and all(
                        delta < 1e-8 for delta in errors
                    )
            })

    return results


def source_mentions():
    pattern = re.compile(
        r"cap_(?:ratio_10_to_1|ratio_50_to_1|"
        r"ratio_100_to_1|mean_first_100_norm|"
        r"std_first_100_norm|slope_first_100_norm|"
        r"fade_fraction_1_to_100)"
    )

    sensitive = re.compile(
        r"password|private.key|api.key|"
        r"access.token|secret|credential",
        re.I
    )

    result = []

    for base in (
        ROOT / "scripts",
        ROOT / "external_validation/scripts"
    ):
        if not base.is_dir():
            continue

        for path in sorted(base.rglob("*.py")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size > 1500000
                or path.name.startswith("stage17")
            ):
                continue

            lines = path.read_text(
                encoding="utf-8",
                errors="replace"
            ).splitlines()

            for i, line in enumerate(lines):
                if (
                    not pattern.search(line)
                    or sensitive.search(line)
                ):
                    continue

                context = " | ".join(
                    f"L{j+1}: {lines[j].strip()[:160]}"
                    for j in range(
                        max(0, i - 1),
                        min(len(lines), i + 2)
                    )
                    if not sensitive.search(lines[j])
                )

                result.append({
                    "file":
                        path.relative_to(ROOT).as_posix(),
                    "line": i + 1,
                    "context": context[:600]
                })

    return result


def main():
    sources = (
        T, MODEL, LOCK, Y,
        Z, AA, ZC, AC
    )

    for path in sources:
        require(
            path.is_file(),
            f"Missing source: {path}"
        )

    for path in (OUT, EVIDENCE, REPORT):
        require(
            not path.exists(),
            f"Refusing overwrite: {path}"
        )

    lock = json.loads(LOCK.read_text())
    y = json.loads(Y.read_text())
    z = json.loads(Z.read_text())
    aa = json.loads(AA.read_text())

    require(
        lock["training_567_sha256"] == sha(T)
        and lock["original_stage16a_script_sha256"]
        == sha(MODEL),
        "Frozen training/model source changed"
    )

    require(
        aa.get("physical_cells") == 66
        and z.get("cells_in_archive") == 66
        and y.get("csv_entries") == 66,
        "Original NCA cohort mismatch"
    )

    for relative, expected in aa.get(
        "source_sha256", {}
    ).items():
        part = Path(relative)

        require(
            not part.is_absolute()
            and ".." not in part.parts,
            "Unsafe provenance path"
        )

        actual = (ROOT / part).resolve()

        require(
            actual.is_relative_to(ROOT)
            and actual.is_file()
            and sha(actual) == expected,
            f"Stage17AA source changed: {relative}"
        )

    training = read_csv(
        T,
        [
            "cell_id", "dataset_id",
            "final_capacity_retention"
        ] + ANCHORS + FEATURES
    )

    cells = read_csv(
        AC,
        [
            "cell_id", "program",
            "strict_first100_gate_DIAGNOSTIC"
        ]
    )

    cycles = read_csv(
        ZC,
        [
            "cell_id", "instrument_cycle",
            "discharge_max_mAh_DIAGNOSTIC",
            "voltage_min_V"
        ]
    )

    require(
        len(training) == 567
        and len(cells) == 66
        and len(cycles) == z["output_cycle_rows"],
        "Input row count mismatch"
    )

    require(
        len({row["cell_id"] for row in cells}) == 66,
        "Duplicate NCA physical cell"
    )

    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for row in cycles:
        cycle = num(row["instrument_cycle"])

        require(
            cycle is not None
            and cycle >= 0
            and float(cycle).is_integer(),
            "Invalid cycle number"
        )

        grouped[row["cell_id"]][int(cycle)].append(row)

    require(
        set(grouped) == {
            row["cell_id"] for row in cells
        },
        "Cell roster mismatch"
    )

    results = []

    for row in sorted(
        cells, key=lambda value: value["cell_id"]
    ):
        cell_id = row["cell_id"]

        gate = (
            row["strict_first100_gate_DIAGNOSTIC"]
            .lower() == "true"
        )

        by_cycle = grouped[cell_id]
        capacities = []
        voltages = []

        for cycle in range(1, 101):
            block = by_cycle.get(cycle, [])

            if len(block) != 1:
                capacities.append(None)
                voltages.append(None)
            else:
                capacities.append(
                    num(
                        block[0][
                            "discharge_max_mAh_DIAGNOSTIC"
                        ]
                    )
                )
                voltages.append(
                    num(block[0]["voltage_min_V"])
                )

        positive = [
            cap for cap in capacities
            if cap is not None and cap > 0
        ]

        count_low = sum(
            cap is not None and cap < 2500
            for cap in capacities
        )

        count_high = sum(
            cap is not None and cap > 3500
            for cap in capacities
        )

        near_cutoff = sum(
            voltage is not None
            and 2.55 <= voltage <= 2.75
            for voltage in voltages
        )

        record = {
            "cell_id": cell_id,
            "program": row["program"],
            "stage17aa_first100_gate": gate,
            "first100_positive_count": len(positive),
            "first100_q_below_2500mAh": count_low,
            "first100_q_above_3500mAh": count_high,
            "first100_minvoltage_2p55_to_2p75V":
                near_cutoff,
            "first100_q_median_mAh":
                float(np.median(positive))
                if positive else None,
            "q_cycle_1_mAh": capacities[0],
            "q_cycle_10_mAh": capacities[9],
            "q_cycle_50_mAh": capacities[49],
            "q_cycle_100_mAh": capacities[99],
            "published_author_2500_to_3500_filter_all_first100":
                all(
                    cap is not None
                    and 2500 <= cap <= 3500
                    for cap in capacities
                ),
            "model_feature_contract_certified": False,
            "terminal_target_inspected": False
        }

        results.append(record)

    comparisons = training_formula_tests(training)
    mentions = source_mentions()

    summary = {
        "all_cells": len(cells),
        "stage17aa_diagnostic_gate": sum(
            r["stage17aa_first100_gate"]
            for r in results
        ),
        "gate_cells_with_any_q_below2500": sum(
            r["stage17aa_first100_gate"]
            and r["first100_q_below_2500mAh"] > 0
            for r in results
        ),
        "gate_cells_with_any_q_above3500": sum(
            r["stage17aa_first100_gate"]
            and r["first100_q_above_3500mAh"] > 0
            for r in results
        ),
        "gate_cells_all_first100_q_in_2500_to_3500":
            sum(
                r["stage17aa_first100_gate"]
                and r[
                    "published_author_2500_to_3500_filter_all_first100"
                ]
                for r in results
            ),
        "gate_cells_with_100_voltage_min_near_2p65V":
            sum(
                r["stage17aa_first100_gate"]
                and r[
                    "first100_minvoltage_2p55_to_2p75V"
                ] == 100
                for r in results
            )
    }

    require(
        summary["stage17aa_diagnostic_gate"]
        == aa["quality_counts"][
            "strict_first100_gate_DIAGNOSTIC"
        ],
        "Stage17AA gate count changed"
    )

    save_csv(
        OUT, results, list(results[0])
    )

    save_csv(
        EVIDENCE,
        mentions[:250],
        ["file", "line", "context"]
    )

    report = {
        "stage":
            "17AB_NCA_FIRST100_FROZEN_FEATURE_CONTRACT",
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha(path)
            for path in sources
        },
        "first100_quality_summary": summary,
        "training_567_candidate_feature_formula_identity":
            comparisons,
        "source_code_mentions_total": len(mentions),
        "source_code_mentions_first25":
            mentions[:25],
        "author_capacity_filter":
            "2500 to 3500 mAh; diagnostic comparison only",
        "full_discharge_completeness_certified": False,
        "external_model_feature_equivalence_certified":
            False,
        "terminal_targets_read": False,
        "models_fitted": False,
        "inference": False,
        "prism_updated": False
    }

    with REPORT.open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(
            report,
            stream,
            ensure_ascii=False,
            indent=2
        )

    print(
        "=== STAGE 17AB NCA FIRST100 FEATURE CONTRACT ==="
    )

    print(
        "QUALITY_SUMMARY=",
        json.dumps(summary)
    )

    print(
        "TRAINING_567_FEATURE_FORMULA_IDENTITY=",
        json.dumps(comparisons)
    )

    print(
        "SOURCE_CODE_FORMULA_MENTIONS=",
        json.dumps(
            mentions[:18],
            ensure_ascii=False
        )
    )

    print("CAPACITY_SUPPORT_CSV=", OUT)
    print("CODE_EVIDENCE_CSV=", EVIDENCE)
    print("REPORT_JSON=", REPORT)

    print(
        "TARGET_LABELS_READ=NO "
        "MODEL_TRAINING=NO "
        "INFERENCE=NO "
        "PRISM_UPDATED=NO"
    )

    print("=== END STAGE 17AB ===")


if __name__ == "__main__":
    main()
