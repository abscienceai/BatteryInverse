#!/usr/bin/env python3
"""Stage20O: combine verified source-target and temporal evidence for 249 cells."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path.cwd()
REP = ROOT / "external_validation/reports"
BENCH = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"

H = REP / "stage20h_batterylife_249_temporal_cells.csv"
J = REP / "stage20j_mich_target_reconciliation.csv"
N = REP / "stage20n_xjtu_full_cells.csv"
N_REPORT = REP / "stage20n_xjtu_full_report.json"
CHECKPOINTS = REP / "stage20n_xjtu_checkpoints"

OUT_CSV = REP / "stage20o_unified_endpoint_cells.csv"
OUT_JSON = REP / "stage20o_unified_endpoint_report.json"

EXPECTED_SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)

EXPECTED_DOMAINS = {
    "CALB": 27,
    "HNEI": 14,
    "MICH": 40,
    "MICH_EXP": 18,
    "NA-ion": 64,
    "SNL": 61,
    "UL_PUR": 2,
    "XJTU": 23,
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def indexed(rows, fields):
    result = {}

    for r in rows:
        key = tuple(r[f] for f in fields)

        if key in result:
            raise ValueError("Duplicate key: " + repr(key))

        result[key] = r

    return result


def statistics(values):
    if not values:
        return None

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "maximum": max(values),
    }


def main():
    if OUT_CSV.exists() or OUT_JSON.exists():
        raise SystemExit(
            "Stage20O outputs already exist; refusing overwrite"
        )

    actual_sha = hashlib.sha256(BENCH.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_SHA

    benchmark = read_csv(BENCH)
    assert len(benchmark) == 567

    selected = indexed(
        [
            r for r in benchmark
            if r["source_family"] == "BatteryLife_selected"
        ],
        ("dataset_id", "cell_id")
    )
    assert len(selected) == 249

    h = indexed(read_csv(H), ("dataset", "cell_id"))
    j = indexed(read_csv(J), ("cell_id",))
    n = indexed(read_csv(N), ("cell_id",))

    assert len(h) == 249
    assert len(j) == 40
    assert len(n) == 23

    full_report = json.loads(N_REPORT.read_text(encoding="utf-8"))
    assert full_report["xjtu_verified_targets"] == 23

    combined = full_report["overall_batterylife_summary"]
    assert combined["status"] == "ALL_SOURCE_TARGETS_RECONSTRUCTED"
    assert combined["batterylife_cells"] == 249

    domain_counts = Counter(
        dataset for dataset, _ in selected
    )
    assert dict(domain_counts) == EXPECTED_DOMAINS

    records = []

    for (dataset, cell), b in sorted(selected.items()):
        key = (dataset, cell)
        assert key in h

        previous = h[key]
        target = float(b["final_capacity_retention"])
        q100_ratio = float(b["cap_ratio_100_to_1"])

        quality_endpoint_nonpositive = 0
        quality_delta = None
        exact100 = None
        nearest100_after100 = None

        if dataset == "XJTU":
            x = n[(cell,)]

            assert x["status"] == "SUCCESS"
            assert int(x["target_match"]) == 1

            checkpoint = json.loads(
                (CHECKPOINTS / (cell + ".json")).read_text(
                    encoding="utf-8"
                )
            )

            assert checkpoint["cell_id"] == cell
            assert checkpoint["benchmark_sha256"] == EXPECTED_SHA
            assert checkpoint["source_file"] == (
                b["source_file"].replace("\\", "/")
            )
            assert abs(float(checkpoint["target"]) - target) < 1e-10

            result = checkpoint["result"]
            assert result["status"] == "SUCCESS"

            m = result["measurement"]
            assert m["source_identity_verified"] is True
            assert m["cell_id"] == cell
            assert m["target_match_1e6"] is True

            reconstructed_diff = float(
                m["target_abs_difference"]
            )

            assert abs(
                reconstructed_diff
                - float(x["target_abs_difference"])
            ) < 1e-10

            post100 = int(m["post100_positive_records"])
            last_cycle = int(m["last_observed_cycle"])
            exact100 = int(m["exact_cycle100_positive"])

            additional_nonpositive = int(
                m["nonpositive_capacity_records"]
            )

            target_formula = "XJTU_RESTRICTED_RECONSTRUCTION"

        else:
            assert previous["decode_status"] == "ACCEPTED_PRIMITIVES"
            assert int(previous["content_id_match"]) == 1
            assert int(previous["valid_capacity_records"]) > 0

            post100 = int(previous["post100_records"])
            last_cycle = int(previous["last_observed_cycle"])

            exact100 = int(previous["exact_cycle_100_count"])
            nearest100_after100 = int(
                previous["nearest100_after100"]
            )

            additional_nonpositive = None

            if dataset == "MICH":
                assert (cell,) in j
                item = j[(cell,)]

                assert int(
                    item["original_formula_matches_1e6"]
                ) == 1

                reconstructed_diff = float(
                    item["original_formula_absdiff"]
                )

                quality_endpoint_nonpositive = int(
                    int(
                        item["nonpositive_in_first_last_three"]
                    ) > 0
                )

                quality_delta = abs(
                    float(item["original_formula_retention"])
                    - float(item["positive_only_retention"])
                )

                additional_nonpositive = int(
                    item["nonpositive_capacity_records"]
                )

                target_formula = "ORIGINAL_FINITE_CAPACITY_MEDIAN"

            else:
                reconstructed_diff = float(
                    previous["target_rebuild_absdiff"]
                )

                target_formula = "SOURCE_RECONSTRUCTED"

        assert reconstructed_diff < 1e-6, (
            dataset, cell, reconstructed_diff
        )

        assert post100 >= 0

        if post100 == 0:
            horizon = "NO_POSITIVE_POST100"
        elif post100 < 3:
            horizon = "ONE_OR_TWO_POSITIVE_POST100"
        else:
            horizon = "AT_LEAST_THREE_POSITIVE_POST100"

        records.append({
            "dataset": dataset,
            "cell_id": cell,
            "source_file": b["source_file"],
            "source_target_absdiff": reconstructed_diff,
            "target_formula": target_formula,
            "last_observed_cycle": last_cycle,
            "positive_post100_records": post100,
            "horizon_category": horizon,
            "exact_cycle100_positive_count": exact100,
            "nearest100_after100": (
                nearest100_after100
                if nearest100_after100 is not None else ""
            ),
            "q100_ratio_target_absdiff": abs(
                q100_ratio - target
            ),
            "mich_nonpositive_endpoint_flag":
                quality_endpoint_nonpositive,
            "mich_positive_only_sensitivity_absdiff": (
                quality_delta
                if quality_delta is not None else ""
            ),
            "observed_nonpositive_records": (
                additional_nonpositive
                if additional_nonpositive is not None else ""
            ),
        })

    assert len(records) == 249
    assert len(set(
        (r["dataset"], r["cell_id"]) for r in records
    )) == 249

    by_domain = defaultdict(list)

    for r in records:
        by_domain[r["dataset"]].append(r)

    domain_results = {}

    for domain in EXPECTED_DOMAINS:
        rows = by_domain[domain]
        assert len(rows) == EXPECTED_DOMAINS[domain]

        zero = [
            r for r in rows
            if r["positive_post100_records"] == 0
        ]

        one_two = [
            r for r in rows
            if 1 <= r["positive_post100_records"] <= 2
        ]

        three_plus = [
            r for r in rows
            if r["positive_post100_records"] >= 3
        ]

        domain_results[domain] = {
            "cells": len(rows),
            "no_positive_post100": len(zero),
            "one_or_two_positive_post100": len(one_two),
            "at_least_three_positive_post100": len(three_plus),
            "exact_cycle100_single_positive": sum(
                r["exact_cycle100_positive_count"] == 1
                for r in rows
            ),
            "q100_target_gap_no_future": statistics([
                r["q100_ratio_target_absdiff"]
                for r in zero
            ]),
            "mich_nonpositive_endpoint_flags": sum(
                r["mich_nonpositive_endpoint_flag"]
                for r in rows
            ),
        }

    no_future = [
        r for r in records
        if r["positive_post100_records"] == 0
    ]

    one_two = [
        r for r in records
        if 1 <= r["positive_post100_records"] <= 2
    ]

    three_plus = [
        r for r in records
        if r["positive_post100_records"] >= 3
    ]

    quality = [
        r for r in records
        if r["mich_nonpositive_endpoint_flag"]
    ]

    flagged_keys = {
        (r["dataset"], r["cell_id"])
        for r in no_future + quality
    }

    assert len(no_future) == 29
    assert len(quality) == 14
    assert len(three_plus) + len(one_two) == 220

    sensitivity = statistics([
        float(r["mich_positive_only_sensitivity_absdiff"])
        for r in quality
    ])

    report = {
        "stage": "20O",
        "benchmark_sha256": EXPECTED_SHA,
        "scope": "249 BatteryLife cells, not 567 full-benchmark cells",
        "verified_source_targets": len(records),
        "no_positive_post100": len(no_future),
        "one_or_two_positive_post100": len(one_two),
        "at_least_three_positive_post100": len(three_plus),
        "mich_nonpositive_endpoint_flags": len(quality),
        "union_of_two_tested_flags": len(flagged_keys),
        "without_these_two_flags": 249 - len(flagged_keys),
        "q100_target_gap_no_future": statistics([
            r["q100_ratio_target_absdiff"]
            for r in no_future
        ]),
        "mich_positive_only_sensitivity": sensitivity,
        "by_domain": domain_results,
        "limitations": [
            "Post-100 positive counts are an observation-coverage proxy.",
            "Three post-100 positive records do not prove harmonized physical endpoints.",
            "The final-three target window was not independently time-audited for every cell.",
            "The nonpositive endpoint flag was specifically established for MICH.",
            "Exact-cycle-100 presence is not proof of all feature extraction timestamps.",
            "The 318 BatteryML_main cells remain outside this source audit.",
            "No model was retrained and no original benchmark was modified."
        ]
    }

    REP.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

    with OUT_JSON.open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(
            report, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")

    print("=== STAGE20O UNIFIED RESULTS ===")
    print("VERIFIED_SOURCE_TARGETS", len(records))
    print("NO_POSITIVE_POST100", len(no_future))
    print("POST100_ONE_OR_TWO", len(one_two))
    print("POST100_THREE_PLUS", len(three_plus))
    print("MICH_NONPOSITIVE_ENDPOINT_FLAGS", len(quality))
    print("UNION_OF_TWO_FLAGS", len(flagged_keys))
    print(
        "Q100_GAP_NO_FUTURE",
        report["q100_target_gap_no_future"]
    )
    print(
        "MICH_TARGET_SENSITIVITY",
        sensitivity
    )

    for domain, values in domain_results.items():
        print("DOMAIN", domain, values)

    print("STAGE20O_UNIFIED_AUDIT_COMPLETE=YES")
    print("REPORT", OUT_JSON)


if __name__ == "__main__":
    main()
