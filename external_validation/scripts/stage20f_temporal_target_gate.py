#!/usr/bin/env python3
"""Stage20F: audited pilot temporal boundary and simple Q100/Q1 comparison."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path.cwd()

BENCH = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
PILOT = ROOT / "external_validation/reports/stage20e_pilot_observed_cycles.csv"

OUT = ROOT / "external_validation/reports"
CELL_OUT = OUT / "stage20f_temporal_target_gate.csv"
REPORT_OUT = OUT / "stage20f_temporal_target_report.json"

EXPECTED = "8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c"

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def metric(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "maximum": max(values)
    }

def main():

    if CELL_OUT.exists() or REPORT_OUT.exists():
        raise SystemExit("Stage20F outputs already exist; no overwrite")

    if hashlib.sha256(BENCH.read_bytes()).hexdigest() != EXPECTED:
        raise SystemExit("Benchmark SHA256 mismatch")

    bench = read_csv(BENCH)
    pilot = read_csv(PILOT)

    assert len(bench) == 567
    assert len(pilot) == 29

    lookup = {
        (r["dataset_id"], r["cell_id"]): r
        for r in bench
    }

    assert len(lookup) == 567

    records = []

    for p in pilot:

        domain = p["dataset"]
        cell = p["cell_id"]
        key = (domain, cell)

        assert key in lookup

        b = lookup[key]

        assert b["source_family"] == "BatteryLife_selected"
        assert int(p["identity_match"]) == 1
        assert p["pickle_status"] == "accepted_primitive_only"

        rebuild_diff = float(p["retention_rebuild_absdiff"])
        assert rebuild_diff < 1e-6

        last = int(float(p["last_observed_cycle"]))
        post100 = int(p["observed_post100_cycles"])
        exact100 = int(p["exact_cycle_100_count"])

        assert exact100 == 1
        assert (last > 100) == (post100 > 0)

        feature = float(b["cap_ratio_100_to_1"])
        target = float(b["final_capacity_retention"])

        assert math.isfinite(feature)
        assert math.isfinite(target)

        no_future = post100 == 0

        records.append({
            "dataset": domain,
            "cell_id": cell,
            "last_observed_cycle": last,
            "post100_capacity_records": post100,
            "exact_cycle100_observed": exact100,
            "target_within_first100": int(no_future),
            "temporal_status": (
                "NO_POST100_TARGET_HORIZON"
                if no_future
                else "POST100_OBSERVATIONS_PRESENT"
            ),
            "q100_over_q1": feature,
            "final_capacity_retention": target,
            "q100_ratio_abs_error": abs(feature - target),
            "source_target_rebuild_absdiff": rebuild_diff,
        })

    assert len(set((r["dataset"], r["cell_id"]) for r in records)) == 29

    grouped = defaultdict(list)

    for r in records:
        grouped[r["dataset"]].append(r)

    result = {}

    for domain, items in sorted(grouped.items()):

        nonfuture = [
            r for r in items if r["target_within_first100"]
        ]

        future = [
            r for r in items if not r["target_within_first100"]
        ]

        result[domain] = {
            "cells": len(items),
            "last_observation_at_or_before100": len(nonfuture),
            "post100_observations_present": len(future),
            "last_cycles": sorted(
                r["last_observed_cycle"] for r in items
            ),
            "q100_ratio_abs_error_no_future": metric([
                r["q100_ratio_abs_error"] for r in nonfuture
            ]),
            "q100_ratio_abs_error_post100": metric([
                r["q100_ratio_abs_error"] for r in future
            ])
        }

    assert result["CALB"]["cells"] == 27
    assert result["UL_PUR"]["cells"] == 2
    assert result["CALB"]["last_observation_at_or_before100"] == 26
    assert result["UL_PUR"]["last_observation_at_or_before100"] == 0

    report = {
        "stage": "20F",
        "scope": "29 source-verified pilot cells, not the full 567-cell benchmark",
        "benchmark_sha256": EXPECTED,
        "domains": result,
        "all_pilot_cells": 29,
        "no_post100_horizon_cells": sum(
            r["target_within_first100"] for r in records
        ),
        "interpretation": [
            "For cells with last observed cycle 100, the final-retention target falls within the feature-observation window.",
            "Q100/Q1 comparison quantifies direct contemporaneous feature-target proximity, not predictive generalization.",
            "Post-100 measurements alone do not establish protocol comparability or a prospectively valid model.",
            "The remaining 538 benchmark cells are not covered by this pilot audit."
        ],
    }

    with CELL_OUT.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    REPORT_OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print("=== STAGE20F TEMPORAL TARGET AUDIT ===")

    for domain, entry in result.items():
        print("DOMAIN", domain)
        print("  CELLS", entry["cells"])
        print(
            "  NO_POST100_TARGET_HORIZON",
            entry["last_observation_at_or_before100"]
        )
        print(
            "  POST100_OBSERVATIONS_PRESENT",
            entry["post100_observations_present"]
        )
        print(
            "  Q100_RATIO_ERROR_NO_FUTURE",
            entry["q100_ratio_abs_error_no_future"]
        )
        print("  LAST_CYCLE_VALUES", entry["last_cycles"])

    print("PILOT_CELLS", len(records))
    print(
        "TOTAL_NO_POST100_TARGET_HORIZON",
        report["no_post100_horizon_cells"]
    )
    print("STAGE20F_VERIFIED=YES")

if __name__ == "__main__":
    main()
