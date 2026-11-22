#!/usr/bin/env python3
"""Stage20J: reconcile MICH endpoint with the original BatteryLife builder."""

import csv
import hashlib
import importlib.util
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path.cwd()
BENCH = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
ARCHIVE = ROOT / "data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original/MICH.zip"
OLD = ROOT / "external_validation/reports/stage20h_batterylife_249_temporal_cells.csv"
DECODER = ROOT / "external_validation/scripts/stage20h_batterylife_temporal_audit.py"
OUT = ROOT / "external_validation/reports"

CSV_OUT = OUT / "stage20j_mich_target_reconciliation.csv"
JSON_OUT = OUT / "stage20j_mich_target_report.json"

SHA = "8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c"
MD5 = "c8f7514deee3487d6195b4419097ef5f"

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def retention(pairs):
    if not pairs:
        return None, None, None

    # Original builder sorts by observed cycle number.
    ordered = sorted(pairs, key=lambda x: (x[0], x[1]))
    caps = [x[2] for x in ordered]

    first = median(caps[:3])
    last = median(caps[-3:])

    value = last / first if first != 0 else None

    return first, last, value

def main():
    if CSV_OUT.exists() or JSON_OUT.exists():
        raise SystemExit("Stage20J reports already exist; no overwrite")

    assert digest(BENCH, "sha256") == SHA
    assert digest(ARCHIVE, "md5") == MD5

    assert DECODER.is_file()

    spec = importlib.util.spec_from_file_location("stage20h_decoder", DECODER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bench = {
        r["cell_id"]: r
        for r in read_csv(BENCH)
        if r["dataset_id"] == "MICH"
        and r["source_family"] == "BatteryLife_selected"
    }

    old = {
        r["cell_id"]: r
        for r in read_csv(OLD)
        if r["dataset"] == "MICH"
    }

    assert len(bench) == 40 and len(old) == 40

    results = []

    with zipfile.ZipFile(ARCHIVE) as z:
        for cell, row in sorted(bench.items()):
            member = row["source_file"].replace("\\", "/")
            obj, status = mod.decode(z, z.getinfo(member))

            assert status == "ACCEPTED_PRIMITIVES", (cell, status)
            assert str(obj.get("cell_id")) == cell

            pairs = []

            for i, cyc in enumerate(obj.get("cycle_data", [])):
                if not isinstance(cyc, dict):
                    continue

                # Follow the source builder's cycle-number fallback.
                try:
                    cycle = int(cyc.get("cycle_number", i + 1))
                except (ValueError, TypeError, OverflowError):
                    cycle = i + 1

                cap = mod.capacity(cyc.get("discharge_capacity_in_Ah"))

                if cap is not None and math.isfinite(cap):
                    pairs.append((cycle, i, cap))

            all_first, all_last, all_ret = retention(pairs)

            positive = [p for p in pairs if p[2] > 0]
            pos_first, pos_last, pos_ret = retention(positive)

            target = float(row["final_capacity_retention"])

            all_difference = (
                abs(all_ret - target) if all_ret is not None else None
            )

            positive_difference = (
                abs(pos_ret - target) if pos_ret is not None else None
            )

            old_difference = float(
                old[cell]["target_rebuild_absdiff"]
            )

            assert positive_difference is not None
            assert abs(positive_difference - old_difference) < 1e-9, (
                "Stage20H replay mismatch", cell
            )

            ordered = sorted(pairs, key=lambda p: (p[0], p[1]))
            ends = ordered[:3] + ordered[-3:]

            result = {
                "cell_id": cell,
                "target": target,
                "finite_capacity_records": len(pairs),
                "positive_capacity_records": len(positive),
                "nonpositive_capacity_records": sum(
                    p[2] <= 0 for p in pairs
                ),
                "nonpositive_in_first_last_three": sum(
                    p[2] <= 0 for p in ends
                ),
                "duplicate_cycle_numbers": (
                    len(pairs) - len(set(p[0] for p in pairs))
                ),
                "original_formula_first": all_first,
                "original_formula_last": all_last,
                "original_formula_retention": all_ret,
                "original_formula_absdiff": all_difference,
                "positive_only_retention": pos_ret,
                "positive_only_absdiff": positive_difference,
                "previous_stage20h_absdiff": old_difference,
                "original_formula_matches_1e6": int(
                    all_difference is not None and all_difference < 1e-6
                ),
            }

            results.append(result)

    previous_mismatch = sum(
        r["positive_only_absdiff"] >= 1e-6
        for r in results
    )

    original_matches = sum(
        r["original_formula_matches_1e6"] for r in results
    )

    nonpositive_cells = sum(
        r["nonpositive_capacity_records"] > 0
        for r in results
    )

    report = {
        "stage": "20J",
        "cells": len(results),
        "previous_positive_only_mismatches": previous_mismatch,
        "original_formula_matches": original_matches,
        "remaining_original_formula_mismatches": 40-original_matches,
        "cells_with_nonpositive_capacities": nonpositive_cells,
        "cells_with_nonpositive_first_last_values": sum(
            r["nonpositive_in_first_last_three"] > 0
            for r in results
        ),
        "duplicate_cycle_cells": sum(
            r["duplicate_cycle_numbers"] > 0
            for r in results
        ),
        "source_builder_semantics": (
            "Finite capacities including zero/negative values; "
            "median of first and last three valid capacities "
            "after cycle-number sorting"
        ),
        "models_retrained": False,
        "benchmark_modified": False,
        "results": results,
    }

    assert len(results) == 40
    assert previous_mismatch == 14

    with CSV_OUT.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    JSON_OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print("=== STAGE20J MICH RESULTS ===")
    print("CELLS", len(results))
    print("PREVIOUS_MISMATCHES", previous_mismatch)
    print("ORIGINAL_FORMULA_MATCHES", original_matches)
    print("REMAINING_MISMATCHES", 40-original_matches)
    print("CELLS_WITH_NONPOSITIVE_CAPACITIES", nonpositive_cells)
    print(
        "NONPOSITIVE_IN_ENDPOINT_WINDOWS",
        report["cells_with_nonpositive_first_last_values"]
    )
    print("DUPLICATE_CYCLE_CELLS", report["duplicate_cycle_cells"])

    for r in results:
        if r["positive_only_absdiff"] >= 1e-6:
            print(
                "RECONCILIATION",
                r["cell_id"],
                "old_diff", round(r["positive_only_absdiff"], 9),
                "new_diff",
                round(r["original_formula_absdiff"], 9)
                if r["original_formula_absdiff"] is not None
                else "UNDEFINED",
                "nonpositive",
                r["nonpositive_capacity_records"]
            )

    print("STAGE20J_COMPLETED=YES")
    print("REPORT", JSON_OUT)

if __name__ == "__main__":
    main()
