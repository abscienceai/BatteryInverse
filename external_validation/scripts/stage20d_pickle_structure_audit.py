#!/usr/bin/env python3
"""Stage20D: verified cell lineage and non-executing pickle structure inspection."""
import csv
import hashlib
import json
import pickletools
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
RAW = ROOT / "data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original"
OUT = ROOT / "external_validation/reports/stage20d_batterylife_pilot_audit.json"

EXPECTED_DATA = "8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c"
EXPECTED_ARCHIVES = {
    "CALB": "86480187bf7ab3b447d8e7553b01cb59",
    "UL_PUR": "7a5e4a3e7ecfbcd6c189d97c2379f50a",
}
EXPECTED_CELLS = {"CALB": 27, "UL_PUR": 2}
KEYS = {
    "cycle_data",
    "cycle_number",
    "discharge_capacity_in_Ah",
    "charge_capacity_in_Ah",
    "nominal_capacity_in_Ah",
    "cell_id",
    "charge_protocol",
    "discharge_protocol",
}
MAX_PICKLE_SIZE = 32 * 1024 * 1024

def hash_file(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect_pickle(z, info):
    result = {
        "member": info.filename,
        "uncompressed_bytes": info.file_size,
        "status": "not_inspected",
    }
    if info.file_size > MAX_PICKLE_SIZE:
        result["status"] = "skipped_size_limit"
        return result

    with z.open(info) as f:
        payload = f.read(MAX_PICKLE_SIZE + 1)
    if len(payload) > MAX_PICKLE_SIZE:
        result["status"] = "skipped_size_limit"
        return result

    strings = set()
    globals_found = set()
    stack_globals = 0
    operations = 0

    try:
        for opcode, arg, _ in pickletools.genops(payload):
            operations += 1
            if opcode.name in {
                "SHORT_BINUNICODE", "BINUNICODE",
                "BINUNICODE8", "UNICODE", "STRING",
                "BINSTRING", "SHORT_BINSTRING",
            }:
                if isinstance(arg, bytes):
                    arg = arg.decode("utf-8", errors="replace")
                if isinstance(arg, str) and arg in KEYS:
                    strings.add(arg)

            if opcode.name == "GLOBAL":
                globals_found.add(str(arg))
            elif opcode.name == "STACK_GLOBAL":
                stack_globals += 1

        result.update({
            "status": "static_parse_complete",
            "operations": operations,
            "recognized_keys": sorted(strings),
            "global_references": sorted(globals_found)[:30],
            "stack_global_operations": stack_globals,
        })
    except Exception as exc:
        result.update({
            "status": "static_parse_error",
            "error": type(exc).__name__ + ": " + str(exc)[:200],
        })

    return result

def main():
    assert not OUT.exists(), "Stage20D result already exists"
    assert hash_file(DATA, "sha256") == EXPECTED_DATA

    with DATA.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    selected = [
        r for r in rows
        if r["source_family"] == "BatteryLife_selected"
        and r["dataset_id"] in EXPECTED_ARCHIVES
    ]

    assert len(selected) == 29
    assert len(set((r["dataset_id"], r["cell_id"]) for r in selected)) == 29

    results = {}
    all_matches = 0

    for domain, expected_md5 in EXPECTED_ARCHIVES.items():
        path = RAW / (domain + ".zip")
        assert path.is_file()
        assert hash_file(path, "md5") == expected_md5

        group = [r for r in selected if r["dataset_id"] == domain]
        assert len(group) == EXPECTED_CELLS[domain]

        with zipfile.ZipFile(path) as z:
            members = {
                i.filename: i
                for i in z.infolist()
                if not i.is_dir() and i.filename.lower().endswith(".pkl")
            }

            matched = []
            for r in group:
                name = r["source_file"].replace("\\", "/")
                assert name in members, (
                    "Missing source member: " + domain + " " + name
                )
                matched.append(name)

            assert len(set(matched)) == len(group)
            assert len(members) == EXPECTED_CELLS[domain]

            all_matches += len(matched)

            samples = sorted(
                (members[name] for name in matched),
                key=lambda i: i.file_size
            )[:2]

            structure = [
                inspect_pickle(z, info)
                for info in samples
            ]

            results[domain] = {
                "benchmark_cells": len(group),
                "unique_source_matches": len(set(matched)),
                "archive_members": len(members),
                "md5_verified": True,
                "sample_structure": structure,
            }

    assert all_matches == 29

    record = {
        "stage": "20D",
        "unique_benchmark_cells": len(selected),
        "unique_source_matches": all_matches,
        "archives": results,
        "pickle_deserialized": False,
        "capacity_values_extracted": False,
        "physical_endpoint_comparability_verified": False,
        "limitations": [
            "File identity is not measurement-content validation.",
            "Static pickle inspection does not establish code safety.",
            "No cycle measurements have yet been reconstructed.",
        ],
    }

    OUT.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print("=== STAGE20D RESULTS ===")
    print("UNIQUE_BENCHMARK_CELLS", len(selected))
    print("UNIQUE_SOURCE_MATCHES", all_matches)

    for domain, result in results.items():
        print(
            "DOMAIN", domain,
            "cells", result["benchmark_cells"],
            "matches", result["unique_source_matches"]
        )
        for sample in result["sample_structure"]:
            print(
                "PICKLE_SAMPLE",
                domain,
                sample["status"],
                sample.get("recognized_keys", []),
                "globals=",
                sample.get("global_references", []),
                "stack_globals=",
                sample.get("stack_global_operations", 0)
            )

    print("STAGE20D_STRUCTURE_AUDIT_COMPLETE=YES")
    print("REPORT", OUT)

if __name__ == "__main__":
    main()
