#!/usr/bin/env python3
"""Stage20K: bounded static examination of XJTU pickle streams."""

import csv
import hashlib
import io
import json
import pickletools
import zipfile
from collections import Counter, deque
from pathlib import Path

ROOT = Path.cwd()
RAW = ROOT / "data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original/XJTU.zip"
BENCH = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
STAGE_H = ROOT / "external_validation/reports/stage20h_batterylife_249_temporal_cells.csv"
REPORT = ROOT / "external_validation/reports"

CSV_OUT = REPORT / "stage20k_xjtu_pickle_inventory.csv"
JSON_OUT = REPORT / "stage20k_xjtu_static_report.json"

EXPECTED_MD5 = "b45b0df2be827f332528b28fbf2a5e37"
EXPECTED_SHA = "8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c"

PREFIX_LIMIT = 3 * 1024 * 1024
OPERATION_LIMIT = 100000

OBJECT_OPERATIONS = {
    "GLOBAL", "STACK_GLOBAL", "REDUCE",
    "BUILD", "NEWOBJ", "NEWOBJ_EX",
    "INST", "OBJ", "EXT1", "EXT2", "EXT4",
}

TEXT_OPERATIONS = {
    "SHORT_BINUNICODE", "BINUNICODE",
    "BINUNICODE8", "UNICODE",
}

def hash_file(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def inspect_prefix(z, info):
    with z.open(info) as stream:
        payload = stream.read(PREFIX_LIMIT)

    counts = Counter()
    recent_text = deque(maxlen=8)
    references = []
    parsed = 0
    stop_seen = False
    reason = "PREFIX_INCOMPLETE"

    try:
        for op, arg, position in pickletools.genops(
            io.BytesIO(payload)
        ):
            parsed += 1
            counts[op.name] += 1

            if op.name in TEXT_OPERATIONS and isinstance(arg, str):
                if len(arg) <= 180:
                    recent_text.append(arg)

            if op.name in ("GLOBAL", "STACK_GLOBAL"):
                if len(references) < 12:
                    references.append({
                        "opcode": op.name,
                        "position": position,
                        "argument": (
                            str(arg)[:180]
                            if arg is not None else None
                        ),
                        "recent_text_context": list(recent_text)[-5:],
                    })

            if op.name == "STOP":
                stop_seen = True
                reason = "FULL_STREAM_IN_PREFIX"
                break

            if parsed >= OPERATION_LIMIT:
                reason = "OPERATION_LIMIT"
                break

    except Exception as exc:
        reason = (
            "TRUNCATED_PREFIX_OR_PARSE_ERROR:"
            + type(exc).__name__
        )

    return {
        "prefix_bytes": len(payload),
        "operations_inspected": parsed,
        "prefix_status": reason,
        "stop_seen": stop_seen,
        "object_opcodes": {
            k: counts[k]
            for k in sorted(OBJECT_OPERATIONS)
            if counts[k]
        },
        "referenced_global_context": references,
        "top_opcodes": counts.most_common(12),
    }

def main():
    if CSV_OUT.exists() or JSON_OUT.exists():
        raise SystemExit(
            "Stage20K outputs exist; refusing overwrite"
        )

    assert hash_file(RAW, "md5") == EXPECTED_MD5
    assert hash_file(BENCH, "sha256") == EXPECTED_SHA

    benchmark = {
        r["cell_id"]: r
        for r in read_csv(BENCH)
        if r["dataset_id"] == "XJTU"
        and r["source_family"] == "BatteryLife_selected"
    }

    old = {
        r["cell_id"]: r
        for r in read_csv(STAGE_H)
        if r["dataset"] == "XJTU"
    }

    assert len(benchmark) == 23
    assert len(old) == 23

    rows = []
    details = []

    with zipfile.ZipFile(RAW) as z:
        infos = {
            i.filename: i
            for i in z.infolist()
            if not i.is_dir()
            and i.filename.lower().endswith(".pkl")
        }

        assert len(infos) == 23

        for cell, b in sorted(benchmark.items()):
            member = b["source_file"].replace("\\", "/")
            assert member in infos

            info = infos[member]
            structure = inspect_prefix(z, info)

            row = {
                "cell_id": cell,
                "source_file": member,
                "uncompressed_MB": round(
                    info.file_size / 1048576, 3
                ),
                "compressed_MB": round(
                    info.compress_size / 1048576, 3
                ),
                "exceeds_previous_128MB_limit": int(
                    info.file_size > 128 * 1048576
                ),
                "previous_decoder_status": old[cell]["decode_status"],
                "operations_inspected": structure["operations_inspected"],
                "prefix_status": structure["prefix_status"],
                "stack_global_in_prefix": structure["object_opcodes"].get(
                    "STACK_GLOBAL", 0
                ),
                "reduce_in_prefix": structure["object_opcodes"].get(
                    "REDUCE", 0
                ),
            }

            rows.append(row)

            details.append({
                **row,
                "static_structure": structure,
            })

            print(
                "XJTU_CELL",
                cell,
                "MB", row["uncompressed_MB"],
                "PREVIOUS", row["previous_decoder_status"],
                "GLOBALS", row["stack_global_in_prefix"],
                "REDUCE", row["reduce_in_prefix"],
                flush=True
            )

    total_bytes = sum(
        int(infos[r["source_file"]].file_size)
        for r in rows
    )

    report = {
        "stage": "20K",
        "source_archive_md5_verified": True,
        "benchmark_sha256_verified": True,
        "cells": len(rows),
        "total_uncompressed_MB": round(
            total_bytes / 1048576, 3
        ),
        "over_128MB": sum(
            r["exceeds_previous_128MB_limit"]
            for r in rows
        ),
        "previous_decoder_statuses": dict(Counter(
            r["previous_decoder_status"] for r in rows
        )),
        "details": details,
        "limitations": [
            "Only the first 3 MiB per pickle was statically inspected.",
            "Operation limit may stop inspection earlier.",
            "Global contexts are hints, not a complete symbolic stack evaluation.",
            "Static inspection does not establish that a pickle is safe to execute.",
            "No pickle object was deserialized.",
            "No capacity data were modified or reconstructed."
        ]
    }

    with CSV_OUT.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        w.writeheader()
        w.writerows(rows)

    JSON_OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print("=== STAGE20K SUMMARY ===")
    print("XJTU_CELLS", len(rows))
    print("OVER_128MB", report["over_128MB"])
    print("TOTAL_UNCOMPRESSED_MB", report["total_uncompressed_MB"])
    print("PREVIOUS_STATUSES", report["previous_decoder_statuses"])

    for item in details[:5]:
        print(
            "GLOBAL_CONTEXT",
            item["cell_id"],
            item["static_structure"]["referenced_global_context"][:4]
        )

    print("STAGE20K_STATIC_AUDIT_COMPLETE=YES")

if __name__ == "__main__":
    main()
