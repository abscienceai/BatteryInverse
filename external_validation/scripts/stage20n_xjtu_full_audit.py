#!/usr/bin/env python3
"""Stage 20N: resumable, restricted XJTU 23-cell audit."""

import csv
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import zipfile

from collections import Counter
from pathlib import Path

ROOT = Path("/home/msoylu/alper/BatteryInverse")
SCRIPTS = ROOT / "external_validation/scripts"
REPORTS = ROOT / "external_validation/reports"

BENCH = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)
ARCHIVE = ROOT / (
    "data/00_external_raw/aging/"
    "BATTERYLIFE_PROCESSED/original/XJTU.zip"
)

DECODER = SCRIPTS / "stage20l_xjtu_restricted_pilot.py"
PILOT_L = REPORTS / "stage20l_xjtu_restricted_pilot.json"
PILOT_M = REPORTS / "stage20m_xjtu_large_pilot.json"
STAGE_H = REPORTS / "stage20h_batterylife_249_temporal_cells.csv"
STAGE_J = REPORTS / "stage20j_mich_target_report.json"

CHECKPOINTS = REPORTS / "stage20n_xjtu_checkpoints"
FINAL = REPORTS / "stage20n_xjtu_full_report.json"
CSV_OUT = REPORTS / "stage20n_xjtu_full_cells.csv"

SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)
MD5 = "b45b0df2be827f332528b28fbf2a5e37"

MAX_OPS = 120_000_000
MEMORY_GIB = 5
CPU_SECONDS = 720
WALL_SECONDS = 900


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_decoder():
    spec = importlib.util.spec_from_file_location(
        "stage20l_restricted", DECODER
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Change the operation budget only.
    # Preserve the exact opcode and global allowlists.
    mod.MAX_OPS = MAX_OPS
    return mod


def worker(member, cell, target):

    limit = MEMORY_GIB * 1024**3

    resource.setrlimit(
        resource.RLIMIT_AS, (limit, limit)
    )
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (CPU_SECONDS, CPU_SECONDS)
    )

    mod = load_decoder()

    with zipfile.ZipFile(ARCHIVE) as z:
        info = z.getinfo(member)

        gate = mod.static_gate(z, info)

        with z.open(info) as stream:
            obj = mod.RestrictedUnpickler(stream).load()

    measurement = mod.examine(
        obj, cell, float(target)
    )

    return {
        "status": "SUCCESS",
        "static_gate": gate,
        "measurement": measurement,
    }


def valid_result(result, cell, target):
    try:
        m = result["measurement"]

        return (
            result["status"] == "SUCCESS"
            and m["cell_id"] == cell
            and m["source_identity_verified"] is True
            and m["target_match_1e6"] is True
            and abs(
                float(m["benchmark_target"]) - float(target)
            ) < 1e-10
            and float(m["target_abs_difference"]) < 1e-6
            and int(m["finite_capacity_records"]) >= 3
            and int(m["post100_positive_records"]) >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def previously_verified(cell, target):

    if cell == "XJTU_3C_battery-11" and PILOT_L.exists():
        data = json.loads(PILOT_L.read_text())
        result = data.get("results", {}).get(cell)

        if result and valid_result(result, cell, target):
            return result, "REUSED_STAGE20L"

    if cell == "XJTU_2C_battery-1" and PILOT_M.exists():
        data = json.loads(PILOT_M.read_text())

        if data.get("cell") == cell:
            result = data.get("result")

            if result and valid_result(result, cell, target):
                return result, "REUSED_STAGE20M"

    return None, None


def launch_worker(member, cell, target):

    command = [
        sys.executable,
        "-I",
        "-S",
        str(Path(__file__).resolve()),
        "--worker",
        member,
        cell,
        str(target)
    ]

    try:
        process = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=WALL_SECONDS,
            cwd="/"
        )

        if process.returncode != 0:
            return {
                "status": "PROCESS_FAILED",
                "returncode": process.returncode,
                "stderr": process.stderr[-400:]
            }

        return json.loads(process.stdout.strip())

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}

    except Exception as exc:
        return {
            "status": "PROCESS_ERROR",
            "error": (
                type(exc).__name__ + ": " + str(exc)[:250]
            )
        }


def checkpoint_path(cell):
    if not all(
        c.isalnum() or c in "._-"
        for c in cell
    ):
        raise ValueError("Unexpected cell ID")

    return CHECKPOINTS / (cell + ".json")


def store_checkpoint(path, record):

    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            record, indent=2,
            ensure_ascii=False, allow_nan=False
        ) + "\n",
        encoding="utf-8"
    )

    os.replace(temporary, path)


def get_checkpoint(cell, member, target):

    path = checkpoint_path(cell)

    if not path.exists():
        return None

    record = json.loads(path.read_text())

    if (
        record.get("cell_id") != cell
        or record.get("source_file") != member
        or record.get("benchmark_sha256") != SHA
        or record.get("archive_md5") != MD5
        or abs(
            float(record.get("target")) - float(target)
        ) > 1e-10
    ):
        raise ValueError(
            "Stale or inconsistent checkpoint: " + cell
        )

    return record


def consolidated_summary(results):

    all_verified = all(
        valid_result(
            r["result"], r["cell_id"], r["target"]
        )
        for r in results
    )

    if not all_verified:
        return {
            "status": "INCOMPLETE",
            "reason": "Some XJTU cells are unresolved"
        }

    old = [
        r for r in read_csv(STAGE_H)
        if r["dataset"] != "XJTU"
    ]

    if len(old) != 226:
        raise ValueError("Unexpected Stage20H population")

    for r in old:
        if (
            r["decode_status"] != "ACCEPTED_PRIMITIVES"
            or int(r["content_id_match"]) != 1
            or int(r["valid_capacity_records"]) < 1
        ):
            raise ValueError(
                "Unverified Stage20H cell: " + r["cell_id"]
            )

    mich = json.loads(STAGE_J.read_text())

    if (
        mich["cells"] != 40
        or mich["original_formula_matches"] != 40
    ):
        raise ValueError(
            "MICH original-target reconciliation incomplete"
        )

    mich_verified = {
        r["cell_id"]
        for r in mich["results"]
        if r["original_formula_matches_1e6"] == 1
    }

    if len(mich_verified) != 40:
        raise ValueError("MICH cell verification mismatch")

    for r in old:
        if r["dataset"] == "MICH":
            if r["cell_id"] not in mich_verified:
                raise ValueError(
                    "Unverified MICH target"
                )
        elif not (
            r["target_rebuild_absdiff"]
            and float(
                r["target_rebuild_absdiff"]
            ) < 1e-6
        ):
            raise ValueError(
                "Other-source target mismatch: "
                + r["cell_id"]
            )

    old_post = sum(
        int(r["post100_records"]) > 0
        for r in old
    )

    new_post = sum(
        int(
            r["result"]["measurement"][
                "post100_positive_records"
            ]
        ) > 0
        for r in results
    )

    total = len(old) + len(results)
    with_post = old_post + new_post

    return {
        "status": "ALL_SOURCE_TARGETS_RECONSTRUCTED",
        "batterylife_cells": total,
        "verified_previous_cells": len(old),
        "verified_xjtu_cells": len(results),
        "cells_with_positive_post100_observations": with_post,
        "cells_without_positive_post100_observations": (
            total - with_post
        ),
        "mich_target_formula_reconciled_cells": 40,
        "interpretation": (
            "This establishes source-target reconstruction "
            "and observed temporal coverage, not common "
            "physical endpoint equivalence."
        )
    }


def parent():

    if FINAL.exists() or CSV_OUT.exists():
        raise SystemExit(
            "Stage20N final report already exists"
        )

    if digest(BENCH, "sha256") != SHA:
        raise SystemExit("Benchmark checksum mismatch")

    if digest(ARCHIVE, "md5") != MD5:
        raise SystemExit("XJTU archive checksum mismatch")

    if not DECODER.exists():
        raise SystemExit("Stage20L decoder missing")

    records = read_csv(BENCH)

    xjtu = {
        r["cell_id"]: r
        for r in records
        if r["dataset_id"] == "XJTU"
        and r["source_family"] == "BatteryLife_selected"
    }

    if len(xjtu) != 23:
        raise SystemExit("Unexpected XJTU cell count")

    with zipfile.ZipFile(ARCHIVE) as z:
        members = {
            i.filename
            for i in z.infolist()
            if not i.is_dir()
            and i.filename.lower().endswith(".pkl")
        }

    if len(members) != 23:
        raise SystemExit("Unexpected XJTU archive members")

    CHECKPOINTS.mkdir(
        parents=True, exist_ok=True
    )

    results = []

    print("=== STAGE20N XJTU 23-CELL AUDIT ===")

    for cell, row in sorted(xjtu.items()):

        member = row["source_file"].replace("\\", "/")
        target = float(
            row["final_capacity_retention"]
        )

        if member not in members:
            raise SystemExit(
                "Missing archive member: " + member
            )

        record = get_checkpoint(
            cell, member, target
        )

        if record is None:

            result, origin = previously_verified(
                cell, target
            )

            if result is None:
                origin = "NEW_RESTRICTED_AUDIT"

                print(
                    "PROCESSING", cell,
                    flush=True
                )

                result = launch_worker(
                    member, cell, target
                )

            record = {
                "cell_id": cell,
                "source_file": member,
                "target": target,
                "benchmark_sha256": SHA,
                "archive_md5": MD5,
                "origin": origin,
                "result": result
            }

            store_checkpoint(
                checkpoint_path(cell),
                record
            )

        good = valid_result(
            record["result"], cell, target
        )

        print(
            "CELL", cell,
            "ORIGIN", record["origin"],
            "STATUS", record["result"].get("status"),
            "TARGET_MATCH", good,
            flush=True
        )

        if not good:
            print(
                "  DETAIL",
                record["result"].get(
                    "error",
                    record["result"].get(
                        "stderr", ""
                    )
                ),
                flush=True
            )

        results.append(record)

    assert len(results) == 23

    successful = [
        r for r in results
        if valid_result(
            r["result"], r["cell_id"], r["target"]
        )
    ]

    group_summary = {}

    for group in ("2C", "3C"):
        group_cells = [
            r for r in results
            if r["cell_id"].startswith(
                "XJTU_" + group + "_"
            )
        ]

        good = [
            r for r in group_cells
            if valid_result(
                r["result"], r["cell_id"], r["target"]
            )
        ]

        group_summary[group] = {
            "cells": len(group_cells),
            "source_target_matches": len(good),
            "positive_post100_cells": sum(
                r["result"]["measurement"][
                    "post100_positive_records"
                ] > 0
                for r in good
            ),
            "without_positive_post100": sum(
                r["result"]["measurement"][
                    "post100_positive_records"
                ] == 0
                for r in good
            ),
            "exact_cycle200_positive_cells": sum(
                r["result"]["measurement"][
                    "exact_cycle200_positive"
                ] == 1
                for r in good
            )
        }

    combined = consolidated_summary(results)

    report = {
        "stage": "20N",
        "source_record": "Zenodo 14934405 v2",
        "benchmark_sha256": SHA,
        "xjtu_archive_md5": MD5,
        "xjtu_total_cells": len(results),
        "xjtu_verified_targets": len(successful),
        "reused_pilots": sum(
            r["origin"].startswith("REUSED_")
            for r in results
        ),
        "xjtu_groups": group_summary,
        "overall_batterylife_summary": combined,
        "unresolved": [
            {
                "cell_id": r["cell_id"],
                "result": r["result"]
            }
            for r in results
            if not valid_result(
                r["result"], r["cell_id"], r["target"]
            )
        ],
        "pickle_controls": {
            "opcode_budget": MAX_OPS,
            "memory_limit_GiB": MEMORY_GIB,
            "cpu_seconds": CPU_SECONDS,
            "workers_sequential": True,
            "only_Stage20L_allowed_classes": True
        },
        "limitations": [
            "Restricted unpickling is not a complete security sandbox.",
            "Target reconstruction is not evidence of predictive accuracy.",
            "Cycle indices alone do not establish physical comparability.",
            "No battery model was trained or modified."
        ]
    }

    with FINAL.open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(
            report, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")

    with CSV_OUT.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "cell_id",
            "source_file",
            "origin",
            "status",
            "target_match",
            "last_observed_cycle",
            "post100_positive_records",
            "exact_cycle200_positive",
            "target_abs_difference"
        ]

        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()

        for r in results:
            result = r["result"]
            m = result.get("measurement", {})

            writer.writerow({
                "cell_id": r["cell_id"],
                "source_file": r["source_file"],
                "origin": r["origin"],
                "status": result.get("status"),
                "target_match": int(valid_result(
                    result, r["cell_id"], r["target"]
                )),
                "last_observed_cycle": m.get(
                    "last_observed_cycle", ""
                ),
                "post100_positive_records": m.get(
                    "post100_positive_records", ""
                ),
                "exact_cycle200_positive": m.get(
                    "exact_cycle200_positive", ""
                ),
                "target_abs_difference": m.get(
                    "target_abs_difference", ""
                )
            })

    print("=== STAGE20N RESULTS ===")
    print("XJTU_CELLS", len(results))
    print("SOURCE_TARGET_MATCHES", len(successful))

    for group, summary in group_summary.items():
        print("GROUP", group, summary)

    print("BATTERYLIFE_SUMMARY", combined)

    if len(successful) == 23:
        print("STAGE20N_FULL_AUDIT_COMPLETE=YES")
    else:
        print("STAGE20N_FULL_AUDIT_COMPLETE=NO")

    print("REPORT", FINAL)


def main():

    if len(sys.argv) > 1 and sys.argv[1] == "--worker":

        _, _, member, cell, target = sys.argv

        try:
            result = worker(
                member, cell, float(target)
            )
        except Exception as exc:
            result = {
                "status": "BLOCKED_OR_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:300]
            }

        print(json.dumps(
            result,
            allow_nan=False
        ))
        return

    parent()


if __name__ == "__main__":
    main()
