#!/usr/bin/env python3
"""Stage20M: bounded large-XJTU pilot using Stage20L restrictions."""

import csv
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/home/msoylu/alper/BatteryInverse")
ORIGINAL = ROOT / "external_validation/scripts/stage20l_xjtu_restricted_pilot.py"
BENCH = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"
ARCHIVE = ROOT / "data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original/XJTU.zip"
REPORT = ROOT / "external_validation/reports/stage20m_xjtu_large_pilot.json"

SHA = "8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c"
MD5 = "b45b0df2be827f332528b28fbf2a5e37"

CELL = "XJTU_2C_battery-1"
MAX_OPS = 120_000_000
MEMORY_GIB = 5
CPU_LIMIT = 720

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def load_restricted_decoder():
    if not ORIGINAL.is_file():
        raise FileNotFoundError(str(ORIGINAL))

    spec = importlib.util.spec_from_file_location(
        "stage20l_restricted", ORIGINAL
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Only increase the resource accounting threshold.
    # Retain the exact allowed opcode and global-class lists.
    mod.MAX_OPS = MAX_OPS

    return mod

def run_worker(member, target):

    address_space = MEMORY_GIB * 1024**3

    resource.setrlimit(
        resource.RLIMIT_AS,
        (address_space, address_space)
    )
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (CPU_LIMIT, CPU_LIMIT)
    )

    mod = load_restricted_decoder()

    with zipfile.ZipFile(ARCHIVE) as z:
        info = z.getinfo(member)

        # Preserve original opcode allowlist and file-size check.
        gate = mod.static_gate(z, info)

        print(
            "STATIC_OPCODE_GATE_PASSED",
            gate["static_operations"],
            file=sys.stderr,
            flush=True
        )

        # Only Stage20L's explicitly allowed scalar/dtype objects.
        with z.open(info) as stream:
            obj = mod.RestrictedUnpickler(stream).load()

    measurement = mod.examine(
        obj, CELL, float(target)
    )

    return {
        "status": "SUCCESS",
        "static_gate": gate,
        "measurement": measurement,
        "opcode_limit": MAX_OPS,
        "memory_limit_gib": MEMORY_GIB
    }

def main():

    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        member, target = sys.argv[2], sys.argv[3]

        try:
            result = run_worker(member, target)
        except Exception as exc:
            result = {
                "status": "BLOCKED_OR_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:300]
            }

        print(json.dumps(result, allow_nan=False))
        return

    if REPORT.exists():
        raise SystemExit(
            "Stage20M report already exists; refusing overwrite"
        )

    assert digest(BENCH, "sha256") == SHA
    assert digest(ARCHIVE, "md5") == MD5

    with BENCH.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    matches = [
        r for r in rows
        if r["dataset_id"] == "XJTU"
        and r["source_family"] == "BatteryLife_selected"
        and r["cell_id"] == CELL
    ]

    assert len(matches) == 1

    row = matches[0]
    member = row["source_file"].replace("\\", "/")

    with zipfile.ZipFile(ARCHIVE) as z:
        info = z.getinfo(member)

    available_kib = None
    meminfo = Path("/proc/meminfo")

    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break

    print("=== STAGE20M PREFLIGHT ===")
    print("CELL", CELL)
    print(
        "UNCOMPRESSED_MIB",
        round(info.file_size / 1048576, 3)
    )

    if available_kib is not None:
        print(
            "MEM_AVAILABLE_GIB",
            round(available_kib / 1024**2, 3)
        )

        if available_kib < 6 * 1024**2:
            raise SystemExit(
                "Insufficient available memory for bounded pilot"
            )

    print("MAX_OPCODES", MAX_OPS)
    print("MEMORY_LIMIT_GIB", MEMORY_GIB)
    print("PROCESSING", CELL, flush=True)

    cmd = [
        sys.executable,
        "-I",
        "-S",
        str(Path(__file__).resolve()),
        "--worker",
        member,
        row["final_capacity_retention"]
    ]

    try:
        process = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            timeout=900,
            cwd="/",
        )

        if process.returncode == 0:
            result = json.loads(
                process.stdout.strip()
            )
        else:
            result = {
                "status": "PROCESS_FAILED",
                "returncode": process.returncode
            }

    except subprocess.TimeoutExpired:
        result = {"status": "TIMEOUT"}
    except Exception as exc:
        result = {
            "status": "PROCESS_ERROR",
            "error": str(exc)[:300]
        }

    output = {
        "stage": "20M",
        "cell": CELL,
        "benchmark_verified": True,
        "archive_verified": True,
        "result": result,
        "models_retrained": False,
        "benchmark_modified": False
    }

    REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with REPORT.open("x", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )
        f.write("\n")

    print("=== STAGE20M RESULTS ===")
    print("STATUS", result["status"])

    if result["status"] == "SUCCESS":

        m = result["measurement"]

        print(
            "STATIC_OPCODES",
            result["static_gate"]["static_operations"]
        )
        print("LAST_CYCLE", m["last_observed_cycle"])
        print(
            "POST100",
            m["post100_positive_records"]
        )
        print(
            "EXACT_CYCLE200",
            m["exact_cycle200_positive"]
        )
        print(
            "TARGET_DIFFERENCE",
            m["target_abs_difference"]
        )
        print(
            "TARGET_MATCH",
            m["target_match_1e6"]
        )

    else:
        print(
            "FAILURE_DETAIL",
            result.get(
                "error",
                result.get("returncode", "Unknown")
            )
        )

    print("STAGE20M_PILOT_COMPLETE=YES")
    print("REPORT", REPORT)

if __name__ == "__main__":
    main()
