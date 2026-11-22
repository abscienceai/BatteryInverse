#!/usr/bin/env python3
"""Stage20L: bounded, restricted NumPy-scalar XJTU pilot."""

import csv
import hashlib
import json
import math
import os
import pickle
import pickletools
import resource
import struct
import subprocess
import sys
import zipfile

from pathlib import Path
from statistics import median

ROOT = Path("/home/msoylu/alper/BatteryInverse")
BENCH = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)
ARCHIVE = ROOT / (
    "data/00_external_raw/aging/"
    "BATTERYLIFE_PROCESSED/original/XJTU.zip"
)
REPORT = ROOT / (
    "external_validation/reports/"
    "stage20l_xjtu_restricted_pilot.json"
)

EXPECTED_SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)
EXPECTED_MD5 = "b45b0df2be827f332528b28fbf2a5e37"

MAX_MEMBER = 180 * 1024 * 1024
MAX_OPS = 30_000_000

ALLOWED_OPS = set("""
PROTO FRAME STOP MARK POP POP_MARK DUP
NONE NEWTRUE NEWFALSE
INT BININT BININT1 BININT2 LONG LONG1 LONG4
FLOAT BINFLOAT
STRING BINSTRING SHORT_BINSTRING UNICODE BINUNICODE
SHORT_BINUNICODE BINUNICODE8
BINBYTES SHORT_BINBYTES BINBYTES8 BYTEARRAY8
EMPTY_TUPLE TUPLE TUPLE1 TUPLE2 TUPLE3
EMPTY_LIST LIST APPEND APPENDS
EMPTY_DICT DICT SETITEM SETITEMS
EMPTY_SET ADDITEMS FROZENSET
MEMOIZE BINPUT LONG_BINPUT PUT
BINGET LONG_BINGET GET
STACK_GLOBAL REDUCE BUILD
""".split())

FORMATS = {
    "f8": "d", "f4": "f", "f2": "e",
    "i8": "q", "i4": "i", "i2": "h", "i1": "b",
    "u8": "Q", "u4": "I", "u2": "H", "u1": "B",
    "b1": "?",
}


class SafeDType:
    def __init__(self, code, align=False, copy=False):
        if type(code) is not str or code not in FORMATS:
            raise pickle.UnpicklingError(
                "Unsupported NumPy dtype"
            )
        if type(align) is not bool or type(copy) is not bool:
            raise pickle.UnpicklingError(
                "Invalid dtype construction flags"
            )
        self.code = code
        self.endian = "<"

    def __setstate__(self, state):
        if (
            type(state) is not tuple
            or len(state) != 8
            or state[0] not in (3, 4)
            or state[1] not in ("<", ">", "=", "|")
        ):
            raise pickle.UnpicklingError(
                "Unsupported dtype state"
            )
        self.endian = state[1]


def safe_scalar(dtype, payload):
    if (
        type(dtype) is not SafeDType
        or type(payload) not in (bytes, bytearray)
    ):
        raise pickle.UnpicklingError(
            "Invalid scalar arguments"
        )

    fmt = FORMATS[dtype.code]

    if len(payload) != struct.calcsize(fmt):
        raise pickle.UnpicklingError(
            "Scalar byte count mismatch"
        )

    endian = dtype.endian

    if endian == "|":
        if len(payload) != 1:
            raise pickle.UnpicklingError(
                "Invalid endian-independent scalar"
            )
        endian = "<"

    if endian == "=":
        endian = (
            "<" if sys.byteorder == "little" else ">"
        )

    return struct.unpack(
        endian + fmt, payload
    )[0]


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (
            module in (
                "numpy.core.multiarray",
                "numpy._core.multiarray"
            )
            and name == "scalar"
        ):
            return safe_scalar

        if module == "numpy" and name == "dtype":
            return SafeDType

        raise pickle.UnpicklingError(
            f"Forbidden global: {module}.{name}"
        )

    def persistent_load(self, pid):
        raise pickle.UnpicklingError(
            "Persistent objects forbidden"
        )


def digest(path, name):
    h = hashlib.new(name)
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def static_gate(z, info):
    if info.file_size > MAX_MEMBER:
        raise ValueError("Member exceeds size limit")

    operations = 0
    globals_seen = 0
    last = None

    with z.open(info) as stream:
        for op, arg, position in pickletools.genops(
            stream
        ):
            if op.name not in ALLOWED_OPS:
                raise ValueError(
                    "Forbidden opcode: " + op.name
                )

            operations += 1

            if operations > MAX_OPS:
                raise ValueError(
                    "Exceeded opcode limit"
                )

            if op.name == "STACK_GLOBAL":
                globals_seen += 1

            last = op.name

    if last != "STOP":
        raise ValueError("Missing pickle STOP")

    return {
        "static_operations": operations,
        "stack_global_opcodes": globals_seen
    }


def cycle_number(value):
    try:
        x = float(value)
        if (
            math.isfinite(x)
            and x >= 1
            and x.is_integer()
        ):
            return int(x)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def capacity_max(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]

    if len(values) > 2_000_000:
        raise ValueError("Oversized capacity series")

    result = None

    for x in values:
        if type(x) not in (int, float, bool):
            continue

        v = float(x)

        if not math.isfinite(v):
            continue

        if result is None or v > result:
            result = v

    return result


def examine(obj, cell, target):
    if type(obj) is not dict:
        raise ValueError("Root is not a dictionary")

    if str(obj.get("cell_id")) != cell:
        raise ValueError("Source cell identity mismatch")

    cycles = obj.get("cycle_data")

    if type(cycles) not in (list, tuple):
        raise ValueError("Invalid cycle_data")

    if len(cycles) > 1_000_000:
        raise ValueError("Excessive cycle count")

    measured = []
    invalid_cycles = 0

    for position, record in enumerate(cycles):
        if type(record) is not dict:
            invalid_cycles += 1
            continue

        # Preserve source-builder fallback semantics.
        raw_cycle = record.get(
            "cycle_number", position + 1
        )

        n = cycle_number(raw_cycle)

        if n is None:
            invalid_cycles += 1
            continue

        cap = capacity_max(
            record.get("discharge_capacity_in_Ah")
        )

        if cap is None:
            continue

        measured.append((n, position, cap))

    if len(measured) < 3:
        raise ValueError(
            "Too few valid capacity records"
        )

    measured.sort(
        key=lambda item: (item[0], item[1])
    )

    caps = [r[2] for r in measured]

    first = median(caps[:3])
    last = median(caps[-3:])

    if first == 0:
        raise ValueError(
            "Zero capacity normalization denominator"
        )

    rebuilt = last / first

    positive = [
        (n, value)
        for n, _, value in measured
        if value > 0
    ]

    return {
        "cell_id": cell,
        "source_identity_verified": True,
        "cycle_records": len(cycles),
        "finite_capacity_records": len(measured),
        "positive_capacity_records": len(positive),
        "nonpositive_capacity_records": sum(
            cap <= 0 for cap in caps
        ),
        "invalid_cycle_records": invalid_cycles,
        "first_observed_cycle": measured[0][0],
        "last_observed_cycle": measured[-1][0],
        "post100_positive_records": sum(
            n > 100 for n, _ in positive
        ),
        "exact_cycle100_positive": sum(
            n == 100 for n, _ in positive
        ),
        "exact_cycle200_positive": sum(
            n == 200 for n, _ in positive
        ),
        "exact_cycle300_positive": sum(
            n == 300 for n, _ in positive
        ),
        "target_reconstructed": rebuilt,
        "benchmark_target": target,
        "target_abs_difference": abs(
            rebuilt - target
        ),
        "target_match_1e6": (
            abs(rebuilt - target) < 1e-6
        ),
    }


def worker(member, cell, target):
    # Per-process resource limits, not server-wide limits.
    memory = 2560 * 1024 * 1024
    resource.setrlimit(
        resource.RLIMIT_AS, (memory, memory)
    )
    resource.setrlimit(
        resource.RLIMIT_CPU, (240, 240)
    )

    with zipfile.ZipFile(ARCHIVE) as z:
        info = z.getinfo(member)

        gate = static_gate(z, info)

        with z.open(info) as stream:
            obj = RestrictedUnpickler(
                stream
            ).load()

    result = examine(obj, cell, target)

    return {
        "status": "SUCCESS",
        "static_gate": gate,
        "measurement": result,
    }


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _, _, member, cell, target = sys.argv

        try:
            result = worker(
                member, cell, float(target)
            )
        except Exception as exc:
            result = {
                "status": "BLOCKED_OR_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:250],
            }

        print(json.dumps(
            result, ensure_ascii=False,
            allow_nan=False
        ))
        return

    if REPORT.exists():
        raise SystemExit(
            "Stage20L report already exists"
        )

    assert digest(BENCH, "sha256") == EXPECTED_SHA
    assert digest(ARCHIVE, "md5") == EXPECTED_MD5

    with BENCH.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    xjtu = {
        r["cell_id"]: r
        for r in rows
        if r["dataset_id"] == "XJTU"
        and r["source_family"] == "BatteryLife_selected"
    }

    assert len(xjtu) == 23

    pilot_cells = [
        "XJTU_3C_battery-11",
        "XJTU_2C_battery-1",
    ]

    results = {}

    for cell in pilot_cells:
        assert cell in xjtu
        row = xjtu[cell]

        member = row["source_file"].replace(
            "\\", "/"
        )

        cmd = [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).resolve()),
            "--worker",
            member,
            cell,
            row["final_capacity_retention"],
        ]

        print(
            "PROCESSING", cell,
            flush=True
        )

        try:
            process = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=300,
                cwd="/",
            )

            if process.returncode == 0:
                result = json.loads(
                    process.stdout.strip()
                )
            else:
                result = {
                    "status": "PROCESS_FAILED",
                    "returncode": process.returncode,
                    "stderr": process.stderr[-500:],
                }

        except subprocess.TimeoutExpired:
            result = {
                "status": "TIMEOUT"
            }
        except Exception as exc:
            result = {
                "status": "PROCESS_ERROR",
                "error": str(exc)[:250],
            }

        results[cell] = result

        print(
            "RESULT",
            cell,
            result["status"],
            flush=True
        )

        if result["status"] == "SUCCESS":
            m = result["measurement"]
            print(
                "  LAST_CYCLE",
                m["last_observed_cycle"]
            )
            print(
                "  POST100",
                m["post100_positive_records"]
            )
            print(
                "  TARGET_DIFFERENCE",
                m["target_abs_difference"]
            )

        else:
            print(
                "  DETAIL",
                result.get(
                    "error",
                    result.get("stderr", "")
                )
            )

    output = {
        "stage": "20L",
        "pilot_cells": pilot_cells,
        "results": results,
        "all_pilot_successful": all(
            r["status"] == "SUCCESS"
            for r in results.values()
        ),
        "pickle_restrictions": (
            "Static opcode allowlist; controlled numeric "
            "scalar and dtype reconstruction only; "
            "no external NumPy callable resolution"
        ),
        "resource_limits": {
            "max_address_space_MiB": 2560,
            "cpu_limit_seconds_per_worker": 240,
        },
        "limitations": [
            "Restricted unpickling reduces risk but is not a complete sandbox.",
            "Only two XJTU cells were inspected.",
            "No model training or benchmark modification occurred.",
        ],
    }

    REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT.write_text(
        json.dumps(
            output, indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )

    print("=== STAGE20L SUMMARY ===")
    print(
        "SUCCESSFUL_PILOT_CELLS",
        sum(
            r["status"] == "SUCCESS"
            for r in results.values()
        )
    )
    print(
        "PILOT_SOURCE_TARGET_MATCHES",
        sum(
            r["status"] == "SUCCESS"
            and r["measurement"]["target_match_1e6"]
            for r in results.values()
        )
    )
    print(
        "STAGE20L_PILOT_COMPLETE=YES"
    )
    print("REPORT", REPORT)


if __name__ == "__main__":
    main()
