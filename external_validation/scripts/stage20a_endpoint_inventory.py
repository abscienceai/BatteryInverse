#!/usr/bin/env python3
"""Stage 20A: read-only source metadata and endpoint feasibility inventory."""

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path.cwd().resolve()
REPORT = ROOT / "external_validation/reports"
DATA = ROOT / "data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv"

OUT = {
    "profile": REPORT / "stage20a_dataset_profile.csv",
    "candidates": REPORT / "stage20a_longitudinal_candidates.csv",
    "scripts": REPORT / "stage20a_source_script_hints.csv",
    "summary": REPORT / "stage20a_endpoint_inventory.json",
}

EXPECTED_SHA = "8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c"

SEARCH_ROOTS = (
    "data", "external_validation", "raw", "raw_data",
    "source_data", "datasets", "downloads", "archives",
    "inputs", "upstream", "external_data",
    "reports/data_audits",
)

SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".cache",
    ".mypy_cache", ".pytest_cache",
    "node_modules", "site-packages", "build", "dist",
}

EXT = {
    ".csv", ".tsv", ".txt", ".json", ".jsonl",
    ".ndjson", ".parquet", ".feather",
    ".pkl", ".pickle", ".h5", ".hdf5",
    ".mat", ".npz", ".npy", ".zip", ".gz",
    ".7z", ".tar", ".xlsx", ".xls",
    ".yaml", ".yml",
}

CYCLE = (
    "cycle", "cyc", "rpt", "time",
    "timestamp", "step", "test_index",
)

CAPACITY = (
    "capacity", "cap_", "charge_ah",
    "discharge_ah", "q_dis", "q_charge", "qmax",
)

def fail(message):
    raise SystemExit("STAGE20A_ERROR: " + message)

def probe_header(path):
    try:
        if path.name.lower().endswith(".gz"):
            import gzip
            with gzip.open(path, "rb") as f:
                head = f.read(16384)
        else:
            with path.open("rb") as f:
                head = f.read(16384)

        line = head.decode(
            "utf-8-sig", errors="replace"
        ).splitlines()[0]

        delimiter = max(
            (",", ";", "\t", "|"),
            key=line.count
        )

        columns = next(
            csv.reader([line], delimiter=delimiter)
        )

        return [
            c.strip()[:80] for c in columns[:60]
        ] if len(columns) > 1 else []

    except (OSError, UnicodeError, ValueError,
            IndexError, csv.Error):
        return []

def write_csv(path, records, columns):
    with path.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)

def main():

    if not DATA.is_file():
        fail("Canonical benchmark not found")

    digest = hashlib.sha256(
        DATA.read_bytes()
    ).hexdigest()

    if digest != EXPECTED_SHA:
        fail("Original dataset SHA256 mismatch")

    if any(p.exists() for p in OUT.values()):
        fail("Existing Stage20A results; no overwrite")

    with DATA.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    if len(rows) != 567 or len(columns) != 38:
        fail("Unexpected benchmark dimensions")

    required = {
        "dataset_id", "cell_id",
        "source_family", "source_file",
        "final_capacity_retention",
        "cap_cycle_100",
    }

    if not required.issubset(columns):
        fail("Required benchmark columns missing")

    if len({
        (r["dataset_id"], r["cell_id"])
        for r in rows
    }) != 567:
        fail("Duplicate dataset/cell combinations")

    groups = defaultdict(list)
    hints = defaultdict(Counter)
    all_hints = Counter()

    for r in rows:
        domain = r["dataset_id"]
        groups[domain].append(r)

        value = (r.get("source_file") or "").strip()

        if value and value.lower() not in (
            "nan", "none", "null"
        ):
            basename = Path(
                value.replace("\\", "/")
            ).name.lower()

            hints[domain][basename] += 1
            all_hints[basename] += 1

    starts = [
        ROOT / name
        for name in SEARCH_ROOTS
        if (ROOT / name).is_dir()
    ]

    candidates = []
    names_found = set()
    examined = 0
    truncated = False

    for start in starts:
        for dirname, dirs, files in os.walk(
            start, followlinks=False
        ):

            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS
                and not (Path(dirname) / d).is_symlink()
            ]

            for name in files:

                examined += 1

                if examined > 150000:
                    truncated = True
                    break

                path = Path(dirname) / name

                if path.is_symlink() or not path.is_file():
                    continue

                lower = name.lower()

                if (
                    path.suffix.lower() not in EXT
                    and not lower.endswith(
                        (".csv.gz", ".tar.gz")
                    )
                ):
                    continue

                rel = path.relative_to(ROOT).as_posix()
                low = rel.lower()

                score = 0

                if any(k in low for k in (
                    "raw", "original", "upstream"
                )):
                    score += 3

                if any(k in low for k in (
                    "processed", "longitudinal"
                )):
                    score += 2

                if any(k in low for k in (
                    "cycle", "rpt", "capacity",
                    "trajectory", "timeseries",
                    "aging", "degrad"
                )):
                    score += 2

                if any(k in low for k in (
                    "matr", "hust", "calce", "rwth",
                    "calb", "hnei", "mich", "na-ion",
                    "xjtu", "ul_pur", "snl"
                )):
                    score += 2

                if lower in all_hints:
                    score += 7

                if "/reports/" in low:
                    score -= 3

                if lower.startswith((
                    "stage17", "stage18",
                    "stage19", "stage20"
                )):
                    score -= 3

                if rel == DATA.relative_to(
                    ROOT
                ).as_posix():
                    score -= 10

                if score < 2:
                    continue

                try:
                    size = path.stat().st_size
                except OSError:
                    continue

                names_found.add(lower)

                candidates.append({
                    "path": rel,
                    "bytes": size,
                    "extension": path.suffix.lower(),
                    "score": score,
                    "source_hint_matches": all_hints[lower],
                    "schema_probe": "",
                    "cycle_header_hint": False,
                    "capacity_header_hint": False,
                })

            if truncated:
                break

        if truncated:
            break

    candidates.sort(
        key=lambda c: (
            -c["score"],
            -c["source_hint_matches"],
            c["path"]
        )
    )

    total_candidates = len(candidates)
    selected = candidates[:5000]

    # Inspect only small header segments, not full files.
    for c in selected[:300]:

        path = ROOT / c["path"]

        if path.suffix.lower() not in (
            ".csv", ".tsv", ".txt", ".gz"
        ):
            continue

        header = probe_header(path)
        tokens = [
            x.lower().replace(" ", "_")
            for x in header
        ]

        c["schema_probe"] = " | ".join(header)[:900]

        c["cycle_header_hint"] = any(
            any(k in t for k in CYCLE)
            for t in tokens
        )

        c["capacity_header_hint"] = any(
            any(k in t for k in CAPACITY)
            for t in tokens
        )

    profiles = []

    for domain, items in sorted(groups.items()):

        values = [
            float(r["final_capacity_retention"])
            for r in items
        ]

        profiles.append({
            "dataset": domain,
            "family": items[0]["source_family"],
            "cells": len(items),
            "source_file_hints": sum(
                hints[domain].values()
            ),
            "unique_source_hints": len(hints[domain]),
            "matched_hint_rows": sum(
                count
                for basename, count in hints[domain].items()
                if basename in names_found
            ),
            "target_min": min(values),
            "target_median": median(values),
            "target_max": max(values),
            "observed_post100_certified": "NO",
            "common_protocol_certified": "NO",
            "raw_cell_lineage_certified": "NO",
        })

    script_hints = []

    for directory in (
        "scripts", "external_validation/scripts"
    ):
        folder = ROOT / directory

        if not folder.is_dir():
            continue

        for path in folder.glob("*.py"):

            try:
                if path.stat().st_size > 2000000:
                    continue

                src = path.read_text(
                    encoding="utf-8",
                    errors="replace"
                ).lower()
            except OSError:
                continue

            terms = [
                k for k in (
                    "source_file", "read_pickle",
                    "read_csv", "read_hdf",
                    "read_parquet", "rpt", "cycle",
                    "batteryml", "batterylife",
                    "raw_data", "source_data",
                    ".pkl", ".mat",
                    "final_capacity_retention"
                )
                if k in src
            ]

            if not terms:
                continue

            score = (
                3 * ("source_file" in terms)
                + 2 * ("read_pickle" in terms)
                + 2 * ("rpt" in terms)
                + 2 * ("raw_data" in terms)
                + int("cycle" in terms)
                + int("batterylife" in terms)
                + int("batteryml" in terms)
            )

            script_hints.append({
                "script_path": path.relative_to(
                    ROOT
                ).as_posix(),
                "score": score,
                "source_terms": " | ".join(terms),
            })

    script_hints.sort(
        key=lambda x: (-x["score"], x["script_path"])
    )

    report = {
        "analysis": "Stage20A physical endpoint inventory",
        "benchmark_sha256": digest,
        "rows": len(rows),
        "columns": len(columns),
        "datasets": len(groups),
        "source_hints_present": sum(
            p["source_file_hints"] for p in profiles
        ),
        "search_roots": [
            p.relative_to(ROOT).as_posix()
            for p in starts
        ],
        "files_examined": examined,
        "candidate_files": total_candidates,
        "candidate_rows_saved": len(selected),
        "scan_truncated": truncated,
        "candidate_list_truncated":
            total_candidates > len(selected),
        "script_candidates": len(script_hints),
        "header_probes": min(300, len(selected)),
        "next_gate": "SOURCE_LINEAGE_AND_ENDPOINT_NOT_YET_VERIFIED",
        "limitations": [
            "File names and headers do not prove cell-level lineage.",
            "The final-retention field is not a standardized endpoint.",
            "First-100 features are not post-100 targets.",
            "No target interpolation or synthetic target is used.",
            "Only directories within the project were scanned.",
        ],
    }

    write_csv(
        OUT["profile"], profiles,
        list(profiles[0].keys())
    )

    write_csv(
        OUT["candidates"], selected,
        [
            "path", "bytes", "extension", "score",
            "source_hint_matches", "schema_probe",
            "cycle_header_hint",
            "capacity_header_hint"
        ]
    )

    write_csv(
        OUT["scripts"], script_hints,
        ["script_path", "score", "source_terms"]
    )

    with OUT["summary"].open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(
            report, f, indent=2, ensure_ascii=False
        )
        f.write("\n")

    print("=== STAGE20A BENCHMARK ===")
    print("BENCHMARK_SHA256=VERIFIED")
    print("BENCHMARK_ROWS=567")
    print("BENCHMARK_COLUMNS=38")

    for p in profiles:
        print(
            f"DOMAIN {p['dataset']:9s} "
            f"cells={p['cells']:3d} "
            f"hints={p['source_file_hints']:3d} "
            f"matched={p['matched_hint_rows']:3d}"
        )

    print("=== FILE INVENTORY ===")
    print("FILES_EXAMINED=", examined)
    print("CANDIDATE_FILES=", total_candidates)
    print("SCAN_TRUNCATED=", truncated)

    for s in script_hints[:15]:
        print(
            "SOURCE_SCRIPT",
            s["score"], s["script_path"]
        )

    for c in selected[:25]:
        print(
            "CANDIDATE",
            c["score"],
            f"{c['bytes']/1048576:.2f}MB",
            "cycle=" + str(int(c["cycle_header_hint"])),
            "capacity=" + str(
                int(c["capacity_header_hint"])
            ),
            c["path"][:130]
        )

    print("COMMON_PHYSICAL_ENDPOINT_VERIFIED=NO")
    print("STAGE20A_INVENTORY_COMPLETE=YES")

    for key, path in OUT.items():
        print(key.upper() + "_FILE=" + str(path))

if __name__ == "__main__":
    main()
