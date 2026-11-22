#!/usr/bin/env python3
"""Stage20P: audit quality flags and discover saved row-level predictions."""

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
REP = ROOT / "external_validation/reports"

BENCH = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)

FLAGS = REP / "stage20o_unified_endpoint_cells.csv"
REPORT = REP / "stage20o_unified_endpoint_report.json"
OUT = REP / "stage20p_prediction_artifact_inventory.json"

EXPECTED_SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)

ROOTS = [
    "external_validation",
    "data",
    "results",
    "outputs",
    "experiments",
    "runs",
    "artifacts",
    "releases",
]

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv",
    "node_modules", "raw", "00_external_raw",
    "stage20n_xjtu_checkpoints",
}

EXTENSIONS = {
    ".csv", ".parquet", ".feather",
    ".json", ".jsonl", ".npz",
}

def read_csv(path):
    with path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))

def columns_from_csv(path):
    try:
        with path.open(
            newline="", encoding="utf-8-sig",
            errors="replace"
        ) as f:
            return next(csv.reader(f), [])[:120]
    except Exception:
        return []

def columns_from_parquet(path):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(
            path
        ).schema_arrow.names[:120]
    except Exception:
        return []

def score_file(relative, columns):
    name = relative.lower()
    lower = [c.lower() for c in columns]
    score = 0

    terms = {
        "lodo": 4,
        "prediction": 4,
        "predict": 3,
        "holdout": 3,
        "heldout": 3,
        "out_of_fold": 3,
        "oof": 3,
        "stage17": 2,
        "stage19": 2,
        "baseline": 2,
        "fold": 1,
    }

    for term, weight in terms.items():
        if term in name:
            score += weight

    if "cell_id" in lower:
        score += 3

    if "dataset_id" in lower or "dataset" in lower:
        score += 2

    if any(
        "pred" in c or "yhat" in c
        for c in lower
    ):
        score += 6

    if any(
        c in lower
        for c in (
            "y_true", "true", "target",
            "actual", "final_capacity_retention"
        )
    ):
        score += 1

    if any(
        "prior" in c or "median" in c
        for c in lower
    ):
        score += 2

    return score

def main():
    if OUT.exists():
        raise SystemExit(
            "Stage20P inventory exists; refusing overwrite"
        )

    assert hashlib.sha256(
        BENCH.read_bytes()
    ).hexdigest() == EXPECTED_SHA

    benchmark = read_csv(BENCH)
    flags = read_csv(FLAGS)
    report = json.loads(
        REPORT.read_text(encoding="utf-8")
    )

    assert len(benchmark) == 567
    assert len(flags) == 249
    assert report["verified_source_targets"] == 249

    keys = [
        (r["dataset"], r["cell_id"])
        for r in flags
    ]

    assert len(set(keys)) == 249

    no_future = []
    sparse = []
    nonpositive = []
    candidates = []

    by_domain = defaultdict(
        lambda: Counter()
    )

    for r in flags:
        domain = r["dataset"]
        cell = r["cell_id"]

        post100 = int(
            r["positive_post100_records"]
        )

        quality = int(
            r["mich_nonpositive_endpoint_flag"]
        )

        assert float(
            r["source_target_absdiff"]
        ) < 1e-6

        assert post100 >= 0

        by_domain[domain]["total"] += 1

        if post100 == 0:
            no_future.append((domain, cell))
            by_domain[domain]["no_future"] += 1

        elif post100 < 3:
            sparse.append((domain, cell))
            by_domain[domain]["sparse"] += 1

        if quality == 1:
            nonpositive.append((domain, cell))
            by_domain[domain]["nonpositive"] += 1

        if post100 >= 3 and quality == 0:
            candidates.append((domain, cell))
            by_domain[domain]["candidate"] += 1

    combined_flags = set(
        no_future + sparse + nonpositive
    )

    assert len(no_future) == 29
    assert len(sparse) == 1
    assert len(nonpositive) == 14
    assert len(combined_flags) == 44
    assert len(candidates) == 205
    assert len(set(candidates)) == 205

    found = []
    scanned = 0

    for directory in ROOTS:
        base = ROOT / directory

        if not base.is_dir():
            continue

        for current, dirs, files in os.walk(
            base, followlinks=False
        ):
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS
                and not (
                    Path(current) / d
                ).is_symlink()
            ]

            for filename in files:
                path = Path(current) / filename

                if path.is_symlink():
                    continue

                if path.suffix.lower() not in EXTENSIONS:
                    continue

                scanned += 1

                relative = str(
                    path.relative_to(ROOT)
                )

                if path.suffix.lower() == ".csv":
                    columns = columns_from_csv(path)
                elif path.suffix.lower() == ".parquet":
                    columns = columns_from_parquet(path)
                else:
                    columns = []

                score = score_file(
                    relative, columns
                )

                if score < 4:
                    continue

                found.append({
                    "path": relative,
                    "size_mb": round(
                        path.stat().st_size / 1048576,
                        4
                    ),
                    "score": score,
                    "columns": columns,
                    "has_cell_id": (
                        "cell_id" in [
                            c.lower() for c in columns
                        ]
                    ),
                    "has_prediction_column": any(
                        "pred" in c.lower()
                        or "yhat" in c.lower()
                        for c in columns
                    ),
                })

    found.sort(
        key=lambda r: (
            -r["score"],
            r["path"]
        )
    )

    output = {
        "stage": "20P",
        "benchmark_sha256": EXPECTED_SHA,
        "audited_batterylife_cells": 249,
        "no_positive_post100": 29,
        "one_or_two_positive_post100": 1,
        "mich_nonpositive_endpoint": 14,
        "union_of_three_flags": 44,
        "candidate_without_these_flags": 205,
        "unaudited_batteryml_cells": 318,
        "by_domain": {
            d: dict(c)
            for d, c in sorted(
                by_domain.items()
            )
        },
        "files_scanned": scanned,
        "prediction_artifact_candidates": found,
        "methodological_constraints": [
            "No predictions were generated or altered.",
            "Only file metadata and CSV/Parquet schema were inspected.",
            "A valid fixed-prediction analysis requires exact cell IDs, fold identity, model branch, predictions, and original outcomes.",
            "The training-median prior must come from the original training fold.",
            "Filtering held-out evaluation cells is not equivalent to model retraining.",
            "The 318 BatteryML_main cells are not source-quality-certified.",
        ]
    }

    with OUT.open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )
        f.write("\n")

    print("=== STAGE20P QUALITY PREFLIGHT ===")
    print("BATTERYLIFE_CELLS", len(flags))
    print("NO_POSITIVE_POST100", len(no_future))
    print("POST100_ONE_OR_TWO", len(sparse))
    print("MICH_NONPOSITIVE_ENDPOINT", len(nonpositive))
    print("UNION_OF_THREE_FLAGS", len(combined_flags))
    print("CANDIDATE_CELLS", len(candidates))
    print("UNAUDITED_BATTERYML_CELLS", 318)

    for domain, counts in sorted(by_domain.items()):
        print("DOMAIN", domain, dict(counts))

    print("=== STAGE20P PREDICTION DISCOVERY ===")
    print("FILES_SCANNED", scanned)
    print("CANDIDATE_FILES", len(found))

    for item in found[:45]:
        print("FILE", item["path"])
        print(
            "  SCORE", item["score"],
            "SIZE_MB", item["size_mb"]
        )
        print(
            "  COLUMNS",
            item["columns"][:35]
        )

    print("STAGE20P_INVENTORY_COMPLETE=YES")
    print("REPORT", OUT)

if __name__ == "__main__":
    main()
