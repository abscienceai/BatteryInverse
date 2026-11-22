#!/usr/bin/env python3
"""Locate original benchmark-cell-level LODO predictions."""

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()

BENCH = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)

OUT = ROOT / (
    "external_validation/reports/"
    "stage20q_lodo_prediction_lineage.json"
)

EXPECTED_SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)

SEARCH_ROOTS = [
    "external_validation",
    "data",
    "results",
    "outputs",
    "experiments",
    "runs",
    "artifacts",
    "releases",
]

SKIP = {
    ".git", "__pycache__", ".venv", "venv",
    "node_modules", "raw", "00_external_raw",
    "stage20n_xjtu_checkpoints"
}

PRED_RE = re.compile(
    r"predic|(^|_)pred($|_)|yhat|y_pred",
    re.IGNORECASE
)

SOURCE_RE = re.compile(
    r"predict\(|to_csv|writerow|csv\.writer|"
    r"prediction|joblib|dump\(|parquet",
    re.IGNORECASE
)


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def get_column(columns, possibilities):
    lookup = {
        c.lower(): c for c in columns
    }

    for name in possibilities:
        if name in lookup:
            return lookup[name]

    return None


def inspect_csv(path, canonical, id_domains):
    try:
        with path.open(
            newline="",
            encoding="utf-8-sig",
            errors="replace"
        ) as f:

            reader = csv.DictReader(f)
            columns = reader.fieldnames or []

            cell_col = get_column(
                columns, ["cell_id", "battery_id"]
            )

            pred_cols = [
                c for c in columns
                if PRED_RE.search(c)
            ]

            if not cell_col or not pred_cols:
                return None

            domain_col = get_column(
                columns,
                ["dataset_id", "dataset",
                 "heldout_dataset", "heldout_domain"]
            )

            model_col = get_column(
                columns, ["model", "method"]
            )

            matched_keys = set()
            matched_rows = 0
            total_rows = 0
            ambiguous_ids = 0
            model_names = Counter()

            for row in reader:
                total_rows += 1

                if total_rows > 100000:
                    break

                cell = str(
                    row.get(cell_col, "")
                ).strip()

                if not cell:
                    continue

                if domain_col:
                    domain = str(
                        row.get(domain_col, "")
                    ).strip()

                    key = (domain, cell)

                    if key not in canonical:
                        continue

                else:
                    domains = id_domains.get(
                        cell, set()
                    )

                    if len(domains) != 1:
                        if len(domains) > 1:
                            ambiguous_ids += 1
                        continue

                    key = (
                        next(iter(domains)),
                        cell
                    )

                matched_keys.add(key)
                matched_rows += 1

                if model_col:
                    model_names[
                        str(row.get(model_col, ""))
                    ] += 1

            return {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "columns": columns,
                "prediction_columns": pred_cols,
                "cell_column": cell_col,
                "domain_column": domain_col,
                "model_column": model_col,
                "rows_scanned": total_rows,
                "matched_rows": matched_rows,
                "unique_benchmark_cells": len(
                    matched_keys
                ),
                "unique_batterylife_cells": sum(
                    domain in {
                        "CALB", "HNEI", "MICH",
                        "MICH_EXP", "NA-ion",
                        "SNL", "UL_PUR", "XJTU"
                    }
                    for domain, _ in matched_keys
                ),
                "ambiguous_cell_ids": ambiguous_ids,
                "model_labels": dict(
                    model_names.most_common(15)
                ),
                "scan_truncated": (
                    total_rows > 100000
                )
            }

    except Exception as exc:
        return {
            "path": str(
                path.relative_to(ROOT)
            ),
            "scan_error": (
                type(exc).__name__ + ": "
                + str(exc)[:150]
            )
        }


def inspect_scripts():
    base = ROOT / "external_validation/scripts"

    patterns = [
        "stage17ak*.py",
        "stage17an*.py",
        "stage19*.py",
    ]

    results = []

    for pattern in patterns:
        for path in sorted(base.glob(pattern)):

            lines = path.read_text(
                encoding="utf-8",
                errors="replace"
            ).splitlines()

            hits = []

            for number, line in enumerate(
                lines, 1
            ):
                if SOURCE_RE.search(line):
                    hits.append({
                        "line": number,
                        "text": line.strip()[:190]
                    })

            results.append({
                "path": str(
                    path.relative_to(ROOT)
                ),
                "matching_lines": hits[:50],
                "total_matching_lines": len(hits)
            })

    return results


def main():
    if OUT.exists():
        raise SystemExit(
            "Stage20Q report exists; refusing overwrite"
        )

    assert hashlib.sha256(
        BENCH.read_bytes()
    ).hexdigest() == EXPECTED_SHA

    benchmark = read_csv(BENCH)

    assert len(benchmark) == 567

    canonical = {
        (r["dataset_id"], r["cell_id"])
        for r in benchmark
    }

    assert len(canonical) == 567

    id_domains = defaultdict(set)

    for domain, cell in canonical:
        id_domains[cell].add(domain)

    inspected = []
    scanned = 0

    for dirname in SEARCH_ROOTS:

        base = ROOT / dirname

        if not base.is_dir():
            continue

        for current, dirs, files in os.walk(
            base, followlinks=False
        ):
            dirs[:] = [
                d for d in dirs
                if d not in SKIP
                and not (
                    Path(current) / d
                ).is_symlink()
            ]

            for filename in files:

                path = Path(current) / filename

                if (
                    path.is_symlink()
                    or path.suffix.lower() != ".csv"
                ):
                    continue

                scanned += 1

                result = inspect_csv(
                    path, canonical, id_domains
                )

                if result is not None:
                    inspected.append(result)

    inspected.sort(
        key=lambda r: (
            -r.get(
                "unique_benchmark_cells", 0
            ),
            -r.get("matched_rows", 0),
            r["path"]
        )
    )

    matched = [
        r for r in inspected
        if r.get(
            "unique_benchmark_cells", 0
        ) > 0
    ]

    script_hints = inspect_scripts()

    report = {
        "stage": "20Q",
        "benchmark_sha256": EXPECTED_SHA,
        "canonical_cells": len(canonical),
        "csv_files_scanned": scanned,
        "files_with_cell_and_prediction_columns": len(
            inspected
        ),
        "files_with_benchmark_cell_matches": len(
            matched
        ),
        "maximum_unique_benchmark_cell_coverage": max(
            (
                r.get(
                    "unique_benchmark_cells", 0
                )
                for r in matched
            ),
            default=0
        ),
        "candidate_files": inspected,
        "relevant_script_output_hints": script_hints,
        "limitations": [
            "The audit checks CSV cell identifiers and schemas.",
            "A matching cell ID is not sufficient to validate a LODO prediction.",
            "Matching model branch, fold, target and prior values must be verified before scoring.",
            "Absence from the scanned files does not prove that predictions were never generated.",
            "No models were fitted or predictions changed."
        ]
    }

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )

    print("=== STAGE20Q PREDICTION LINEAGE ===")
    print("CANONICAL_CELLS", len(canonical))
    print("CSV_FILES_SCANNED", scanned)
    print(
        "FILES_WITH_PREDICTION_COLUMNS",
        len(inspected)
    )
    print(
        "FILES_WITH_BENCHMARK_MATCHES",
        len(matched)
    )
    print(
        "MAX_BENCHMARK_COVERAGE",
        report[
            "maximum_unique_benchmark_cell_coverage"
        ]
    )

    for r in matched[:20]:
        print(
            "MATCHED_FILE",
            r["path"]
        )
        print(
            "  UNIQUE_CELLS",
            r["unique_benchmark_cells"],
            "ROWS",
            r["matched_rows"]
        )
        print(
            "  PREDICTION_COLUMNS",
            r["prediction_columns"]
        )
        print(
            "  MODEL_LABELS",
            r["model_labels"]
        )

    print("=== STAGE20Q SCRIPT OUTPUT HINTS ===")

    for item in script_hints:
        print("SCRIPT", item["path"])
        print(
            "  MATCHING_LINES",
            item["total_matching_lines"]
        )

        for hit in item["matching_lines"][:15]:
            print(
                "  L" + str(hit["line"]),
                hit["text"]
            )

    print("STAGE20Q_LINEAGE_COMPLETE=YES")
    print("REPORT", OUT)


if __name__ == "__main__":
    main()
