import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / (
    "data/05_model_ready/aging/"
    "expanded_aging_features_with_batterylife.csv"
)
RAW = ROOT / (
    "data/00_external_raw/aging/"
    "BATTERYLIFE_PROCESSED/original"
)
OUT = ROOT / "external_validation/reports"

CSV_OUT = OUT / "stage20g_source_lineage.csv"
JSON_OUT = OUT / "stage20g_source_lineage_report.json"

EXPECTED_SHA = (
    "8bc5296425fbb0735faceb56fe1cf52c6d0292"
    "bec2e361be8870fc1626aa5c7c"
)

COUNTS = {
    "HNEI": 14,
    "MICH": 40,
    "MICH_EXP": 18,
    "NA-ion": 64,
    "SNL": 61,
    "XJTU": 23,
}

def norm(s):
    return str(s or "").replace("\\", "/").strip().lower()

assert not CSV_OUT.exists()
assert not JSON_OUT.exists()

assert hashlib.sha256(
    DATA.read_bytes()
).hexdigest() == EXPECTED_SHA

with DATA.open(newline="", encoding="utf-8-sig") as f:
    bench = list(csv.DictReader(f))

assert len(bench) == 567

selected = [
    row for row in bench
    if row["source_family"] == "BatteryLife_selected"
    and row["dataset_id"] in COUNTS
]

assert len(selected) == 220

assert len({
    (r["dataset_id"], r["cell_id"])
    for r in selected
}) == 220

records = []
domain_results = {}

for domain, count in COUNTS.items():

    group = [
        r for r in selected
        if r["dataset_id"] == domain
    ]

    assert len(group) == count

    path = RAW / (domain + ".zip")

    with zipfile.ZipFile(path) as z:
        members = [
            info
            for info in z.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(".pkl")
        ]

    full = Counter(
        norm(info.filename) for info in members
    )

    basename = Counter(
        Path(norm(info.filename)).name
        for info in members
    )

    stems = Counter(
        Path(norm(info.filename)).stem
        for info in members
    )

    for row in group:

        source = norm(row.get("source_file"))
        cell_id = norm(row["cell_id"])

        source_base = (
            Path(source).name if source else ""
        )

        exact = bool(
            source and full[source] == 1
        )

        unique_base = bool(
            source_base and basename[source_base] == 1
        )

        unique_stem = bool(
            cell_id and stems[cell_id] == 1
        )

        records.append({
            "dataset": domain,
            "cell_id": row["cell_id"],
            "source_file": row.get("source_file", ""),
            "exact_member_match": int(exact),
            "unique_basename_match": int(unique_base),
            "unique_cell_stem_match": int(unique_stem),
            "source_name_match": int(
                exact or unique_base
            ),
        })

    subset = [
        r for r in records
        if r["dataset"] == domain
    ]

    domain_results[domain] = {
        "benchmark_cells": len(group),
        "archive_pickle_members": len(members),
        "exact_member_matches": sum(
            r["exact_member_match"]
            for r in subset
        ),
        "unique_basename_matches": sum(
            r["unique_basename_match"]
            for r in subset
        ),
        "source_name_matches": sum(
            r["source_name_match"]
            for r in subset
        ),
        "unique_cell_stem_matches": sum(
            r["unique_cell_stem_match"]
            for r in subset
        ),
    }

with CSV_OUT.open(
    "x", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f, fieldnames=list(records[0])
    )

    writer.writeheader()
    writer.writerows(records)

report = {
    "stage": "20G",
    "benchmark_sha256": EXPECTED_SHA,
    "source_record": "Zenodo 14934405 v2",
    "audited_benchmark_cells": len(records),
    "domains": domain_results,
    "unique_source_name_matches": sum(
        r["source_name_match"]
        for r in records
    ),
    "pickle_deserialization_performed": False,
    "capacity_values_extracted": False,
    "endpoint_comparability_established": False,
    "limitations": [
        "Filename matching is not content verification.",
        "ZIP directory inspection does not establish measurement validity.",
        "No common physical endpoint has been certified."
    ],
}

JSON_OUT.write_text(
    json.dumps(
        report, indent=2, ensure_ascii=False
    ) + "\n",
    encoding="utf-8"
)

print("=== STAGE20G RESULTS ===")

for domain, value in domain_results.items():
    print("DOMAIN", domain, value)

print("AUDITED_CELLS", len(records))
print(
    "SOURCE_NAME_MATCHES",
    report["unique_source_name_matches"]
)

print("STAGE20G_LINEAGE_AUDIT_COMPLETE=YES")
