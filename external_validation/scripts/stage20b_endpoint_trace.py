#!/usr/bin/env python3
"""Stage 20B: index original BatteryLife archives and exact observed cycle anchors.
No pickle deserialization, no interpolation, and no changes to existing data.
"""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, zipfile

root = Path.cwd()
data = root/'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'
summary = root/'data/03_processed/BATTERYLIFE_SELECTED/batterylife_selected_cell_summary.csv'
trajectory = root/'data/03_processed/BATTERYLIFE_SELECTED/batterylife_selected_capacity_trajectory.csv'
raw = root/'data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original'
report = root/'external_validation/reports'

per_cell_file = report/'stage20b_batterylife_observed_anchor_coverage.csv'
report_file = report/'stage20b_batterylife_lineage_report.json'

assert not per_cell_file.exists() and not report_file.exists(), \
    'Stage20B results exist; refusing overwrite'

expected = '8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c'
assert data.is_file() and hashlib.sha256(data.read_bytes()).hexdigest() == expected, \
    'Benchmark SHA256 mismatch'

with data.open(newline='', encoding='utf-8-sig') as f:
    bench = list(csv.DictReader(f))

assert len(bench) == 567

bl = {
    (r['dataset_id'], r['cell_id']): r
    for r in bench
    if r['source_family'] == 'BatteryLife_selected'
}
assert len(bl) == 249
assert len(bench) - len(bl) == 318

# Only ZIP central-directory metadata is inspected.
archive_members = {}
archive_errors = {}

for zp in sorted(raw.glob('*.zip')) if raw.is_dir() else []:
    try:
        with zipfile.ZipFile(zp) as z:
            archive_members[zp.name] = {
                x.filename for x in z.infolist()
                if x.filename.lower().endswith('.pkl')
            }
    except (OSError, zipfile.BadZipFile) as ex:
        archive_errors[zp.name] = str(ex)

processed = {}
summary_columns = []

if summary.is_file():
    with summary.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        summary_columns = reader.fieldnames or []
        for row in reader:
            key = (row.get('dataset_id', ''), row.get('cell_id', ''))
            if key in bl:
                processed.setdefault(key, []).append(row)

anchors = (1, 100, 150, 200, 300)

records = {
    k: {
        'n_capacity_rows_post100': 0,
        'max_observed_cycle': None,
        'nearest_to_100': None,
        'exact': defaultdict(list)
    }
    for k in bl
}

trajectory_columns = []
read_rows = 0
matched_rows = 0

if trajectory.is_file():
    with trajectory.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        trajectory_columns = reader.fieldnames or []

        needed = {
            'dataset_id', 'cell_id',
            'cycle_number', 'discharge_capacity_Ah'
        }
        if not needed.issubset(trajectory_columns):
            raise RuntimeError(
                'Unexpected trajectory schema: ' + str(trajectory_columns)
            )

        for row in reader:
            read_rows += 1
            key = (row.get('dataset_id', ''), row.get('cell_id', ''))

            if key not in records:
                continue

            try:
                cyc = float(row['cycle_number'])
                capacity = float(row['discharge_capacity_Ah'])

                if not (
                    capacity > 0
                    and cyc >= 1
                    and cyc.is_integer()
                ):
                    continue

                cyc = int(cyc)

            except (ValueError, TypeError, OverflowError):
                continue

            matched_rows += 1
            rec = records[key]

            if (
                rec['max_observed_cycle'] is None
                or cyc > rec['max_observed_cycle']
            ):
                rec['max_observed_cycle'] = cyc

            if cyc > 100:
                rec['n_capacity_rows_post100'] += 1

            near = rec['nearest_to_100']

            if near is None or (
                abs(cyc-100), cyc
            ) < (
                abs(near-100), near
            ):
                rec['nearest_to_100'] = cyc

            if cyc in anchors:
                rec['exact'][cyc].append(capacity)

out = []

for key, b in sorted(bl.items()):
    source = (b.get('source_file') or '').strip()
    matches = processed.get(key, [])
    p = matches[0] if len(matches) == 1 else {}

    archive = p.get('source_archive', '')
    member = p.get('source_file', '') or source

    member_in_zip = (
        archive in archive_members
        and member in archive_members[archive]
    )

    rec = records[key]

    line = {
        'dataset_id': key[0],
        'cell_id': key[1],
        'benchmark_source_file': source,
        'processed_summary_rows': len(matches),
        'source_archive': archive,
        'archive_member_exact_match': int(member_in_zip),
        'has_trajectory_capacity_rows': int(
            rec['max_observed_cycle'] is not None
        ),
        'max_observed_cycle':
            rec['max_observed_cycle']
            if rec['max_observed_cycle'] is not None else '',
        'post100_capacity_rows': rec['n_capacity_rows_post100'],
        'nearest_measured_cycle_to_100':
            rec['nearest_to_100']
            if rec['nearest_to_100'] is not None else '',
        'nearest_to100_is_after100': int(
            rec['nearest_to_100'] is not None
            and rec['nearest_to_100'] > 100
        ),
    }

    for cyc in anchors:
        vals = rec['exact'][cyc]
        line[f'exact_cycle_{cyc}_count'] = len(vals)
        line[f'exact_cycle_{cyc}_single_valid'] = int(len(vals) == 1)

    out.append(line)

counts = {}

for domain in sorted(set(k[0] for k in bl)):
    group = [r for r in out if r['dataset_id'] == domain]

    counts[domain] = {
        'cells': len(group),
        'matched_processed_summary': sum(
            r['processed_summary_rows'] == 1 for r in group
        ),
        'matched_zip_member': sum(
            r['archive_member_exact_match'] for r in group
        ),
        'with_observed_post100': sum(
            r['post100_capacity_rows'] > 0 for r in group
        ),
        'nearest_to100_after100': sum(
            r['nearest_to100_is_after100'] for r in group
        ),
        **{
            f'exact_{cyc}_single_valid': sum(
                r[f'exact_cycle_{cyc}_single_valid']
                for r in group
            )
            for cyc in anchors
        }
    }

report_obj = {
    'stage': '20B',
    'scope': (
        '249 BatteryLife_selected benchmark cells; '
        '318 BatteryML_main cells not yet audited '
        'for endpoint availability'
    ),
    'benchmark_sha256': expected,
    'file_presence': {
        'processed_summary': summary.is_file(),
        'processed_trajectory': trajectory.is_file(),
        'raw_archive_dir': raw.is_dir()
    },
    'zip_archives': {
        name: len(members)
        for name, members in archive_members.items()
    },
    'zip_errors': archive_errors,
    'processed_summary_columns': summary_columns,
    'processed_trajectory_columns': trajectory_columns,
    'trajectory_rows_scanned': read_rows,
    'trajectory_rows_matched_with_valid_capacity': matched_rows,
    'per_dataset': counts,
    'limitations': [
        'Exact cycle presence does not prove physical endpoint equivalence.',
        'ZIP member name matching does not verify pickle content.',
        'No pickle files were deserialized.',
        'No measurement was interpolated, imputed or synthesized.',
        'No common-endpoint predictive experiment was performed.'
    ]
}

report.mkdir(parents=True, exist_ok=True)

with per_cell_file.open('x', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(out[0]))
    writer.writeheader()
    writer.writerows(out)

with report_file.open('x', encoding='utf-8') as f:
    json.dump(report_obj, f, indent=2, ensure_ascii=False)
    f.write('\n')

print('=== STAGE20B INPUT COVERAGE ===')

for name, present in report_obj['file_presence'].items():
    print(name, present)

print('ZIP_ARCHIVES', report_obj['zip_archives'])
print('TRAJECTORY_ROWS_SCANNED', read_rows)
print('VALID_MATCHED_ROWS', matched_rows)

for domain, result in counts.items():
    print('DOMAIN', domain, result)

print(
    'CELLS_WITH_EXACT_200',
    sum(r['exact_cycle_200_single_valid'] for r in out)
)

print(
    'CELLS_NEAREST_TO_100_AFTER_100',
    sum(r['nearest_to100_is_after100'] for r in out)
)

print('STAGE20B_INVENTORY_COMPLETE=YES')
print('REPORT', report_file)
print('PER_CELL', per_cell_file)
