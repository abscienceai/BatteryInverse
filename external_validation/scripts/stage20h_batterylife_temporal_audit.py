#!/usr/bin/env python3
"""Stage 20H: source-verified temporal endpoint audit of 249 BatteryLife cells."""

import csv
import hashlib
import io
import json
import math
import pickle
import pickletools
import zipfile
from pathlib import Path
from statistics import median

ROOT = Path.cwd()
BASE = ROOT / 'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'
RAW = ROOT / 'data/00_external_raw/aging/BATTERYLIFE_PROCESSED/original'
REP = ROOT / 'external_validation/reports'
OUT_CSV = REP / 'stage20h_batterylife_249_temporal_cells.csv'
OUT_JSON = REP / 'stage20h_batterylife_249_temporal_report.json'
PILOT = REP / 'stage20e_pilot_observed_cycles.csv'

EXPECTED_SHA = '8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c'

ARCHIVES = {
    'CALB': ('86480187bf7ab3b447d8e7553b01cb59', 27),
    'HNEI': ('1e09e22b08af3616322f0e25756100fd', 14),
    'MICH': ('c8f7514deee3487d6195b4419097ef5f', 40),
    'MICH_EXP': ('7d04165080c441e8680efa8250ef4b93', 18),
    'NA-ion': ('5fc0e3bea452dce4427a5f1e770493a9', 64),
    'SNL': ('33c9c7dd3692e85c8315eb4c43ce0bbd', 61),
    'UL_PUR': ('7a5e4a3e7ecfbcd6c189d97c2379f50a', 2),
    'XJTU': ('b45b0df2be827f332528b28fbf2a5e37', 23),
}

SAFE_OPS = set("""
PROTO FRAME STOP MARK POP POP_MARK DUP NONE NEWTRUE NEWFALSE
INT BININT BININT1 BININT2 LONG LONG1 LONG4 FLOAT BINFLOAT
STRING BINSTRING SHORT_BINSTRING UNICODE BINUNICODE
SHORT_BINUNICODE BINUNICODE8 BINBYTES SHORT_BINBYTES BINBYTES8
BYTEARRAY8 EMPTY_TUPLE TUPLE TUPLE1 TUPLE2 TUPLE3
EMPTY_LIST LIST APPEND APPENDS EMPTY_DICT DICT SETITEM
SETITEMS EMPTY_SET ADDITEMS FROZENSET MEMOIZE BINPUT
LONG_BINPUT PUT BINGET LONG_BINGET GET
""".split())

MAX_BYTES = 128 * 1024 * 1024
MAX_OPS = 10_000_000
ANCHORS = (1, 100, 150, 200, 300)


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Class resolution prohibited")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent references prohibited")


def digest(path, kind):
    h = hashlib.new(kind)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def decode(z, info):
    if info.file_size > MAX_BYTES:
        return None, 'SKIPPED_OVERSIZED_PICKLE'

    try:
        with z.open(info) as src:
            payload = src.read(MAX_BYTES + 1)

        if len(payload) > MAX_BYTES:
            return None, 'SKIPPED_OVERSIZED_PICKLE'

        count = 0
        last = None

        for opcode, _, _ in pickletools.genops(payload):
            name = opcode.name

            if name not in SAFE_OPS:
                return None, 'BLOCKED_PICKLE_OPCODE_' + name

            count += 1

            if count > MAX_OPS:
                return None, 'SKIPPED_OPCODE_LIMIT'

            last = name

        if last != 'STOP':
            return None, 'INVALID_PICKLE_STOP'

        obj = PrimitiveUnpickler(io.BytesIO(payload)).load()

        if not isinstance(obj, dict):
            return None, 'INVALID_ROOT_TYPE'

        return obj, 'ACCEPTED_PRIMITIVES'

    except Exception as exc:
        return None, 'PARSE_ERROR_' + type(exc).__name__


def capacity(value):
    items = value if isinstance(value, (list, tuple)) else [value]

    if len(items) > 2_000_000:
        raise ValueError("Capacity vector exceeded safety limit")

    numbers = []

    for item in items:
        if isinstance(item, (dict, list, tuple, set, bytes, bytearray)):
            continue

        try:
            v = float(item)
            if math.isfinite(v):
                numbers.append(v)
        except (ValueError, TypeError, OverflowError):
            pass

    return max(numbers) if numbers else None


def cycle_index(value):
    try:
        n = float(value)

        if math.isfinite(n) and n >= 1 and n.is_integer():
            return int(n)

    except (ValueError, TypeError, OverflowError):
        pass

    return None


def audit(domain, cell, row, z, info):
    result = dict(
        dataset=domain,
        cell_id=cell,
        source_file=info.filename,
        decode_status='',
        content_id_match=0,
        cycle_records=0,
        valid_capacity_records=0,
        post100_records=0,
        last_observed_cycle='',
        nearest_valid_cycle_to_100='',
        nearest100_after100='',
        target_rebuild_absdiff='',
        cap_cycle100_absdiff='',
        q100_ratio_target_absdiff=''
    )

    for k in ANCHORS:
        result[f'exact_cycle_{k}_count'] = 0

    obj, status = decode(z, info)
    result['decode_status'] = status

    if obj is None:
        return result

    result['content_id_match'] = int(
        str(obj.get('cell_id')) == cell
    )

    cycles = obj.get('cycle_data')

    if not isinstance(cycles, (list, tuple)) or len(cycles) > 1_000_000:
        result['decode_status'] = 'INVALID_CYCLE_DATA'
        return result

    result['cycle_records'] = len(cycles)
    values = []

    try:
        for r in cycles:
            if not isinstance(r, dict):
                continue

            num = cycle_index(r.get('cycle_number'))

            if num is None:
                continue

            cap = capacity(r.get('discharge_capacity_in_Ah'))

            if cap is not None and cap > 0:
                values.append((num, cap))

    except Exception as exc:
        result['decode_status'] = (
            'CAPACITY_PARSE_ERROR_' + type(exc).__name__
        )
        return result

    result['valid_capacity_records'] = len(values)

    result['post100_records'] = sum(
        num > 100 for num, _ in values
    )

    if values:
        result['last_observed_cycle'] = max(
            c for c, _ in values
        )

        nearest = min(
            values,
            key=lambda x: (abs(x[0] - 100), x[0])
        )

        result['nearest_valid_cycle_to_100'] = nearest[0]
        result['nearest100_after100'] = int(nearest[0] > 100)

        ordered = [
            v for _, v in sorted(values)
        ]

        first = median(ordered[:3])
        last = median(ordered[-3:])

        if first > 0:
            result['target_rebuild_absdiff'] = abs(
                float(row['final_capacity_retention'])
                - last / first
            )

        result['cap_cycle100_absdiff'] = abs(
            float(row['cap_cycle_100']) - nearest[1]
        )

        result['q100_ratio_target_absdiff'] = abs(
            float(row['cap_ratio_100_to_1'])
            - float(row['final_capacity_retention'])
        )

        for n in ANCHORS:
            result[f'exact_cycle_{n}_count'] = sum(
                c == n for c, _ in values
            )

    return result


def main():
    if OUT_CSV.exists() or OUT_JSON.exists():
        raise SystemExit(
            'Stage20H outputs already exist; refusing overwrite'
        )

    if digest(BASE, 'sha256') != EXPECTED_SHA:
        raise SystemExit('Canonical benchmark hash mismatch')

    original = read_csv(BASE)

    if len(original) != 567:
        raise SystemExit('Unexpected benchmark length')

    selected = {
        (r['dataset_id'], r['cell_id']): r
        for r in original
        if r['source_family'] == 'BatteryLife_selected'
    }

    if len(selected) != 249:
        raise SystemExit('Unexpected BatteryLife cohort')

    results = []

    for domain, (md5, n) in ARCHIVES.items():
        archive = RAW / (domain + '.zip')

        if digest(archive, 'md5') != md5:
            raise SystemExit(
                f'Archive checksum mismatch: {domain}'
            )

        group = {
            (d, c): r
            for (d, c), r in selected.items()
            if d == domain
        }

        if len(group) != n:
            raise SystemExit(
                f'Domain population mismatch: {domain}'
            )

        with zipfile.ZipFile(archive) as z:
            infos = {
                i.filename: i
                for i in z.infolist()
                if not i.is_dir()
                and i.filename.lower().endswith('.pkl')
            }

            if len(infos) != n:
                raise SystemExit(
                    f'Archive member count mismatch: {domain}'
                )

            for (_, cell), row in sorted(group.items()):
                member = row['source_file'].replace('\\', '/')

                if member not in infos:
                    raise SystemExit(
                        f'Missing matched member: {domain} {cell}'
                    )

                results.append(
                    audit(domain, cell, row, z, infos[member])
                )

        good = sum(
            r['decode_status'] == 'ACCEPTED_PRIMITIVES'
            and r['content_id_match']
            for r in results
            if r['dataset'] == domain
        )

        print(
            f'COMPLETED_DOMAIN={domain} '
            f'CELLS={n} CONTENT_MATCHES={good}',
            flush=True
        )

    if len(results) != 249:
        raise SystemExit('Audit population mismatch')

    # Reproduce the previous 29-cell pilot before saving results.
    pilot = read_csv(PILOT)

    if len(pilot) != 29:
        raise SystemExit(
            'Stage20E pilot is missing or incomplete'
        )

    index = {
        (r['dataset'], r['cell_id']): r
        for r in results
    }

    for p in pilot:
        q = index[(p['dataset'], p['cell_id'])]

        if (
            q['decode_status'] != 'ACCEPTED_PRIMITIVES'
            or not q['content_id_match']
        ):
            raise SystemExit(
                'Pilot identity/replay mismatch'
            )

        for old, new in [
            ('last_observed_cycle', 'last_observed_cycle'),
            ('observed_post100_cycles', 'post100_records'),
            ('exact_cycle_100_count', 'exact_cycle_100_count')
        ]:
            if int(p[old]) != int(q[new]):
                raise SystemExit(
                    f'Pilot metric mismatch: {old}'
                )

        if abs(
            float(p['retention_rebuild_absdiff'])
            - q['target_rebuild_absdiff']
        ) > 1e-9:
            raise SystemExit(
                'Pilot target reconstruction mismatch'
            )

    summarized = {}

    for domain in ARCHIVES:
        rows = [
            r for r in results
            if r['dataset'] == domain
        ]

        valid = [
            r for r in rows
            if r['decode_status'] == 'ACCEPTED_PRIMITIVES'
            and r['content_id_match']
            and r['valid_capacity_records'] > 0
        ]

        no_future = [
            r for r in valid
            if r['post100_records'] == 0
        ]

        gaps = [
            r['q100_ratio_target_absdiff']
            for r in no_future
        ]

        summarized[domain] = {
            'benchmark_cells': len(rows),

            'decoded_primitives': sum(
                r['decode_status'] == 'ACCEPTED_PRIMITIVES'
                for r in rows
            ),

            'content_ids_matched': sum(
                r['content_id_match']
                for r in rows
            ),

            'valid_capacity_cells': len(valid),

            'no_post100_observation': len(no_future),

            'with_post100_observation': sum(
                r['post100_records'] > 0
                for r in valid
            ),

            'target_rebuild_within_1e_6': sum(
                isinstance(r['target_rebuild_absdiff'], float)
                and r['target_rebuild_absdiff'] < 1e-6
                for r in valid
            ),

            'nearest100_after100': sum(
                r['nearest100_after100']
                for r in valid
            ),

            'q100_target_gap_median_no_future': (
                median(gaps) if gaps else None
            ),

            **{
                f'exact_cycle_{n}': sum(
                    r[f'exact_cycle_{n}_count'] == 1
                    for r in valid
                )
                for n in ANCHORS
            },

            'blocked_or_unreadable': (
                len(rows) - len(valid)
            )
        }

    report = {
        'stage': '20H',
        'benchmark_sha256': EXPECTED_SHA,
        'scope': (
            '249 BatteryLife_selected cells, eight datasets; '
            '318 BatteryML_main cells not included'
        ),
        'pilot_replay': '29 of 29 Stage20E cells matched',
        'per_dataset': summarized,
        'unreadable_details': [
            {
                'dataset': r['dataset'],
                'cell_id': r['cell_id'],
                'reason': r['decode_status']
            }
            for r in results
            if r['decode_status'] != 'ACCEPTED_PRIMITIVES'
        ],
        'limitations': [
            'No post-100 observations means no demonstrated post-100 target horizon.',
            'A common cycle index does not establish physical endpoint equivalence.',
            'Restricted primitive pickle decoding was used.',
            'No missing capacities were imputed.',
            'No predictive models were trained.'
        ]
    }

    REP.mkdir(exist_ok=True, parents=True)

    with OUT_CSV.open(
        'x', newline='', encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    with OUT_JSON.open('x', encoding='utf-8') as f:
        json.dump(
            report, f, indent=2, ensure_ascii=False
        )
        f.write('\n')

    print('=== STAGE20H TEMPORAL SUMMARY ===')

    for domain, r in summarized.items():
        print('DOMAIN', domain, r)

    print(
        'TOTAL_VALID_CONTENT_IDS',
        sum(r['content_id_match'] for r in results)
    )

    print(
        'TOTAL_WITHOUT_POST100',
        sum(
            r['no_post100_observation']
            for r in summarized.values()
        )
    )

    print(
        'TOTAL_WITH_POST100',
        sum(
            r['with_post100_observation']
            for r in summarized.values()
        )
    )

    print('STAGE20H_AUDIT_COMPLETE=YES')


if __name__ == '__main__':
    main()
