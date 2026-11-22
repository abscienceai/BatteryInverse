#!/usr/bin/env python3
"""Stage20E: restricted source decoding and observed capacity audit."""

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

BENCH = ROOT / (
    'data/05_model_ready/aging/'
    'expanded_aging_features_with_batterylife.csv'
)

RAW = ROOT / (
    'data/00_external_raw/aging/'
    'BATTERYLIFE_PROCESSED/original'
)

OUT = ROOT / 'external_validation/reports'

CSV = OUT / 'stage20e_pilot_observed_cycles.csv'
JSON = OUT / 'stage20e_pilot_measurement_report.json'

SHA = (
    '8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361'
    'be8870fc1626aa5c7c'
)

ARCHIVES = {
    'CALB': '86480187bf7ab3b447d8e7553b01cb59',
    'UL_PUR': '7a5e4a3e7ecfbcd6c189d97c2379f50a'
}

COUNTS = {'CALB': 27, 'UL_PUR': 2}

MAX_BYTES = 128 * 1024 * 1024
ANCHORS = (1, 100, 150, 200, 300)

SAFE_OPS = {
    'PROTO', 'FRAME', 'STOP', 'MARK', 'POP', 'POP_MARK',
    'DUP', 'NONE', 'NEWTRUE', 'NEWFALSE',
    'INT', 'BININT', 'BININT1', 'BININT2',
    'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT',
    'STRING', 'BINSTRING', 'SHORT_BINSTRING',
    'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE',
    'BINUNICODE8', 'BINBYTES', 'SHORT_BINBYTES',
    'BINBYTES8', 'BYTEARRAY8',
    'EMPTY_TUPLE', 'TUPLE', 'TUPLE1', 'TUPLE2',
    'TUPLE3', 'EMPTY_LIST', 'LIST', 'APPEND',
    'APPENDS', 'EMPTY_DICT', 'DICT', 'SETITEM',
    'SETITEMS', 'EMPTY_SET', 'ADDITEMS',
    'FROZENSET', 'MEMOIZE', 'BINPUT', 'LONG_BINPUT',
    'PUT', 'BINGET', 'LONG_BINGET', 'GET'
}


class PrimitiveOnlyUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            'External classes forbidden'
        )

    def persistent_load(self, pid):
        raise pickle.UnpicklingError(
            'Persistent objects forbidden'
        )


def hashed(path, algorithm):
    h = hashlib.new(algorithm)

    with path.open('rb') as f:
        for block in iter(
            lambda: f.read(1048576), b''
        ):
            h.update(block)

    return h.hexdigest()


def finite_max(value):

    if isinstance(value, (list, tuple)):
        nums = value
    elif value is None:
        nums = []
    else:
        nums = [value]

    if len(nums) > 2000000:
        raise ValueError('Oversized capacity vector')

    found = []

    for v in nums:
        if isinstance(
            v, (dict, list, tuple, set, bytes, bytearray)
        ):
            continue

        try:
            z = float(v)
            if math.isfinite(z):
                found.append(z)
        except (TypeError, ValueError, OverflowError):
            pass

    return max(found) if found else None


def numeric_cycle(value):

    try:
        v = float(value)

        if math.isfinite(v) and v >= 1 and v.is_integer():
            return int(v)

    except (ValueError, TypeError, OverflowError):
        pass

    return None


def scan_and_load(z, info):

    if info.file_size > MAX_BYTES:
        return None, 'skipped_oversize', []

    try:
        with z.open(info) as f:
            payload = f.read(MAX_BYTES + 1)

        if len(payload) > MAX_BYTES:
            return None, 'skipped_oversize', []

        operations = [
            opcode.name
            for opcode, _, _ in pickletools.genops(payload)
        ]

        unsupported = sorted(set(operations) - SAFE_OPS)

        if unsupported:
            return None, 'blocked_opcodes', unsupported

        if not operations or operations[-1] != 'STOP':
            return None, 'missing_stop', []

        obj = PrimitiveOnlyUnpickler(
            io.BytesIO(payload)
        ).load()

        if not isinstance(obj, dict):
            return None, 'invalid_root_type', []

        return obj, 'accepted_primitive_only', []

    except Exception as exc:
        return (
            None,
            'parse_error_' + type(exc).__name__,
            []
        )


def per_cell(domain, bench, z, info):

    result = {
        'dataset': domain,
        'cell_id': bench['cell_id'],
        'source_file': info.filename,
        'pickle_status': '',
        'identity_match': 0,
        'total_cycle_records': 0,
        'explicit_cycle_records': 0,
        'valid_positive_cycles': 0,
        'observed_post100_cycles': 0,
        'last_observed_cycle': '',
        'nearest_valid_cycle_to_100': '',
        'nearest100_after100': 0,
        'benchmark_cap100_absdiff': '',
        'retention_rebuild_absdiff': '',
        'first_exact_capacity_Ah': ''
    }

    for n in ANCHORS:
        result[f'exact_cycle_{n}_count'] = 0

    obj, status, bad = scan_and_load(z, info)

    result['pickle_status'] = (
        status if not bad
        else status + ':' + ','.join(bad[:8])
    )

    if obj is None:
        return result

    result['identity_match'] = int(
        str(obj.get('cell_id')) == str(bench['cell_id'])
    )

    cycles = obj.get('cycle_data')

    if not isinstance(cycles, (list, tuple)):
        result['pickle_status'] = 'invalid_cycle_data_type'
        return result

    if len(cycles) > 1000000:
        result['pickle_status'] = 'oversize_cycle_count'
        return result

    result['total_cycle_records'] = len(cycles)

    measured = []

    for cyc in cycles:

        if not isinstance(cyc, dict):
            continue

        cycle = (
            numeric_cycle(cyc.get('cycle_number'))
            if 'cycle_number' in cyc
            else None
        )

        # Never invent a cycle index from list position.
        if cycle is None:
            continue

        result['explicit_cycle_records'] += 1

        cap = finite_max(
            cyc.get('discharge_capacity_in_Ah')
        )

        if cap is None or cap <= 0:
            continue

        measured.append((cycle, cap))

    result['valid_positive_cycles'] = len(measured)

    result['observed_post100_cycles'] = sum(
        cycle > 100 for cycle, _ in measured
    )

    if measured:

        result['last_observed_cycle'] = max(
            cycle for cycle, _ in measured
        )

        nearest = min(
            measured,
            key=lambda item: (
                abs(item[0] - 100),
                item[0]
            )
        )

        result['nearest_valid_cycle_to_100'] = nearest[0]

        result['nearest100_after100'] = int(
            nearest[0] > 100
        )

        ordered = [
            cap for _, cap in sorted(
                measured, key=lambda item: item[0]
            )
        ]

        first = median(ordered[:3])
        last = median(ordered[-3:])

        if first > 0:
            try:
                result['retention_rebuild_absdiff'] = abs(
                    float(bench['final_capacity_retention'])
                    - last / first
                )
            except (TypeError, ValueError):
                pass

        if bench.get('cap_cycle_100'):
            try:
                result['benchmark_cap100_absdiff'] = abs(
                    float(bench['cap_cycle_100'])
                    - nearest[1]
                )
            except (TypeError, ValueError):
                pass

        for n in ANCHORS:

            matches = [
                cap for cycle, cap in measured
                if cycle == n
            ]

            result[f'exact_cycle_{n}_count'] = len(matches)

            if n == 1 and len(matches) == 1:
                result['first_exact_capacity_Ah'] = matches[0]

    return result


def main():

    if CSV.exists() or JSON.exists():
        raise SystemExit(
            'Stage20E results already exist; refusing overwrite'
        )

    if hashed(BENCH, 'sha256') != SHA:
        raise SystemExit('Benchmark checksum mismatch')

    with BENCH.open(
        newline='', encoding='utf-8-sig'
    ) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 567

    selected = {
        (r['dataset_id'], r['cell_id']): r
        for r in rows
        if r['source_family'] == 'BatteryLife_selected'
        and r['dataset_id'] in ARCHIVES
    }

    assert len(selected) == 29

    results = []

    for domain, digest in ARCHIVES.items():

        path = RAW / (domain + '.zip')

        assert hashed(path, 'md5') == digest, (
            'Archive hash mismatch'
        )

        with zipfile.ZipFile(path) as z:

            infos = {
                x.filename: x
                for x in z.infolist()
                if not x.is_dir()
                and x.filename.endswith('.pkl')
            }

            group = [
                r for (d, _), r in selected.items()
                if d == domain
            ]

            assert len(group) == COUNTS[domain]
            assert len(infos) == COUNTS[domain]

            assert len({
                r['source_file'] for r in group
            }) == len(group)

            for r in group:

                name = r['source_file'].replace('\\', '/')

                if name not in infos:
                    raise SystemExit(
                        'Missing archive member: ' + name
                    )

                record = per_cell(
                    domain, r, z, infos[name]
                )

                results.append(record)

                print(
                    'CELL',
                    domain,
                    r['cell_id'],
                    record['pickle_status'],
                    'cycle_records',
                    record['total_cycle_records'],
                    flush=True
                )

    counts = {}

    for domain in ARCHIVES:

        subs = [
            r for r in results
            if r['dataset'] == domain
        ]

        counts[domain] = {
            'cells': len(subs),

            'primitive_only': sum(
                r['pickle_status'] == 'accepted_primitive_only'
                for r in subs
            ),

            'content_id_matches': sum(
                r['identity_match'] for r in subs
            ),

            'post100_observed': sum(
                r['observed_post100_cycles'] > 0
                for r in subs
            ),

            'nearest100_after100': sum(
                r['nearest100_after100'] for r in subs
            ),

            'retention_rebuild_close_1e-6': sum(
                isinstance(
                    r['retention_rebuild_absdiff'], float
                )
                and r['retention_rebuild_absdiff'] < 1e-6
                for r in subs
            ),

            **{
                f'exact_{n}': sum(
                    r[f'exact_cycle_{n}_count'] == 1
                    for r in subs
                )
                for n in ANCHORS
            }
        }

    OUT.mkdir(parents=True, exist_ok=True)

    with CSV.open(
        'x', newline='', encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    report = {
        'stage': '20E',
        'input_sha256': SHA,
        'archive_md5': ARCHIVES,
        'domains': counts,
        'constraints': [
            'Only permitted primitive pickle opcodes were decoded.',
            'No inferred cycle numbers were used.',
            'Exact-cycle presence does not prove protocol equivalence.',
            'No interpolation or model training was performed.'
        ]
    }

    JSON.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ) + '\n',
        encoding='utf-8'
    )

    print('=== STAGE20E SUMMARY ===')

    for domain, values in counts.items():
        print('DOMAIN', domain, values)

    print('STAGE20E_COMPLETED=YES')
    print('REPORT', JSON)


if __name__ == '__main__':
    main()
