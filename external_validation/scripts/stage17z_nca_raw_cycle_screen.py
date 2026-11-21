#!/usr/bin/env python3
"""Stage17Z: prospective, read-only Tongji/KIT NCA cycle integrity screen.

No training, predictions, leakage-based exclusion or final target certification.
All 66 cells are inventoried. Reads the authenticated zip without extraction.
"""
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path.cwd().resolve()
EV = ROOT / 'external_validation'
RAW = EV / 'raw/tongji_zenodo6405084/Dataset_1_NCA_battery.zip'
LOCK = EV / 'reports/stage17y_nca_pre_extraction_protocol.json'
Y = EV / 'reports/stage17y_tongji_nca_archive_inventory.json'
TRAIN = ROOT / 'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'
MODEL = EV / 'scripts/stage16a_isu_exploratory_transfer.py'
REPORTS = EV / 'reports'
CYCLES = REPORTS / 'stage17z_nca_observed_cycle_segments.csv'
CELLS = REPORTS / 'stage17z_nca_physical_cell_screen.csv'
REPORT = REPORTS / 'stage17z_nca_raw_cycle_screen_report.json'
EXPECTED_MD5 = 'cbd6d7eab2a296b52385e025e8d0f42a'
COLS = ['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h',
        'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
CYCLE_COLS = ['cell_id', 'segment_index', 'instrument_cycle', 'rows',
              'discharge_valid_rows', 'charge_valid_rows', 'voltage_valid_rows',
              'time_valid_rows', 'discharge_max_mAh_DIAGNOSTIC',
              'charge_max_mAh_DIAGNOSTIC', 'voltage_min_V', 'voltage_max_V',
              'q_large_decreases', 'time_backward_steps', 'q_negative_rows',
              'q_max_exceeds_4500_mAh', 'q_max_positive',
              'has_both_charge_discharge_positive', 'recording_ambiguity']


def require(condition, msg):
    if not condition:
        raise RuntimeError(msg)


def digest(path, kind):
    h = hashlib.new(kind)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def number(raw):
    try:
        x = float(raw)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError, OverflowError):
        return None


def cycle_number(raw):
    x = number(raw)
    if x is None or x < 0 or abs(x - round(x)) > 1e-6:
        return None
    return int(round(x))


def new_segment(cell, cycle, n):
    return dict(cell_id=cell, segment_index=n, instrument_cycle=cycle,
                rows=0, discharge_valid_rows=0, charge_valid_rows=0,
                voltage_valid_rows=0, time_valid_rows=0,
                discharge_max_mAh_DIAGNOSTIC=None, charge_max_mAh_DIAGNOSTIC=None,
                voltage_min_V=None, voltage_max_V=None, q_large_decreases=0,
                time_backward_steps=0, q_negative_rows=0,
                q_max_exceeds_4500_mAh=False, q_max_positive=False,
                has_both_charge_discharge_positive=False,
                recording_ambiguity=False)


def finish(s):
    s['q_max_positive'] = bool((s['discharge_max_mAh_DIAGNOSTIC'] or 0) > 0)
    s['q_max_exceeds_4500_mAh'] = bool((s['discharge_max_mAh_DIAGNOSTIC'] or 0) > 4500)
    s['has_both_charge_discharge_positive'] = bool(
        s['q_max_positive'] and (s['charge_max_mAh_DIAGNOSTIC'] or 0) > 0)
    s['recording_ambiguity'] = bool(s['time_backward_steps'] or s['q_large_decreases'])
    return s


def screen(zf, entry, writer):
    cell = Path(entry.filename).stem
    total_rows = 0
    bad_cycle_rows = 0
    bad_width_rows = 0
    segments = 0
    q_decreases = 0
    time_backwards = 0
    unusual_q = 0
    last_instrument_cycle = None
    seen_cycles = set()
    repeated_cycle_blocks = 0
    backward_cycle_transitions = 0
    first_ids = []
    q_for_ids = {}
    segment_ids = []
    s = None
    prior_q = None
    prior_time = None

    with zf.open(entry) as binary:
        stream = io.TextIOWrapper(binary, encoding='utf-8-sig', errors='replace', newline='')
        reader = csv.reader(stream)
        head = next(reader, None)
        require(head == COLS, f'Unexpected CSV header for {cell}: {head!r}')

        for row in reader:
            total_rows += 1
            if len(row) != len(COLS):
                bad_width_rows += 1
                continue

            cy = cycle_number(row[8])
            if cy is None:
                bad_cycle_rows += 1
                continue

            if s is None or cy != last_instrument_cycle:
                if s is not None:
                    completed = finish(s)
                    writer.writerow(completed)
                    q_for_ids.setdefault(completed['instrument_cycle'], []).append(
                        completed['discharge_max_mAh_DIAGNOSTIC'])

                if cy in seen_cycles:
                    repeated_cycle_blocks += 1

                if last_instrument_cycle is not None and cy < last_instrument_cycle:
                    backward_cycle_transitions += 1

                seen_cycles.add(cy)
                segments += 1
                s = new_segment(cell, cy, segments)
                segment_ids.append(cy)

                if len(first_ids) < 12:
                    first_ids.append(cy)

                prior_q = None
                prior_time = None
                last_instrument_cycle = cy

            s['rows'] += 1
            q = number(row[4])
            qc = number(row[5])
            volt = number(row[2])
            tt = number(row[0])

            if q is not None:
                s['discharge_valid_rows'] += 1
                m = s['discharge_max_mAh_DIAGNOSTIC']
                s['discharge_max_mAh_DIAGNOSTIC'] = q if m is None else max(m, q)

                if q < 0:
                    s['q_negative_rows'] += 1

                if prior_q is not None and q + 100 < prior_q:
                    s['q_large_decreases'] += 1
                    q_decreases += 1

                prior_q = q

            if qc is not None:
                s['charge_valid_rows'] += 1
                m = s['charge_max_mAh_DIAGNOSTIC']
                s['charge_max_mAh_DIAGNOSTIC'] = qc if m is None else max(m, qc)

            if volt is not None:
                s['voltage_valid_rows'] += 1
                lo = s['voltage_min_V']
                hi = s['voltage_max_V']
                s['voltage_min_V'] = volt if lo is None else min(lo, volt)
                s['voltage_max_V'] = volt if hi is None else max(hi, volt)

            if tt is not None:
                s['time_valid_rows'] += 1
                if prior_time is not None and tt + 1 < prior_time:
                    s['time_backward_steps'] += 1
                    time_backwards += 1
                prior_time = tt

        if s is not None:
            completed = finish(s)
            writer.writerow(completed)
            q_for_ids.setdefault(completed['instrument_cycle'], []).append(
                completed['discharge_max_mAh_DIAGNOSTIC'])

    require(total_rows > 0 and segments > 0, f'No valid cycles in {cell}')

    unique = len(seen_cycles)

    for values in q_for_ids.values():
        unusual_q += sum(v is not None and v > 4500 for v in values)

    probe = {}

    for target in (1, 10, 50, 100):
        values = q_for_ids.get(target, [])
        probe[f'cycle_{target}_segments'] = len(values)
        probe[f'cycle_{target}_q_max_mAh_DIAGNOSTIC'] = (
            values[0] if len(values) == 1 else None)

    return dict(
        cell_id=cell,
        archive_member=entry.filename,
        compressed_bytes=entry.compress_size,
        source_csv_bytes=entry.file_size,
        raw_rows=total_rows,
        malformed_width_rows=bad_width_rows,
        invalid_cycle_number_rows=bad_cycle_rows,
        contiguous_cycle_segments=segments,
        unique_instrument_cycles=unique,
        instrument_cycle_min=min(seen_cycles),
        instrument_cycle_max=max(seen_cycles),
        cycle_number_revisited_blocks=repeated_cycle_blocks,
        cycle_number_backward_transitions=backward_cycle_transitions,
        within_cycle_time_backward_steps=time_backwards,
        within_cycle_q_drop_over100mAh=q_decreases,
        segments_q_over4500mAh=unusual_q,
        raw_cycle_first12=first_ids,
        raw_cycle_last12=segment_ids[-12:],
        has_100_or_more_distinct_instrument_cycles=unique >= 100,
        has_exact_instrument_cycles_1_10_50_100=all(
            len(q_for_ids.get(c, [])) == 1 and (q_for_ids[c][0] or 0) > 0
            for c in (1, 10, 50, 100)),
        raw_recording_ambiguity=bool(
            bad_width_rows or bad_cycle_rows or repeated_cycle_blocks
            or backward_cycle_transitions or time_backwards or q_decreases),
        **probe
    )


def main():
    for path in (RAW, LOCK, Y, TRAIN, MODEL):
        require(path.is_file(), f'Missing input: {path}')

    for out in (CYCLES, CELLS, REPORT):
        require(not out.exists(), f'Refusing overwrite: {out}')

    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    y = json.loads(Y.read_text(encoding='utf-8'))

    require(
        y.get('md5_verified') is True and y.get('csv_entries') == 66,
        'Stage17Y archive screen not verified'
    )

    require(
        lock.get('training_567_sha256') == digest(TRAIN, 'sha256')
        and lock.get('original_stage16a_script_sha256') == digest(MODEL, 'sha256'),
        'Locked original training/model changed'
    )

    require(
        lock.get('source_doi') == '10.5281/zenodo.6405084',
        'Wrong source DOI'
    )

    require(
        y.get('archive_sha256') == digest(RAW, 'sha256')
        and digest(RAW, 'md5') == EXPECTED_MD5,
        'ZIP hash does not match Stage17Y'
    )

    REPORTS.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(RAW) as z, tempfile.TemporaryDirectory(
        prefix='stage17z_tmp_', dir=str(REPORTS)
    ) as temp:

        members = sorted([
            i for i in z.infolist()
            if not i.is_dir() and i.filename.lower().endswith('.csv')
        ], key=lambda x: x.filename)

        require(
            len(members) == 66,
            f'Expected 66 physical CSV files, got {len(members)}'
        )

        names = [Path(e.filename).stem for e in members]
        require(len(names) == len(set(names)), 'Duplicate cell file stems')

        tmp = Path(temp)
        cycle_temp = tmp / 'cycles.csv'
        cell_temp = tmp / 'cells.csv'
        report_temp = tmp / 'report.json'

        summaries = []

        with cycle_temp.open('w', newline='', encoding='utf-8') as f:
            cw = csv.DictWriter(f, fieldnames=CYCLE_COLS, extrasaction='raise')
            cw.writeheader()

            for i, item in enumerate(members, 1):
                info = screen(z, item, cw)
                summaries.append(info)

                print(
                    f'CELL_{i:02d}/66={info["cell_id"]} '
                    f'cycle_segments={info["contiguous_cycle_segments"]} '
                    f'unique_cycles={info["unique_instrument_cycles"]} '
                    f'has_1_10_50_100={info["has_exact_instrument_cycles_1_10_50_100"]} '
                    f'ambiguous={info["raw_recording_ambiguity"]}',
                    flush=True
                )

        with cell_temp.open('w', newline='', encoding='utf-8') as f:
            cols = list(summaries[0].keys())
            cw = csv.DictWriter(f, fieldnames=cols)
            cw.writeheader()

            for item in summaries:
                row = dict(item)
                for k in ('raw_cycle_first12', 'raw_cycle_last12'):
                    row[k] = json.dumps(row[k])
                cw.writerow(row)

        total = Counter()

        for item in summaries:
            for field in (
                'contiguous_cycle_segments',
                'unique_instrument_cycles',
                'raw_rows',
                'malformed_width_rows',
                'invalid_cycle_number_rows',
                'cycle_number_revisited_blocks',
                'cycle_number_backward_transitions',
                'within_cycle_time_backward_steps',
                'within_cycle_q_drop_over100mAh'
            ):
                total[field] += item[field]

        first100 = sum(
            i['has_exact_instrument_cycles_1_10_50_100']
            for i in summaries
        )

        ambiguous = sum(
            i['raw_recording_ambiguity']
            for i in summaries
        )

        report = dict(
            stage='17Z_NCA_RAW_CYCLE_STRUCTURAL_SCREEN',
            raw_archive_sha256=digest(RAW, 'sha256'),
            protocol_lock_sha256=digest(LOCK, 'sha256'),
            stage17y_sha256=digest(Y, 'sha256'),
            training_sha256=digest(TRAIN, 'sha256'),
            original_model_code_sha256=digest(MODEL, 'sha256'),
            cells_in_archive=66,
            raw_totals=dict(total),
            cells_with_unique_positive_cycle_1_10_50_100=first100,
            cells_with_recording_ambiguity=ambiguous,
            cells_below_100_distinct_cycle_ids=sum(
                i['unique_instrument_cycles'] < 100
                for i in summaries
            ),
            source_code_capacity_method=(
                'per-instrument-cycle max Q discharge/mA.h; diagnostic only'
            ),
            NOTE=(
                'Structural screen only: positive Q and cycle presence do not '
                'prove a full discharge or a comparable target.'
            ),
            physically_eligible_cells_NOT_YET_CERTIFIED=True,
            outcome_based_filtering=False,
            target_equivalence_certified=False,
            models_fitted=False,
            predictions_produced=False,
            prism_updated=False,
            output_cycle_rows=total['contiguous_cycle_segments']
        )

        report_temp.write_text(
            json.dumps(report, indent=2),
            encoding='utf-8'
        )

        require(
            not any(x.exists() for x in (CYCLES, CELLS, REPORT)),
            'Output appeared during processing'
        )

        os.replace(cycle_temp, CYCLES)
        os.replace(cell_temp, CELLS)
        os.replace(report_temp, REPORT)

    print('=== STAGE 17Z NCA RAW CYCLE STRUCTURAL SCREEN ===')
    print('PHYSICAL_CELLS=66')
    print('RAW_COUNTS=', json.dumps(dict(total)))
    print('CELLS_WITH_CYCLES_1_10_50_100_POSITIVE=', first100)
    print('CELLS_WITH_ANY_RECORDING_AMBIGUITY=', ambiguous)
    print(
        'CELLS_BELOW_100_DISTINCT_INSTRUMENT_CYCLES=',
        report['cells_below_100_distinct_cycle_ids']
    )
    print('ALL_66_CELLS_RETAINED_IN_SCREEN=YES')
    print('CYCLE_SEGMENT_CSV=', CYCLES)
    print('PHYSICAL_CELL_CSV=', CELLS)
    print('REPORT_JSON=', REPORT)
    print(
        'PHYSICAL_COMPLETENESS_CERTIFIED=NO '
        'TARGET_EQUIVALENCE_CERTIFIED=NO '
        'MODEL_FITTING=NO INFERENCE=NO PRISM_UPDATED=NO'
    )
    print('=== END STAGE 17Z ===')


if __name__ == '__main__':
    main()
