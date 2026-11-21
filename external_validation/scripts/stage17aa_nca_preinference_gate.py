#!/usr/bin/env python3
"""Stage17AA: prospective NCA first-100-cycle and raw recording-gap gate.
No prediction, target-based model selection, training, or overwritten inputs.
"""
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd().resolve()
EV = ROOT / 'external_validation'
R = EV / 'reports'

RAW = EV / 'raw/tongji_zenodo6405084/Dataset_1_NCA_battery.zip'
LOCK = R / 'stage17y_nca_pre_extraction_protocol.json'
Y = R / 'stage17y_tongji_nca_archive_inventory.json'
Z = R / 'stage17z_nca_raw_cycle_screen_report.json'
CELLS = R / 'stage17z_nca_physical_cell_screen.csv'
CYCLES = R / 'stage17z_nca_observed_cycle_segments.csv'

TRAIN = ROOT / (
    'data/05_model_ready/aging/'
    'expanded_aging_features_with_batterylife.csv'
)
MODEL = EV / 'scripts/stage16a_isu_exploratory_transfer.py'

OUT_CSV = R / 'stage17aa_nca_66cell_preinference_gate.csv'
OUT_JSON = R / 'stage17aa_nca_preinference_gate_report.json'

HEADER = [
    'time/s', 'control/V/mA', 'Ecell/V', '<I>/mA',
    'Q discharge/mA.h', 'Q charge/mA.h',
    'control/V', 'control/mA', 'cycle number'
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    require(path.is_file(), f'Missing input: {path}')
    with path.open(
        encoding='utf-8-sig', newline=''
    ) as f:
        rd = csv.DictReader(f)
        return list(rd), list(rd.fieldnames or [])


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def yes(value):
    return str(value).strip().lower() == 'true'


def scan_time_gaps(archive, entry):
    total = 0
    within = 0
    active = 0
    early = 0
    rows = 0
    examples = []
    previous = None

    with archive.open(entry) as binary:
        stream = io.TextIOWrapper(
            binary, encoding='utf-8-sig',
            newline='', errors='replace'
        )
        reader = csv.reader(stream)

        require(
            next(reader, None) == HEADER,
            f'Header mismatch: {entry.filename}'
        )

        for row in reader:
            rows += 1

            if len(row) != 9:
                previous = None
                continue

            t = number(row[0])
            current = number(row[3])
            qd = number(row[4])
            qc = number(row[5])
            raw_cycle = number(row[8])

            if t is None or raw_cycle is None:
                previous = None
                continue

            if abs(raw_cycle - round(raw_cycle)) > 1e-6:
                previous = None
                continue

            cycle = int(round(raw_cycle))

            if previous is not None:
                pt, pc, pcur, pqd, pqc = previous
                gap = t - pt

                if gap > 3600:
                    total += 1
                    same_cycle = cycle == pc

                    current_active = (
                        abs(current or 0) > 10
                        or abs(pcur or 0) > 10
                    )

                    counter_advanced = (
                        qd is not None
                        and pqd is not None
                        and qd - pqd > 10
                    ) or (
                        qc is not None
                        and pqc is not None
                        and qc - pqc > 10
                    )

                    if same_cycle:
                        within += 1

                    if same_cycle and (
                        current_active or counter_advanced
                    ):
                        active += 1
                        if cycle <= 100:
                            early += 1

                    if len(examples) < 5:
                        examples.append({
                            'row': rows,
                            'cycle_before': pc,
                            'cycle_after': cycle,
                            'gap_seconds': round(gap, 2),
                            'same_cycle': same_cycle,
                            'active_signal': bool(
                                current_active or counter_advanced
                            )
                        })

            previous = (t, cycle, current, qd, qc)

    return {
        'raw_rows_rescanned': rows,
        'gaps_over_1h_all': total,
        'gaps_over_1h_within_cycle': within,
        'gaps_over_1h_active_within_cycle': active,
        'gaps_over_1h_active_first100': early,
        'gap_examples': examples
    }


def main():
    inputs = (
        RAW, LOCK, Y, Z, CELLS,
        CYCLES, TRAIN, MODEL
    )

    for path in inputs:
        require(path.is_file(), f'Missing: {path}')

    for path in (OUT_CSV, OUT_JSON):
        require(
            not path.exists(),
            f'Refusing overwrite: {path}'
        )

    y = json.loads(Y.read_text(encoding='utf-8'))
    z = json.loads(Z.read_text(encoding='utf-8'))
    lock = json.loads(LOCK.read_text(encoding='utf-8'))

    require(
        y.get('csv_entries') == 66
        and y.get('md5_verified') is True,
        'Stage17Y verification mismatch'
    )

    require(
        z.get('cells_in_archive') == 66
        and z.get('models_fitted') is False,
        'Stage17Z scope mismatch'
    )

    require(
        y['archive_sha256']
        == z['raw_archive_sha256']
        == sha(RAW),
        'Archive hash changed'
    )

    require(
        z['protocol_lock_sha256'] == sha(LOCK),
        'Protocol lock changed'
    )

    require(
        lock['training_567_sha256'] == sha(TRAIN)
        and lock['original_stage16a_script_sha256']
            == sha(MODEL),
        'Original training source or model code changed'
    )

    cells, _ = read_csv(CELLS)
    segments, columns = read_csv(CYCLES)

    require(
        len(cells) == 66
        and len(segments) == z['output_cycle_rows'],
        'Stage17Z cohort/count mismatch'
    )

    need_columns = [
        'cell_id', 'instrument_cycle', 'segment_index',
        'discharge_max_mAh_DIAGNOSTIC',
        'q_large_decreases', 'time_backward_steps',
        'voltage_min_V'
    ]

    require(
        set(need_columns).issubset(columns),
        'Missing Stage17Z cycle columns'
    )

    grouped = defaultdict(list)

    for segment in segments:
        grouped[segment['cell_id']].append(segment)

    require(
        len(grouped) == 66
        and set(grouped) == {
            cell['cell_id'] for cell in cells
        },
        'Cell identity mismatch'
    )

    results = []

    with zipfile.ZipFile(RAW) as archive:
        members = {
            Path(item.filename).stem: item
            for item in archive.infolist()
            if (
                item.filename.lower().endswith('.csv')
                and not item.is_dir()
            )
        }

        require(
            len(members) == 66
            and set(members) == set(grouped),
            'Raw ZIP cohort mismatch'
        )

        for index, cell_row in enumerate(
            sorted(cells, key=lambda x: x['cell_id']), 1
        ):
            cell_id = cell_row['cell_id']
            blocks = grouped[cell_id]

            by_cycle = defaultdict(list)

            for block in blocks:
                cy = int(float(block['instrument_cycle']))
                by_cycle[cy].append(block)

            def positive_cycle(n):
                return (
                    len(by_cycle[n]) == 1
                    and (
                        number(
                            by_cycle[n][0][
                                'discharge_max_mAh_DIAGNOSTIC'
                            ]
                        ) or 0
                    ) > 0
                )

            first100_single = all(
                len(by_cycle[n]) == 1
                for n in range(1, 101)
            )

            first100_positive = all(
                positive_cycle(n)
                for n in range(1, 101)
            )

            q_low = 0
            q_high = 0
            voltage_near_cutoff = 0

            for n in range(1, 101):
                if len(by_cycle[n]) != 1:
                    continue

                block = by_cycle[n]
                cap = number(
                    block[0][
                        'discharge_max_mAh_DIAGNOSTIC'
                    ]
                )

                voltage = number(
                    block[0]['voltage_min_V']
                )

                if cap is not None:
                    q_low += int(0 < cap < 100)
                    q_high += int(cap > 4500)

                if voltage is not None:
                    voltage_near_cutoff += int(
                        2.4 <= voltage <= 2.9
                    )

            early_anomalies = sum(
                int(float(block['q_large_decreases']))
                + int(float(block['time_backward_steps']))
                for block in blocks
                if int(float(block['instrument_cycle'])) <= 100
            )

            last_cycle = max(by_cycle)
            last_cycle_positive = positive_cycle(last_cycle)

            original_four = yes(
                cell_row[
                    'has_exact_instrument_cycles_1_10_50_100'
                ]
            )

            original_ambiguity = yes(
                cell_row['raw_recording_ambiguity']
            )

            require(
                original_four == all(
                    positive_cycle(n)
                    for n in (1, 10, 50, 100)
                ),
                f'Original Stage17Z flag mismatch: {cell_id}'
            )

            gaps = scan_time_gaps(
                archive, members[cell_id]
            )

            require(
                gaps['raw_rows_rescanned']
                    == int(cell_row['raw_rows']),
                f'Raw row-count mismatch: {cell_id}'
            )

            original_structural_group = (
                original_four
                and not original_ambiguity
            )

            early_gate = (
                first100_single
                and first100_positive
                and not original_ambiguity
                and q_low == 0
                and q_high == 0
                and early_anomalies == 0
                and gaps['gaps_over_1h_active_first100'] == 0
            )

            reasons = []

            if not original_four:
                reasons.append(
                    'FOUR_REQUIRED_CYCLES_NOT_UNIQUE_POSITIVE'
                )

            if original_ambiguity:
                reasons.append(
                    'STAGE17Z_RECORDING_AMBIGUITY'
                )

            if not first100_single:
                reasons.append(
                    'MISSING_OR_REPEATED_FIRST100_CYCLE'
                )

            if not first100_positive:
                reasons.append(
                    'MISSING_OR_NONPOSITIVE_FIRST100_CAPACITY'
                )

            if q_low:
                reasons.append(
                    'EARLY_CAPACITY_BELOW_100MAH'
                )

            if q_high:
                reasons.append(
                    'EARLY_CAPACITY_ABOVE_4500MAH'
                )

            if early_anomalies:
                reasons.append(
                    'EARLY_COUNTER_OR_TIME_ANOMALY'
                )

            if gaps['gaps_over_1h_active_first100']:
                reasons.append(
                    'EARLY_LONG_ACTIVE_RECORDING_GAP'
                )

            if not last_cycle_positive:
                reasons.append(
                    'LAST_CYCLE_POSITIVE_CAPACITY_NOT_AVAILABLE'
                )

            result = {
                'cell_id': cell_id,
                'program': cell_id.split('-#')[0],
                'distinct_cycle_ids': len(by_cycle),
                'last_instrument_cycle': last_cycle,
                'stage17z_four_required_positive': original_four,
                'stage17z_recording_ambiguity': original_ambiguity,
                'stage17z_initial_structural_47_group':
                    original_structural_group,
                'every_first100_cycle_exactly_once':
                    first100_single,
                'every_first100_capacity_positive':
                    first100_positive,
                'first100_q_below_100mAh': q_low,
                'first100_q_above_4500mAh': q_high,
                'first100_cycles_voltage_min_2p4_to_2p9V':
                    voltage_near_cutoff,
                'first100_counter_clock_anomalies':
                    early_anomalies,
                'last_cycle_positive_unique':
                    last_cycle_positive,
                'strict_first100_gate_DIAGNOSTIC':
                    early_gate,
                'full_discharge_completeness_certified': False,
                'terminal_endpoint_certified': False,
                'target_equivalence_certified': False,
                'issues': '|'.join(reasons),
                **{
                    k: v
                    for k, v in gaps.items()
                    if k != 'gap_examples'
                },
                'gap_examples_json':
                    json.dumps(gaps['gap_examples'])
            }

            results.append(result)

            print(
                f'CELL_{index:02d}/66={cell_id} '
                f'first100_single={first100_single} '
                f'early_gate={early_gate} '
                f'active_gaps={gaps["gaps_over_1h_active_within_cycle"]}',
                flush=True
            )

    summary = {
        'stage17z_four_cycles_no_ambiguity': sum(
            r['stage17z_initial_structural_47_group']
            for r in results
        ),
        'cells_with_100_unique_contiguous_cycles': sum(
            r['every_first100_cycle_exactly_once']
            for r in results
        ),
        'cells_with_100_positive_capacities': sum(
            r['every_first100_capacity_positive']
            for r in results
        ),
        'strict_first100_gate_DIAGNOSTIC': sum(
            r['strict_first100_gate_DIAGNOSTIC']
            for r in results
        ),
        'cells_with_active_gap_within_cycle_over_1h': sum(
            r['gaps_over_1h_active_within_cycle'] > 0
            for r in results
        ),
        'cells_with_active_gap_first100_over_1h': sum(
            r['gaps_over_1h_active_first100'] > 0
            for r in results
        ),
        'cells_with_positive_unique_last_cycle': sum(
            r['last_cycle_positive_unique']
            for r in results
        )
    }

    require(
        summary['stage17z_four_cycles_no_ambiguity'] == 47,
        'Stage17Z baseline cohort discrepancy'
    )

    report = {
        'stage': '17AA_NCA_PREINFERENCE_RAW_GAP_GATE',
        'source_doi': lock['source_doi'],
        'physical_cells': 66,
        'source_sha256': {
            path.relative_to(ROOT).as_posix(): sha(path)
            for path in inputs
        },
        'quality_counts': summary,
        'one_hour_gap_is_diagnostic_not_proof_of_incompleteness':
            True,
        'no_external_target_used_for_model_selection': True,
        'full_discharge_completeness_certified': False,
        'target_equivalence_certified': False,
        'models_fitted': False,
        'inference': False,
        'prism_updated': False
    }

    with tempfile.TemporaryDirectory(
        prefix='stage17aa_',
        dir=R
    ) as folder:
        temp = Path(folder)
        temp_csv = temp / 'cells.csv'
        temp_json = temp / 'report.json'

        with temp_csv.open(
            'w', newline='', encoding='utf-8'
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(results[0])
            )
            writer.writeheader()
            writer.writerows(results)

        temp_json.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2
            ),
            encoding='utf-8'
        )

        require(
            not OUT_CSV.exists()
            and not OUT_JSON.exists(),
            'Output appeared during processing'
        )

        os.replace(temp_csv, OUT_CSV)
        os.replace(temp_json, OUT_JSON)

    flagged = [
        {
            'cell_id': r['cell_id'],
            'issues': r['issues'],
            'long_active_gaps':
                r['gaps_over_1h_active_within_cycle']
        }
        for r in results
        if (
            r['issues']
            or r['gaps_over_1h_active_within_cycle']
        )
    ]

    print(
        '=== STAGE 17AA NCA PRE-INFERENCE QUALITY GATE ==='
    )
    print('PHYSICAL_CELLS=66')
    print('ORIGINAL_ARCHIVE_HASH_UNCHANGED=YES')

    print(
        'QUALITY_COUNTS=',
        json.dumps(summary)
    )

    print(
        'FLAGGED_CELLS=',
        json.dumps(
            flagged, ensure_ascii=False
        )
    )

    print('CELL_GATE_CSV=', OUT_CSV)
    print('REPORT_JSON=', OUT_JSON)

    print(
        'COMPLETE_DISCHARGE_CERTIFIED=NO '
        'TARGET_EQUIVALENCE_CERTIFIED=NO '
        'MODEL_TRAINING=NO INFERENCE=NO '
        'PRISM_UPDATED=NO'
    )

    print('=== END STAGE 17AA ===')


if __name__ == '__main__':
    main()
