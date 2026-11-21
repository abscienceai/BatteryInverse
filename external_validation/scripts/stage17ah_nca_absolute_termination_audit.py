#!/usr/bin/env python3
"""Stage17AH: post-outcome absolute-capacity and horizon clustering audit.

Reads only existing reports. No NCA inference, model training or retuning.
Findings are descriptive, not stopping-rule certification.
"""
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
R = ROOT / 'external_validation/reports'

AF = R / 'stage17af_nca_unsealed_outcome_evidence.json'
AF_CELLS = R / 'stage17af_nca_30cell_last_cycle_diagnostic_and_locked_predictions.csv'
AG = R / 'stage17ag_nca_stopping_horizon_report.json'
AG_CELLS = R / 'stage17ag_nca_30cell_stopping_horizon_diagnostics.csv'
AA = R / 'stage17aa_nca_66cell_preinference_gate.csv'
ZC = R / 'stage17z_nca_observed_cycle_segments.csv'
AE = R / 'stage17ae_nca_preoutcome_model_lock_report.json'
PRED = R / 'stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv'

OUT_CELL = R / 'stage17ah_nca_66cell_absolute_last_capacity_audit.csv'
OUT_GROUP = R / 'stage17ah_nca_group_endpoint_clusters.csv'
OUT_REPORT = R / 'stage17ah_nca_absolute_endpoint_and_censoring_report.json'

OUTS = (OUT_CELL, OUT_GROUP, OUT_REPORT)

PRED_SHA = (
    'cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757'
)


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path, cols):
    require(path.is_file(), f'Missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        require(
            set(cols).issubset(rd.fieldnames or []),
            f'Columns missing from {path.name}: '
            f'{set(cols)-set(rd.fieldnames or [])}'
        )
        return list(rd)


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def integer(value, label):
    x = number(value)
    require(
        x is not None and x >= 0 and x.is_integer(),
        f'Invalid integer {label}={value}'
    )
    return int(x)


def truth(value):
    return str(value).strip().lower() == 'true'


def summary(values):
    a = np.asarray(
        [x for x in values if x is not None],
        dtype=float
    )
    if not len(a):
        return {'n': 0}
    return {
        'n': int(len(a)),
        'min': float(a.min()),
        'p05': float(np.quantile(a, .05)),
        'median': float(np.median(a)),
        'p95': float(np.quantile(a, .95)),
        'max': float(a.max())
    }


def write_csv(path, rows):
    require(bool(rows), f'Empty output {path}')
    with path.open('x', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def verify_locks():
    sources = (
        AF, AF_CELLS, AG, AG_CELLS,
        AA, ZC, AE, PRED
    )

    for path in sources:
        require(path.is_file(), f'Missing source: {path}')

    for path in OUTS:
        require(
            not path.exists(),
            f'Refusing output overwrite: {path}'
        )

    af = json.loads(AF.read_text(encoding='utf-8'))
    ag = json.loads(AG.read_text(encoding='utf-8'))
    ae = json.loads(AE.read_text(encoding='utf-8'))

    require(
        af.get('NCA_locked_test_cells') == 30
        and af.get('NCA_positive_last_cycle_cells') == 30,
        'Stage17AF 30-cell cohort mismatch'
    )

    require(
        ag.get('physical_cells') == 30
        and ag.get('model_training') is False
        and ag.get('prediction_changes') is False,
        'Stage17AG provenance mismatch'
    )

    require(
        ae.get('NCA_terminal_labels_accessed') is False
        and ae.get(
            'NCA_predictions_locked_before_terminal_outcome_review'
        ) is True,
        'Stage17AE was not preoutcome-locked'
    )

    require(
        sha(PRED) == PRED_SHA
        == af.get('preoutcome_predictions_sha256'),
        'Frozen prediction SHA256 mismatch'
    )

    require(
        sha(ZC) == af.get('raw_cycle_segments_sha256'),
        'Stage17AF raw cycle summary changed'
    )

    require(
        sha(AA) == af.get('first100_gate_sha256'),
        'Stage17AF quality gate changed'
    )

    for relative, expected in ag.get(
        'input_sha256', {}
    ).items():
        rp = Path(relative)

        require(
            not rp.is_absolute() and '..' not in rp.parts,
            'Unsafe Stage17AG source path'
        )

        p = (ROOT / rp).resolve()

        require(
            p.is_relative_to(ROOT)
            and p.is_file()
            and sha(p) == expected,
            f'Stage17AG input changed: {relative}'
        )

    return af, ag


def main():
    af, ag = verify_locks()

    gates = read_csv(
        AA,
        [
            'cell_id',
            'program',
            'strict_first100_gate_DIAGNOSTIC'
        ]
    )

    af_cells = read_csv(
        AF_CELLS,
        [
            'cell_id',
            'program',
            'first_cycle_capacity_mAh',
            'last_instrument_cycle',
            'last_cycle_capacity_mAh',
            'measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL'
        ]
    )

    ag_cells = read_csv(
        AG_CELLS,
        [
            'cell_id',
            'program',
            'first_below_80pct_cycle',
            'first_3consecutive_below_80pct_cycle',
            'last_recorded_capacity_ratio'
        ]
    )

    segments = read_csv(
        ZC,
        [
            'cell_id',
            'instrument_cycle',
            'segment_index',
            'discharge_max_mAh_DIAGNOSTIC'
        ]
    )

    require(
        len(gates) == 66
        and len(af_cells) == 30
        and len(ag_cells) == 30,
        'Unexpected row count'
    )

    gateids = [str(r['cell_id']) for r in gates]

    require(
        len(set(gateids)) == 66,
        'Duplicate gate cell IDs'
    )

    af_index = {
        r['cell_id']: r for r in af_cells
    }
    ag_index = {
        r['cell_id']: r for r in ag_cells
    }

    require(
        len(af_index) == len(ag_index) == 30
        and set(af_index) == set(ag_index),
        'Selected roster mismatch'
    )

    selected = {
        r['cell_id']
        for r in gates
        if truth(r['strict_first100_gate_DIAGNOSTIC'])
    }

    require(
        selected == set(af_index),
        '30-cell gate and frozen prediction roster mismatch'
    )

    known = set(gateids)
    bycell = defaultdict(list)

    for r in segments:
        require(
            r['cell_id'] in known,
            'Unknown segment cell'
        )
        bycell[r['cell_id']].append(r)

    require(
        set(bycell) == known,
        'Missing raw segments'
    )

    rows = []

    for gate in sorted(
        gates, key=lambda r: r['cell_id']
    ):
        cid = gate['cell_id']
        seq = bycell[cid]

        seq.sort(
            key=lambda r: integer(
                r['segment_index'],
                f'{cid} segment'
            )
        )

        positions = [
            integer(s['segment_index'], cid)
            for s in seq
        ]

        last = seq[-1]

        last_cycle = integer(
            last['instrument_cycle'],
            f'{cid} last cycle'
        )

        last_q = number(
            last['discharge_max_mAh_DIAGNOSTIC']
        )

        positive = [
            (
                integer(s['instrument_cycle'], cid),
                number(s['discharge_max_mAh_DIAGNOSTIC'])
            )
            for s in seq
        ]

        positives = [
            (cy, q)
            for cy, q in positive
            if q is not None and q > 0
        ]

        last_valid_q = (
            positives[-1][1]
            if positives else None
        )

        recent = [
            q for _, q in positives[-6:-1]
        ]

        q2500dist = (
            last_q - 2500
            if last_q is not None else None
        )

        # The 2500-mAh grid is a post-outcome
        # descriptive check, not a prespecified
        # model endpoint or certified stop rule.
        row = {
            'cell_id': cid,
            'program': gate['program'],
            'selected_preoutcome_30': cid in selected,
            'first100_gate': truth(
                gate['strict_first100_gate_DIAGNOSTIC']
            ),
            'last_instrument_cycle': last_cycle,
            'last_segment_discharge_mAh': last_q,
            'last_segment_positive':
                last_q is not None and last_q > 0,
            'last_positive_segment_discharge_mAh_DIAGNOSTIC':
                last_valid_q,
            'last_segment_equal_last_positive':
                last_q is not None
                and last_valid_q is not None
                and abs(last_q - last_valid_q) < 1e-8,
            'last_mAh_minus_2500': q2500dist,
            'last_within_5mAh_of_2500':
                q2500dist is not None
                and abs(q2500dist) <= 5,
            'last_within_10mAh_of_2500':
                q2500dist is not None
                and abs(q2500dist) <= 10,
            'last_within_25mAh_of_2500':
                q2500dist is not None
                and abs(q2500dist) <= 25,
            'last_within_50mAh_of_2500':
                q2500dist is not None
                and abs(q2500dist) <= 50,
            'last_within_100mAh_of_2500':
                q2500dist is not None
                and abs(q2500dist) <= 100,
            'last_minus_previous5_positive_median_mAh':
                (
                    last_q - float(np.median(recent))
                    if last_q is not None and recent
                    else None
                ),
            'last_5_positive_cycle_count':
                len(recent),
            'instrument_cycle_revisited':
                len({cy for cy, _ in positive})
                < len(positive),
            'segment_index_repeated':
                len(set(positions)) < len(positions),
            'last_instrument_cycle_multiple_of_100':
                last_cycle % 100 == 0,
            'first80pct_crossing_cycle_if_selected':
                None,
            'three_consecutive_80pct_crossing_if_selected':
                None,
            'cycles_after_first80pct_crossing_if_selected':
                None,
            'selected_last_ratio': None,
            'selected_cycle1_mAh': None,
            'observational_2500mAh_is_not_verified_stop_rule':
                True
        }

        if cid in selected:
            a = af_index[cid]
            g = ag_index[cid]

            require(
                a['program'] == gate['program']
                == g['program'],
                f'Program mismatch: {cid}'
            )

            qaf = number(a['last_cycle_capacity_mAh'])
            lastaf = integer(
                a['last_instrument_cycle'], cid
            )
            q1 = number(
                a['first_cycle_capacity_mAh']
            )
            ratio = number(
                a[
                    'measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL'
                ]
            )

            require(
                last_q is not None
                and qaf is not None
                and abs(qaf - last_q) < 1e-8
                and lastaf == last_cycle,
                f'Stage17AF terminal mismatch: {cid}'
            )

            require(
                q1 is not None
                and q1 > 0
                and ratio is not None
                and abs(last_q / q1 - ratio) < 1e-9,
                f'Stage17AF retention mismatch: {cid}'
            )

            gr = number(
                g['last_recorded_capacity_ratio']
            )

            require(
                gr is not None
                and abs(gr - ratio) < 1e-9,
                f'Stage17AG retention mismatch: {cid}'
            )

            cy80 = number(
                g['first_below_80pct_cycle']
            )
            cy3 = number(
                g['first_3consecutive_below_80pct_cycle']
            )

            row.update({
                'first80pct_crossing_cycle_if_selected':
                    int(cy80)
                    if cy80 is not None else None,
                'three_consecutive_80pct_crossing_if_selected':
                    int(cy3)
                    if cy3 is not None else None,
                'cycles_after_first80pct_crossing_if_selected':
                    last_cycle - int(cy80)
                    if cy80 is not None else None,
                'selected_last_ratio': ratio,
                'selected_cycle1_mAh': q1
            })

        rows.append(row)

    cohorts = [
        ('ALL66_ARCHIVE', rows),
        (
            'SELECTED30_PREOUTCOME_GATE',
            [
                x for x in rows
                if x['selected_preoutcome_30']
            ]
        ),
        (
            'EXCLUDED36_OTHER',
            [
                x for x in rows
                if not x['selected_preoutcome_30']
            ]
        )
    ]

    for program in sorted({
        r['program'] for r in rows
    }):
        cohorts.append((
            'SELECTED30_' + program,
            [
                r for r in rows
                if r['selected_preoutcome_30']
                and r['program'] == program
            ]
        ))

        cohorts.append((
            'ALL66_' + program,
            [
                r for r in rows
                if r['program'] == program
            ]
        ))

    groupout = []

    for name, group in cohorts:
        if not group:
            continue

        valid = [
            r for r in group
            if r['last_segment_positive']
        ]

        groupout.append({
            'cohort': name,
            'cells': len(group),
            'last_positive_cells': len(valid),
            'last_q_mAh_summary_json':
                json.dumps(summary([
                    r['last_segment_discharge_mAh']
                    for r in valid
                ])),
            'last_cycle_summary_json':
                json.dumps(summary([
                    r['last_instrument_cycle']
                    for r in group
                ])),
            'last_q_within_5mAh_2500_count':
                sum(r['last_within_5mAh_of_2500']
                    for r in group),
            'last_q_within_10mAh_2500_count':
                sum(r['last_within_10mAh_of_2500']
                    for r in group),
            'last_q_within_25mAh_2500_count':
                sum(r['last_within_25mAh_of_2500']
                    for r in group),
            'last_q_within_50mAh_2500_count':
                sum(r['last_within_50mAh_of_2500']
                    for r in group),
            'last_q_within_100mAh_2500_count':
                sum(r['last_within_100mAh_of_2500']
                    for r in group),
            'last_cycle_multiple100_count':
                sum(
                    r['last_instrument_cycle_multiple_of_100']
                    for r in group
                ),
            'observed_first80pct_crossing_count':
                sum(
                    r['first80pct_crossing_cycle_if_selected']
                    is not None for r in group
                ),
            'median_cycles_after_first80pct_crossing':
                summary([
                    r[
                        'cycles_after_first80pct_crossing_if_selected'
                    ]
                    for r in group
                ]).get('median'),
            'recording_cycle_revisit_cells':
                sum(
                    r['instrument_cycle_revisited']
                    for r in group
                )
        })

    report = {
        'stage':
            '17AH_POSTOUTCOME_ABSOLUTE_CAPACITY_TERMINATION_CLUSTER_AUDIT',
        'reference_hypothesis_mAh': 2500,
        'hypothesis_selected_after_outcome_review':
            True,
        'reference_percent_of_3500mAh_nominal_only':
            2500 / 3500,
        'nominal_capacity_3500mAh_verified_for_each_cell':
            False,
        'original_experiment_stopping_rule_verified':
            False,
        'nonpositive_last_segments_are_not_replaced_silently':
            True,
        'preoutcome_locked_predictions_unchanged':
            True,
        'prediction_sha256': sha(PRED),
        'original_66_archive_count': 66,
        'selected_30_count': 30,
        'source_sha256': {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                AF, AF_CELLS, AG, AG_CELLS,
                AA, ZC, AE, PRED
            )
        },
        'cohort_summaries': groupout,
        'right_censoring_note':
            'Absence of an observed threshold crossing '
            'is not proof that it never occurs after '
            'recording stops.',
        'quality_note':
            'Instrument-cycle maximum discharge Q '
            'is diagnostic; late recording gaps may '
            'compromise measurement completeness.',
        'new_training': False,
        'new_model_inference': False,
        'prism_updated': False
    }

    with tempfile.TemporaryDirectory(
        prefix='stage17ah_', dir=R
    ) as folder:
        t = Path(folder)

        write_csv(t / 'cells.csv', rows)
        write_csv(t / 'groups.csv', groupout)

        (t / 'report.json').write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2
            ),
            encoding='utf-8'
        )

        require(
            not any(p.exists() for p in OUTS),
            'Output appeared; refusing overwrite'
        )

        os.replace(t / 'cells.csv', OUT_CELL)
        os.replace(t / 'groups.csv', OUT_GROUP)
        os.replace(t / 'report.json', OUT_REPORT)

    print(
        '=== STAGE 17AH NCA ABSOLUTE-ENDPOINT AUDIT ==='
    )

    print(
        'PREDICTION_LOCK_SHA256_MATCH=YES '
        'SELECTED30_UNCHANGED=YES ALL66_INCLUDED=YES'
    )

    print(
        'TERMINAL_ABSOLUTE_CAPACITY_GROUP_SUMMARY=',
        json.dumps(groupout, ensure_ascii=False)
    )

    print(
        'SELECTED30_NEAR_2500mAh_IDS=',
        json.dumps([
            r['cell_id']
            for r in rows
            if r['selected_preoutcome_30']
            and r['last_within_25mAh_of_2500']
        ])
    )

    print(
        'SELECTED30_FIRST80_CROSSING_LAGS=',
        json.dumps(summary([
            r[
                'cycles_after_first80pct_crossing_if_selected'
            ]
            for r in rows
            if r['selected_preoutcome_30']
        ]))
    )

    print('CELL_AUDIT_CSV=', OUT_CELL)
    print('GROUP_SUMMARY_CSV=', OUT_GROUP)
    print('REPORT_JSON=', OUT_REPORT)

    print(
        'STOP_RULE_CERTIFIED=NO '
        'MODEL_TRAINING=NO '
        'MODEL_INFERENCE=NO '
        'PRISM_UPDATED=NO'
    )

    print('=== END STAGE 17AH ===')


if __name__ == '__main__':
    main()
