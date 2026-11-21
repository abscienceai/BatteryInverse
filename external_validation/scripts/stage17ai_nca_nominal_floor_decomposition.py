#!/usr/bin/env python3
"""Stage17AI: post-outcome mechanistic explanation, NEVER a new predictor.

Use original frozen predictions and Stage17AF/AH outputs unchanged.
The paper's 2500 mAh (~71% of 3500 mAh nominal) is an
analysis/aging-window reference, not proof of each CSV's stopping command.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
R = ROOT / 'external_validation/reports'
AH = R / 'stage17ah_nca_absolute_endpoint_and_censoring_report.json'
AH_CELLS = R / 'stage17ah_nca_66cell_absolute_last_capacity_audit.csv'
AF = R / 'stage17af_nca_unsealed_outcome_evidence.json'
AF_CELLS = R / 'stage17af_nca_30cell_last_cycle_diagnostic_and_locked_predictions.csv'
AE = R / 'stage17ae_nca_preoutcome_model_lock_report.json'
PRED = R / 'stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv'
OUT_CELLS = R / 'stage17ai_nca_nominal_floor_mechanistic_decomposition_30.csv'
OUT_GROUPS = R / 'stage17ai_nca_nominal_floor_diagnostic_groups.csv'
OUT_REPORT = R / 'stage17ai_nca_2500mAh_endpoint_interpretation.json'
OUTPUTS = (OUT_CELLS, OUT_GROUPS, OUT_REPORT)
PRED_SHA = 'cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757'
FLOOR_MAH = 2500.0
NOMINAL_MAH = 3500.0


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for buf in iter(lambda: stream.read(1048576), b''):
            h.update(buf)
    return h.hexdigest()


def readcsv(path, required):
    need(path.is_file(), f'Missing input: {path}')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        need(
            set(required) <= set(reader.fieldnames or []),
            f'Missing columns in {path}: '
            f'{set(required) - set(reader.fieldnames or [])}'
        )
        return list(reader)


def idx(rows, description):
    d = {}
    for r in rows:
        cid = r['cell_id'].strip()
        need(
            bool(cid) and cid not in d,
            f'Duplicate/blank cell ID in {description}'
        )
        d[cid] = r
    return d


def n(x, label):
    try:
        val = float(x)
    except (ValueError, TypeError, OverflowError) as ex:
        raise RuntimeError(
            f'Invalid {label}: {x!r}'
        ) from ex

    need(math.isfinite(val), f'Non-finite {label}')
    return val


def true(x):
    return str(x).strip().lower() == 'true'


def dist(seq):
    a = np.asarray(seq, dtype=float)
    need(
        len(a) and np.isfinite(a).all(),
        'Bad distribution'
    )
    return {
        'n': len(a),
        'min': float(a.min()),
        'median': float(np.median(a)),
        'mean': float(a.mean()),
        'p95': float(np.quantile(a, .95)),
        'max': float(a.max())
    }


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    need(
        len(y) == len(p)
        and len(y) >= 1
        and np.isfinite(y).all()
        and np.isfinite(p).all(),
        'Bad score input'
    )

    delta = p - y
    den = np.sum((y - y.mean()) ** 2)

    return {
        'MAE': float(np.mean(np.abs(delta))),
        'RMSE': float(np.sqrt(np.mean(delta**2))),
        'signed_bias': float(np.mean(delta)),
        'R2_descriptive': (
            float(1 - np.sum(delta**2) / den)
            if den > 1e-22 else None
        )
    }


def writecsv(path, rows):
    need(rows, f'No rows for {path}')

    with path.open(
        'x', newline='', encoding='utf-8'
    ) as stream:
        w = csv.DictWriter(
            stream, fieldnames=list(rows[0])
        )
        w.writeheader()
        w.writerows(rows)


def main():
    sources = (
        AH, AH_CELLS, AF,
        AF_CELLS, AE, PRED
    )

    for p in sources:
        need(p.is_file(), f'Missing input: {p}')

    for p in OUTPUTS:
        need(
            not p.exists(),
            f'Refusing overwrite: {p}'
        )

    ah = json.loads(
        AH.read_text(encoding='utf-8')
    )
    af = json.loads(
        AF.read_text(encoding='utf-8')
    )
    ae = json.loads(
        AE.read_text(encoding='utf-8')
    )

    need(
        ah.get('original_66_archive_count') == 66
        and ah.get('selected_30_count') == 30
        and ah.get(
            'preoutcome_locked_predictions_unchanged'
        ) is True,
        '17AH audit mismatch'
    )

    need(
        af.get('NCA_locked_test_cells') == 30
        and af.get('NCA_positive_last_cycle_cells') == 30,
        '17AF cohort mismatch'
    )

    need(
        ae.get('training_cells') == 567
        and ae.get('NCA_terminal_labels_accessed') is False
        and ae.get(
            'NCA_predictions_locked_before_terminal_outcome_review'
        ) is True,
        '17AE pre-outcome mismatch'
    )

    need(
        sha(PRED) == PRED_SHA
        == ah.get('prediction_sha256')
        == af.get('preoutcome_predictions_sha256'),
        'Frozen prediction hash changed'
    )

    for rel, expected in ah.get(
        'source_sha256', {}
    ).items():
        rp = Path(rel)

        need(
            not rp.is_absolute()
            and '..' not in rp.parts,
            f'Unsafe evidence path {rel}'
        )

        path = (ROOT / rp).resolve()

        need(
            path.is_relative_to(ROOT)
            and path.is_file()
            and sha(path) == expected,
            f'Changed Stage17AH input: {rel}'
        )

    ah_rows = readcsv(
        AH_CELLS,
        [
            'cell_id', 'program',
            'selected_preoutcome_30',
            'last_segment_discharge_mAh',
            'selected_cycle1_mAh',
            'selected_last_ratio',
            'last_instrument_cycle'
        ]
    )

    af_rows = readcsv(
        AF_CELLS,
        [
            'cell_id', 'program',
            'first_cycle_capacity_mAh',
            'last_instrument_cycle',
            'last_cycle_capacity_mAh',
            'measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL',
            'training_median_prior',
            'random_forest_positive_loss',
            'extra_trees_positive_loss',
            'random_forest_negative_signed',
            'extra_trees_negative_signed'
        ]
    )

    pred_rows = readcsv(
        PRED,
        [
            'cell_id', 'program',
            'training_median_prior_prediction'
        ]
    )

    need(
        len(ah_rows) == 66
        and len(af_rows) == 30
        and len(pred_rows) == 30,
        'Unexpected cohort sizes'
    )

    ah_i = idx(ah_rows, '17AH')
    af_i = idx(af_rows, '17AF')
    pred_i = idx(pred_rows, '17AE')

    selected = {
        cid for cid, r in ah_i.items()
        if true(r['selected_preoutcome_30'])
    }

    need(
        selected == set(af_i) == set(pred_i)
        and len(selected) == 30,
        'Locked 30-cell roster differs'
    )

    allq = [
        n(r['last_segment_discharge_mAh'], 'Qlast')
        for r in ah_rows
    ]

    need(
        all(q >= FLOOR_MAH - 1e-8 for q in allq),
        '2500 mAh lower-edge pattern not universal; '
        'inspect and report rather than force'
    )

    out = []

    for cid in sorted(selected):
        a = ah_i[cid]
        b = af_i[cid]
        p = pred_i[cid]

        need(
            a['program'] == b['program'] == p['program'],
            f'Program mismatch: {cid}'
        )

        q1 = n(
            a['selected_cycle1_mAh'], 'Q1'
        )
        last = n(
            a['last_segment_discharge_mAh'], 'Qlast'
        )
        y = n(
            a['selected_last_ratio'], 'last ratio'
        )

        need(
            q1 > 0
            and abs(last / q1 - y) < 1e-10,
            f'Last ratio mismatch: {cid}'
        )

        need(
            abs(
                q1 - n(
                    b['first_cycle_capacity_mAh'],
                    '17AF Q1'
                )
            ) < 1e-8
            and abs(
                last - n(
                    b['last_cycle_capacity_mAh'],
                    '17AF Qlast'
                )
            ) < 1e-8
            and abs(
                y - n(
                    b[
                        'measured_last_cycle_ratio_DIAGNOSTIC_NOT_EOL'
                    ],
                    '17AF y'
                )
            ) < 1e-10,
            f'17AF identity mismatch: {cid}'
        )

        need(
            abs(
                n(
                    b['training_median_prior'],
                    '17AF prior'
                )
                - n(
                    p['training_median_prior_prediction'],
                    '17AE prior'
                )
            ) < 1e-12,
            f'Prior changed: {cid}'
        )

        floor_proxy = FLOOR_MAH / q1
        excess = (last - FLOOR_MAH) / q1

        need(
            abs(y - floor_proxy - excess) < 1e-10,
            f'Mechanistic identity fails: {cid}'
        )

        rec = {
            'cell_id': cid,
            'program': a['program'],
            'last_instrument_cycle':
                int(n(a['last_instrument_cycle'],
                      'last cycle')),
            'first_cycle_Q1_mAh': q1,
            'last_Qmax_mAh': last,
            'last_excess_above_2500mAh':
                last - FLOOR_MAH,
            'observed_last_ratio_Qlast_over_Q1_NOT_EOL':
                y,
            'published_window_lower_edge_over_Q1_DIAGNOSTIC_NOT_PREDICTION':
                floor_proxy,
            'residual_excess_over_Q1': excess,
            'training_median_prior':
                n(b['training_median_prior'], 'prior'),
            'RF_positive_loss_frozen':
                n(b['random_forest_positive_loss'],
                  'RF+'),
            'ET_positive_loss_frozen':
                n(b['extra_trees_positive_loss'],
                  'ET+'),
            'RF_negative_signed_frozen':
                n(b['random_forest_negative_signed'],
                  'RF-'),
            'ET_negative_signed_frozen':
                n(b['extra_trees_negative_signed'],
                  'ET-'),
            'within25mAh_lower_window_edge':
                0 <= last - FLOOR_MAH <= 25,
            'reference_floor_2500mAh_is_postoutcome_interpretation':
                True,
            'actual_stopping_command_certified':
                False
        }

        out.append(rec)

    cohorts = [('ALL30_LOCKED', out)]

    cohorts.extend(
        (
            f'PROGRAM_{pr}',
            [
                r for r in out
                if r['program'] == pr
            ]
        )
        for pr in sorted({
            r['program'] for r in out
        })
    )

    cohorts += [
        (
            'WITHIN_25mAh_POSTOUTCOME',
            [
                r for r in out
                if r['within25mAh_lower_window_edge']
            ]
        ),
        (
            'ABOVE_25mAh_POSTOUTCOME',
            [
                r for r in out
                if not r['within25mAh_lower_window_edge']
            ]
        )
    ]

    groups = []

    for name, rows in cohorts:
        if not rows:
            continue

        y = [
            r['observed_last_ratio_Qlast_over_Q1_NOT_EOL']
            for r in rows
        ]

        def scores(k):
            return metrics(
                y, [r[k] for r in rows]
            )

        group = {
            'cohort': name,
            'physical_cells': len(rows),
            'median_Q1_mAh':
                float(np.median([
                    r['first_cycle_Q1_mAh']
                    for r in rows
                ])),
            'median_last_Qmax_mAh':
                float(np.median([
                    r['last_Qmax_mAh']
                    for r in rows
                ])),
            'median_last_ratio':
                float(np.median(y)),
            'within25mAh_count':
                sum(
                    r['within25mAh_lower_window_edge']
                    for r in rows
                ),
            'published_2500mAh_window_floor_ratio_median':
                float(np.median([
                    r[
                        'published_window_lower_edge_over_Q1_DIAGNOSTIC_NOT_PREDICTION'
                    ]
                    for r in rows
                ])),
            'published_window_floor_proxy_DESCRIPTIVE_ONLY_MAE':
                scores(
                    'published_window_lower_edge_over_Q1_DIAGNOSTIC_NOT_PREDICTION'
                )['MAE'],
            'published_window_floor_proxy_DESCRIPTIVE_ONLY_RMSE':
                scores(
                    'published_window_lower_edge_over_Q1_DIAGNOSTIC_NOT_PREDICTION'
                )['RMSE'],
            'prior_MAE':
                scores('training_median_prior')['MAE'],
            'RF_positive_loss_MAE':
                scores('RF_positive_loss_frozen')['MAE'],
            'ET_positive_loss_MAE':
                scores('ET_positive_loss_frozen')['MAE'],
            'RF_negative_signed_MAE':
                scores('RF_negative_signed_frozen')['MAE'],
            'ET_negative_signed_MAE':
                scores('ET_negative_signed_frozen')['MAE']
        }

        groups.append(group)

    report = {
        'stage':
            '17AI_POSTOUTCOME_NCA_NOMINAL_FLOOR_MECHANISTIC_DECOMPOSITION',
        'source_publication':
            'Zhu et al., Nature Communications (2022) 13:2261, doi:10.1038/s41467-022-29837-w',
        'source_dataset_doi':
            '10.5281/zenodo.6405084',
        'source_nominal_NCA_capacity_mAh':
            NOMINAL_MAH,
        'source_reported_lower_aging_window_mAh':
            FLOOR_MAH,
        'source_reported_lower_aging_window_fraction':
            FLOOR_MAH / NOMINAL_MAH,
        'source_author_cycle_extraction_uses_2500_to_3500mAh_filter':
            True,
        'exact_per_cell_experimental_stopping_command_verified':
            False,
        'raw_archive_last_Qmax_all_66_at_or_above_2500':
            True,
        'raw_archive_last_Qmax_distribution':
            dist(allq),
        'raw_archive_within_25mAh_above_2500_count':
            sum(
                0 <= q - FLOOR_MAH <= 25
                for q in allq
            ),
        'locked_30_selected_count': 30,
        'exact_algebraic_identity':
            'Qlast/Q1 = 2500/Q1 + (Qlast-2500)/Q1',
        'postoutcome_explanatory_proxy_NOT_a_preoutcome_baseline':
            '2500/Q1',
        'methodological_warning':
            'The 2500 mAh reference is interpreted '
            'after outcomes and from the published '
            'aging window. It is not a prospectively '
            'evaluated predictor and must not be '
            'ranked as a new forecasting model. '
            'Neither 2500 mAh crossing times nor '
            'EOL are certified.',
        'source_sha256': {
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in sources
        },
        'group_diagnostics': groups,
        'model_refitted': False,
        'frozen_predictions_changed': False,
        'prism_updated': False
    }

    writecsv(OUT_CELLS, out)
    writecsv(OUT_GROUPS, groups)

    with OUT_REPORT.open(
        'x', encoding='utf-8'
    ) as stream:
        json.dump(
            report,
            stream,
            ensure_ascii=False,
            indent=2
        )

    print(
        '=== STAGE 17AI POSTOUTCOME NCA FLOOR MECHANISM ==='
    )
    print(
        'FROZEN_PREDICTION_SHA256_VERIFIED=YES '
        'NCA_30_IDENTITY=YES ALL66_ORIGINAL=YES'
    )
    print(
        'AUTHOR_NOMINAL_MAH=3500 '
        'AUTHOR_AGING_WINDOW_LOWER_EDGE_MAH=2500'
    )
    print(
        'ALL66_LAST_CAPACITY_DISTRIBUTION=',
        json.dumps(dist(allq))
    )
    print(
        'ALL66_WITHIN25mAh_ABOVE_2500=',
        report[
            'raw_archive_within_25mAh_above_2500_count'
        ]
    )
    print(
        'MECHANICAL_LOWER_EDGE_DECOMPOSITION=',
        json.dumps(groups, ensure_ascii=False)
    )
    print(
        'POSTOUTCOME_FLOOR_PROXY_IS_PREDICTIVE_MODEL=NO'
    )
    print(
        'EXACT_STOP_RULE_CERTIFIED=NO '
        'END_OF_LIFE_CERTIFIED=NO'
    )
    print('CELL_CSV=', OUT_CELLS)
    print('GROUP_CSV=', OUT_GROUPS)
    print('REPORT_JSON=', OUT_REPORT)
    print(
        'NEW_TRAINING=NO '
        'NEW_INFERENCE=NO PRISM_UPDATED=NO'
    )
    print('=== END STAGE 17AI ===')


if __name__ == '__main__':
    main()
