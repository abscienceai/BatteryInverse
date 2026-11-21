#!/usr/bin/env python3
"""Stage17AC: original feature algebra and provisional NCA matrix.

No external terminal target, fitting, inference, or model choice.
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

TRAIN = ROOT / (
    'data/05_model_ready/aging/'
    'expanded_aging_features_with_batterylife.csv'
)
MODEL = ROOT / (
    'external_validation/scripts/'
    'stage16a_isu_exploratory_transfer.py'
)
BUILDER = ROOT / (
    'scripts/build_expanded_aging_features_with_batterylife.py'
)

LOCK = R / 'stage17y_nca_pre_extraction_protocol.json'
AB = R / 'stage17ab_first100_feature_contract_report.json'
AC = R / 'stage17aa_nca_66cell_preinference_gate.csv'
ZC = R / 'stage17z_nca_observed_cycle_segments.csv'

OUT = R / 'stage17ac_nca_30cell_PROVISIONAL_features.csv'
IDENT = R / 'stage17ac_original_feature_algebra_by_family.csv'
REPORT = R / 'stage17ac_feature_lineage_and_provisional_matrix.json'

FEATURES = [
    'cap_ratio_10_to_1',
    'cap_ratio_50_to_1',
    'cap_ratio_100_to_1',
    'cap_mean_first_100_norm',
    'cap_std_first_100_norm',
    'cap_slope_first_100_norm',
    'cap_fade_fraction_1_to_100'
]

RAW = {
    'cap_mean_first_100_norm': 'cap_mean_first_100',
    'cap_std_first_100_norm': 'cap_std_first_100',
    'cap_slope_first_100_norm': 'cap_slope_first_100',
    'cap_fade_fraction_1_to_100': 'cap_fade_1_to_100'
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()

    with path.open('rb') as f:
        for block in iter(
            lambda: f.read(1048576), b''
        ):
            h.update(block)

    return h.hexdigest()


def rows(path, columns):
    need(
        path.is_file(),
        'Missing source: ' + str(path)
    )

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []

        need(
            set(columns) <= set(names),
            f'Missing columns in {path.name}: '
            f'{set(columns)-set(names)}'
        )

        return list(reader), names


def n(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def describe(x):
    a = np.asarray(x, dtype=float)

    need(
        len(a) > 0 and np.isfinite(a).all(),
        'Empty/nonfinite distribution'
    )

    return {
        'n': len(a),
        'min': float(a.min()),
        'p05': float(np.quantile(a, .05)),
        'median': float(np.median(a)),
        'p95': float(np.quantile(a, .95)),
        'max': float(a.max())
    }


def csvsave(path, things):
    need(bool(things), 'No output rows')

    with path.open(
        'x',
        encoding='utf-8',
        newline=''
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(things[0])
        )
        writer.writeheader()
        writer.writerows(things)


def main():
    sources = (
        TRAIN, MODEL, BUILDER,
        LOCK, AB, AC, ZC
    )

    for path in sources:
        need(
            path.is_file(),
            'Missing input: ' + str(path)
        )

    for path in (OUT, IDENT, REPORT):
        need(
            not path.exists(),
            'Refusing overwrite: ' + str(path)
        )

    lock = json.loads(
        LOCK.read_text(encoding='utf-8')
    )

    ab = json.loads(
        AB.read_text(encoding='utf-8')
    )

    need(
        lock['training_567_sha256'] == sha(TRAIN)
        and lock['original_stage16a_script_sha256']
            == sha(MODEL),
        'Frozen training/model source changed'
    )

    need(
        ab.get('terminal_targets_read') is False
        and ab.get('inference') is False,
        'Stage17AB was not pre-inference'
    )

    for rel, expected in ab.get(
        'source_sha256', {}
    ).items():
        part = Path(rel)

        need(
            not part.is_absolute()
            and '..' not in part.parts,
            'Unsafe provenance path: ' + rel
        )

        path = (ROOT / part).resolve()

        need(
            path.is_relative_to(ROOT)
            and path.is_file()
            and sha(path) == expected,
            'Changed input since 17AB: ' + rel
        )

    train, tcols = rows(
        TRAIN,
        [
            'source_family', 'cell_id',
            'cap_cycle_1', 'cap_cycle_10',
            'cap_cycle_50', 'cap_cycle_100'
        ] + FEATURES
    )

    cells, _ = rows(
        AC,
        [
            'cell_id', 'program',
            'strict_first100_gate_DIAGNOSTIC'
        ]
    )

    cycles, _ = rows(
        ZC,
        [
            'cell_id', 'instrument_cycle',
            'discharge_max_mAh_DIAGNOSTIC'
        ]
    )

    need(
        len(train) == 567
        and len(cells) == 66,
        'Original training or NCA roster mismatch'
    )

    need(
        len({c['cell_id'] for c in cells}) == 66,
        'Duplicate NCA physical cells'
    )

    need(
        set(r['source_family'] for r in train)
        == {'BatteryML_main', 'BatteryLife_selected'},
        'Unknown training source families'
    )

    algebra = []

    for family in (
        'BatteryML_main',
        'BatteryLife_selected'
    ):
        subset = [
            r for r in train
            if r['source_family'] == family
        ]

        for feat in FEATURES:
            candidates = (
                [('four_anchor_equation', feat)]
                if feat.startswith('cap_ratio_')
                else []
            )

            if feat in RAW:
                candidates.append((
                    'stored_raw_divided_by_cap_cycle_1',
                    RAW[feat]
                ))

            if feat == 'cap_fade_fraction_1_to_100':
                candidates.append((
                    'one_minus_q100_divided_by_q1',
                    None
                ))

            for formula, raw in candidates:
                diffs = []

                for r in subset:
                    base = n(r.get('cap_cycle_1'))
                    actual = n(r.get(feat))

                    if (
                        base is None
                        or base <= 0
                        or actual is None
                    ):
                        continue

                    if formula == 'four_anchor_equation':
                        c = (
                            'cap_cycle_'
                            + feat.split('_')[2]
                        )
                        val = n(r.get(c))
                        est = (
                            val / base
                            if val is not None
                            else None
                        )

                    elif formula == (
                        'stored_raw_divided_by_cap_cycle_1'
                    ):
                        val = n(r.get(raw))
                        est = (
                            val / base
                            if val is not None
                            else None
                        )

                    else:
                        val = n(
                            r.get('cap_cycle_100')
                        )
                        est = (
                            1 - val / base
                            if val is not None
                            else None
                        )

                    if est is not None:
                        diffs.append(
                            abs(est - actual)
                        )

                algebra.append({
                    'source_family': family,
                    'feature': feat,
                    'candidate_formula': formula,
                    'raw_column': raw or '',
                    'checked': len(diffs),
                    'matched_within_1e_8': sum(
                        d < 1e-8 for d in diffs
                    ),
                    'max_abs_error':
                        max(diffs)
                        if diffs else None,
                    'all_source_cells_match':
                        bool(diffs)
                        and len(diffs) == len(subset)
                        and all(
                            d < 1e-8
                            for d in diffs
                        )
                })

    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for row in cycles:
        cy = n(row['instrument_cycle'])

        need(
            cy is not None and int(cy) == cy,
            'Invalid instrument cycle'
        )

        grouped[row['cell_id']][
            int(cy)
        ].append(row)

    need(
        set(grouped)
        == {c['cell_id'] for c in cells},
        'NCA cycle roster mismatch'
    )

    out = []

    for cell in sorted(
        cells,
        key=lambda r: r['cell_id']
    ):
        if (
            cell[
                'strict_first100_gate_DIAGNOSTIC'
            ].strip().lower()
            != 'true'
        ):
            continue

        bycycle = grouped[cell['cell_id']]
        q = []

        for index in range(1, 101):
            need(
                len(bycycle[index]) == 1,
                'Missing/repeated first100 cycle: '
                + cell['cell_id']
                + f', {index}'
            )

            val = n(
                bycycle[index][0][
                    'discharge_max_mAh_DIAGNOSTIC'
                ]
            )

            need(
                val is not None
                and 2500 <= val <= 3500,
                'Invalid first100 capacity: '
                + cell['cell_id']
                + f', {index}'
            )

            q.append(val)

        a = np.asarray(q, dtype=float)
        norm = a / a[0]

        values = {
            'cap_ratio_10_to_1':
                float(norm[9]),
            'cap_ratio_50_to_1':
                float(norm[49]),
            'cap_ratio_100_to_1':
                float(norm[99]),
            'cap_mean_first_100_norm':
                float(norm.mean()),
            'cap_std_first_100_norm':
                float(norm.std(ddof=0)),
            'cap_slope_first_100_norm':
                float(np.polyfit(
                    np.arange(1, 101),
                    norm, 1
                )[0]),
            'cap_fade_fraction_1_to_100':
                float(1 - norm[99])
        }

        out.append({
            'cell_id': cell['cell_id'],
            'program': cell['program'],
            'source':
                'Tongji_KIT_NCA_Zenodo6405084',
            'feature_status':
                'PROVISIONAL_NOT_CERTIFIED',
            'cycle1_capacity_mAh_DIAGNOSTIC':
                float(a[0]),
            'cycle100_capacity_mAh_DIAGNOSTIC':
                float(a[-1]),
            **values
        })

    need(
        len(out) == 30,
        f'Expected 30 strict gate cells, got {len(out)}'
    )

    support = []

    for feature in FEATURES:
        ta = [
            n(row[feature])
            for row in train
        ]

        need(
            all(v is not None for v in ta),
            'Nonfinite training feature: '
            + feature
        )

        ea = [
            row[feature]
            for row in out
        ]

        original = describe(ta)
        external = describe(ea)

        support.append({
            'feature': feature,
            'train': original,
            'NCA_30_provisional': external,
            'NCA_outside_train_minmax': sum(
                x < original['min']
                or x > original['max']
                for x in ea
            ),
            'NCA_outside_train_p05p95': sum(
                x < original['p05']
                or x > original['p95']
                for x in ea
            )
        })

    builder = BUILDER.read_text(
        encoding='utf-8'
    ).splitlines()

    excerpt = [
        {
            'line': i + 1,
            'text': line.strip()[:200]
        }
        for i, line in enumerate(
            builder[:100]
        )
        if (
            'cap_' in line
            or 'eps' in line
            or 'def ' in line
            or 'base' in line
        )
    ]

    report = {
        'stage':
            '17AC_FEATURE_ORIGIN_AND_NCA_30_PROVISIONAL_MATRIX',
        'training_cells': 567,
        'NCA_screened_cells': 66,
        'NCA_candidate_cells': len(out),
        'original_builder_sha256':
            sha(BUILDER),
        'stage17ab_sha256': sha(AB),
        'frozen_model_sha256':
            sha(MODEL),
        'original_source_family_feature_algebra':
            algebra,
        'original_builder_relevant_lines':
            excerpt,
        'NCA_provisional_feature_support':
            support,
        'provisional_mean_std_slope_from_full100':
            True,
        'cross_family_semantic_feature_equivalence_certified':
            False,
        'source_318_upstream_csv_missing':
            True,
        'no_external_terminal_targets_read':
            True,
        'no_model_fitting': True,
        'no_external_predictions': True,
        'prism_updated': False
    }

    csvsave(OUT, out)
    csvsave(IDENT, algebra)

    with REPORT.open(
        'x',
        encoding='utf-8'
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        '=== STAGE 17AC SOURCE FEATURE ALGEBRA + NCA PROVISIONAL ==='
    )
    print(
        'NCA_PROVISIONAL_30_CELL_FEATURE_ROWS=',
        len(out)
    )
    print(
        'TRAINING_ALGEBRA_BY_FAMILY=',
        json.dumps(
            algebra,
            ensure_ascii=False
        )
    )
    print(
        'ORIGINAL_BUILDER_RELEVANT_LINES=',
        json.dumps(
            excerpt,
            ensure_ascii=False
        )
    )
    print(
        'NCA_30_FEATURE_DISTRIBUTION_SUPPORT=',
        json.dumps(
            support,
            ensure_ascii=False
        )
    )
    print('PROVISIONAL_FEATURE_MATRIX=', OUT)
    print('ALGEBRA_CSV=', IDENT)
    print('REPORT_JSON=', REPORT)
    print(
        'TRAINING=NO EXTERNAL_TARGET_READ=NO '
        'PREDICTIONS=NO PRISM_UPDATED=NO'
    )
    print('=== END STAGE 17AC ===')


if __name__ == '__main__':
    main()
