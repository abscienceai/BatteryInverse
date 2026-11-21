#!/usr/bin/env python3
"""Stage17AD: lock NCA inputs, audit fade semantics and inspect original recipe."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

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
NCA = R / 'stage17ac_nca_30cell_PROVISIONAL_features.csv'
AC = R / 'stage17ac_feature_lineage_and_provisional_matrix.json'
LOCK = R / 'stage17y_nca_pre_extraction_protocol.json'

FADE_CSV = R / 'stage17ad_fade_formula_comparison_by_family.csv'
RECIPE_CSV = R / 'stage17ad_original_model_recipe_code_context.csv'
REPORT = R / 'stage17ad_preinference_locked_recipe_and_fade.json'

FEATURES = [
    'cap_ratio_10_to_1',
    'cap_ratio_50_to_1',
    'cap_ratio_100_to_1',
    'cap_mean_first_100_norm',
    'cap_std_first_100_norm',
    'cap_slope_first_100_norm',
    'cap_fade_fraction_1_to_100'
]


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(
            lambda: f.read(1048576), b''
        ):
            h.update(chunk)
    return h.hexdigest()


def readcsv(path, required):
    need(path.is_file(), f'Missing input: {path}')

    with path.open(
        newline='',
        encoding='utf-8-sig'
    ) as f:
        reader = csv.DictReader(f)
        names = set(reader.fieldnames or [])

        need(
            set(required) <= names,
            f'{path.name} missing {set(required)-names}'
        )

        return list(reader), names


def num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def csvwrite(path, records, columns):
    with path.open(
        'x',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction='ignore'
        )
        writer.writeheader()
        writer.writerows(records)


def compare_fade(rows):
    forms = {
        'stored_raw_fade_divided_by_Q1':
            lambda q1, q100, raw: raw/q1,

        'one_minus_Q100_divided_by_Q1':
            lambda q1, q100, raw: 1-q100/q1,

        'Q100_divided_by_Q1_minus_one':
            lambda q1, q100, raw: q100/q1-1,

        'absolute_normalized_Q1_Q100_change':
            lambda q1, q100, raw: abs(1-q100/q1),

        'negative_stored_raw_fade_divided_by_Q1':
            lambda q1, q100, raw: -raw/q1,

        'Q1_minus_Q100_absolute':
            lambda q1, q100, raw: q1-q100,

        'Q100_minus_Q1_absolute':
            lambda q1, q100, raw: q100-q1
    }

    results = []

    for family in (
        'BatteryML_main',
        'BatteryLife_selected'
    ):
        subset = [
            row for row in rows
            if row['source_family'] == family
        ]

        expected = (
            318 if family == 'BatteryML_main' else 249
        )

        need(
            len(subset) == expected,
            'Training family roster changed'
        )

        for formula, func in forms.items():
            differences = []

            for row in subset:
                q1 = num(row['cap_cycle_1'])
                q100 = num(row['cap_cycle_100'])
                raw = num(row['cap_fade_1_to_100'])
                actual = num(
                    row['cap_fade_fraction_1_to_100']
                )

                if (
                    None in (q1, q100, raw, actual)
                    or q1 <= 0
                ):
                    continue

                estimate = func(q1, q100, raw)

                differences.append(
                    abs(actual - estimate)
                )

            ordered = sorted(differences)

            results.append({
                'source_family': family,
                'candidate_equation': formula,
                'matched_rows': len(ordered),
                'matching_at_1e_8': sum(
                    x < 1e-8 for x in ordered
                ),
                'median_abs_error':
                    ordered[len(ordered)//2]
                    if ordered else None,
                'max_abs_error':
                    max(ordered)
                    if ordered else None
            })

    return results


def extract_model_recipe(path):
    content = path.read_text(
        encoding='utf-8'
    )

    tree = ast.parse(
        content,
        filename=str(path)
    )

    lines = content.splitlines()

    interesting = {
        'RandomForestRegressor',
        'ExtraTreesRegressor',
        'SimpleImputer',
        'Pipeline',
        'StandardScaler',
        'MinMaxScaler',
        'fit',
        'predict',
        'fit_transform',
        'transform',
        'fillna',
        'clip',
        'median'
    }

    found = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func

        if isinstance(func, ast.Name):
            name = func.id

        elif isinstance(func, ast.Attribute):
            name = func.attr

        else:
            continue

        if name not in interesting:
            continue

        found.append(
            (node.lineno, name)
        )

    records = []
    visited = set()

    for line_no, name in sorted(found):
        if (line_no, name) in visited:
            continue

        visited.add((line_no, name))

        start = max(0, line_no - 7)
        end = min(len(lines), line_no + 6)

        context = '\n'.join(
            f'L{i+1}: {lines[i]}'
            for i in range(start, end)
        )

        records.append({
            'line': line_no,
            'call': name,
            'code_context': context[:2500]
        })

    return records


def main():
    inputs = (
        TRAIN, MODEL, BUILDER,
        NCA, AC, LOCK
    )

    for path in inputs:
        need(
            path.is_file(),
            f'Missing source: {path}'
        )

    for path in (FADE_CSV, RECIPE_CSV, REPORT):
        need(
            not path.exists(),
            f'Output exists; refusing overwrite: {path}'
        )

    previous = json.loads(
        AC.read_text(encoding='utf-8')
    )

    lock = json.loads(
        LOCK.read_text(encoding='utf-8')
    )

    need(
        previous.get('training_cells') == 567
        and previous.get('NCA_candidate_cells') == 30,
        'Stage17AC cohort mismatch'
    )

    need(
        previous.get('no_external_terminal_targets_read') is True
        and previous.get('no_model_fitting') is True
        and previous.get('no_external_predictions') is True,
        'Stage17AC no-outcome status mismatch'
    )

    need(
        previous.get('original_builder_sha256') == sha(BUILDER)
        and previous.get('frozen_model_sha256') == sha(MODEL),
        'Original Stage17AC source changed'
    )

    need(
        lock.get('training_567_sha256') == sha(TRAIN)
        and lock.get('original_stage16a_script_sha256')
            == sha(MODEL),
        'Original model or training lock changed'
    )

    training, _ = readcsv(
        TRAIN,
        [
            'cell_id',
            'source_family',
            'cap_cycle_1',
            'cap_cycle_100',
            'cap_fade_1_to_100',
            'cap_fade_fraction_1_to_100'
        ] + FEATURES
    )

    nca, nca_cols = readcsv(
        NCA,
        ['cell_id', 'feature_status'] + FEATURES
    )

    need(
        len(training) == 567 and len(nca) == 30,
        'Unexpected training/NCA cohort size'
    )

    train_ids = {
        row['cell_id'] for row in training
    }

    nca_ids = {
        row['cell_id'] for row in nca
    }

    need(
        len(train_ids) == 567
        and len(nca_ids) == 30,
        'Duplicate physical cell identifiers'
    )

    need(
        not (train_ids & nca_ids),
        'Exact training/NCA cell-ID collision'
    )

    forbidden = {
        'final_capacity_retention',
        'target',
        'label',
        'terminal_capacity',
        'terminal_retention'
    }

    need(
        not (nca_cols & forbidden),
        'Target column unexpectedly present in NCA input'
    )

    for row in nca:
        need(
            row['feature_status']
                == 'PROVISIONAL_NOT_CERTIFIED',
            'NCA feature status changed'
        )

        need(
            all(num(row[col]) is not None for col in FEATURES),
            'Nonfinite NCA feature values'
        )

    comparisons = compare_fade(training)
    recipe = extract_model_recipe(MODEL)

    need(
        any(
            row['call'] in (
                'RandomForestRegressor',
                'ExtraTreesRegressor'
            )
            for row in recipe
        ),
        'Original estimator source not identified'
    )

    csvwrite(
        FADE_CSV,
        comparisons,
        [
            'source_family',
            'candidate_equation',
            'matched_rows',
            'matching_at_1e_8',
            'median_abs_error',
            'max_abs_error'
        ]
    )

    csvwrite(
        RECIPE_CSV,
        recipe,
        ['line', 'call', 'code_context']
    )

    report = {
        'stage':
            '17AD_PREINFERENCE_RECIPE_AND_FADE_AUDIT',

        'training_n': 567,
        'nca_candidates_n': 30,

        'source_sha256': {
            str(path.relative_to(ROOT)): sha(path)
            for path in inputs
        },

        'fade_equation_comparisons':
            comparisons,

        'original_model_recipe_call_context':
            recipe,

        'nca_input_feature_order':
            FEATURES,

        'nca_features_provisional_not_certified':
            True,

        'label_free_input_matrix_locked':
            True,

        'terminal_targets_read': False,
        'model_fitted': False,
        'external_predictions_produced': False,
        'prism_updated': False
    }

    with REPORT.open(
        'x',
        encoding='utf-8'
    ) as f:
        json.dump(
            report, f,
            ensure_ascii=False,
            indent=2
        )

    print(
        '=== STAGE 17AD INPUT LOCK + MODEL RECIPE ==='
    )

    print(
        'ORIGINAL_TRAIN_CELLS=567 '
        'NCA_PROVISIONAL_CELLS=30'
    )

    print(
        'FADE_FORMULA_COMPARISONS=',
        json.dumps(
            comparisons,
            ensure_ascii=False
        )
    )

    print(
        'ORIGINAL_ESTIMATOR_AND_PREPROCESSING_CALLS=',
        json.dumps([
            {
                'line': row['line'],
                'call': row['call']
            }
            for row in recipe
        ])
    )

    print(
        'ORIGINAL_RECIPE_CODE_CONTEXT_FIRST12=',
        json.dumps(
            recipe[:12],
            ensure_ascii=False
        )
    )

    print('NCA_FEATURES_SHA256=', sha(NCA))
    print('FADE_EQUATIONS_CSV=', FADE_CSV)
    print('EXACT_RECIPE_SOURCE_CONTEXT_CSV=', RECIPE_CSV)
    print('REPORT_JSON=', REPORT)

    print(
        'NCA_TARGET_READ=NO MODEL_FIT=NO '
        'NCA_PREDICTIONS=NO PRISM_UPDATED=NO'
    )

    print('=== END STAGE 17AD ===')


if __name__ == '__main__':
    main()
