#!/usr/bin/env python3
"""Stage 16B: read-only failure-mode audit of frozen Stage 16A predictions.
No fitting, model selection, recalibration or external label approval.
"""
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P = ROOT / 'external_validation/processed'
R = ROOT / 'external_validation/reports'
TRAIN = ROOT / 'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'
FEATURES_FILE = P / 'stage15c6_isu_242_capacity15_provisional.csv'
PRED = P / 'stage16a_isu_exploratory_predictions.csv'
PREV = R / 'stage16a_isu_exploratory_report.json'
CELL_OUT = P / 'stage16b_isu_cellwise_error_diagnostics.csv'
REPORT_OUT = R / 'stage16b_isu_failure_mode_report.json'
LOCKED_SHA = '8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c'

FEATURES = [
    'cap_ratio_10_to_1',
    'cap_ratio_50_to_1',
    'cap_ratio_100_to_1',
    'cap_mean_first_100_norm',
    'cap_std_first_100_norm',
    'cap_slope_first_100_norm',
    'cap_fade_fraction_1_to_100'
]

PRED_COLS = {
    'median_prior': 'training_median_prior_prediction',
    'random_forest': 'random_forest_prediction',
    'extra_trees': 'extra_trees_prediction'
}

Y_COLS = {
    'C5': 'c5_reference_candidate_NOT_FINAL',
    'C2': 'c2_reference_candidate_NOT_FINAL'
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def indexed(path, fields):
    frame = pd.read_csv(path)
    require(
        set(fields + ['cell_id']).issubset(frame.columns),
        f'Missing columns in {path}'
    )
    keys = frame['cell_id'].astype(str).str.strip()
    require(
        keys.ne('').all() and keys.is_unique,
        f'Duplicate/empty cells in {path}'
    )
    frame['cell_id'] = keys
    return frame.set_index('cell_id')


def metrics(y, pred):
    n = len(y)
    residual = pred - y
    bias = float(np.mean(residual))
    centered = residual - bias

    def corr(a, b):
        if (
            len(a) < 3
            or np.std(a) < 1e-12
            or np.std(b) < 1e-12
        ):
            return None
        return float(np.corrcoef(a, b)[0, 1])

    rank_y = pd.Series(y).rank(
        method='average'
    ).to_numpy(float)

    rank_pred = pd.Series(pred).rank(
        method='average'
    ).to_numpy(float)

    return {
        'n_cells': n,
        'observed_mean': float(np.mean(y)),
        'observed_median': float(np.median(y)),
        'predicted_mean': float(np.mean(pred)),
        'predicted_median': float(np.median(pred)),
        'mae': float(np.mean(np.abs(residual))),
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'signed_bias_pred_minus_observed': bias,
        'centered_residual_rmse_DESCRIPTIVE_ONLY':
            float(np.sqrt(np.mean(centered ** 2))),
        'overprediction_count': int(np.sum(residual > 0)),
        'underprediction_count': int(np.sum(residual < 0)),
        'pearson': corr(y, pred),
        'spearman': corr(rank_y, rank_pred)
    }


def clustered_bootstrap(mae_diff_cell, clusters, B=3000):
    groups = sorted(set(clusters))
    sums = []
    sizes = []

    for group in groups:
        values = np.asarray([
            value
            for value, cell_group in zip(mae_diff_cell, clusters)
            if cell_group == group
        ], dtype=float)

        sums.append(float(values.sum()))
        sizes.append(len(values))

    sums = np.asarray(sums)
    sizes = np.asarray(sizes)

    rng = np.random.default_rng(20261119)
    picks = rng.integers(
        0, len(groups),
        size=(B, len(groups))
    )

    diffs = (
        sums[picks].sum(axis=1)
        / sizes[picks].sum(axis=1)
    )

    lo, hi = np.quantile(diffs, [0.025, 0.975])

    return {
        'point_delta_mae_model_minus_prior':
            float(np.mean(mae_diff_cell)),
        'group_bootstrap_95pct_interval':
            [float(lo), float(hi)],
        'groups': len(groups),
        'resamples': B,
        'method':
            'Descriptive G-group cluster bootstrap; no model tuning'
    }


def main():
    for path in (TRAIN, FEATURES_FILE, PRED, PREV):
        require(
            path.is_file(),
            f'Missing required input: {path}'
        )

    require(
        sha(TRAIN) == LOCKED_SHA,
        'Frozen 567-cell training table SHA256 changed'
    )

    for path in (CELL_OUT, REPORT_OUT):
        require(
            not path.exists(),
            f'Refusing to overwrite: {path}'
        )

    previous = json.loads(
        PREV.read_text(encoding='utf-8')
    )

    require(
        previous.get('training_cells') == 567
        and previous.get('ISU_cells') == 242,
        'Stage16A report inventory mismatch'
    )

    require(
        previous.get('features') == FEATURES,
        'Stage16A feature list differs'
    )

    require(
        previous.get('training_sha256') == LOCKED_SHA,
        'Stage16A training SHA mismatch'
    )

    require(
        previous.get('ISU_features_sha256')
        == sha(FEATURES_FILE),
        'Stage16A external-feature file changed'
    )

    train = pd.read_csv(TRAIN)

    require(
        len(train) == 567,
        'Training row count mismatch'
    )

    xtrain = (
        train[FEATURES]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
    )

    witness = indexed(FEATURES_FILE, FEATURES)

    pred = indexed(
        PRED,
        list(Y_COLS.values()) + list(PRED_COLS.values())
    )

    require(
        len(pred) == 242
        and len(witness) == 242
        and set(pred.index) == set(witness.index),
        'Prediction/feature cell identities differ'
    )

    witness = witness.loc[pred.index]

    xtest = witness[FEATURES].apply(
        pd.to_numeric, errors='coerce'
    )

    require(
        np.isfinite(xtest.to_numpy(float)).all(),
        'External features not finite'
    )

    ytrain = pd.to_numeric(
        train['final_capacity_retention'],
        errors='coerce'
    ).to_numpy(float)

    require(
        np.isfinite(ytrain).all(),
        'Invalid original training targets'
    )

    prior_median = float(np.median(ytrain))

    for col in PRED_COLS.values():
        require(
            np.isfinite(pred[col].to_numpy(float)).all(),
            f'Invalid prediction: {col}'
        )

    require(
        np.allclose(
            pred[PRED_COLS['median_prior']].to_numpy(float),
            prior_median,
            atol=1e-12
        ),
        'Frozen training median mismatch'
    )

    for col in Y_COLS.values():
        require(
            np.isfinite(pred[col].to_numpy(float)).all(),
            f'Invalid target candidate: {col}'
        )

    ood_columns = []

    for col in FEATURES:
        original = xtrain[col].dropna().to_numpy(float)

        require(
            len(original) > 0,
            'Empty training feature: ' + col
        )

        external = xtest[col].to_numpy(float)

        ood_columns.append(
            (external < original.min())
            | (external > original.max())
        )

    ood_matrix = np.column_stack(ood_columns)
    ood_any = ood_matrix.any(axis=1)

    require(
        int(ood_any.sum())
        == previous.get(
            'ISU_outside_training_full_feature_range_any'
        ),
        'OOD count differs from Stage16A'
    )

    clusters = []

    for cell in pred.index:
        match = re.fullmatch(
            r'(G\d+)C\d+',
            cell,
            re.IGNORECASE
        )

        clusters.append(
            match.group(1).upper()
            if match
            else 'SINGLE_' + cell
        )

    output = pd.DataFrame({
        'cell_id': pred.index,
        'protocol_group': clusters,
        'outside_train_minmax_any': ood_any,
        'num_features_outside_train_minmax':
            ood_matrix.sum(axis=1).astype(int),
        'outside_train_features': [
            ','.join(
                col for col, flag in zip(FEATURES, row)
                if flag
            )
            for row in ood_matrix
        ]
    })

    for name, col in PRED_COLS.items():
        output[name + '_prediction'] = (
            pred[col].to_numpy(float)
        )

    for label, col in Y_COLS.items():
        observed = pred[col].to_numpy(float)

        output[label + '_candidate_NOT_FINAL'] = observed

        for name, pred_col in PRED_COLS.items():
            residual = (
                pred[pred_col].to_numpy(float) - observed
            )

            output[
                label + '_' + name + '_signed_error'
            ] = residual

            output[
                label + '_' + name + '_abs_error'
            ] = np.abs(residual)

    score_table = {
        (rec['target'], rec['model']): rec
        for rec in previous['scores']
    }

    diagnoses = {}
    intervals = {}

    for label, col in Y_COLS.items():
        observed = pred[col].to_numpy(float)

        diagnoses[label] = {}

        original_label = (
            'C5_PRIMARY_CANDIDATE'
            if label == 'C5'
            else 'C2_SENSITIVITY'
        )

        cohorts = [
            (
                'all_242',
                np.ones(len(observed), dtype=bool)
            ),
            ('in_training_minmax_175', ~ood_any),
            ('outside_training_minmax_67', ood_any)
        ]

        for cohort_name, mask in cohorts:
            diagnoses[label][cohort_name] = {}

            for name, pred_col in PRED_COLS.items():
                predicted = (
                    pred[pred_col].to_numpy(float)[mask]
                )

                result = metrics(
                    observed[mask],
                    predicted
                )

                diagnoses[label][cohort_name][name] = result

                if cohort_name == 'all_242':
                    model_name = (
                        'training_median_prior'
                        if name == 'median_prior'
                        else name
                    )

                    original = score_table[
                        (original_label, model_name)
                    ]

                    comparisons = [
                        ('mae', 'mae'),
                        ('rmse', 'rmse'),
                        (
                            'signed_bias_pred_minus_observed',
                            'signed_bias_pred_minus_true'
                        )
                    ]

                    for new_key, original_key in comparisons:
                        require(
                            math.isclose(
                                result[new_key],
                                original[original_key],
                                rel_tol=1e-9,
                                abs_tol=1e-10
                            ),
                            f'Stage16A score mismatch: '
                            f'{label}/{name}/{new_key}'
                        )

        prior = pred[
            PRED_COLS['median_prior']
        ].to_numpy(float)

        intervals[label] = {}

        for name in ('random_forest', 'extra_trees'):
            model_pred = pred[
                PRED_COLS[name]
            ].to_numpy(float)

            paired_difference = (
                np.abs(model_pred - observed)
                - np.abs(prior - observed)
            )

            intervals[label][name] = clustered_bootstrap(
                paired_difference,
                clusters
            )

    report = {
        'experiment':
            'STAGE16B_DESCRIPTIVE_FAILURE_MODE_AUDIT',
        'source_stage16a_report_sha256': sha(PREV),
        'source_stage16a_predictions_sha256': sha(PRED),
        'source_training_sha256': LOCKED_SHA,
        'cells': len(pred),
        'protocol_group_count': len(set(clusters)),
        'ISU_cells_outside_training_full_feature_range_any':
            int(ood_any.sum()),
        'ISU_cells_inside_training_full_feature_range_all':
            int((~ood_any).sum()),
        'statistics_by_candidate_target_and_subgroup':
            diagnoses,
        'paired_group_bootstrap_MAE_difference':
            intervals,
        'interpretation_limitations': [
            'C5/C2 remain unfinalized candidate endpoints.',
            'OOD subgroups are post-hoc descriptive.',
            'Do not remove OOD cells to inflate performance.',
            'Centered residual RMSE is not a recalibrated score.',
            'G-prefix bootstrap assumes independent groups.',
            'Capacity units, DoD, and horizon not fully harmonized.'
        ],
        'external_labels_used_for_training_or_tuning':
            False,
        'predictions_recomputed': False,
        'model_training': False,
        'source_files_modified': False
    }

    R.mkdir(parents=True, exist_ok=True)
    P.mkdir(parents=True, exist_ok=True)

    with CELL_OUT.open(
        'x', newline='', encoding='utf-8'
    ) as handle:
        output.to_csv(handle, index=False)

    with REPORT_OUT.open(
        'x', encoding='utf-8'
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False
        )

    print('=== STAGE 16B FAILURE MODE DIAGNOSTICS ===')
    print(
        'CELLS=', len(pred),
        'PROTOCOL_GROUPS=', len(set(clusters))
    )

    print(
        'OUT_OF_TRAIN_FULL_RANGE=',
        int(ood_any.sum()),
        'IN_RANGE=',
        int((~ood_any).sum())
    )

    print(
        'FULL_METRICS=',
        json.dumps({
            k: v['all_242']
            for k, v in diagnoses.items()
        })
    )

    print(
        'OOD_SPLIT_METRICS=',
        json.dumps({
            k: {
                'in_range':
                    v['in_training_minmax_175'],
                'outside':
                    v['outside_training_minmax_67']
            }
            for k, v in diagnoses.items()
        })
    )

    print(
        'PAIRED_BOOTSTRAP_MAE_DIFF=',
        json.dumps(intervals)
    )

    print('CELLWISE_CSV=', CELL_OUT)
    print('REPORT_JSON=', REPORT_OUT)
    print('EXTERNAL_LABELS_USED_FOR_TRAINING=NO')
    print('MODEL_TRAINING=NO')
    print('FINAL_EXTERNAL_VALIDATION=NO')
    print('=== END STAGE 16B ===')


if __name__ == '__main__':
    main()
