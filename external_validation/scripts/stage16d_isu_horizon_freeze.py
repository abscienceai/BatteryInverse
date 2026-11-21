#!/usr/bin/env python3
"""Stage 16D: fixed-prediction horizon decomposition; descriptive only."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P = ROOT / 'external_validation/processed'
R = ROOT / 'external_validation/reports'

PAIR = P / 'stage16c_isu_endpoint_horizon_paired.csv'
C16 = R / 'stage16c_isu_endpoint_horizon_report.json'
C16B = R / 'stage16b_isu_failure_mode_report.json'
OUT = R / 'stage16d_isu_horizon_closure.json'

TRAIN = ROOT / (
    'data/05_model_ready/aging/'
    'expanded_aging_features_with_batterylife.csv'
)

LOCKED = (
    '8bc5296425fbb0735faceb56fe1cf52'
    'c6d0292bec2e361be8870fc1626aa5c7c'
)

MODELS = [
    'median_prior',
    'random_forest',
    'extra_trees'
]


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()

    with path.open('rb') as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b''
        ):
            h.update(chunk)

    return h.hexdigest()


def describe(a):
    a = np.asarray(a, dtype=float)

    need(
        np.isfinite(a).all() and len(a) > 0,
        'Invalid statistic input'
    )

    return {
        'n': int(len(a)),
        'min': float(a.min()),
        'p05': float(np.quantile(a, .05)),
        'median': float(np.median(a)),
        'p95': float(np.quantile(a, .95)),
        'max': float(a.max())
    }


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy(float)
    rb = pd.Series(b).rank().to_numpy(float)

    if np.std(ra) == 0 or np.std(rb) == 0:
        return None

    return float(np.corrcoef(ra, rb)[0, 1])


def group_bootstrap(values, groups, repeats=4000):
    frame = pd.DataFrame({
        'group': groups,
        'value': values
    })

    summary = (
        frame.groupby('group', sort=True)['value']
        .agg(['sum', 'count'])
    )

    sums = summary['sum'].to_numpy(float)
    counts = summary['count'].to_numpy(int)

    rng = np.random.default_rng(20261119)

    draws = rng.integers(
        0,
        len(sums),
        size=(repeats, len(sums))
    )

    boots = (
        sums[draws].sum(axis=1)
        / counts[draws].sum(axis=1)
    )

    interval = np.quantile(
        boots, [.025, .975]
    )

    return {
        'delta_mae_terminal_minus_first':
            float(np.mean(values)),

        'G_group_bootstrap_95pct_interval':
            [float(v) for v in interval],

        'n_G_groups': int(len(sums)),
        'bootstrap_replicates': repeats,

        'note':
            'Descriptive; provisional endpoints'
    }


def main():
    for path in (PAIR, C16, C16B, TRAIN):
        need(path.is_file(), f'Missing input: {path}')

    need(
        not OUT.exists(),
        'Stage16D report exists; no overwrite'
    )

    need(
        sha256(TRAIN) == LOCKED,
        'Frozen training table changed'
    )

    old = json.loads(
        C16.read_text(encoding='utf-8')
    )

    prev = json.loads(
        C16B.read_text(encoding='utf-8')
    )

    need(
        old.get('source_stage16b_report_sha256')
        == sha256(C16B),
        'Stage16B report modified'
    )

    need(
        old.get('model_training') is False
        and prev.get('cells') == 242,
        'Stage history mismatch'
    )

    df = pd.read_csv(
        PAIR,
        dtype={
            'cell_id': str,
            'protocol_group': str
        }
    )

    need(
        len(df) == 242
        and df['cell_id'].notna().all()
        and df['cell_id'].is_unique,
        'Unexpected cell cohort'
    )

    need(
        df['protocol_group'].nunique() == 62,
        'Expected 62 G-groups'
    )

    required = [
        'first_post100_c5_ratio_V7_NOT_FINAL',
        'terminal_c5_ratio_B0_NOT_FINAL',
        'rpt_index_gap_last_minus_first_post100',
        'first_post100_target_hours'
    ] + [
        f'{m}_pred' for m in MODELS
    ]

    need(
        set(required).issubset(df.columns),
        'Missing Stage16C columns'
    )

    for col in required:
        df[col] = pd.to_numeric(
            df[col], errors='coerce'
        )

        need(
            np.isfinite(
                df[col].to_numpy(float)
            ).all(),
            f'Nonfinite values: {col}'
        )

    first = df[
        'first_post100_c5_ratio_V7_NOT_FINAL'
    ].to_numpy(float)

    final = df[
        'terminal_c5_ratio_B0_NOT_FINAL'
    ].to_numpy(float)

    gap = df[
        'rpt_index_gap_last_minus_first_post100'
    ].to_numpy(float)

    drop = first - final

    need(
        np.all(gap >= 0)
        and int((gap > 0).sum()) == 240
        and int((gap == 0).sum()) == 2,
        'Unexpected RPT gap counts'
    )

    need(
        np.all(drop >= -1e-10)
        and int((drop > 1e-10).sum()) == 240,
        'Unexpected capacity trajectory'
    )

    need(
        math.isclose(
            float(np.median(drop)),
            .48252514,
            rel_tol=0,
            abs_tol=.00001
        ),
        'Stage16C gap median changed'
    )

    strata = {}

    subsets = [
        ('same_RPT_index', gap == 0),
        ('gap_1_to_8',
            (gap >= 1) & (gap <= 8)),
        ('gap_9_to_16',
            (gap >= 9) & (gap <= 16)),
        ('gap_17_to_32',
            (gap >= 17) & (gap <= 32)),
        ('gap_33_plus', gap >= 33)
    ]

    for name, mask in subsets:
        if mask.sum() == 0:
            continue

        strata[name] = {
            'n_cells': int(mask.sum()),

            'median_first_minus_terminal_pp':
                float(np.median(drop[mask]) * 100),

            'median_first_ratio':
                float(np.median(first[mask])),

            'median_terminal_ratio':
                float(np.median(final[mask]))
        }

    comparisons = {}

    for model in MODELS:
        prediction = df[
            f'{model}_pred'
        ].to_numpy(float)

        first_error = prediction - first
        final_error = prediction - final

        delta_abs = (
            np.abs(final_error)
            - np.abs(first_error)
        )

        comparisons[model] = {
            'first_RPT': {
                'mae':
                    float(np.mean(np.abs(first_error))),

                'signed_bias':
                    float(np.mean(first_error)),

                'spearman':
                    spearman(first, prediction)
            },

            'terminal_RPT': {
                'mae':
                    float(np.mean(np.abs(final_error))),

                'signed_bias':
                    float(np.mean(final_error)),

                'spearman':
                    spearman(final, prediction)
            },

            'underpredict_first_and_overpredict_terminal_cells':
                int(
                    (
                        (first_error < 0)
                        & (final_error > 0)
                    ).sum()
                ),

            'MAE_horizon_difference':
                group_bootstrap(
                    delta_abs,
                    df['protocol_group'].tolist()
                )
        }

        frozen = old['results'][
            'frozen_prediction_diagnostics'
        ][model]

        prev_first = frozen[
            'against_first_post100_C5_POSTHOC_DIAGNOSTIC'
        ]['mae']

        prev_final = frozen[
            'against_terminal_C5_STAGE16A'
        ]['mae']

        need(
            np.isclose(
                comparisons[model]['first_RPT']['mae'],
                prev_first,
                atol=1e-10
            ),
            f'{model}: first RPT MAE mismatch'
        )

        need(
            np.isclose(
                comparisons[model]['terminal_RPT']['mae'],
                prev_final,
                atol=1e-10
            ),
            f'{model}: terminal MAE mismatch'
        )

    report = {
        'stage': '16D_ISU_DESCRIPTIVE_HORIZON_CLOSURE',

        'n_cells': len(df),

        'n_G_groups':
            int(df['protocol_group'].nunique()),

        'source_stage16c_sha256':
            sha256(PAIR),

        'source_stage16c_report_sha256':
            sha256(C16),

        'training_sha256': LOCKED,

        'rpt_index_gap':
            describe(gap),

        'paired_capacity_drop_percentage_points':
            describe(100 * drop),

        'gap_vs_capacity_drop_spearman':
            spearman(gap, drop),

        'gap_strata': strata,

        'frozen_model_horizon_diagnostics':
            comparisons,

        'cautions': [
            'First post-100 RPT endpoint is post-hoc.',
            'RPT index gap is not elapsed time.',
            'Bootstrap assumes independent G groups.',
            'Cross-dataset target equivalence unconfirmed.',
            'These results do not prove causality.'
        ],

        'model_training': False,
        'predictions_recomputed': False,
        'external_target_approved': False
    }

    R.mkdir(parents=True, exist_ok=True)

    with OUT.open(
        'x', encoding='utf-8'
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        '=== STAGE 16D ISU HORIZON CLOSURE ==='
    )

    print('CELLS=242 G_GROUPS=62')

    print(
        'GAP_VS_CAPACITY_DROP_SPEARMAN=',
        report['gap_vs_capacity_drop_spearman']
    )

    print(
        'GAP_STRATA=',
        json.dumps(strata)
    )

    print(
        'FROZEN_MODEL_COMPARISONS=',
        json.dumps(comparisons)
    )

    print('REPORT_JSON=', OUT)

    print(
        'TRAINING=NO PREDICTIONS_CHANGED=NO '
        'TARGET_FINALIZED=NO'
    )

    print('=== END STAGE 16D ===')


if __name__ == '__main__':
    main()
