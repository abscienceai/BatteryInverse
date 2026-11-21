#!/usr/bin/env python3
"""Stage 16C: fixed-prediction ISU endpoint-horizon audit (post-hoc)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P = ROOT / 'external_validation/processed'
R = ROOT / 'external_validation/reports'

V7 = P / 'stage15a_v7_candidate_cohorts.csv'
B0 = P / 'stage15b0_isu_terminal_endpoint_audit.csv'
V6 = P / 'stage15a_v6_prospective_rpt_audit.csv'
PRED = P / 'stage16b_isu_cellwise_error_diagnostics.csv'
S16B = R / 'stage16b_isu_failure_mode_report.json'
TRAIN = ROOT / 'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'

OUT = P / 'stage16c_isu_endpoint_horizon_paired.csv'
SUMMARY = R / 'stage16c_isu_endpoint_horizon_report.json'

LOCKED_SHA = (
    '8bc5296425fbb0735faceb56fe1cf52'
    'c6d0292bec2e361be8870fc1626aa5c7c'
)


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b''
        ):
            h.update(block)
    return h.hexdigest()


def load(path, cols):
    check(path.is_file(), f'Input missing: {path}')

    df = pd.read_csv(
        path, dtype={'cell_id': str}
    )

    check(
        set(cols + ['cell_id']).issubset(df.columns),
        f'Missing columns: {path}'
    )

    df['cell_id'] = df['cell_id'].str.strip()

    check(
        df.cell_id.notna().all()
        and df.cell_id.ne('').all()
        and df.cell_id.is_unique,
        f'Duplicate or empty cell IDs: {path}'
    )

    return df.set_index('cell_id')


def describe(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if not len(x):
        return {'n': 0}

    return {
        'n': int(len(x)),
        'min': float(x.min()),
        'p05': float(np.quantile(x, 0.05)),
        'median': float(np.median(x)),
        'p95': float(np.quantile(x, 0.95)),
        'max': float(x.max()),
    }


def correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        len(a) < 3
        or np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return None

    return float(np.corrcoef(a, b)[0, 1])


def scores(observed, predicted):
    residual = predicted - observed

    rank_y = pd.Series(
        observed
    ).rank().to_numpy()

    rank_p = pd.Series(
        predicted
    ).rank().to_numpy()

    return {
        'mae': float(
            np.mean(np.abs(residual))
        ),
        'rmse': float(
            np.sqrt(np.mean(residual ** 2))
        ),
        'signed_bias': float(
            np.mean(residual)
        ),
        'spearman': correlation(
            rank_y, rank_p
        ),
    }


def main():
    for path in (
        V7, B0, V6, PRED, S16B, TRAIN
    ):
        check(
            path.is_file(),
            f'Missing: {path}'
        )

    check(
        sha(TRAIN) == LOCKED_SHA,
        'Frozen benchmark checksum changed; STOP'
    )

    check(
        not OUT.exists()
        and not SUMMARY.exists(),
        'Stage16C output already exists; STOP'
    )

    previous = json.loads(
        S16B.read_text(encoding='utf-8')
    )

    check(
        previous.get('cells') == 242
        and not previous.get('model_training'),
        'Stage16B report inconsistent'
    )

    check(
        previous.get('source_training_sha256')
        == LOCKED_SHA,
        'Stage16B benchmark hash inconsistent'
    )

    check(
        previous.get(
            'ISU_cells_outside_training_full_feature_range_any'
        ) == 67,
        'Stage16B OOD count inconsistent'
    )

    v7 = load(
        V7,
        [
            'candidate_class',
            'target_rpt_index',
            'target_baseline_c5_ratio',
            'hours_to_target',
        ]
    )

    b0 = load(
        B0,
        [
            'strict_terminal_observed_candidate',
            'first_post100_rpt_index',
            'last_recorded_rpt_index',
            'terminal_c5_to_initial_ratio',
        ]
    )

    v6 = load(
        V6,
        [
            'candidate_rpt_index',
            'target_rpt_after_first100',
            'baseline_rpt_before_input',
        ]
    )

    pred = load(
        PRED,
        [
            'C5_candidate_NOT_FINAL',
            'median_prior_prediction',
            'random_forest_prediction',
            'extra_trees_prediction',
            'protocol_group',
            'outside_train_minmax_any',
        ]
    )

    check(
        len(pred) == 242
        and len(v7) == 251
        and len(b0) == 251
        and len(v6) == 251,
        'Unexpected input inventories'
    )

    check(
        set(pred.index).issubset(v7.index)
        and set(pred.index).issubset(b0.index)
        and set(pred.index).issubset(v6.index),
        'Cohort cell IDs mismatch'
    )

    idx = pred.index

    v7 = v7.loc[idx]
    b0 = b0.loc[idx]
    v6 = v6.loc[idx]

    check(
        v7.candidate_class.eq(
            'STRICT_PROSPECTIVE_CANDIDATE'
        ).all(),
        'Non-strict v7 cell'
    )

    check(
        b0.strict_terminal_observed_candidate
        .astype(str).str.lower()
        .isin(['true', '1', 'yes']).all(),
        'Non-strict terminal cell'
    )

    for col in (
        'target_rpt_after_first100',
        'baseline_rpt_before_input'
    ):
        check(
            v6[col].astype(str).str.lower()
            .isin(['true', '1', 'yes']).all(),
            'Chronology failed: ' + col
        )

    first = pd.to_numeric(
        v7.target_baseline_c5_ratio,
        errors='coerce'
    ).to_numpy(float)

    terminal = pd.to_numeric(
        b0.terminal_c5_to_initial_ratio,
        errors='coerce'
    ).to_numpy(float)

    prev_target = pd.to_numeric(
        pred.C5_candidate_NOT_FINAL,
        errors='coerce'
    ).to_numpy(float)

    check(
        np.isfinite(first).all()
        and (first > 0).all(),
        'Invalid first-post100 C5 ratio'
    )

    check(
        np.isfinite(terminal).all()
        and (terminal > 0).all(),
        'Invalid terminal C5 ratio'
    )

    check(
        np.allclose(
            terminal, prev_target,
            atol=1e-9, rtol=1e-9
        ),
        'Terminal C5 disagrees with Stage16B labels'
    )

    selected_index = pd.to_numeric(
        v7.target_rpt_index,
        errors='coerce'
    )

    early_index = pd.to_numeric(
        b0.first_post100_rpt_index,
        errors='coerce'
    )

    final_index = pd.to_numeric(
        b0.last_recorded_rpt_index,
        errors='coerce'
    )

    v6_index = pd.to_numeric(
        v6.candidate_rpt_index,
        errors='coerce'
    )

    comparable = (
        selected_index.notna()
        & early_index.notna()
    )

    match_early = (
        comparable
        & selected_index.eq(early_index)
    )

    v6_comparable = (
        selected_index.notna()
        & v6_index.notna()
    )

    match_v6 = (
        v6_comparable
        & selected_index.eq(v6_index)
    )

    gap = final_index - early_index

    hours = pd.to_numeric(
        v7.hours_to_target,
        errors='coerce'
    )

    result = pd.DataFrame({
        'cell_id': idx,
        'protocol_group':
            pred.protocol_group.to_numpy(),
        'first_post100_c5_ratio_V7_NOT_FINAL':
            first,
        'terminal_c5_ratio_B0_NOT_FINAL':
            terminal,
        'first_minus_terminal_ratio':
            first - terminal,
        'first_minus_terminal_percentage_points':
            100 * (first - terminal),
        'first_post100_target_hours':
            hours.to_numpy(),
        'v7_target_rpt_index':
            selected_index.to_numpy(),
        'b0_first_post100_rpt_index':
            early_index.to_numpy(),
        'b0_last_recorded_rpt_index':
            final_index.to_numpy(),
        'rpt_index_gap_last_minus_first_post100':
            gap.to_numpy(),
        'v7_index_matches_b0_first':
            match_early.to_numpy(),
        'v7_index_matches_v6_candidate':
            match_v6.to_numpy(),
        'outside_training_feature_range_any':
            pred.outside_train_minmax_any.to_numpy(),
    })

    for model in (
        'median_prior',
        'random_forest',
        'extra_trees'
    ):
        p = pd.to_numeric(
            pred[f'{model}_prediction'],
            errors='coerce'
        ).to_numpy(float)

        check(
            np.isfinite(p).all(),
            'Invalid prediction values'
        )

        result[f'{model}_pred'] = p

        result[
            f'{model}_abs_error_first_post100_DIAGNOSTIC'
        ] = np.abs(p - first)

        result[
            f'{model}_abs_error_terminal'
        ] = np.abs(p - terminal)

    horizon = {
        'n_cells': len(idx),

        'first_post100_c5_ratio':
            describe(first),

        'terminal_c5_ratio':
            describe(terminal),

        'first_minus_terminal_pp':
            describe(100 * (first - terminal)),

        'count_first_ratio_greater_than_terminal':
            int((first > terminal).sum()),

        'count_first_ratio_less_than_terminal':
            int((first < terminal).sum()),

        'count_abs_difference_ge_3pp':
            int(
                (np.abs(first - terminal) >= .03).sum()
            ),

        'count_abs_difference_ge_5pp':
            int(
                (np.abs(first - terminal) >= .05).sum()
            ),

        'first_post100_target_hours':
            describe(hours),

        'rpt_index_gap':
            describe(gap),

        'count_later_rpt_index_after_first_post100':
            int((gap > 0).sum()),

        'count_first_post100_same_as_terminal_index':
            int((gap == 0).sum()),

        'v7_to_b0_first_index_comparable':
            int(comparable.sum()),

        'v7_to_b0_first_index_match':
            int(match_early.sum()),

        'v7_to_v6_candidate_index_comparable':
            int(v6_comparable.sum()),

        'v7_to_v6_candidate_index_match':
            int(match_v6.sum()),
    }

    horizon[
        'frozen_prediction_diagnostics'
    ] = {}

    for model in (
        'median_prior',
        'random_forest',
        'extra_trees'
    ):
        p = result[
            f'{model}_pred'
        ].to_numpy(float)

        horizon[
            'frozen_prediction_diagnostics'
        ][model] = {
            'against_first_post100_C5_POSTHOC_DIAGNOSTIC':
                scores(first, p),

            'against_terminal_C5_STAGE16A':
                scores(terminal, p),
        }

    report = {
        'study':
            'STAGE16C_EXPLORATORY_ENDPOINT_HORIZON_AUDIT',

        'source_stage16b_report_sha256':
            sha(S16B),

        'source_predictions_sha256':
            sha(PRED),

        'source_v7_sha256':
            sha(V7),

        'source_b0_sha256':
            sha(B0),

        'results': horizon,

        'limits': [
            'The V7 post-100 RPT candidate remains provisional.',
            'RPT index conventions require source verification.',
            'hours_to_target concerns the first post-100 candidate, not terminal RPT.',
            'This post-hoc comparison cannot replace the original target.',
            'Capacity-unit and cross-dataset endpoint equivalence remain unconfirmed.',
        ],

        'model_training': False,
        'predictions_recomputed': False,
        'new_target_approved': False,
    }

    P.mkdir(
        parents=True, exist_ok=True
    )
    R.mkdir(
        parents=True, exist_ok=True
    )

    with OUT.open(
        'x', encoding='utf-8', newline=''
    ) as f:
        result.to_csv(f, index=False)

    with SUMMARY.open(
        'x', encoding='utf-8'
    ) as f:
        json.dump(
            report, f,
            indent=2,
            ensure_ascii=False
        )

    print(
        '=== STAGE 16C ISU ENDPOINT HORIZON AUDIT ==='
    )

    print(
        'RPT_INDEX_COMPARABILITY=',
        json.dumps({
            k: v for k, v in horizon.items()
            if 'index' in k
        })
    )

    print(
        'HORIZON_AND_TARGET_GAP=',
        json.dumps({
            k: v for k, v in horizon.items()
            if (
                k != 'frozen_prediction_diagnostics'
                and 'index' not in k
            )
        })
    )

    print(
        'FROZEN_PREDICTION_HORIZON_COMPARISON=',
        json.dumps(
            horizon['frozen_prediction_diagnostics']
        )
    )

    print('CELLWISE_CSV=', OUT)
    print('REPORT_JSON=', SUMMARY)
    print('MODEL_TRAINING=NO')
    print('FINAL_EXTERNAL_VALIDATION=NO')
    print('=== END STAGE 16C ===')


if __name__ == '__main__':
    main()
