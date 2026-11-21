#!/usr/bin/env python3
"""Stage17AM final synthesis of existing 17AK/17AL folds; no new fitting."""
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path.cwd().resolve()
R = ROOT / 'external_validation/reports'
T = ROOT / 'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'
AK = R / 'stage17ak_harmonization_ablation_report.json'
AKC = R / 'stage17ak_harmonization_source_and_lodo_folds.csv'
AL = R / 'stage17al_sixfeature_redundancy_control_report.json'
ALC = R / 'stage17al_sixfeature_deletion_fold_comparison.csv'
AJ = R / 'stage17aj_v2_figure_only_provenance.json'
PRED = R / 'stage17ae_nca_30cell_PREOUTCOME_dual_sign_predictions.csv'

OUT = R / 'stage17am_three_condition_fold_score_evidence.csv'
AGG = R / 'stage17am_macro_and_cell_weighted_results.csv'
REPORT = R / 'stage17am_exploratory_ablation_consolidated_evidence.json'
LEDGER = R / 'stage17am_deferred_prism_update_ledger.md'

FROZEN = 'cf34a8085d57924f1e6f7e9377b2707092452e0c383b680d82d4e5a27a588757'


def need(x, m):
    if not x:
        raise RuntimeError(m)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def readcsv(p, fields):
    need(p.is_file(), 'Missing: ' + str(p))
    with p.open(newline='', encoding='utf-8-sig') as f:
        d = csv.DictReader(f)
        need(
            set(fields) <= set(d.fieldnames or []),
            'Missing columns in ' + str(p)
        )
        return list(d)


def number(v, name):
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError) as e:
        raise RuntimeError(
            'Invalid ' + name + ': ' + str(v)
        ) from e
    need(math.isfinite(x), 'Nonfinite ' + name)
    return x


def writecsv(path, vals):
    need(bool(vals), 'No CSV rows')
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(vals[0]))
        w.writeheader()
        w.writerows(vals)


def main():
    sources = (T, AK, AKC, AL, ALC, AJ, PRED)

    for p in sources:
        need(p.is_file(), 'Missing: ' + str(p))

    for p in (OUT, AGG, REPORT, LEDGER):
        need(
            not p.exists(),
            'Refusing overwrite: ' + str(p)
        )

    ak = json.loads(AK.read_text(encoding='utf-8'))
    al = json.loads(AL.read_text(encoding='utf-8'))
    aj = json.loads(AJ.read_text(encoding='utf-8'))

    need(
        ak.get('original_training_n') == 567
        and al.get('training_rows') == 567,
        '567 training lock mismatch'
    )

    need(
        ak.get('split_grid', {}).get('total_fits') == 56
        and al.get('new_models_fit') == 28,
        'Original model-run count mismatch'
    )

    need(
        al.get('no_new_ISU_or_NCA_inference') is True
        and al.get('original_training_table_modified') is False
        and ak.get('original_training_table_modified') is False,
        'Earlier experiment provenance inconsistent'
    )

    need(
        sha(PRED) == FROZEN
        == ak.get('frozen_NCA_prediction_sha256')
        == al.get('frozen_NCA_prediction_sha256')
        == aj.get('input_sha256', {}).get(
            PRED.relative_to(ROOT).as_posix()
        ),
        'Prediction hash mismatch'
    )

    for obj, label in ((ak, '17AK'), (al, '17AL')):
        hashes = obj.get(
            'input_sha256',
            obj.get('source_sha256', {})
        )

        for rel, hashval in hashes.items():
            q = Path(rel)
            need(
                not q.is_absolute() and '..' not in q.parts,
                'Unsafe ' + label + ' path'
            )
            path = (ROOT / q).resolve()
            need(
                path.is_relative_to(ROOT)
                and path.is_file()
                and sha(path) == hashval,
                label + ' input changed: ' + rel
            )

    for rel, hashval in aj.get('figure_sha256', {}).items():
        p = (ROOT / rel).resolve()
        need(
            p.is_relative_to(ROOT)
            and p.is_file()
            and sha(p) == hashval,
            'Figure file mismatch: ' + rel
        )

    need(
        aj.get('archive_cells') == 66
        and aj.get('selected_cells') == 30,
        '17AJ figure roster mismatch'
    )

    original = readcsv(
        T, ['cell_id', 'dataset_id', 'source_family']
    )

    need(
        len(original) == 567
        and len(set(x['cell_id'] for x in original)) == 567,
        'Training size/unique cell check failed'
    )

    family_counts = Counter(
        x['source_family'] for x in original
    )
    dataset_counts = Counter(
        x['dataset_id'] for x in original
    )

    need(
        family_counts == {
            'BatteryML_main': 318,
            'BatteryLife_selected': 249
        }
        and len(dataset_counts) == 12,
        'Training domain roster changed'
    )

    folds_ak = readcsv(AKC, [
        'split_type', 'heldout', 'model',
        'feature_condition', 'test_n',
        'train_n', 'MAE', 'baseline_MAE'
    ])

    folds_al = readcsv(ALC, [
        'split_type', 'heldout', 'model',
        'test_n', 'train_n', 'prior_MAE',
        'original_mixed_7_MAE_from17AK',
        'harmonized_7_MAE_from17AK',
        'no_fade_6_MAE'
    ])

    need(
        len(folds_ak) == 56
        and len(folds_al) == 28,
        'Missing ablation fold results'
    )

    idx = {}

    for x in folds_ak:
        k = (
            x['split_type'],
            x['heldout'],
            x['model'],
            x['feature_condition']
        )
        need(k not in idx, 'Duplicated Stage17AK score')
        idx[k] = x

    models = {'random_forest', 'extra_trees'}
    trials = {}

    for x in folds_al:
        kind = x['split_type']
        hold = x['heldout']
        model = x['model']

        need(
            model in models
            and kind in (
                'SOURCE_FAMILY_HOLDOUT',
                'DATASET_LODO'
            ),
            'Unknown method/split'
        )

        group = (kind, hold, model)
        need(group not in trials, 'Repeated Stage17AL group')

        n = int(number(x['test_n'], 'n'))
        nt = int(number(x['train_n'], 'nt'))

        need(
            n >= 2 and nt + n == 567,
            'Fold size invalid'
        )

        if kind == 'SOURCE_FAMILY_HOLDOUT':
            need(
                hold in family_counts
                and family_counts[hold] == n,
                'Family size mismatch'
            )
        else:
            need(
                hold in dataset_counts
                and dataset_counts[hold] == n,
                'Dataset size mismatch'
            )

        orig = idx.get(
            (*group, 'ORIGINAL_MIXED_SIGN')
        )
        harmon = idx.get(
            (*group, 'HARMONIZED_POSITIVE_LOSS')
        )

        need(
            orig is not None and harmon is not None,
            'Stage17AK match missing'
        )

        for row, which in (
            (orig, 'original_mixed_7_MAE_from17AK'),
            (harmon, 'harmonized_7_MAE_from17AK')
        ):
            need(
                int(row['test_n']) == n
                and int(row['train_n']) == nt,
                'Group membership changed'
            )

            need(
                abs(
                    number(row['MAE'], which)
                    - number(x[which], which)
                ) < 1e-10,
                '17AK/17AL MAE discrepancy: ' + str(group)
            )

            need(
                abs(
                    number(row['baseline_MAE'], 'prior')
                    - number(x['prior_MAE'], 'prior')
                ) < 1e-10,
                'Prior mismatch'
            )

        trials[group] = {
            'test_n': n,
            'train_n': nt,
            'prior': number(x['prior_MAE'], 'prior'),
            'original7': number(
                x['original_mixed_7_MAE_from17AK'], 'orig'
            ),
            'harmonized7': number(
                x['harmonized_7_MAE_from17AK'], 'harm'
            ),
            'no_fade6': number(
                x['no_fade_6_MAE'], 'no_fade'
            )
        }

    need(len(trials) == 28, 'Incomplete paired grid')

    for kind, expected in (
        ('SOURCE_FAMILY_HOLDOUT', set(family_counts)),
        ('DATASET_LODO', set(dataset_counts))
    ):
        for model in sorted(models):
            have = {
                h for k, h, m in trials
                if k == kind and m == model
            }
            need(
                have == expected,
                'Incomplete fold coverage: ' + kind
            )

            count = sum(
                v['test_n']
                for (k, h, m), v in trials.items()
                if k == kind and m == model
            )

            need(
                count == 567,
                'Partition coverage error'
            )

    foldrows = []

    for (kind, hold, model), v in sorted(trials.items()):
        foldrows.append({
            'split_type': kind,
            'heldout': hold,
            'model': model,
            'train_n': v['train_n'],
            'test_n': v['test_n'],
            'prior_MAE': v['prior'],
            'original7_MAE': v['original7'],
            'harmonized7_MAE': v['harmonized7'],
            'no_fade6_MAE': v['no_fade6'],
            'gain_original7_minus_harmonized7':
                v['original7'] - v['harmonized7'],
            'gain_original7_minus_no_fade6':
                v['original7'] - v['no_fade6'],
            'gain_prior_minus_original7':
                v['prior'] - v['original7'],
            'gain_prior_minus_no_fade6':
                v['prior'] - v['no_fade6']
        })

    agg = []

    for kind in (
        'SOURCE_FAMILY_HOLDOUT',
        'DATASET_LODO'
    ):
        for model in sorted(models):
            for subset in ('ALL', 'EXCLUDE_UL_PUR'):
                a = [
                    v for (k, h, m), v in trials.items()
                    if k == kind
                    and m == model
                    and (
                        subset == 'ALL'
                        or h != 'UL_PUR'
                    )
                ]

                if (
                    not a
                    or (
                        kind == 'SOURCE_FAMILY_HOLDOUT'
                        and subset == 'EXCLUDE_UL_PUR'
                    )
                ):
                    continue

                weight = sum(
                    z['test_n'] for z in a
                )

                record = {
                    'split_type': kind,
                    'model': model,
                    'subset': subset,
                    'folds': len(a),
                    'test_cells_total': weight
                }

                for cond in (
                    'prior',
                    'original7',
                    'harmonized7',
                    'no_fade6'
                ):
                    record['macro_MAE_' + cond] = sum(
                        z[cond] for z in a
                    ) / len(a)

                    record[
                        'cell_weighted_MAE_' + cond
                    ] = sum(
                        z[cond] * z['test_n']
                        for z in a
                    ) / weight

                record[
                    'folds_orig7_better_prior'
                ] = sum(
                    z['original7'] < z['prior'] - 1e-12
                    for z in a
                )

                record[
                    'folds_no_fade6_better_orig7'
                ] = sum(
                    z['no_fade6'] < z['original7'] - 1e-12
                    for z in a
                )

                agg.append(record)

    report = {
        'stage': '17AM_FROZEN_SENSITIVITY_SYNTHESIS',
        'method':
            'aggregation_of_existing_17AK_and_17AL_folds_only',
        'status':
            'POST_OUTCOME_EXPLORATORY_NOT_EXTERNAL_PROSPECTIVE',
        'original_training_cells': 567,
        'datasets': 12,
        'source_families': 2,
        'three_feature_conditions': [
            'original7',
            'harmonized7',
            'no_fade6'
        ],
        'interpretation':
            'Macro averages equally across datasets/domains. '
            'Cell-weighted MAE weights each holdout by its '
            'physical-cell count.',
        'fold_level_comparisons': 28,
        'aggregate': agg,
        'original_models_or_predictions_modified': False,
        'NCA_predictions_sha256': sha(PRED),
        'figure_17AJ_V2_verified': True,
        'no_new_model_fitting': True,
        'no_new_outcome_scoring': True,
        'prism_updated': False,
        'sources_sha256': {
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in sources
        }
    }

    ledger = '''# BatteryInverse — deferred Prism update ledger

Do not edit the manuscript yet. This ledger is for a future
consolidated revision.

## Evidence already represented in the current draft

- 242 measured ISU transfer observations and horizon caveats.
- 30 NCA locked predictions scored against last-recorded,
  non-EOL capacity. Preserve existing Tables 6 and 7.
- Source-family opposite-sign capacity-change feature.
- NCA 2500 mAh observation-window clustering and the
  post-outcome algebraic diagnostic.

## New items for the consolidated Prism update

- Stage 17AK: 567 cells, 2 source-family and 12 dataset
  holdouts, RF/ET, original mixed-sign 7 versus harmonized
  positive-loss 7. Harmonization did not consistently
  reduce transfer error.
- Stage 17AL: Same folds with capacity-change feature
  removed (6 features). Cross-family transfer remains
  below the train-median baseline. LODO is heterogeneous.
- Stage 17AM: Report both domain-macro MAE and physical-
  cell-weighted MAE. Do not selectively choose results.
  Use a supplementary table if relevant to reviewers.
- Stage 17AJ-V2: Two-panel vector PDF candidate for
  supplementary figure: 66 archive endpoints and
  30 selected-cell diagnostic ratios.

## Essential interpretation restrictions

- Harmonized fade = 1-Q100/Q1, a deterministic transform
  of an existing feature.
- Removing this feature does not causally identify why
  transfer fails.
- Stages 17AK–17AL are post-outcome exploratory ablations,
  not prospectively prespecified independent validation.
- 2500/Q1 is not a forecasting baseline.
- The exact per-cell NCA stopping command and physical
  EOL remain uncertified.
- Preserve all previous numerical results, citations,
  figures, SHA-256 locks and existing manuscript content.

No manuscript modifications are performed by this script.
'''

    with tempfile.TemporaryDirectory(
        prefix='stage17am_',
        dir=R
    ) as folder:
        d = Path(folder)

        writecsv(d / 'fold.csv', foldrows)
        writecsv(d / 'agg.csv', agg)

        (d / 'report.json').write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ),
            encoding='utf-8'
        )

        (d / 'ledger.md').write_text(
            ledger,
            encoding='utf-8'
        )

        need(
            not any(
                p.exists()
                for p in (OUT, AGG, REPORT, LEDGER)
            ),
            'Result path appeared; refusing overwrite'
        )

        for src, dest in (
            (d / 'fold.csv', OUT),
            (d / 'agg.csv', AGG),
            (d / 'report.json', REPORT),
            (d / 'ledger.md', LEDGER)
        ):
            os.replace(src, dest)

    print('=== STAGE17AM FINAL SENSITIVITY SYNTHESIS ===')
    print('ORIGINAL567=YES LODO_DATASETS=12 SOURCE_HOLDOUTS=2')
    print('FROZEN_NCA_PREDICTIONS_UNCHANGED=YES')

    print(
        'AGGREGATES_MACRO_AND_CELL_WEIGHTED=',
        json.dumps(agg, ensure_ascii=False)
    )

    print('EVIDENCE_CSV=', OUT)
    print('AGGREGATE_CSV=', AGG)
    print('EVIDENCE_JSON=', REPORT)
    print('DEFERRED_PRISM_LEDGER=', LEDGER)

    print(
        'NEW_FITS=NO NEW_EXTERNAL_INFERENCE=NO '
        'PRISM_UPDATED=NO'
    )
    print('=== END STAGE17AM ===')


if __name__ == '__main__':
    main()
