#!/usr/bin/env python3
"""Exploratory BatteryInverse Stage19: test-only, within-dataset whole-feature-block swap."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

ROOT = Path.cwd()
REP = ROOT / 'external_validation/reports'
DATA = ROOT / 'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'
OLD = REP / 'stage17ak_harmonization_source_and_lodo_folds.csv'
RECIPE = ROOT / 'external_validation/scripts/stage16a_isu_exploratory_transfer.py'
SHA_DATA = '8bc5296425fbb0735faceb56fe1cf52c6d0292bec2e361be8870fc1626aa5c7c'
SHA_RECIPE = '537b148602a6c191b1964ee93cd4fee697156772ce43f61d6803808078e29e61'
F = ['cap_ratio_10_to_1','cap_ratio_50_to_1','cap_ratio_100_to_1',
     'cap_mean_first_100_norm','cap_std_first_100_norm',
     'cap_slope_first_100_norm','cap_fade_fraction_1_to_100']
P = 'stage19_featureblock_swap'
OUT = {k: REP / (P + '_' + k + ('.json' if k=='report' else '.csv'))
       for k in ['folds','permutations','summary','influence','report']}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''): h.update(block)
    return h.hexdigest()

def check(ok, msg):
    if not ok: raise RuntimeError(msg)

def pipeline(reg):
    return Pipeline([('preprocessor', ColumnTransformer([
        ('num', Pipeline([('imputer',SimpleImputer(strategy='median')),
                          ('scaler',StandardScaler())]), F)])),
        ('model',reg)])

def swap_order(n, rng):
    if n==2: return np.array([1,0])
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)): return p

def main():
    check(DATA.is_file() and sha(DATA)==SHA_DATA, 'Input data hash mismatch')
    check(RECIPE.is_file() and sha(RECIPE)==SHA_RECIPE, 'Original recipe mismatch')
    check(OLD.is_file(), 'Original Stage17AK fold CSV missing')
    check(all(not path.exists() for path in OUT.values()), 'Stage19 outputs already exist; no overwrite')
    d = pd.read_csv(DATA)
    check(d.shape==(567,38) and d.cell_id.is_unique and d.dataset_id.nunique()==12,
          'Unexpected original benchmark population')
    for f in F+['final_capacity_retention']:
        check(np.isfinite(pd.to_numeric(d[f],errors='raise')).all(), 'Nonfinite '+f)
    old = pd.read_csv(OLD)
    old = old[(old.split_type=='DATASET_LODO') &
              (old.feature_condition=='ORIGINAL_MIXED_SIGN')]
    check(len(old)==24, 'Expected 24 original fold metrics')
    y = d.final_capacity_retention.to_numpy(float)
    data_datasets = d.dataset_id.astype(str).to_numpy()
    domains = sorted(d.dataset_id.unique())
    folds, reps, max_rf_drift, max_et_drift = [], [], 0., 0.
    for domain_idx, domain in enumerate(domains):
        test = data_datasets==domain
        n = int(test.sum())
        check(n>=2, 'Cannot swap singleton domain')
        ytr, yte = y[~test], y[test]
        Xtr, Xte = d.loc[~test,F], d.loc[test,F]
        prior = float(np.median(ytr))
        prior_mae = float(np.mean(np.abs(yte-prior)))
        orders = [swap_order(n,np.random.default_rng(20261121+domain_idx*100000+s))
                  for s in range(100)]
        Xswapped = pd.DataFrame(np.vstack([Xte.to_numpy()[p] for p in orders]), columns=F)
        for model, klass in [('RF',RandomForestRegressor),('ET',ExtraTreesRegressor)]:
            m = pipeline(klass(n_estimators=700,min_samples_leaf=2,random_state=42,n_jobs=1))
            m.fit(Xtr,ytr)
            orig_mae = float(np.abs(m.predict(Xte)-yte).mean())
            expected = old[(old.heldout==domain)&
                           (old.model==('random_forest' if model=='RF' else 'extra_trees'))]
            check(len(expected)==1, 'Missing baseline '+domain+' '+model)
            drift = abs(orig_mae-float(expected.iloc[0].MAE))
            check(abs(prior_mae-float(expected.iloc[0].baseline_MAE))<1e-10,
                  'Prior replay mismatch')
            check(drift < (3e-4 if model=='RF' else 1e-10),
                  'Model replay differs beyond documented cross-version tolerance')
            if model=='RF': max_rf_drift=max(max_rf_drift,drift)
            else: max_et_drift=max(max_et_drift,drift)
            swapped = m.predict(Xswapped).reshape(100,n)
            errors = np.abs(swapped-yte[None,:]).mean(axis=1)
            folds.append({'domain':domain,'model':model,'n':n,'original_mae':orig_mae,
                          'prior_mae':prior_mae,'swap_mae_mean':float(errors.mean()),
                          'swap_minus_orig_mean':float(errors.mean()-orig_mae),
                          'replay_abs_delta':drift})
            for s,err in enumerate(errors):
                reps.append({'domain':domain,'model':model,'permutation':s,'n':n,
                             'original_mae':orig_mae,'prior_mae_unchanged':prior_mae,
                             'swapped_mae':float(err),'swap_minus_orig':float(err-orig_mae)})
        print('COMPLETED_DOMAIN',domain,n,flush=True)
    fdf = pd.DataFrame(folds)
    rdf = pd.DataFrame(reps)
    summaries, influences = [], []
    for model in ['RF','ET']:
        for cohort in ['all12','without_UL_PUR']:
            folds_m = fdf[(fdf.model==model) & ((fdf.domain!='UL_PUR') if cohort!='all12' else True)]
            domains_m = sorted(folds_m.domain.tolist())
            base = folds_m.set_index('domain').loc[domains_m]
            values = rdf[rdf.model==model].pivot(index='permutation',columns='domain',values='swapped_mae')[domains_m]
            for weighting in ['equal_domain','cell_weighted']:
                w = np.ones(len(base)) if weighting=='equal_domain' else base.n.to_numpy(float)
                w = w/w.sum()
                old_mae = float(w@base.original_mae.to_numpy())
                prior = float(w@base.prior_mae.to_numpy())
                new_mae = values.to_numpy()@w
                diff = new_mae-old_mae
                summaries.append({'model':model,'cohort':cohort,'weighting':weighting,
                                  'n_domains':len(base),'n_cells':int(base.n.sum()),
                                  'orig_mae':old_mae,'prior_mae_unchanged':prior,
                                  'swap_mae_mean':float(new_mae.mean()),
                                  'delta_mean':float(diff.mean()),
                                  'delta_p025':float(np.quantile(diff,.025)),
                                  'delta_p975':float(np.quantile(diff,.975)),
                                  'positive_permutations':int((diff>0).sum())})
        for removed in sorted(fdf.domain.unique()):
            part=fdf[(fdf.model==model)&(fdf.domain!=removed)]
            for wt in ['equal_domain','cell_weighted']:
                w = np.ones(len(part)) if wt=='equal_domain' else part.n.to_numpy(float)
                influences.append({'model':model,'excluded_domain':removed,'weighting':wt,
                                   'remaining_domains':len(part),
                                   'mean_delta':float(np.average(part.swap_minus_orig_mean,weights=w))})
    summary=pd.DataFrame(summaries)
    for model,value in [('RF',0.20908484562427967),('ET',0.20814730816213847)]:
        got=summary[(summary.model==model)&(summary.cohort=='all12')&
                    (summary.weighting=='equal_domain')].iloc[0].orig_mae
        check(abs(got-value)<(3e-4 if model=='RF' else 1e-10),'LODO macro reproduction mismatch')
    # Nothing is written until every fold and cross-check has passed.
    for key,df in [('folds',fdf),('permutations',rdf),
                   ('summary',summary),('influence',pd.DataFrame(influences))]:
        df.to_csv(OUT[key],index=False,mode='x')
    report={'study':'EXPLORATORY_POST_OUTCOME_NOT_PROSPECTIVE',
            'model_recipe':'original seven-feature RF/ET Stage17AK LODO, fixed fitted models',
            'input_data_sha256':sha(DATA),'stage16a_recipe_sha256':sha(RECIPE),
            'stage17ak_fold_sha256':sha(OLD),'scikit_learn_version':sklearn.__version__,
            'permutations_per_dataset':100,'all_test_domains':12,
            'test_only_block_derangement':True,'training_targets_unchanged':True,
            'test_targets_unchanged':True,'source_membership_unchanged':True,
            'prior_unchanged':True,'rf_max_replay_drift':max_rf_drift,
            'et_max_replay_drift':max_et_drift,'results':summaries,
            'limits':['No population confidence intervals',
                      'Two-cell UL_PUR has one unique non-identity swap',
                      'Descriptive within-domain feature-target association; not causal evidence']}
    OUT['report'].write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('STAGE19_VALIDATED=YES',flush=True)
    print('RF_REPLAY_MAX_DELTA',max_rf_drift,'ET_REPLAY_MAX_DELTA',max_et_drift,flush=True)
    print(summary.to_string(index=False,float_format=lambda z:f'{z:.6f}'),flush=True)

if __name__=='__main__': main()
