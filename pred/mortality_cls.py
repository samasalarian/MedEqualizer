#!/usr/bin/env python3
"""
mortality_cls.py  –  MedEqualizer  
═════════════════════════════════════════════════════════════════════════════════════════

TRTR, TSTR Original and TSTR MedEqualizer AUROC with 95% intervals, for the six
finalized configurations, on the fixed patient-level held-out real test set.
Reads the existing leakage-free splits and synthetic datasets in
results/tstr_trtr/ (no GAN is trained). Outcome: any recorded death
(MORTALITY = 1, i.e. a recorded date of death).

  Features     GENDER, AGE_GROUP, ETHNICITY, ADMISSION_LOCATION, ADMISSION_TYPE,
               INSURANCE, DISEASE, FIRST_CAREUNIT, LAST_CAREUNIT, LOS
  Preprocessing (fit on each training set only): most-frequent imputation +
               ordinal encoding (unseen test levels -> -1) for categoricals,
               median imputation for LOS; KNN additionally standardized.
  Classifiers  KNN (k=5); Random Forest and Extra Trees (100 trees,
               scikit-learn defaults); seed 42.

  TRTR   one fit on the real training data; AUROC with a 95% percentile
         bootstrap CI over test patients (1000 resamples).
  TSTR   one fit per GAN run (seeds 42, 52, 62, 72, 82); mean AUROC with a
         t-based 95% CI over the five runs (df = 4). It reflects GAN-run
         variability only, not test-set sampling uncertainty.

Output: one CSV (default results/mortality_cls_auroc.csv).

Usage
─────
  python pred/mortality_cls.py --workers 8
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results' / 'tstr_trtr'
CONFIGS = json.loads((REPO / 'reproducibility' / 'final-configurations.json').read_text())['experiments']
LABEL = 'MORTALITY'
FEATURES = ['GENDER', 'AGE_GROUP', 'ETHNICITY', 'ADMISSION_LOCATION', 'ADMISSION_TYPE',
            'INSURANCE', 'DISEASE', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'LOS']
NUMERIC = ['LOS']
GAN_SEEDS = [42, 52, 62, 72, 82]
SEED = 42
N_BOOT = 1000
CLASSIFIERS = ['KNN', 'Random Forest', 'Extra Trees']
METHOD = {'real': 'TRTR', 'original': 'TSTR Original', 'medequalizer': 'TSTR MedEqualizer'}


def get_classifiers():
    return {'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=1)}


def split_paths(dataset):
    return RESULTS / dataset / 'real_train.csv', RESULTS / dataset / 'real_test.csv'


def synthetic_path(cfg, arm, seed):
    return RESULTS / f"{cfg['dataset']}_{cfg['generator']}_tau{cfg['tau']}" / f'{arm}_seed{seed}.csv'


def xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors='coerce') if c in NUMERIC else X[c].astype(str)
    return X, pd.to_numeric(df[LABEL]).round().astype(int).values


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit imputers/encoder/scaler on the training data only; categorical columns first."""
    cat = [c for c in X_train.columns if c not in NUMERIC]
    imp_c = SimpleImputer(strategy='most_frequent')
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    imp_n = SimpleImputer(strategy='median')
    tr = np.hstack([enc.fit_transform(imp_c.fit_transform(X_train[cat])), imp_n.fit_transform(X_train[NUMERIC])])
    te = np.hstack([enc.transform(imp_c.transform(X_test[cat])), imp_n.transform(X_test[NUMERIC])])
    scaler = StandardScaler().fit(tr)
    return tr, te, scaler.transform(tr), scaler.transform(te)


def score(job) -> list:
    """Fit the three classifiers on one training set; AUROC on the real test set."""
    from threadpoolctl import threadpool_limits
    meta, train_path, test_path, boot = job
    Xtr, ytr = xy(pd.read_csv(train_path, low_memory=False))
    Xte, yte = xy(pd.read_csv(test_path))
    tr, te, tr_sc, te_sc = preprocess(Xtr, Xte)
    rows = []
    with threadpool_limits(1):
        for name, clf in get_classifiers().items():
            a, b = (tr_sc, te_sc) if name == 'KNN' else (tr, te)
            p = clf.fit(a, ytr).predict_proba(b)[:, 1]
            row = {**meta, 'classifier': name, 'auroc': roc_auc_score(yte, p)}
            if boot:
                rng = np.random.default_rng(SEED)
                aucs = [roc_auc_score(yte[i], p[i]) for i in
                        (rng.integers(0, len(yte), len(yte)) for _ in range(N_BOOT))]
                row['ci_lower'], row['ci_upper'] = np.percentile(aucs, [2.5, 97.5])
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description='Mortality AUROC (TRTR / TSTR) for KNN, RF and ET',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', default=str(REPO / 'results' / 'mortality_cls_auroc.csv'))
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f'Exists, not overwriting: {out}')

    jobs = []
    for d in ('mimic3', 'mimic4'):
        tr, te = split_paths(d)
        overlap = set(pd.read_csv(tr, usecols=['SUBJECT_ID']).SUBJECT_ID) & set(pd.read_csv(te, usecols=['SUBJECT_ID']).SUBJECT_ID)
        if overlap:
            sys.exit(f'{d}: {len(overlap)} patients in both train and test')
        jobs.append(({'dataset': d, 'arm': 'real', 'gan_seed': np.nan}, tr, te, True))
    for cfg in CONFIGS:
        for arm in ('original', 'medequalizer'):
            for seed in GAN_SEEDS:
                f = synthetic_path(cfg, arm, seed)
                if not f.exists():
                    sys.exit(f'Missing synthetic file: {f}')
                jobs.append(({'dataset': cfg['dataset'], 'generator': cfg['generator'], 'arm': arm,
                              'gan_seed': seed}, f, split_paths(cfg['dataset'])[1], False))
    with ProcessPoolExecutor(args.workers) as ex:
        runs = pd.DataFrame([r for rs in ex.map(score, jobs) for r in rs])
    assert len(runs) == len(jobs) * len(CLASSIFIERS)

    # Cross-check against the completed nine-classifier evaluation, when present.
    prev_path = RESULTS / 'classification_metrics.csv'
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        prev['generator'] = prev.generator.replace('none', np.nan)
        key = ['dataset', 'generator', 'arm', 'gan_seed', 'classifier']
        m = runs.merge(prev[key + ['auroc']], on=key, suffixes=('', '_prev'), how='left')
        assert m.auroc_prev.notna().all() and (m.auroc - m.auroc_prev).abs().max() < 1e-9, \
            'AUROC differs from classification_metrics.csv'
        print('AUROC identical to classification_metrics.csv for all runs')

    tcrit = stats.t.ppf(0.975, df=len(GAN_SEEDS) - 1)
    rows = []
    for cfg in CONFIGS:
        for c in CLASSIFIERS:
            base = {'dataset': cfg['dataset'], 'generator': cfg['generator'], 'tau': cfg['tau'], 'classifier': c}
            t = runs[(runs.dataset == cfg['dataset']) & (runs.arm == 'real') & (runs.classifier == c)].iloc[0]
            rows.append({**base, 'method': METHOD['real'], 'auroc': t.auroc, 'ci_lower': t.ci_lower,
                         'ci_upper': t.ci_upper, 'ci_method': f'bootstrap over test patients ({N_BOOT})',
                         'n_runs': 1})
            for arm in ('original', 'medequalizer'):
                a = runs[(runs.dataset == cfg['dataset']) & (runs.generator == cfg['generator'])
                         & (runs.arm == arm) & (runs.classifier == c)].auroc
                assert len(a) == len(GAN_SEEDS)
                half = tcrit * a.std(ddof=1) / np.sqrt(len(a))
                rows.append({**base, 'method': METHOD[arm], 'auroc': a.mean(), 'ci_lower': a.mean() - half,
                             'ci_upper': a.mean() + half, 'ci_method': 't over GAN runs (df=4)',
                             'n_runs': len(a)})
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(f'Saved {out} ({len(res)} rows)')


if __name__ == '__main__':
    main()
