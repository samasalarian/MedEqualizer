#!/usr/bin/env python3
"""
run_pred.py  —  Mortality prediction on synthetic data generated from the
original cohort (before) vs. from the MedEqualizer-augmented cohort (after).

For each seed: one stratified 80/20 train/test split of each file,
preprocessing (imputation, ordinal encoding, scaling) fit on the training
partition only, then every classifier is fit and scored on the test partition.
The 'source' provenance column added by augmentation/cham.py is dropped.

This is the prediction code behind the manuscript mortality AUROC table
(see pred/mortality_classification.py).

Usage
─────
  python pred/run_pred.py \
      --before data/Mimic4/synthetic/ctgan/before.csv \
      --after  "data/Mimic4/synthetic/ctgan/after_tau=12_nu0p2_auto_0p9_random_forest.csv" \
      --label  mimic4_ctgan_tau12 \
      --outdir pred/out/ctgan \
      --seeds  42 52 62 72 82
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

LABEL_COL      = 'MORTALITY'
TEST_SIZE      = 0.2
DEFAULT_SEEDS  = [42, 52, 62, 72, 82]
SCALE_REQUIRED = {'KNN', 'MLP'}
METRICS        = ['roc_auc']


# ── Classifiers ───────────────────────────────────────────────────────────────

def get_classifiers(seed: int) -> dict:
    return {
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=seed),
        'KNN':                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=seed),
        'Hist GB':             HistGradientBoostingClassifier(max_iter=100, random_state=seed),
        'MLP':                 MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                                             random_state=seed, early_stopping=True),
    }


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.upper()
    if LABEL_COL not in df.columns:
        raise ValueError(f"Column '{LABEL_COL}' not found in {path}\n"
                         f"Available: {list(df.columns)}")
    # 'source' (real/synthetic) is an augmentation-provenance artifact, not a
    # clinical feature — drop it so the classifier can't key off it.
    return df.drop(columns=['SOURCE'], errors='ignore')


def preprocess_split(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit imputers/encoder/scaler on the training partition only.
    Returns unscaled and scaled train/test matrices (categorical columns first)."""
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == object]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    Xtr_parts, Xte_parts = [], []
    if cat_cols:
        imputer = SimpleImputer(strategy='most_frequent')
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        Xtr_parts.append(encoder.fit_transform(imputer.fit_transform(X_train[cat_cols].astype(str))))
        Xte_parts.append(encoder.transform(imputer.transform(X_test[cat_cols].astype(str))))
    if num_cols:
        imputer = SimpleImputer(strategy='median')
        Xtr_parts.append(imputer.fit_transform(X_train[num_cols].apply(pd.to_numeric, errors='coerce')))
        Xte_parts.append(imputer.transform(X_test[num_cols].apply(pd.to_numeric, errors='coerce')))

    Xtr = np.hstack(Xtr_parts).astype(float)
    Xte = np.hstack(Xte_parts).astype(float)
    scaler = StandardScaler()
    return Xtr, Xte, scaler.fit_transform(Xtr), scaler.transform(Xte)


# ── Training + evaluation ─────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, seed: int, classifiers=None) -> list:
    """One seeded stratified 80/20 split; returns one row of metrics per classifier."""
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    Xtr, Xte, Xtr_sc, Xte_sc = preprocess_split(X_train, X_test)
    ytr, yte = y_train.values, y_test.values

    rows = []
    for name, clf in get_classifiers(seed).items():
        if classifiers is not None and name not in classifiers:
            continue
        scaled = name in SCALE_REQUIRED
        clf.fit(Xtr_sc if scaled else Xtr, ytr)
        y_proba = clf.predict_proba(Xte_sc if scaled else Xte)[:, 1]
        rows.append({
            'seed': seed, 'Classifier': name,
            'roc_auc': roc_auc_score(yte, y_proba),
        })
    return rows


def run_pair(before: Path, after: Path, seeds=DEFAULT_SEEDS, classifiers=None) -> pd.DataFrame:
    """Per-seed, per-classifier metrics for before vs. after, paired by seed."""
    df_b, df_a = load_data(before), load_data(after)
    out = []
    for seed in seeds:
        print(f"    seed={seed}", flush=True)
        res_b = pd.DataFrame(evaluate(df_b, seed, classifiers))
        res_a = pd.DataFrame(evaluate(df_a, seed, classifiers))
        out.append(res_b.merge(res_a, on=['seed', 'Classifier'], suffixes=('_before', '_after')))
    cmp = pd.concat(out, ignore_index=True)
    for m in METRICS:
        cmp[f'{m}_diff'] = cmp[f'{m}_after'] - cmp[f'{m}_before']
    return cmp


def parse_args():
    p = argparse.ArgumentParser(
        description='Mortality prediction with explicit before/after paths',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--before', required=True, help='Path to before CSV')
    p.add_argument('--after',  required=True, help='Path to after CSV')
    p.add_argument('--label',  required=True,
                   help='Run label used in output filenames (e.g. mimic4_ctgan_tau12)')
    p.add_argument('--outdir', default='pred/out/ctgan', help='Output directory')
    p.add_argument('--seeds',  type=int, nargs='+', default=DEFAULT_SEEDS,
                   help='Split/classifier seeds')
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {args.label}\n  Before : {args.before}\n  After  : {args.after}")
    cmp = run_pair(Path(args.before), Path(args.after), args.seeds)

    csv_path = out_dir / f'{args.label}_model_comparison.csv'
    cmp.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    summary = cmp.groupby('Classifier')[['roc_auc_before', 'roc_auc_after', 'roc_auc_diff']].mean()
    print(f"\n  Mean AUROC over seeds {args.seeds}:")
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
