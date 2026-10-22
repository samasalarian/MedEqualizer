#!/usr/bin/env python3
"""
mortality_classification.py
===========================

Builds the manuscript mortality-classification AUROC table: synthetic data
generated from the original cohort vs. from the MedEqualizer-augmented cohort,
at each dataset/generator pair's final tau:

    MIMIC-III: MedGAN tau=8,  HealthGAN tau=14, CTGAN tau=12
    MIMIC-IV:  MedGAN tau=14, HealthGAN tau=8,  CTGAN tau=12

All prediction is done by pred/run_pred.py (5 seeds, stratified 80/20 split,
train-only preprocessing). This script only selects the classifiers and
summarises: Delta = augmented - original, paired by seed, with a t-based 95% CI
over the paired differences: mean +/- t(0.975, df=n-1) * SD / sqrt(n).

Output: pred/mortality_auroc_table.csv

Usage (from the repo root):
    python pred/mortality_classification.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'pred'))
from run_pred import DEFAULT_SEEDS, run_pair

OUT_CSV = Path(__file__).resolve().parent / 'mortality_auroc_table.csv'

DATASETS   = ['MIMIC-III', 'MIMIC-IV']
GENERATORS = ['MedGAN', 'HealthGAN', 'CTGAN']
CLASSIFIER_ORDER = ['Decision Tree', 'KNN', 'Gradient Boosting', 'Hist GB', 'MLP']
FINAL_TAU = {
    ('MIMIC-III', 'MedGAN'): 8, ('MIMIC-III', 'HealthGAN'): 14, ('MIMIC-III', 'CTGAN'): 12,
    ('MIMIC-IV', 'MedGAN'): 14, ('MIMIC-IV', 'HealthGAN'): 8, ('MIMIC-IV', 'CTGAN'): 12,
}
DATA_DIR = {'MIMIC-III': 'Mimic3', 'MIMIC-IV': 'Mimic4'}


def synthetic_paths(dataset: str, generator: str, tau: int):
    base = REPO_ROOT / 'data' / DATA_DIR[dataset] / 'synthetic' / generator.lower()
    return base / 'before.csv', base / f'after_tau={tau}_nu0p2_auto_0p9_random_forest.csv'


def summarise(cmp: pd.DataFrame, clf: str) -> dict:
    g = cmp[cmp['Classifier'] == clf]
    orig, aug, diff = g['roc_auc_before'], g['roc_auc_after'], g['roc_auc_diff']
    n = len(diff)
    half = stats.t.ppf(0.975, df=n - 1) * diff.std(ddof=1) / np.sqrt(n)
    return dict(
        AUROC_original_mean=round(orig.mean(), 3), AUROC_original_sd=round(orig.std(ddof=1), 3),
        AUROC_augmented_mean=round(aug.mean(), 3), AUROC_augmented_sd=round(aug.std(ddof=1), 3),
        Delta_AUROC=round(diff.mean(), 3),
        CI_lower=round(diff.mean() - half, 3), CI_upper=round(diff.mean() + half, 3),
    )


def main():
    t0 = time.time()
    rows = []
    for dataset in DATASETS:
        for generator in GENERATORS:
            tau = FINAL_TAU[(dataset, generator)]
            before, after = synthetic_paths(dataset, generator, tau)
            print(f'{dataset} / {generator} / tau={tau}', flush=True)
            cmp = run_pair(before, after, DEFAULT_SEEDS, CLASSIFIER_ORDER)
            for clf in CLASSIFIER_ORDER:
                rows.append(dict(Dataset=dataset, Generator=generator, Tau=tau, Classifier=clf,
                                 **summarise(cmp, clf)))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}  ({len(rows)} rows)  in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
