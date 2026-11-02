#!/usr/bin/env python3
"""
mortality_data.py  –  MedEqualizer  |  Leakage-free inputs for pred/mortality_cls.py
═════════════════════════════════════════════════════════════════════════════════════

For each MIMIC dataset, one fixed patient-level, outcome-stratified 80/20 split
of real.csv (seed 42). Everything generative sees ONLY the real training split:

  original      GAN trained on real_train
  medequalizer  augmentation/cham.py (finalized tau/parameters) on real_train,
                then GAN trained on that augmented cohort

Six finalized configurations (reproducibility/final-configurations.json), GAN
seeds 42 52 62 72 82, 100,000 synthetic rows, finalized hyperparameters.
Outcome: MORTALITY = 1 when a date of death is recorded (source column
MORTALITY_WITHINTHIRTYDAYS; not in-hospital or 30-day mortality).

Steps
─────
  split     <out>/<dataset>/real_train.csv, real_test.csv
  augment   --config i : <out>/<config>/medeq_augmented.csv
  generate  --task k   : <out>/<config>/<arm>_seed<s>.csv  (k = 0..59)
  verify    patient-level leakage checks

Usage
─────
  python pred/mortality_data.py split --data_dir /path/to/data   # Mimic3/, Mimic4/ real/real.csv
  for i in 0 1 2 3 4 5; do python pred/mortality_data.py augment --config $i; done
  for k in $(seq 0 59); do python pred/mortality_data.py generate --task $k; done
  python pred/mortality_data.py verify

Run from the `env` conda environment; HealthGAN and MedGAN steps call the
`fenv` interpreter. GAN training does not use a GPU in the original runs, so
set CUDA_VISIBLE_DEVICES="" for comparable results. Outputs are never
overwritten. Patient-level outputs are private; keep them out of version control.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / 'reproducibility' / 'final-configurations.json').read_text())
CONFIGS, COMMON = CONFIG['experiments'], CONFIG['common']
LABEL = 'MORTALITY'
ARMS = ['original', 'medequalizer']
GAN_SEEDS = [42, 52, 62, 72, 82]
SPLIT_SEED = 42
N_SAMPLES = 100000
FENV_PYTHON = Path(sys.executable).resolve().parents[2] / 'fenv' / 'bin' / 'python'


def run_name(cfg):
    return f"{cfg['dataset']}_{cfg['generator']}_tau{cfg['tau']}"


def split_paths(out, dataset):
    return Path(out) / dataset / 'real_train.csv', Path(out) / dataset / 'real_test.csv'


def train_input(out, cfg, arm):
    return (split_paths(out, cfg['dataset'])[0] if arm == 'original'
            else Path(out) / run_name(cfg) / 'medeq_augmented.csv')


def tasks():
    return [(i, arm, s) for i in range(len(CONFIGS)) for arm in ARMS for s in GAN_SEEDS]


def run(cmd, dry_run):
    cmd = [str(c) for c in cmd]
    print('$', ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def private_copy(tmp, *parts):
    """Copy repo code into tmp: HealthGAN/MedGAN write to fixed paths inside their
    folder, so each run uses its own copy to allow parallel jobs."""
    for p in parts:
        shutil.copytree(REPO / p, Path(tmp) / p,
                        ignore=shutil.ignore_patterns('gen_data', 'data', '__pycache__'))


# ── Split ──────────────────────────────────────────────────────────────────────

def split_patients(real: pd.DataFrame):
    """Patient-level, outcome-stratified 80/20 split."""
    real = real.rename(columns={'MORTALITY_WITHINTHIRTYDAYS': LABEL})
    per_patient = real.groupby('SUBJECT_ID')[LABEL].max()
    train_ids, test_ids = train_test_split(per_patient.index, test_size=0.2,
                                           random_state=SPLIT_SEED, stratify=per_patient)
    return (real[real['SUBJECT_ID'].isin(train_ids)].reset_index(drop=True),
            real[real['SUBJECT_ID'].isin(test_ids)].reset_index(drop=True))


def cmd_split(args):
    for dataset in ('mimic3', 'mimic4'):
        tr_path, te_path = split_paths(args.out, dataset)
        if tr_path.exists() or te_path.exists():
            print(f'Exists, not overwriting: {tr_path.parent}')
            continue
        cohort = 'Mimic3' if dataset == 'mimic3' else 'Mimic4'
        train, test = split_patients(pd.read_csv(Path(args.data_dir) / cohort / 'real' / 'real.csv'))
        tr_path.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(tr_path, index=False)
        test.to_csv(te_path, index=False)
        print(f'{dataset}: train {len(train)} (mortality {train[LABEL].mean():.3f}), '
              f'test {len(test)} (mortality {test[LABEL].mean():.3f})')


# ── MedEqualizer augmentation of the training split ───────────────────────────

def cmd_augment(args):
    cfg = CONFIGS[args.config]
    m = cfg['generator']
    dst = train_input(args.out, cfg, 'medequalizer')
    if dst.exists():
        print(f'Exists, not overwriting: {dst}')
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    python = sys.executable if m == 'ctgan' else FENV_PYTHON
    # Same arguments run_pipeline.py passes to cham.py.
    extra = {'healthgan': ['--epochs', cfg['epochs'], '--aug_epochs', cfg['aug_epochs'],
                           '--nodes', cfg['nodes'], '--critic', cfg['critic']],
             'medgan': ['--epochs', cfg['epochs'], '--aug_epochs', cfg['aug_epochs']],
             'ctgan': []}[m]
    tmp = Path(tempfile.mkdtemp(dir=dst.parent))
    try:
        private_copy(tmp, 'augmentation', f'synthetic_data_generation/{m}')
        run([python, tmp / 'augmentation' / 'cham.py',
             '--dataset', cfg['dataset'], '--model', m, '--tau', cfg['tau'],
             '--nu', COMMON['nu'], '--gamma', COMMON['gamma'], '--auroc', COMMON['auroc'],
             '--classifier', COMMON['classifier'], '--workers', cfg['workers'],
             '--batch_size', COMMON['batch_size'], '--max_tries', cfg['max_tries'],
             '--max_attempts', COMMON['max_attempts'], '--seed', COMMON['aug_seed'],
             '--min_subgroup_size', COMMON['min_subgroup_size'], *extra,
             '--input_path', split_paths(args.out, cfg['dataset'])[0],
             '--output_path', tmp / 'medeq.csv', '--log_output_path', dst.with_suffix('.xlsx')],
            args.dry_run)
        if not args.dry_run:
            Path(tmp, 'medeq.csv').replace(dst)
            aug = pd.read_csv(dst, low_memory=False)
            print(f'{run_name(cfg)}: MedEqualizer added {int((aug.source == "synthetic").sum())} records')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── GAN training ───────────────────────────────────────────────────────────────

def cmd_generate(args):
    i, arm, seed = tasks()[args.task] if args.task is not None else (args.config, args.arm, args.seed)
    cfg = CONFIGS[i]
    m, d = cfg['generator'], cfg['dataset']
    out_csv = Path(args.out) / run_name(cfg) / f'{arm}_seed{seed}.csv'
    if out_csv.exists():
        print(f'Exists, not overwriting: {out_csv}')
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    train_csv = train_input(args.out, cfg, arm)
    common = ['--version', d, '--epochs', cfg['epochs'], '--n_samples', N_SAMPLES, '--seed', seed]
    print(f'{run_name(cfg)} arm={arm} gan_seed={seed}', flush=True)
    # mkdtemp + lenient cleanup: on NFS, TensorFlow's event writer can leave a
    # transient .nfs* file that makes a strict rmtree fail after a good run.
    tmp = Path(tempfile.mkdtemp(dir=out_csv.parent))
    try:
        private_copy(tmp, f'synthetic_data_generation/{m}')
        gen_dir = tmp / 'synthetic_data_generation' / m
        gen_out = gen_dir / 'gen_data' / d / 'synthetic_decoded.csv'
        if m == 'ctgan':
            gen_out = tmp / 'synthetic.csv'
            cmd = [sys.executable, gen_dir / 'ctgan_gen.py', '--data', train_csv, *common,
                   '--output_path', gen_out, '--model_output_path', tmp / 'ctgan_model.pkl']
        elif m == 'healthgan':
            cmd = [FENV_PYTHON, gen_dir / 'healthgan_gen.py', '--data', train_csv, *common,
                   '--nodes', cfg['nodes'], '--critic', cfg['critic']]
        else:
            # The finalized MedGAN runs excluded `source` from the one-hot input.
            medgan_csv = tmp / 'train_no_source.csv'
            if not args.dry_run:
                (pd.read_csv(train_csv, low_memory=False).drop(columns='source', errors='ignore')
                 .to_csv(medgan_csv, index=False))
            cmd = [FENV_PYTHON, gen_dir / 'medgan_gen.py', '--data', medgan_csv, *common,
                   '--pretrain_epochs', cfg['pretrain_epochs'], '--batch_size', cfg['final_batch_size']]
        run(cmd, args.dry_run)
        if not args.dry_run:
            shutil.copy(gen_out, tmp / 'final.csv')
            Path(tmp, 'final.csv').replace(out_csv)
            print(f'Saved {out_csv}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Leakage checks ─────────────────────────────────────────────────────────────

def cmd_verify(args):
    fails = []
    for dataset in ('mimic3', 'mimic4'):
        tr_path, te_path = split_paths(args.out, dataset)
        tr = set(pd.read_csv(tr_path, usecols=['SUBJECT_ID']).SUBJECT_ID)
        te = set(pd.read_csv(te_path, usecols=['SUBJECT_ID']).SUBJECT_ID)
        print(f'{dataset}: train {len(tr)} / test {len(te)} patients, overlap {len(tr & te)}')
        if tr & te:
            fails.append(f'{dataset}: {len(tr & te)} patients in both train and test')
        for cfg in (c for c in CONFIGS if c['dataset'] == dataset):
            f = train_input(args.out, cfg, 'medequalizer')
            if not f.exists():
                print(f'{run_name(cfg)}: no MedEqualizer cohort yet')
                continue
            aug = pd.read_csv(f, usecols=['SUBJECT_ID', 'source'])
            bad = []
            if set(aug[aug.source == 'real'].SUBJECT_ID) != tr:
                bad.append(f'{run_name(cfg)}: real rows are not exactly the training patients')
            if te & set(aug.SUBJECT_ID.dropna()):
                bad.append(f'{run_name(cfg)}: contains test patients')
            print(f'{run_name(cfg)}: MedEqualizer cohort {"OK" if not bad else "FAIL"}')
            fails += bad
    print('LEAKAGE CHECKS PASSED' if not fails else 'LEAKAGE CHECKS FAILED:\n' + '\n'.join(fails))
    if fails:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description='Create leakage-free inputs for pred/mortality_cls.py',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('split', 'augment', 'generate', 'verify'):
        s = sub.add_parser(name)
        s.add_argument('--out', default=str(REPO / 'results' / 'tstr_trtr'),
                       help='Output folder (pred/mortality_cls.py reads results/tstr_trtr)')
        s.add_argument('--data_dir', default=str(REPO / 'data'),
                       help='split: folder with Mimic3/ and Mimic4/ real/real.csv')
        s.add_argument('--config', type=int, help='Index into final-configurations.json (0-5)')
        s.add_argument('--arm', choices=ARMS, default='original')
        s.add_argument('--seed', type=int, default=42, help='GAN seed')
        s.add_argument('--task', type=int, help='generate: index 0-59 into (config, arm, GAN seed)')
        s.add_argument('--dry_run', action='store_true', help='augment/generate: print the command only')
    args = p.parse_args()
    if args.command == 'augment' and args.config is None:
        p.error('augment requires --config')
    if args.command == 'generate' and args.task is None and args.config is None:
        p.error('generate requires --task or --config')
    {'split': cmd_split, 'augment': cmd_augment, 'generate': cmd_generate, 'verify': cmd_verify}[args.command](args)


if __name__ == '__main__':
    main()
