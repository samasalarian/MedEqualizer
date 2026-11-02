#!/usr/bin/env python3
"""
ros_baseline.py  –  MedEqualizer  |  Random Oversampling (ROS) baseline
════════════════════════════════════════════════════════════════════════

For each finalized configuration in reproducibility/final-configurations.json:

  augment   Build the ROS cohort: the real rows of the MedEqualizer augmented
            CSV plus, for every GENDER x ETHNICITY x AGE_GROUP subgroup, exactly
            as many real records (drawn from that subgroup with replacement) as
            MedEqualizer added to it.  Added rows get source='synthetic' so the
            generator input format is identical to the MedEqualizer cohort.
  generate  Retrain the generator on the Original (real.csv), ROS or
            MedEqualizer cohort with the finalized hyperparameters (same
            scripts and arguments as run_pipeline.py) for a given seed.
            --task i runs item i of the seed study (Slurm array index).
  evaluate  Label the 80 mortality x race x age x gender leaf subgroups of
            Original (before.csv), ROS and MedEqualizer (after_tau=...) with the
            repository's R log-disparity code and write one consolidated CSV.
  seeds     Seed study (42, 52, 62, 72, 82): GAN retraining variability with
            FIXED augmentation cohorts (augmentation itself is not rerun).
            One long CSV; prints mean/SD and paired MedEqualizer - ROS.
  plot      Bar charts of ROS vs MedEqualizer from the consolidated CSV, one PDF.
  selftest  Validate augment + evaluate on artificial data.

All outputs land in results/ros/.

Usage
─────
  python ros_baseline.py augment  --data_dir /path/to/data
  python ros_baseline.py generate --data_dir /path/to/data --config 0
  python ros_baseline.py evaluate --data_dir /path/to/data
  python ros_baseline.py generate --data_dir /path/to/data --task 0  
  python ros_baseline.py seeds    --data_dir /path/to/data
  python ros_baseline.py plot
  python ros_baseline.py selftest
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()
OUT_DIR = BASE_DIR / 'results' / 'ros'
CONFIG_JSON = BASE_DIR / 'reproducibility' / 'final-configurations.json'
SUNBURST_DIR = BASE_DIR / 'visualization' / 'sunburst_diagram'
CONDA_ENVS = Path(sys.executable).resolve().parents[2]

EQUITY_COLS = ['GENDER', 'ETHNICITY', 'AGE_GROUP']
CATEGORIES = ['Highly Underrepresented', 'Underrepresented', 'Equitable',
              'Overrepresented', 'Highly Overrepresented']
ARMS = ['original', 'ros', 'medequalizer']
SEEDS = [42, 52, 62, 72, 82]
SEED_DIR = OUT_DIR / 'seeds'
PARAM_SUFFIX = '0p2_auto_0p9_random_forest'

R_EVAL = r'''
args <- commandArgs(trailingOnly = TRUE)
sb_dir <- args[1]; dataset <- args[2]; ref_path <- args[3]; syn_path <- args[4]; out_csv <- args[5]
suppressMessages({library(dplyr); library(tidyr)})
options(lifecycle_verbosity = "quiet")
source(file.path(sb_dir, "scripts", "Equity_metrics.R"))
source(file.path(sb_dir, "scripts", "sunburst_process.R"))
for (e in parse(file.path(sb_dir, "create_sunburst.R"))) {
  if (is.call(e) && identical(e[[1]], as.name("<-")) && is.call(e[[3]]) &&
      identical(e[[3]][[1]], as.name("function"))) eval(e)
}
test_sunburst2 <- function(given_background, varList, given_rct, write_filename, sig_t, lower_t, upper_t) {
  write.csv(generate_sunburst_df(given_background, as.list(varList), given_rct, sig_t,
                                 lower_t, upper_t, Log_Disparate_Impact, FALSE), file = write_filename)
}
prep <- function(path) {
  d <- read.csv(path)
  if (!"MORTALITY" %in% names(d)) d$MORTALITY <- d$MORTALITY_WITHINTHIRTYDAYS
  d <- if (dataset == "mimic3") add_race_mimic3(d) else add_race_mimic4(d)
  add_mortality(add_age(d))
}
suppressMessages(preprocess_mimic_data(prep(ref_path), prep(syn_path),
                                       c("mortality", "RACE", "AGE", "GENDER"), out_csv))
'''


def load_configs():
    return json.loads(CONFIG_JSON.read_text())['experiments']


def run_name(cfg):
    return f"{cfg['dataset']}_{cfg['generator']}_tau{cfg['tau']}"


def paths(cfg, data_dir):
    cohort = 'Mimic3' if cfg['dataset'] == 'mimic3' else 'Mimic4'
    m, t = cfg['generator'], cfg['tau']
    d = Path(data_dir) / cohort
    return {
        'real':         d / 'real' / 'real.csv',
        'medeq_aug':    d / 'augmented' / m / f'{t}_{PARAM_SUFFIX}.csv',
        'original':     d / 'synthetic' / m / 'before.csv',
        'medequalizer': d / 'synthetic' / m / f'after_tau={t}_nu{PARAM_SUFFIX}.csv',
        'run_dir':      OUT_DIR / run_name(cfg),
    }


# ── ROS augmentation ───────────────────────────────────────────────────────────

def added_per_subgroup(aug: pd.DataFrame) -> pd.Series:
    # HealthGAN augmentation decodes a few '+81' records as '81' (lost sign).
    added = aug[aug['source'] == 'synthetic'].replace({'AGE_GROUP': {'81': '+81'}})
    return added.groupby(EQUITY_COLS).size().sort_index()


def random_oversample(medeq_aug: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Real rows + per-subgroup duplicates matching MedEqualizer's additions."""
    real = medeq_aug[medeq_aug['source'] == 'real']
    added = added_per_subgroup(medeq_aug)
    rng = np.random.default_rng(seed)
    real_groups = real.groupby(EQUITY_COLS)
    dups = []
    for key, k in added.items():
        pool = real_groups.get_group(key)
        dups.append(pool.iloc[rng.integers(0, len(pool), size=k)])
    dups = pd.concat(dups).assign(source='synthetic')
    return pd.concat([real, dups], ignore_index=True)


def cmd_augment(args):
    for i, cfg in enumerate(load_configs()):
        if args.config is not None and i != args.config:
            continue
        p = paths(cfg, args.data_dir)
        medeq = pd.read_csv(p['medeq_aug'], low_memory=False)
        ros = random_oversample(medeq, seed=args.seed)
        target, got = added_per_subgroup(medeq), added_per_subgroup(ros)
        assert target.equals(got), 'per-subgroup ROS counts do not match'
        p['run_dir'].mkdir(parents=True, exist_ok=True)
        ros.to_csv(p['run_dir'] / 'ros_augmented.csv', index=False)
        print(f"{run_name(cfg)}: {int((ros['source'] == 'real').sum())} real + "
              f"{int(got.sum())} ROS rows over {len(got)} subgroups")


# ── Generator retraining ───────────────────────────────────────────────────────

def seed_tasks():
    """(config index, arm, seed) for the seed study; ROS seed 42 reuses the main run."""
    return [(i, arm, s) for i in range(len(load_configs())) for arm in ARMS for s in SEEDS
            if not (arm == 'ros' and s == 42)]


def synthetic_path(cfg, arm, seed):
    if arm == 'ros' and seed == 42:
        return OUT_DIR / run_name(cfg) / 'ros_synthetic.csv'
    return SEED_DIR / run_name(cfg) / f'{arm}_seed{seed}.csv'


def cmd_generate(args):
    if args.task is not None:
        args.config, args.arm, args.seed = seed_tasks()[args.task]
    cfg = load_configs()[args.config]
    p = paths(cfg, args.data_dir)
    train_csv = {'original': p['real'], 'ros': p['run_dir'] / 'ros_augmented.csv',
                 'medequalizer': p['medeq_aug']}[args.arm]
    print(f'{run_name(cfg)} arm={args.arm} seed={args.seed}', flush=True)
    train_generator(cfg, train_csv, synthetic_path(cfg, args.arm, args.seed), args.seed,
                    epochs=args.epochs, n_samples=args.n_samples,
                    env_python=args.env_python, fenv_python=args.fenv_python)


def train_generator(cfg, train_csv, out_csv, seed, epochs=None, n_samples=100000,
                    env_python=None, fenv_python=None):
    """Train cfg's generator on train_csv with the finalized hyperparameters and
    write n_samples synthetic rows to out_csv (never overwrites)."""
    out_csv = Path(out_csv)
    if out_csv.exists():
        print(f'Exists, not overwriting: {out_csv}')
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    m, d = cfg['generator'], cfg['dataset']
    fenv_python = fenv_python or str(CONDA_ENVS / 'fenv' / 'bin' / 'python')
    common = ['--version', d, '--epochs', epochs or cfg['epochs'],
              '--n_samples', n_samples, '--seed', seed]

    tmp = tempfile.mkdtemp(dir=out_csv.parent)
    try:
        gen_dir = Path(tmp) / m
        shutil.copytree(BASE_DIR / 'synthetic_data_generation' / m, gen_dir,
                        ignore=shutil.ignore_patterns('gen_data', 'data', '__pycache__'))
        gen_out = gen_dir / 'gen_data' / d / 'synthetic_decoded.csv'
        if m == 'ctgan':
            gen_out = Path(tmp) / 'synthetic.csv'
            cmd = [env_python or sys.executable, gen_dir / 'ctgan_gen.py', '--data', train_csv,
                   *common, '--output_path', gen_out,
                   '--model_output_path', Path(tmp) / 'ctgan_model.pkl']
        elif m == 'healthgan':
            cmd = [fenv_python, gen_dir / 'healthgan_gen.py', '--data', train_csv, *common,
                   '--nodes', cfg['nodes'], '--critic', cfg['critic']]
        else:
            # The finalized MedGAN runs excluded `source` from the one-hot input.
            medgan_csv = Path(tmp) / 'train_no_source.csv'
            (pd.read_csv(train_csv, low_memory=False).drop(columns='source', errors='ignore')
             .to_csv(medgan_csv, index=False))
            cmd = [fenv_python, gen_dir / 'medgan_gen.py', '--data', medgan_csv, *common,
                   '--pretrain_epochs', cfg['pretrain_epochs'],
                   '--batch_size', cfg['final_batch_size']]

        cmd = [str(c) for c in cmd]
        print('$', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        shutil.copy(gen_out, Path(tmp) / 'final.csv')
        Path(tmp, 'final.csv').replace(out_csv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'Saved {out_csv}')


# ── Representation evaluation ──────────────────────────────────────────────────

def equity_labels(dataset, ref_csv, syn_csv, out_csv, rscript):
    with tempfile.NamedTemporaryFile('w', suffix='.R', delete=False) as f:
        f.write(R_EVAL)
    try:
        subprocess.run([rscript, f.name, str(SUNBURST_DIR), dataset,
                        str(ref_csv), str(syn_csv), str(out_csv)], check=True)
    finally:
        Path(f.name).unlink()
    return label_counts(out_csv)


def label_counts(plot_csv):
    """Leaf-level label counts, as in run_pipeline.py's equity summary."""
    df = pd.read_csv(plot_csv)
    leaves = df[~df['ids'].isin(df['parents'])]
    labels = leaves['EquityLable'].replace('Equitable(p)', 'Equitable')
    counts = {c: int((labels == c).sum()) for c in CATEGORIES}
    counts['other_labels'] = int((~labels.isin(CATEGORIES)).sum())
    counts['n_leaves'] = len(leaves)
    return counts


def cmd_evaluate(args):
    rscript = args.rscript or str(CONDA_ENVS / 'fenv' / 'bin' / 'Rscript')
    rows = []
    for cfg in load_configs():
        p = paths(cfg, args.data_dir)
        medeq_aug = pd.read_csv(p['medeq_aug'], usecols=['source'])
        n_added = int((medeq_aug['source'] == 'synthetic').sum())
        p['run_dir'].mkdir(parents=True, exist_ok=True)
        syn = {'original': p['original'], 'ros': p['run_dir'] / 'ros_synthetic.csv',
               'medequalizer': p['medequalizer']}
        for arm in ARMS:
            if not syn[arm].exists():
                print(f'MISSING {run_name(cfg)} {arm}: {syn[arm]}')
                continue
            counts = equity_labels(cfg['dataset'], p['real'], syn[arm],
                                   p['run_dir'] / f'plot_{arm}.csv', rscript)
            rows.append({'dataset': cfg['dataset'], 'generator': cfg['generator'],
                         'tau': cfg['tau'], 'arm': arm,
                         'records_added': 0 if arm == 'original' else n_added,
                         'n_synthetic': len(pd.read_csv(syn[arm], usecols=[0])),
                         **counts})
    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / 'ros_comparison.csv', index=False)
    wide = res.pivot_table(index=['dataset', 'generator', 'tau'], columns='arm',
                           values='Equitable').reindex(columns=ARMS)
    print(res.to_string(index=False))
    print('\nEquitably represented subgroups (of 80):')
    print(wide.to_string())
    print(f'\nSaved {OUT_DIR / "ros_comparison.csv"}')


def cmd_seeds(args):
    """Seed study: GAN retraining variability with FIXED augmentation cohorts.

    The Original (real.csv), ROS and MedEqualizer training cohorts are not
    regenerated; only the generator seed changes. Missing runs are reported as
    status='missing' and excluded from the summary, never imputed.
    """
    rscript = args.rscript or str(CONDA_ENVS / 'fenv' / 'bin' / 'Rscript')
    rows = []
    for cfg in load_configs():
        p = paths(cfg, args.data_dir)
        for arm in ARMS:
            for seed in SEEDS:
                syn = synthetic_path(cfg, arm, seed)
                row = {'dataset': cfg['dataset'], 'generator': cfg['generator'],
                       'tau': cfg['tau'], 'arm': arm, 'seed': seed,
                       'variability_source': 'gan_retraining_fixed_augmentation_cohort',
                       'reused_main_run': arm == 'ros' and seed == 42,
                       'synthetic_file': str(syn.relative_to(BASE_DIR))}
                if not syn.exists():
                    rows.append({**row, 'status': 'missing'})
                    print(f'MISSING {syn}')
                    continue
                plot_csv = SEED_DIR / run_name(cfg) / f'plot_{arm}_seed{seed}.csv'
                plot_csv.parent.mkdir(parents=True, exist_ok=True)
                counts = equity_labels(cfg['dataset'], p['real'], syn, plot_csv, rscript)
                rows.append({**row, 'status': 'ok',
                             'n_synthetic': len(pd.read_csv(syn, usecols=[0])), **counts})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / 'ros_seed_study.csv', index=False)

    ok = res[res['status'] == 'ok']
    keys = ['dataset', 'generator', 'tau']
    summary = ok.groupby(keys + ['arm'])['Equitable'].agg(['mean', 'std', 'count']).unstack('arm')
    paired = ok.pivot_table(index=keys + ['seed'], columns='arm', values='Equitable')
    paired = (paired['medequalizer'] - paired['ros']).dropna().rename('medeq_minus_ros')
    print('Equitably represented subgroups (of 80), mean / SD / n seeds:')
    print(summary.round(2).to_string())
    print('\nPaired MedEqualizer - ROS by seed:')
    print(paired.unstack('seed').to_string())
    print('\nPaired difference, mean / SD / n:')
    print(paired.groupby(keys).agg(['mean', 'std', 'count']).round(2).to_string())
    print(f'\n{(res["status"] == "missing").sum()} missing runs')
    print(f'Saved {OUT_DIR / "ros_seed_study.csv"}')


def cmd_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    res = pd.read_csv(OUT_DIR / 'ros_comparison.csv')
    arms = {'ros': ('ROS', '#2a78d6'), 'medequalizer': ('MedEqualizer', '#eb6834')}
    ink, muted = '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                         'xtick.color': muted, 'ytick.color': muted})
    configs = res[['dataset', 'generator', 'tau']].drop_duplicates().values.tolist()
    names = {'ctgan': 'CTGAN', 'healthgan': 'HealthGAN', 'medgan': 'MedGAN',
             'mimic3': 'MIMIC-III', 'mimic4': 'MIMIC-IV'}
    title = lambda d, g, t: f'{names[d]} · {names[g]} (τ={t})'

    def bars(ax, groups, values, width=0.38):
        x = np.arange(len(groups))
        for j, (arm, (label, color)) in enumerate(arms.items()):
            v = values[arm]
            bs = ax.bar(x + (j - 0.5) * width, v, width * 0.92, color=color, label=label, zorder=2)
            for b, val in zip(bs, v):
                if not np.isnan(val):
                    ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f'{int(val)}',
                            ha='center', va='bottom', fontsize=7, color=ink)
        ax.set_xticks(x)
        ax.yaxis.grid(True, color='#e6e5e0', linewidth=0.6, zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    def get(d, g, t, arm, col):
        r = res[(res.dataset == d) & (res.generator == g) & (res.tau == t) & (res.arm == arm)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    out = OUT_DIR / 'ros_vs_medequalizer.pdf'
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        bars(ax, configs, {a: [get(*c, a, 'Equitable') for c in configs] for a in arms})
        ax.set_xticklabels([title(*c).replace(' · ', '\n') for c in configs], fontsize=8)
        ax.set_ylim(0, 80)
        ax.set_ylabel('Equitably represented subgroups (of 80)')
        ax.set_title('Equitably represented intersectional subgroups: ROS vs MedEqualizer',
                     loc='left', fontsize=11, color=ink)
        ax.legend(frameon=False, loc='upper right', ncol=2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True)
        short = ['Highly\nunder', 'Under', 'Equitable', 'Over', 'Highly\nover']
        for ax, c in zip(axes.flat, configs):
            bars(ax, short, {a: [get(*c, a, k) for k in CATEGORIES] for a in arms})
            ax.set_xticklabels(short, fontsize=7.5)
            ax.set_title(title(*c), loc='left', fontsize=9.5, color=ink)
        for ax in axes[:, 0]:
            ax.set_ylabel('Subgroups (of 80)')
        axes[0, 0].set_ylim(0, max(res[CATEGORIES].max().max() + 6, 10))
        fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False,
                   loc='upper right', ncol=2)
        fig.suptitle('Representation categories by configuration (log disparity, 80 subgroups)',
                     x=0.01, ha='left', fontsize=11, color=ink)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)
    print(f'Saved {out}')


# ── Validation on artificial data ──────────────────────────────────────────────

def cmd_selftest(args):
    rscript = args.rscript or str(CONDA_ENVS / 'fenv' / 'bin' / 'Rscript')
    rng = np.random.default_rng(0)
    n = 20000
    real = pd.DataFrame({
        'SUBJECT_ID': np.arange(n),
        'GENDER': rng.choice(['M', 'F'], n),
        'MORTALITY_WITHINTHIRTYDAYS': rng.choice([0, 1], n, p=[0.8, 0.2]),
        'AGE_GROUP': rng.choice(['<= 45', '46-65', '66-80', '+81'], n),
        'ETHNICITY': rng.choice(['WHITE', 'BLACK/AFRICAN AMERICAN', 'ASIAN',
                                 'HISPANIC/LATINO - PUERTO RICAN', 'UNKNOWN/NOT SPECIFIED'],
                                n, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        'LOS': rng.exponential(3, n),
    })
    # Fake MedEqualizer output: synthetic rows in three subgroups.
    targets = {('F', 'ASIAN', '+81'): 7, ('M', 'UNKNOWN/NOT SPECIFIED', '<= 45'): 3,
               ('F', 'BLACK/AFRICAN AMERICAN', '66-80'): 11}
    fake = pd.concat([real[(real[EQUITY_COLS] == list(k)).all(axis=1)].sample(v, replace=True, random_state=1)
                      .assign(SUBJECT_ID=-1) for k, v in targets.items()])
    medeq = pd.concat([real.assign(source='real'), fake.assign(source='synthetic')], ignore_index=True)

    ros = random_oversample(medeq, seed=42)
    added = ros[ros['source'] == 'synthetic']
    assert len(ros) == len(medeq)
    assert ros[ros['source'] == 'real'].drop(columns='source').reset_index(drop=True).equals(real)
    assert {k: int(v) for k, v in added.groupby(EQUITY_COLS).size().items()} == targets
    assert added['SUBJECT_ID'].isin(real['SUBJECT_ID']).all(), 'ROS rows must be real records'
    assert random_oversample(medeq, seed=42).equals(ros), 'ROS must be deterministic'
    print('PASS  ROS: real rows preserved, per-subgroup counts matched, duplicates are real records')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        real.to_csv(tmp / 'real.csv', index=False)
        real.to_csv(tmp / 'same.csv', index=False)
        c = equity_labels('mimic3', tmp / 'real.csv', tmp / 'same.csv', tmp / 'p1.csv', rscript)
        assert c['Equitable'] == c['n_leaves'] == 80, c
        print('PASS  evaluate: identical data -> 80/80 leaves equitable')

        asian_f = (real['ETHNICITY'] == 'ASIAN') & (real['GENDER'] == 'F')
        skew = pd.concat([real[~asian_f], real[asian_f].sample(frac=0.2, random_state=0)])
        skew = pd.concat([skew, skew[skew['ETHNICITY'] == 'WHITE']])
        skew.to_csv(tmp / 'skew.csv', index=False)
        c = equity_labels('mimic3', tmp / 'real.csv', tmp / 'skew.csv', tmp / 'p2.csv', rscript)
        lab = pd.read_csv(tmp / 'p2.csv').set_index('ids')['EquityLable']
        assert lab['Alive - Asian - 46-65 - Female'] == 'Highly Underrepresented'
        assert lab['Alive - White - 46-65 - Male'] in ('Overrepresented', 'Highly Overrepresented')
        print(f'PASS  evaluate: downsampled/duplicated groups labeled as expected {c}')


def parse_args():
    p = argparse.ArgumentParser(description='MedEqualizer – Random Oversampling baseline',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('augment', 'generate', 'evaluate', 'seeds', 'plot', 'selftest'):
        s = sub.add_parser(name)
        s.add_argument('--data_dir', default=str(BASE_DIR / 'data'),
                       help='Folder holding Mimic3/ and Mimic4/ real, augmented and synthetic data')
        s.add_argument('--config', type=int, default=None,
                       help='Index into final-configurations.json experiments')
        s.add_argument('--seed', type=int, default=42)
        s.add_argument('--n_samples', type=int, default=100000)
        s.add_argument('--epochs', type=int, default=None,
                       help='Override finalized epochs (smoke tests only)')
        s.add_argument('--env_python', default=None, help='Python of the `env` conda env (CTGAN)')
        s.add_argument('--fenv_python', default=None, help='Python of the `fenv` conda env')
        s.add_argument('--rscript', default=None, help='Rscript with dplyr/tidyr/plyr')
        s.add_argument('--arm', choices=ARMS, default='ros', help='Training cohort for generate')
        s.add_argument('--task', type=int, default=None,
                       help='Seed-study task index (sets --config/--arm/--seed); see seed_tasks()')
    args = p.parse_args()
    if args.command == 'generate' and args.config is None and args.task is None:
        p.error('generate requires --config or --task')
    return args


def main():
    args = parse_args()
    {'augment': cmd_augment, 'generate': cmd_generate,
     'evaluate': cmd_evaluate, 'seeds': cmd_seeds, 'plot': cmd_plot, 'selftest': cmd_selftest}[args.command](args)


if __name__ == '__main__':
    main()
