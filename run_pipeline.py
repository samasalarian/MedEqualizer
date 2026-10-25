#!/usr/bin/env python3
"""
run_pipeline.py  –  MedEqualizer  |  End-to-end augmentation + visualization pipeline
══════════════════════════════════════════════════════════════════════════════════════

Steps
─────
  1. Augment real data           →  augmentation/cham.py
  2. Generate "after" synthetic  →  synthetic_data_generation/ctgan/ctgan_gen.py
  3. Sunburst — before           →  visualization/sunburst_diagram/create_sunburst.R
  4. Sunburst — after            →  visualization/sunburst_diagram/create_sunburst.R
  5. Subgroup bar chart          →  visualization/sunburst_diagram/create_barchart.py

All outputs for a given parameter set land in:
  visualization/results/<model>/<tag>/

where <tag> encodes the augmentation parameters so runs never overwrite each other.

Usage examples
──────────────
  # Minimal (required args only)
  python run_pipeline.py --dataset mimic3 --model ctgan --tau 150

  # Full parameter spec
  python run_pipeline.py \
      --dataset mimic3 --model ctgan --tau 150 \
      --nu 0.3 --gamma auto --auroc 0.75 --classifier random_forest \
      --workers 4 --epochs 300 --n_samples 100000 \
      --aug_seed 42 --gen_seed 42

  # Skip augmentation (augmented CSV already exists) and regenerate visuals only
  python run_pipeline.py --dataset mimic3 --model ctgan --tau 150 \
      --skip_augment --skip_gen


"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


MODELS_IMPLEMENTED = {'ctgan', 'healthgan', 'medgan'}

CLASSIFIER_ABBR = {
    'random_forest':      'rf',
    'logistic_regression': 'lr',
    'gradient_boosting':  'gb',
    'decision_tree':      'dt',
    'svm':                'svm',
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dot_to_p(value) -> str:
    return str(value).replace('.', 'p')


def build_tag(args) -> str:
    return (
        f"{args.dataset}_{args.model}"
        f"_tau{args.tau}"
        f"_nu{_dot_to_p(args.nu)}"
        f"_{args.gamma}"
        f"_{_dot_to_p(args.auroc)}"
        f"_{CLASSIFIER_ABBR.get(args.classifier, args.classifier)}"
    )


def run_step(cmd: list, label: str, cwd: Path = None) -> None:
    print(f'\n{"═" * 64}')
    print(f'  {label}')
    print(f'  $ {" ".join(str(c) for c in cmd)}')
    print(f'{"═" * 64}')
    result = subprocess.run([str(c) for c in cmd],
                            cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        print(f'\nERROR: step failed (exit {result.returncode})')
        sys.exit(result.returncode)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='MedEqualizer – end-to-end augmentation + visualization pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset / model
    p.add_argument('--dataset', required=True, choices=['mimic3', 'mimic4'],
                   help='Which MIMIC dataset to use')
    p.add_argument('--model',   required=True,
                   choices=['ctgan', 'healthgan', 'medgan'],
                   help='Generative model for augmentation and synthesis')

    # ── Augmentation (cham.py)
    aug = p.add_argument_group('Augmentation (cham.py)')
    aug.add_argument('--tau',          required=True, type=int,
                     help='Representation threshold — groups below tau are augmented')
    aug.add_argument('--nu',           type=float, default=0.05,
                     help='OCSVM outlier fraction bound')
    aug.add_argument('--gamma',        type=str,   default='scale',
                     help="OCSVM gamma: 'scale', 'auto', or a float")
    aug.add_argument('--auroc',        type=float, default=0.75,
                     help='Gate-2 AUROC acceptance threshold')
    aug.add_argument('--classifier',   default='random_forest',
                     choices=['random_forest', 'logistic_regression',
                              'gradient_boosting', 'decision_tree', 'svm'],
                     help='Gate-2 classifier')
    aug.add_argument('--workers',      type=int, default=1,
                     help='Parallel subgroup workers for cham.py')
    aug.add_argument('--batch_size',   type=int, default=500)
    aug.add_argument('--max_tries',    type=int, default=100)
    aug.add_argument('--max_attempts', type=int, default=10)
    aug.add_argument('--aug_epochs',   type=int, default=1000,
                     help='HealthGAN epochs per subgroup augmentation attempt')
    aug.add_argument('--aug_seed',           type=int, default=42,
                     help='Random seed for augmentation (cham.py)')
    aug.add_argument('--min_subgroup_size', type=int, default=10,
                     help='Minimum real rows in a subgroup to attempt augmentation')

    # ── Synthesis (ctgan_gen.py / healthgan_gen.py)
    gen = p.add_argument_group('Synthesis (ctgan_gen.py / healthgan_gen.py)')
    gen.add_argument('--epochs',    type=int, default=300,
                     help='Training epochs (CTGAN: 300, HealthGAN: 10000)')
    gen.add_argument('--n_samples', type=int, default=100000,
                     help='Synthetic rows to generate in final synthesis step')
    gen.add_argument('--nodes',     type=int, default=64,
                     help='Base nodes for HealthGAN architecture')
    gen.add_argument('--critic',    type=int, default=5,
                     help='Critic iterations for HealthGAN')
    gen.add_argument('--gen_seed',  type=int, default=42,
                     help='Random seed for synthesis')

    # ── Step skipping
    skip = p.add_argument_group('Skip steps')
    skip.add_argument('--skip_augment',  action='store_true',
                      help='Skip step 1 — assume augmented CSV already exists')
    skip.add_argument('--force_augment', action='store_true',
                      help='Rerun augmentation even when its output CSV exists')
    skip.add_argument('--skip_gen',      action='store_true',
                      help='Skip step 2 — assume after.csv already exists')
    skip.add_argument('--skip_sunburst', action='store_true',
                      help='Skip steps 3-4 — do not run Rscript')
    skip.add_argument('--skip_barchart', action='store_true',
                      help='Skip step 5 — do not generate bar chart')

    # ── Categorized ethnicity mode
    p.add_argument('--categorized', action='store_true',
                   help='Use real_categorized.csv (5-group ethnicity) and store '
                        'outputs under {model}_cat directories')

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.model not in MODELS_IMPLEMENTED:
        print(f"ERROR: '{args.model}' is not yet implemented. "
              f"Supported: {sorted(MODELS_IMPLEMENTED)}")
        sys.exit(1)

    base_dir    = Path(__file__).parent.resolve()
    dataset_cap = 'Mimic3' if args.dataset == 'mimic3' else 'Mimic4'
    tag         = build_tag(args)

    # When --categorized, all intermediate/output paths use {model}_cat
    model_dir = f"{args.model}_cat" if args.categorized else args.model

    # ── Upfront checks ────────────────────────────────────────────────────────
    if args.min_subgroup_size >= args.tau:
        print(f'\nERROR: --min_subgroup_size ({args.min_subgroup_size}) >= --tau ({args.tau}).')
        print('  No subgroup can ever be augmented. Lower min_subgroup_size or raise tau.')
        sys.exit(1)

    # ── Key paths ──────────────────────────────────────────────────────────────
    synth_dir   = base_dir / 'data' / dataset_cap / 'synthetic' / model_dir
    before_csv  = synth_dir / ('before_categorized.csv' if args.categorized else 'before.csv')

    nu_s    = _dot_to_p(args.nu)
    auroc_s = _dot_to_p(args.auroc)
    gamma_s = _dot_to_p(args.gamma)
    param_suffix = f'nu{nu_s}_{gamma_s}_{auroc_s}_{args.classifier}'
    after_csv = synth_dir / f'after_tau={args.tau}_{param_suffix}.csv'
    aug_csv = (base_dir / 'data' / dataset_cap / 'augmented' / model_dir /
               f'{args.tau}_{nu_s}_{gamma_s}_{auroc_s}_{args.classifier}.csv')

    # Gen script and its fixed output path — model-dependent
    if args.model == 'ctgan':
        gen_script = base_dir / 'synthetic_data_generation' / 'ctgan' / 'ctgan_gen.py'
        gen_out    = base_dir / 'synthetic_data_generation' / 'ctgan' / 'gen_data' / args.dataset / 'synthetic.csv'
    elif args.model == 'healthgan':
        gen_script = base_dir / 'synthetic_data_generation' / 'healthgan' / 'healthgan_gen.py'
        gen_out    = base_dir / 'synthetic_data_generation' / 'healthgan' / 'gen_data' / args.dataset / 'synthetic_decoded.csv'
    elif args.model == 'medgan':
        gen_script = base_dir / 'synthetic_data_generation' / 'medgan' / 'medgan_gen.py'
        gen_out    = base_dir / 'synthetic_data_generation' / 'medgan' / 'gen_data' / args.dataset / 'synthetic_decoded.csv'

    out_dir = base_dir / 'visualization' / 'results' / model_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Log all params (including gen_seed) ────────────────────────────────────
    params_log = {**vars(args), 'tag': tag, 'aug_csv': str(aug_csv), 'out_dir': str(out_dir)}
    (out_dir / 'params.json').write_text(json.dumps(params_log, indent=2))

    # ── Scripts ────────────────────────────────────────────────────────────────
    cham_script      = base_dir / 'augmentation' / 'cham.py'
    sunburst_script  = base_dir / 'visualization' / 'sunburst_diagram' / 'create_sunburst.R'
    barchart_script  = base_dir / 'visualization' / 'sunburst_diagram' / 'create_barchart.py'

    print('=' * 64)
    print('  MedEqualizer Pipeline')
    print('=' * 64)
    print(f'  Dataset    : {args.dataset}')
    print(f'  Model      : {args.model}')
    print(f'  Tag        : {tag}')
    print(f'  Output dir : {out_dir}')
    print('=' * 64)

    # ── Step 1: Augment real data ──────────────────────────────────────────────
    if args.skip_augment:
        print('\n  [Step 1/5 skipped — --skip_augment]')
    elif aug_csv.exists() and not args.force_augment:
        print(f'\n  [Step 1/5 skipped — augmented CSV already exists]')
        print(f'  {aug_csv}')
    else:
        aug_cmd = [sys.executable, cham_script,
                   '--dataset',      args.dataset,
                   '--model',        args.model,
                   '--tau',          args.tau,
                   '--nu',           args.nu,
                   '--gamma',        args.gamma,
                   '--auroc',        args.auroc,
                   '--classifier',   args.classifier,
                   '--workers',      args.workers,
                   '--batch_size',   args.batch_size,
                   '--max_tries',    args.max_tries,
                   '--max_attempts',       args.max_attempts,
                   '--seed',               args.aug_seed,
                   '--min_subgroup_size',  args.min_subgroup_size,
                   ]
        if args.categorized:
            aug_cmd.append('--categorized')
        if args.model == 'healthgan':
            aug_cmd += ['--epochs',     args.epochs,
                        '--aug_epochs', args.aug_epochs,
                        '--nodes',      args.nodes,
                        '--critic',     args.critic]
        elif args.model == 'medgan':
            aug_cmd += ['--epochs',     args.epochs,
                        '--aug_epochs', args.aug_epochs]
        label = 'Step 1/5 — Augmenting real data (cham.py)'
        print(f'\n{"═" * 64}')
        print(f'  {label}')
        print(f'  $ {" ".join(str(c) for c in aug_cmd)}')
        print(f'{"═" * 64}')
        result = subprocess.run([str(c) for c in aug_cmd])
        if result.returncode == 2:
            print('\n  No underrepresented groups found for this tau — pipeline stopped.')
            print(f'  Try a higher --tau value (current: {args.tau}).')
            sys.exit(0)
        elif result.returncode != 0:
            print(f'\nERROR: step failed (exit {result.returncode})')
            sys.exit(result.returncode)

    if not aug_csv.exists():
        print(f'\nERROR: augmented CSV not found: {aug_csv}')
        print('  Run without --skip_augment or check cham.py output.')
        sys.exit(1)

    # ── Zero-augmentation check ────────────────────────────────────────────────
    real_csv = (base_dir / 'data' / dataset_cap / 'real' /
                ('real_categorized.csv' if args.categorized else 'real.csv'))
    if real_csv.exists():
        with open(real_csv) as f:
            real_rows = sum(1 for _ in f) - 1
        with open(aug_csv) as f:
            aug_rows = sum(1 for _ in f) - 1
        added = aug_rows - real_rows
        if added <= 0:
            print(f'\n  No synthetic rows were accepted during augmentation ({added} added).')
            print('  Every targeted subgroup failed the OCSVM/AUROC quality gates.')
            print('  Skipping steps 2-5 — nothing to synthesize or visualize.')
            sys.exit(0)

    # ── Step 2: Generate "after" synthetic data ────────────────────────────────
    if not args.skip_gen:
        if args.model == 'ctgan':
            gen_cmd = [sys.executable, gen_script,
                       '--data',      aug_csv,
                       '--version',   args.dataset,
                       '--epochs',    args.epochs,
                       '--n_samples', args.n_samples,
                       '--seed',      args.gen_seed]
        elif args.model == 'healthgan':
            gen_cmd = [sys.executable, gen_script,
                       '--data',      aug_csv,
                       '--version',   args.dataset,
                       '--epochs',    args.epochs,
                       '--nodes',     args.nodes,
                       '--critic',    args.critic,
                       '--n_samples', args.n_samples,
                       '--seed',      args.gen_seed]
        elif args.model == 'medgan':
            gen_cmd = [sys.executable, gen_script,
                       '--data',      aug_csv,
                       '--version',   args.dataset,
                       '--epochs',    args.epochs,
                       '--n_samples', args.n_samples,
                       '--seed',      args.gen_seed]

        run_step(gen_cmd,
                 label=f'Step 2/5 — Generating "after" synthetic data ({args.model}_gen.py)')

        synth_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(gen_out, after_csv)
        print(f'\n  Copied  {gen_out}')
        print(f'       →  {after_csv}')

        after_params_file = after_csv.with_suffix('').with_suffix('.params.json')
        after_params_file.write_text(json.dumps({
            'dataset':     args.dataset,
            'model':       args.model,
            'categorized': args.categorized,
            'tau':         args.tau,
            'nu':          args.nu,
            'gamma':       args.gamma,
            'auroc':       args.auroc,
            'classifier':  args.classifier,
            'aug_csv':     str(aug_csv),
            'epochs':      args.epochs,
            'n_samples':   args.n_samples,
            'seed':        args.gen_seed,
        }, indent=2))
        print(f'       →  {after_params_file}')
    else:
        print('\n  [Step 2/5 skipped — --skip_gen]')

    if not after_csv.exists():
        print(f'\nERROR: {after_csv.name} not found: {after_csv}')
        print(f'  Run without --skip_gen or check {args.model}_gen.py output.')
        sys.exit(1)

    # ── before.csv check ──────────────────────────────────────────────────────
    before_available = before_csv.exists()
    if not before_available:
        before_name = before_csv.name
        print(f'\nWARNING: {before_name} not found at {before_csv}')
        print('  Step 3 (before sunburst) will be skipped.')

    # ── Step 3: Sunburst — before ──────────────────────────────────────────────
    if not args.skip_sunburst and before_available:
        r_before_cmd = ['Rscript', sunburst_script,
                        '--data',    args.dataset,
                        '--model',   model_dir,
                        '--version', 'before']
        if args.categorized:
            r_before_cmd.append('--categorized')
        run_step(r_before_cmd, cwd=out_dir,
                 label='Step 3/5 — Sunburst: before augmentation')
    else:
        reason = '--skip_sunburst' if args.skip_sunburst else 'before.csv missing'
        print(f'\n  [Step 3/5 skipped — {reason}]')

    # ── Step 4: Sunburst — after ───────────────────────────────────────────────
    if not args.skip_sunburst:
        r_after_cmd = ['Rscript', sunburst_script,
                       '--data',       args.dataset,
                       '--model',      model_dir,
                       '--version',    'after',
                       '--tau',        args.tau,
                       '--after_file', after_csv.name]
        if args.categorized:
            r_after_cmd.append('--categorized')
        run_step(r_after_cmd, cwd=out_dir,
                 label='Step 4/5 — Sunburst: after augmentation')
    else:
        print('\n  [Step 4/5 skipped — --skip_sunburst]')

    # ── Step 5: Equity label bar chart ────────────────────────────────────────
    sunburst_csv_before = out_dir / 'plot_before.csv'
    sunburst_csv_after  = out_dir / 'plot_after.csv'

    if not args.skip_barchart:
        if not sunburst_csv_before.exists():
            print(f'\nWARNING: before sunburst CSV not found — bar chart "before" bars will be empty.')
        if not sunburst_csv_after.exists():
            print(f'\nERROR: after sunburst CSV not found: {sunburst_csv_after}')
            sys.exit(1)
        run_step(
            [sys.executable, barchart_script,
             '--before_csv', sunburst_csv_before,
             '--after_csv',  sunburst_csv_after,
             '--outdir',     out_dir,
             '--tag',        tag,
             ],
            label='Step 5/5 — Equity label distribution bar chart',
        )
    else:
        print('\n  [Step 5/5 skipped — --skip_barchart]')

    # ── Equity summary table ──────────────────────────────────────────────────
    CATEGORIES = [
        'Highly Overrepresented', 'Highly Underrepresented',
        'Overrepresented', 'Equitable', 'Underrepresented',
    ]
    if sunburst_csv_before.exists() and sunburst_csv_after.exists():
        import pandas as pd
        def _equity_counts(path):
            df = pd.read_csv(path)
            df = df[~df['ids'].isin(df['parents'])]
            df['EquityLable'] = df['EquityLable'].replace('Equitable(p)', 'Equitable')
            c = df['EquityLable'].value_counts()
            return {cat: int(c.get(cat, 0)) for cat in CATEGORIES}

        bc = _equity_counts(sunburst_csv_before)
        ac = _equity_counts(sunburst_csv_after)
        col_w = 26
        print('\n' + '═' * 64)
        print('  Equity Label Summary  (leaf nodes only)')
        print('═' * 64)
        print(f"  {'Label':<{col_w}} {'Before':>7} {'After':>7} {'Change':>8}")
        print(f"  {'-'*col_w} {'-'*7} {'-'*7} {'-'*8}")
        for cat in CATEGORIES:
            b, a = bc[cat], ac[cat]
            d = a - b
            arrow = f'+{d}' if d > 0 else str(d)
            print(f"  {cat:<{col_w}} {b:>7} {a:>7} {arrow:>8}")
        print(f"  {'-'*col_w} {'-'*7} {'-'*7} {'-'*8}")
        eq_change = ac['Equitable'] - bc['Equitable']
        direction = 'IMPROVED' if eq_change > 0 else ('WORSENED' if eq_change < 0 else 'NO CHANGE')
        print(f"  Equitable subgroups: {bc['Equitable']} → {ac['Equitable']}  ({direction})")
        print('═' * 64)

    print('\n' + '=' * 64)
    print(f'  Pipeline complete.')
    print(f'  Results → {out_dir}')
    print('=' * 64)


if __name__ == '__main__':
    main()
