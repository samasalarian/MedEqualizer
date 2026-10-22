#!/usr/bin/env python3
"""
create_barchart.py  –  Equity label distribution bar chart: before vs after augmentation

Reads the sunburst CSVs produced by create_sunburst.R, counts how many subgroups
fall in each EquityLable category, and produces a grouped bar chart matching the
Log Disparate Impact colour scheme used in the sunburst diagrams.

Usage
─────
  python visualization/sunburst_diagram/create_barchart.py \
      --before_csv visualization/results/ctgan/my_run/mimic4_ctgan_before_...csv \
      --after_csv  visualization/results/ctgan/my_run/mimic4_ctgan_after_...csv \
      --outdir     visualization/results/ctgan/my_run/ \
      --tag        mimic4_ctgan_tau150_nu0p3_auto_0p75_rf
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Category order and colours ─────────────────────────────────────────────────
# Exact colours from Equity_metrics.R → add_colors().
# Before = solid fill; After = same colour + hatch, so the equity meaning
# is always encoded by hue regardless of before/after.

CATEGORIES = [
    'Highly Overrepresented',
    'Highly Underrepresented',
    'Overrepresented',
    'Equitable',
    'Underrepresented',
]

COLORS = {
    'Highly Overrepresented':  '#00205b',
    'Highly Underrepresented': '#d58570',
    'Overrepresented':         '#a5b0cb',
    'Equitable':               '#d4e6e8',
    'Underrepresented':        '#eabcad',
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_count(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        print(f'  WARNING: {csv_path.name} not found — using zeros.')
        return pd.Series({cat: 0 for cat in CATEGORIES})
    df = pd.read_csv(csv_path)
    # Keep only leaf nodes — rows whose id never appears as a parent
    leaf_mask = ~df['ids'].isin(df['parents'])
    df = df[leaf_mask]
    print(f'  Leaf nodes: {len(df)} (of {len(leaf_mask)} total rows in {csv_path.name})')
    # Equitable(p) = not statistically significant → treat as Equitable
    df['EquityLable'] = df['EquityLable'].replace('Equitable(p)', 'Equitable')
    counts = df['EquityLable'].value_counts()
    return pd.Series({cat: int(counts.get(cat, 0)) for cat in CATEGORIES})


def _label_bar(ax, bar):
    h = bar.get_height()
    if h > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.3,
            str(int(h)),
            ha='center', va='bottom', fontsize=10,
        )


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(before_counts: pd.Series, after_counts: pd.Series,
         outdir: Path, tag: str) -> Path:
    x     = np.arange(len(CATEGORIES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = [COLORS[c] for c in CATEGORIES]

    for bar in ax.bar(
        x - width / 2,
        [before_counts[c] for c in CATEGORIES],
        width,
        color=colors,
        edgecolor='white',
        linewidth=0.5,
        label='Original Data',
    ):
        _label_bar(ax, bar)

    for bar in ax.bar(
        x + width / 2,
        [after_counts[c] for c in CATEGORIES],
        width,
        color=colors,
        alpha=0.5,
        edgecolor='white',
        linewidth=0.5,
        label='Augmented Data',
    ):
        _label_bar(ax, bar)

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=11)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(
        'Representitive Distribution via Log Disparity\n'
        'Left: Original Data, Right: Augmented Data',
        fontsize=13,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    tau_suffix = f'_tau{tag.split("tau")[-1].split("_")[0]}' if 'tau' in tag else ''
    out_path = outdir / f'barchart{tau_suffix}.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Bar chart saved: {out_path}')
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Equity label distribution bar chart: before vs after augmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--before_csv', required=True,
                   help='Sunburst CSV for before augmentation (from create_sunburst.R)')
    p.add_argument('--after_csv',  required=True,
                   help='Sunburst CSV for after augmentation (from create_sunburst.R)')
    p.add_argument('--outdir',     required=True, help='Output directory')
    p.add_argument('--tag',        default='run', help='Run tag for filename')
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('  Loading sunburst CSVs ...')
    before_counts = load_and_count(Path(args.before_csv))
    after_counts  = load_and_count(Path(args.after_csv))

    print('\n  Before augmentation — equity label counts:')
    print(before_counts.to_string())
    print('\n  After augmentation — equity label counts:')
    print(after_counts.to_string())

    plot(before_counts, after_counts, outdir, args.tag)


if __name__ == '__main__':
    main()
