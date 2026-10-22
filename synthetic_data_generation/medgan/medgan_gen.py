"""
MedGAN Synthetic Data Generator
---------------------------------
Takes an augmented CSV, OneHotEncodes all categorical columns into a binary
matrix, trains MedGAN, generates synthetic data, decodes back to a DataFrame,
and saves as synthetic_decoded.csv.

LOS is continuous and not supported by MedGAN — it is excluded during training
and re-sampled from the augmented dataset's LOS distribution after decoding.

Usage:
    python medgan_gen.py --data path/to/augmented.csv --version mimic4

Output:
    synthetic_data_generation/medgan/gen_data/<version>/synthetic_decoded.csv
"""

import os
import sys
import argparse
import subprocess
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# Columns to drop before MedGAN training (identifiers + continuous)
DROP_COLS  = ['SUBJECT_ID', 'HADM_ID', 'ANCHOR_AGE']
CONTINUOUS = ['LOS']


def flush(msg):
    print(msg, flush=True)


def run_pipeline(data_path: str, version: str, epochs: int, n_samples: int,
                 pretrain_epochs: int, batch_size: int, seed: int):

    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    gen_dir  = os.path.join(base_dir, 'gen_data', version)
    os.makedirs(gen_dir, exist_ok=True)

    flush(f"\n{'='*60}")
    flush(f"  MedGAN pipeline: {version.upper()}  |  File: {data_path}")
    flush(f"{'='*60}\n")
    flush(f"  Epochs          : {epochs}")
    flush(f"  Pretrain epochs : {pretrain_epochs}")
    flush(f"  N samples       : {n_samples}")
    flush(f"  Batch size      : {batch_size}")
    flush(f"  Seed            : {seed}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = pd.read_csv(data_path)
    flush(f"[1/5] Loaded data -- shape: {data.shape}")

    drop_cols = [c for c in DROP_COLS if c in data.columns]
    los_values = data['LOS'].dropna().values if 'LOS' in data.columns else None

    data_cat = data.drop(columns=drop_cols + [c for c in CONTINUOUS if c in data.columns])
    flush(f"      Categorical columns for MedGAN: {list(data_cat.columns)}\n")

    # ── 2. OneHotEncode → binary matrix ──────────────────────────────────────
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    mat = encoder.fit_transform(data_cat.astype(str)).astype(np.float32)
    flush(f"[2/5] Encoded matrix shape: {mat.shape}")

    mat_path   = os.path.join(gen_dir, 'input_matrix.npy')
    model_path = os.path.join(gen_dir, 'model', 'output')
    synth_path = os.path.join(gen_dir, 'synthetic_output')

    os.makedirs(os.path.join(gen_dir, 'model'), exist_ok=True)
    pickle.dump(mat.astype(int), open(mat_path, 'wb'), -1)
    flush(f"      Saved binary matrix -> {mat_path}\n")

    medgan_script = os.path.join(base_dir, 'medgan.py')

    # ── 3. Train MedGAN ───────────────────────────────────────────────────────
    flush(f"[3/5] Training MedGAN ...")
    train_cmd = [
        sys.executable, medgan_script,
        mat_path, model_path,
        '--data_type', 'binary',
        '--n_epoch', str(epochs),
        '--n_pretrain_epoch', str(pretrain_epochs),
        '--batch_size', str(batch_size),
        '--save_max_keep', '1',
    ]
    flush(f"  $ {' '.join(train_cmd)}\n")
    result = subprocess.run(train_cmd, cwd=base_dir)
    if result.returncode != 0:
        raise RuntimeError(f"MedGAN training failed (exit {result.returncode})")

    # Find the saved model checkpoint (last epoch)
    model_file = f"{model_path}-{epochs - 1}"
    flush(f"\n      Model file: {model_file}")

    # ── 4/5. Generate + decode, over-requesting until n_samples valid rows ────
    # MedGAN's binary output doesn't always put exactly one "1" in each
    # one-hot column block (it can be all-zero or multi-hot); any row with
    # even one ambiguous block gets dropped by encoder.inverse_transform, so
    # a plain single-shot request always falls short of n_samples by a
    # random amount. Over-request and retry with a growing budget until we
    # actually have n_samples valid decoded rows, then truncate to exactly
    # that. generateData() uses a fixed internal seed, so a larger request
    # is a deterministic superset of a smaller one -- no duplicate/overlap
    # risk from regenerating with a bigger --n_samples each round.
    flush(f"\n[4/5] Generating {n_samples} synthetic samples ...")
    gen_batch_size = min(batch_size, mat.shape[0])
    request        = int(n_samples * 1.30)
    max_rounds     = 3
    decoded        = np.empty((0, len(data_cat.columns)), dtype=object)

    for round_i in range(1, max_rounds + 1):
        gen_cmd = [
            sys.executable, medgan_script,
            mat_path, synth_path,
            '--data_type', 'binary',
            '--generate_data', 'True',
            '--model_file', model_file,
            '--batch_size', str(gen_batch_size),
            '--n_samples', str(request),
        ]
        flush(f"  [round {round_i}] requesting {request} raw samples")
        flush(f"  $ {' '.join(gen_cmd)}\n")
        result = subprocess.run(gen_cmd, cwd=base_dir)
        if result.returncode != 0:
            raise RuntimeError(f"MedGAN generation failed (exit {result.returncode})")

        synth_npy = synth_path + '.npy'
        if not os.path.exists(synth_npy):
            raise FileNotFoundError(f"Expected synthetic output at {synth_npy}")

        # ── 5. Decode → DataFrame ─────────────────────────────────────────────
        raw          = np.load(synth_npy, allow_pickle=True)
        decoded_full = encoder.inverse_transform(raw.astype(int))
        valid_mask   = np.array([not any(x is None for x in row) for row in decoded_full])
        decoded      = decoded_full[valid_mask]
        flush(f"      Valid rows after decoding: {len(decoded)} / {len(raw)}")

        if len(decoded) >= n_samples:
            break
        request = int(request * 1.6)
        flush(f"      Only {len(decoded)}/{n_samples} valid -- retrying with a larger request\n")

    if len(decoded) < n_samples:
        flush(f"      WARNING: only got {len(decoded)}/{n_samples} valid rows after "
              f"{max_rounds} rounds -- returning what we have.")
    decoded = decoded[:n_samples]

    df_synth = pd.DataFrame(decoded, columns=data_cat.columns)

    # Re-attach LOS by sampling from real distribution
    if los_values is not None:
        df_synth['LOS'] = np.random.choice(los_values, size=len(df_synth), replace=True)
        flush(f"      LOS re-sampled from augmented data distribution.")

    df_synth['source'] = 'synthetic'

    out_path = os.path.join(gen_dir, 'synthetic_decoded.csv')
    df_synth.to_csv(out_path, index=False)
    flush(f"\nShape : {df_synth.shape}")
    flush(f"Saved : {out_path}\n")
    flush(str(df_synth.head()))

    return df_synth


def parse_args():
    parser = argparse.ArgumentParser(
        description='MedGAN synthetic data generator for MIMIC-III / MIMIC-IV'
    )
    parser.add_argument('--data',    required=True, help='Path to augmented CSV')
    parser.add_argument('--version', required=True, choices=['mimic3', 'mimic4'],
                        help='Dataset version')
    parser.add_argument('--epochs',          type=int, default=1000)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--n_samples',       type=int, default=100000)
    parser.add_argument('--batch_size',      type=int, default=1000)
    parser.add_argument('--seed',            type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        data_path      = args.data,
        version        = args.version,
        epochs         = args.epochs,
        n_samples      = args.n_samples,
        pretrain_epochs= args.pretrain_epochs,
        batch_size     = args.batch_size,
        seed           = args.seed,
    )
