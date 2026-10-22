"""
HealthGAN Synthetic Data Generator  —  AGE_GROUP discrete variant
-------------------------------------------------------------------
Same as gen_age_continuous.py except:
  - ANCHOR_AGE is dropped (raw age not used)
  - AGE_GROUP (<= 45, 46-65, 66-80, +81) is kept as a categorical column

Requires AGE_GROUP column to already exist in the input CSV.

Usage:
    python -u healthgan_gen.py --data path/to/real.csv --version mimic4
    python -u healthgan_gen.py --data mimic3.csv mimic4.csv --version mimic3 mimic4

Arguments:
    --data      Path(s) to the input CSV file(s)
    --version   Dataset version(s): mimic3 or mimic4 (must match order of --data)
    --epochs    Number of GAN training epochs (default: 10000)
    --nodes     Base nodes for GAN architecture (default: 64)
    --critic    Critic iterations (default: 5)
    --outdir    Output directory name inside healthgan/ (default: gen_data)
    --seed      Random seed for reproducibility (default: 42)

Output:
    synthetic_data_generation/healthgan/<outdir>/<version>/synthetic_decoded.csv

Example:
    conda activate fenv
    python -u synthetic_data_generation/healthgan/healthgan_gen.py \
        --data "data/Mimic4/real/real.csv" --version mimic4
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Column configs ─────────────────────────────────────────────────────────────

# Drop identifiers and raw age (AGE_GROUP is used instead)
DROP_COLS = ['SUBJECT_ID', 'HADM_ID', 'ANCHOR_AGE']


# ── Helpers ───────────────────────────────────────────────────────────────────

def flush(msg):
    """Print and immediately flush so SLURM .out file updates live."""
    print(msg, flush=True)


def set_seed(seed):
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(data_path: str, version: str, epochs: int, nodes: int,
                 critic: int, outdir: str, seed: int, n_samples: int = 100000):

    set_seed(seed)

    flush(f"\n{'='*60}")
    flush(f"  Running pipeline for: {version.upper()}  |  File: {data_path}")
    flush(f"{'='*60}\n")

    print("Model arguments are:")
    flush(f"      Seed         : {seed}")
    flush(f"      Epochs       : {epochs}")
    flush(f"      Base nodes   : {nodes}")
    flush(f"      Critic iters : {critic}")
    flush(f"      N samples    : {n_samples}")
    flush(f"      Output dir   : {outdir}")
    flush(f"      Version      : {version}\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, "data", version)
    gen_dir  = os.path.join(base_dir, outdir, version)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = pd.read_csv(os.path.abspath(data_path))
    flush(f"[1/5] Loaded data -- shape: {data.shape}")
    flush(f"      Columns: {list(data.columns)}\n")

    # Drop identifiers and raw age
    drop_cols = [c for c in DROP_COLS if c in data.columns]
    data = data.drop(columns=drop_cols, axis=1)
    flush(f"      Dropped columns : {drop_cols}")
    flush(f"      Remaining columns: {list(data.columns)}\n")

    # Handle NaN values before encoding
    nan_counts = data.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        flush(f"      NaN counts per column before cleaning:")
        flush(str(nan_counts))

    for col in data.columns:
        if data[col].dtype == object or data[col].nunique() < 20:
            data[col] = data[col].fillna('Unknown')
        else:
            data[col] = data[col].fillna(data[col].median())

    flush(f"\n      NaN values remaining: {data.isnull().sum().sum()}\n")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path,  index=False)
    flush(f"[2/5] Split -- Train: {len(train_data)} rows | Test: {len(test_data)} rows\n")

    # ── 3. Encode to SDV format ───────────────────────────────────────────────
    from synthetic_data.generators.processing import Encoder

    en = Encoder()
    en.encode_train(train_path)
    en.encode_test(test_path, train_path)
    flush("[3/5] Encoding complete.\n")

    train_sdv = os.path.join(data_dir, "train_sdv.csv")
    test_sdv  = os.path.join(data_dir, "test_sdv.csv")

    # ── 4. Train HealthGAN ────────────────────────────────────────────────────
    from synthetic_data.generators.gan import HealthGAN

    epoch_tag     = epochs - 1
    gan_filename  = f"samples_{epoch_tag}_{critic}_{nodes}_synthetic_0.csv"
    gan_output    = os.path.join(base_dir, "gen_data", gan_filename)
    synthetic_raw = os.path.join(gen_dir, gan_filename)

    original_cwd = os.getcwd()
    os.chdir(base_dir)
    flush(f"      Working directory temporarily set to: {base_dir}")

    try:
        gan = HealthGAN(
            train_file   = train_sdv,
            test_file    = test_sdv,
            base_nodes   = nodes,
            critic_iters = critic,
            num_epochs   = epochs,
            n_samples    = n_samples,
        )
        gan.train()
    finally:
        os.chdir(original_cwd)

    flush("[4/5] GAN training complete.\n")

    if os.path.abspath(gan_output) != os.path.abspath(synthetic_raw):
        os.makedirs(gen_dir, exist_ok=True)
        os.replace(gan_output, synthetic_raw)
        flush(f"      Moved GAN output -> {synthetic_raw}\n")

    # ── 5. Decode synthetic data ──────────────────────────────────────────────
    from synthetic_data.generators.processing import Decoder

    de = Decoder()
    de.decode(train_path, synthetic_raw)
    flush("[5/5] Decoding complete.\n")

    # ── 6. Save & display output ──────────────────────────────────────────────
    decoded_file = os.path.join(data_dir, "train_decoded.csv")

    if not os.path.exists(decoded_file):
        raise FileNotFoundError(
            f"Expected decoded output at:\n  {decoded_file}\n"
            f"But it was not found. Check Decoder.decode() completed successfully."
        )

    synthetic_data = pd.read_csv(decoded_file)
    out_path = os.path.join(gen_dir, "synthetic_decoded.csv")
    synthetic_data.to_csv(out_path, index=False)

    flush(f"Synthetic Data Preview ({version.upper()}):")
    flush(str(synthetic_data.head()))
    flush(f"\nShape : {synthetic_data.shape}")
    flush(f"Saved : {out_path}\n")

    return synthetic_data


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HealthGAN synthetic data generator for MIMIC-III / MIMIC-IV (AGE_GROUP discrete)"
    )
    parser.add_argument(
        "--data", nargs="+", required=True,
        help="Path(s) to input CSV file(s). E.g. --data mimic3.csv mimic4.csv"
    )
    parser.add_argument(
        "--version", nargs="+", required=True, choices=["mimic3", "mimic4"],
        help="Dataset version(s) matching --data order. E.g. --version mimic3 mimic4"
    )
    parser.add_argument("--epochs", type=int, default=10000,
                        help="GAN training epochs (default: 10000)")
    parser.add_argument("--nodes",  type=int, default=64,
                        help="Base nodes for GAN (default: 64)")
    parser.add_argument("--critic", type=int, default=5,
                        help="Critic iterations (default: 5)")
    parser.add_argument("--outdir", type=str, default="gen_data",
                        help="Output directory name inside healthgan/ (default: gen_data)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--n_samples", type=int, default=100000,
                        help="Number of synthetic rows to generate (default: 100000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if len(args.data) != len(args.version):
        raise ValueError(
            f"Number of --data paths ({len(args.data)}) must match "
            f"number of --version values ({len(args.version)})."
        )

    results = {}
    for data_path, version in zip(args.data, args.version):
        synthetic = run_pipeline(
            data_path = data_path,
            version   = version,
            epochs    = args.epochs,
            nodes     = args.nodes,
            critic    = args.critic,
            outdir    = args.outdir,
            seed      = args.seed,
            n_samples = args.n_samples,
        )
        results[version] = synthetic

    flush("\nAll pipelines complete!")
    for version, df in results.items():
        flush(f"  {version.upper()} synthetic data shape: {df.shape}")
