"""
CTGAN Synthetic Data Generator  —  AGE_GROUP discrete variant
-----------------------------------------------------------------
Same as gen_age_continuous.py except:
  - ANCHOR_AGE is dropped (raw age not used by CTGAN)
  - AGE_GROUP (<= 45, 46-65, 66-80, +81) is treated as discrete
  - Only LOS is continuous

Requires AGE_GROUP column to already exist in the input CSV.

Usage:
    python -u ctgan_gen.py --data path/to/real.csv --version mimic4
    python -u ctgan_gen.py --data mimic3.csv mimic4.csv --version mimic3 mimic4

Arguments:
    --data      Path(s) to the input CSV file(s)
    --version   Dataset version(s): mimic3 or mimic4 (must match order of --data)
    --n_samples Number of synthetic rows to generate (default: 100000)
    --epochs    CTGAN training epochs (default: 300)
    --seed      Random seed for reproducibility (default: 0)
    --output_path       Optional explicit synthetic CSV path
    --model_output_path Optional explicit fitted-model path

Output:
    synthetic_data_generation/ctgan/gen_data/<version>/synthetic.csv
    synthetic_data_generation/ctgan/gen_data/<version>/ctgan_model.pkl

Continuous columns (NOT treated as discrete by CTGAN):
    LOS

Example:
    conda activate <your_env>
    python -u synthetic_data_generation/ctgan/ctgan_gen.py \
        --data "data/Mimic4/real/real.csv" --version mimic4
    python -u synthetic_data_generation/ctgan/ctgan_gen.py \
        --data "data/Mimic3/real/real.csv" --version mimic3
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import argparse
import pandas as pd
from ctgan import CTGAN


# ── Column configs ─────────────────────────────────────────────────────────────

# Drop identifiers and raw age (AGE_GROUP is used instead)
DROP_COLS = ['SUBJECT_ID', 'HADM_ID', 'ANCHOR_AGE']

# Only LOS is continuous; AGE_GROUP and everything else is discrete/categorical.
CONTINUOUS_COLS = ['LOS']


# ── Helpers ────────────────────────────────────────────────────────────────────

def flush(msg):
    """Print and immediately flush so SLURM .out file updates live."""
    print(msg, flush=True)


def set_seed(seed):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Core pipeline ──────────────────────────────────────────────────────────────

def _resolve_output_path(explicit_path, default_path):
    """Return an absolute output path and create its parent directory."""
    path = os.path.abspath(explicit_path) if explicit_path else default_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def run_pipeline(data_path, version, n_samples, epochs, seed,
                 output_path=None, model_output_path=None, drop_source=False):

    flush("\n" + "=" * 60)
    flush(f"  CTGAN pipeline: {version.upper()}  |  File: {data_path}")
    flush("=" * 60 + "\n")

    flush(f"      Seed         : {seed}")
    flush(f"      Epochs       : {epochs}")
    flush(f"      N samples    : {n_samples}")
    flush(f"      Continuous   : {CONTINUOUS_COLS}")
    flush(f"      Version      : {version}\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "gen_data", version)
    os.makedirs(out_dir, exist_ok=True)
    out_path = _resolve_output_path(
        output_path, os.path.join(out_dir, "synthetic.csv"))
    model_path = _resolve_output_path(
        model_output_path, os.path.join(out_dir, "ctgan_model.pkl"))

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = pd.read_csv(os.path.abspath(data_path))
    flush(f"[1/4] Loaded data -- shape: {data.shape}")
    flush(f"      Columns: {list(data.columns)}\n")

    # ── 2. Drop identifier and raw age columns ────────────────────────────────
    configured_drop_cols = DROP_COLS + (['source'] if drop_source else [])
    drop_cols = [c for c in configured_drop_cols if c in data.columns]
    data = data.drop(columns=drop_cols)
    flush(f"[2/4] Dropped  : {drop_cols}")
    flush(f"      Remaining: {list(data.columns)}\n")

    # Handle NaN values
    nan_counts = data.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        flush(f"      NaN counts before cleaning:\n{nan_counts}")
    for col in data.columns:
        if col in CONTINUOUS_COLS:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna('Unknown')
    flush(f"      NaN values remaining: {data.isnull().sum().sum()}\n")

    # ── 3. Define discrete vs continuous columns ──────────────────────────────
    discrete_columns = [col for col in data.columns if col not in CONTINUOUS_COLS]

    # Cast discrete columns to string
    for col in discrete_columns:
        data[col] = data[col].astype(str)

    # Cast continuous columns to float
    for col in CONTINUOUS_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(data[col].median())

    flush(f"[3/4] Column types:")
    flush(f"      Continuous (float) : {[c for c in CONTINUOUS_COLS if c in data.columns]}")
    flush(f"      Discrete   (string): {discrete_columns}\n")

    # ── 4. Train CTGAN ────────────────────────────────────────────────────────
    flush(f"[4/4] Training CTGAN for {epochs} epochs on {len(data)} rows...\n")

    set_seed(seed)
    import torch
    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    selected_device = 'CUDA' if cuda_available else 'CPU'
    if mps_available and not cuda_available:
        flush("      Device       : MPS detected; CTGAN does not support MPS, using CPU")
    else:
        flush(f"      Device       : {selected_device}")

    ctgan = CTGAN(epochs=epochs, verbose=True, cuda=cuda_available)
    ctgan.fit(data, discrete_columns)

    flush("\nCTGAN training complete.\n")

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(ctgan, f)
    flush(f"      Model saved: {model_path}")

    # ── 5. Generate synthetic data ────────────────────────────────────────────
    flush(f"\nGenerating {n_samples} synthetic rows...")
    synthetic = ctgan.sample(n_samples)

    synthetic.to_csv(out_path, index=False)

    flush(f"\nSynthetic Data Preview ({version.upper()}):")
    flush(str(synthetic.head()))
    flush(f"\nShape : {synthetic.shape}")
    flush(f"Saved : {out_path}\n")

    return synthetic


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CTGAN synthetic data generator for MIMIC-III / MIMIC-IV (AGE_GROUP discrete)"
    )
    parser.add_argument(
        "--data", nargs="+", required=True,
        help="Path(s) to input CSV file(s). E.g. --data mimic3.csv mimic4.csv"
    )
    parser.add_argument(
        "--version", nargs="+", required=True, choices=["mimic3", "mimic4"],
        help="Dataset version(s) matching --data order. E.g. --version mimic3 mimic4"
    )
    parser.add_argument("--n_samples", type=int, default=100000,
                        help="Number of synthetic rows to generate (default: 100000)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="CTGAN training epochs (default: 300)")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Random seed for reproducibility (default: 0)")
    parser.add_argument(
        "--output_path", nargs="+", default=None,
        help="Explicit synthetic CSV path(s), matching --data order. "
             "Defaults to the legacy gen_data/<version>/synthetic.csv path."
    )
    parser.add_argument(
        "--model_output_path", nargs="+", default=None,
        help="Explicit model pickle path(s), matching --data order. "
             "Defaults to the legacy gen_data/<version>/ctgan_model.pkl path."
    )
    parser.add_argument(
        "--drop_source", action="store_true",
        help="Drop the MedEqualizer provenance column before fitting. This is "
             "opt-in so legacy invocations retain their prior behavior."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if len(args.data) != len(args.version):
        raise ValueError(
            f"Number of --data paths ({len(args.data)}) must match "
            f"number of --version values ({len(args.version)})."
        )

    for option, values in [('--output_path', args.output_path),
                           ('--model_output_path', args.model_output_path)]:
        if values is not None and len(values) != len(args.data):
            raise ValueError(
                f"Number of {option} paths ({len(values)}) must match "
                f"number of --data paths ({len(args.data)})."
            )

    results = {}
    output_paths = args.output_path or [None] * len(args.data)
    model_paths = args.model_output_path or [None] * len(args.data)
    for data_path, version, output_path, model_path in zip(
            args.data, args.version, output_paths, model_paths):
        synthetic = run_pipeline(
            data_path = data_path,
            version   = version,
            n_samples = args.n_samples,
            epochs    = args.epochs,
            seed      = args.seed,
            output_path       = output_path,
            model_output_path = model_path,
            drop_source       = args.drop_source,
        )
        results[version] = synthetic

    flush("\nAll pipelines complete!")
    for version, df in results.items():
        flush(f"  {version.upper()} synthetic data shape: {df.shape}")
