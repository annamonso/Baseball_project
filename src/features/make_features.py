#!/usr/bin/env python3
"""
make_features.py
----------------
Extracts PRE-CONTACT features from raw Statcast data.

Usage:
    python -m src.features.make_features --input data_raw/statcast_full.parquet --output data_proc/features.parquet
"""
import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np


def read_input_parquet(input_path: str) -> pd.DataFrame:
    """Read parquet from file or directory."""
    p = Path(input_path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "season=*/pitches.parquet")))
        if not files:
            files = sorted(glob.glob(str(p / "**/*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {input_path}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(p)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pre-contact features only.

    NO post-contact columns allowed:
    - launch_speed, launch_angle, hit_distance_sc
    - barrel, babip_value
    - hc_x, hc_y (these are spray location, post-contact)
    """
    # Numeric pitch features
    numeric_pitch = [
        'release_speed', 'release_pos_x', 'release_pos_z',
        'pfx_x', 'pfx_z', 'spin_rate_deprecated',
        'px', 'pz'
    ]

    # Count and context features
    count_context = [
        'balls', 'strikes', 'inning', 'outs_when_up'
    ]

    # Categorical features
    categorical = ['pitch_type', 'p_throws', 'stand']

    # Identifiers
    identifiers = ['game_date', 'game_pk', 'pitcher', 'batter']

    # Create a clean copy
    out = df.copy()

    # Derived features: runners on base (binary flags)
    if 'on_1b' in out.columns:
        out['on_1b_flag'] = out['on_1b'].notna().astype('int8')
    if 'on_2b' in out.columns:
        out['on_2b_flag'] = out['on_2b'].notna().astype('int8')
    if 'on_3b' in out.columns:
        out['on_3b_flag'] = out['on_3b'].notna().astype('int8')

    # Select final columns
    keep_cols = identifiers + numeric_pitch + count_context + categorical + \
                ['on_1b_flag', 'on_2b_flag', 'on_3b_flag']

    # Filter to only existing columns
    keep_cols = [c for c in keep_cols if c in out.columns]

    return out[keep_cols]


def main(args):
    df = read_input_parquet(args.input)

    # Normalize game_date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    features = extract_features(df)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Save
    features.to_parquet(args.output, index=False)

    print(f"✅ Features saved → {args.output}")
    print(f"   Shape: {features.shape[0]:,} rows × {features.shape[1]} columns")
    print(f"\n   Columns: {list(features.columns)}")

    # Quick stats
    print(f"\n   Missing values:")
    missing = features.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            pct = 100 * count / len(features)
            print(f"     {col}: {count:,} ({pct:.1f}%)")
    else:
        print("     (none)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_raw/statcast_full.parquet",
                    help="Path to raw parquet file or directory")
    ap.add_argument("--output", default="data_proc/features.parquet",
                    help="Output path for features parquet")
    args = ap.parse_args()
    main(args)
