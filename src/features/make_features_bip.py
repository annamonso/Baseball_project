#!/usr/bin/env python3
"""
make_features_bip.py
--------------------
Extracts features for BIP (ball-in-play) outcome prediction.

This includes BOTH pre-contact and post-contact features, since we're modeling
what happens AFTER contact has been made.

Usage:
    python -m src.features.make_features_bip \
        --input data_raw/statcast_full.parquet \
        --output data_proc/features_bip.parquet
"""
import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np


# BIP events (same as in make_labels_bip.py)
BALL_IN_PLAY_EVENTS = {
    "field_out", "single", "double", "triple", "home_run",
    "field_error", "force_out", "grounded_into_double_play",
    "fielders_choice", "fielders_choice_out", "double_play", "triple_play",
    "sac_fly", "sac_bunt"
}


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


def extract_bip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for BIP outcome prediction.

    Includes:
    - Pre-contact features (pitch characteristics, count, game situation)
    - Post-contact features (launch_speed, launch_angle, hit_distance_sc)

    Only returns rows where a ball was put in play.
    """
    # Filter to BIP events only
    events = df["events"].astype("string").str.lower()
    bip = df[events.isin(BALL_IN_PLAY_EVENTS)].copy()

    print(f"  Filtered to {len(bip):,} balls in play (from {len(df):,} total pitches)")

    # Pre-contact: Numeric pitch features
    numeric_pitch = [
        'release_speed', 'release_pos_x', 'release_pos_z',
        'pfx_x', 'pfx_z', 'spin_rate_deprecated',
        'px', 'pz'  # pitch location at plate
    ]

    # Pre-contact: Count and context features
    count_context = [
        'balls', 'strikes', 'inning', 'outs_when_up'
    ]

    # Pre-contact: Categorical features
    categorical = ['pitch_type', 'p_throws', 'stand']

    # Post-contact: Batted ball characteristics
    post_contact = [
        'launch_speed',      # Exit velocity
        'launch_angle',      # Launch angle
        'hit_distance_sc'    # Hit distance
    ]

    # Identifiers
    identifiers = ['game_date', 'game_pk', 'pitcher', 'batter', 'events']

    # Create output dataframe
    out = bip.copy()

    # Derived features: runners on base (binary flags)
    if 'on_1b' in out.columns:
        out['on_1b_flag'] = out['on_1b'].notna().astype('int8')
    if 'on_2b' in out.columns:
        out['on_2b_flag'] = out['on_2b'].notna().astype('int8')
    if 'on_3b' in out.columns:
        out['on_3b_flag'] = out['on_3b'].notna().astype('int8')

    # Select final columns
    keep_cols = (identifiers + numeric_pitch + count_context + categorical +
                 ['on_1b_flag', 'on_2b_flag', 'on_3b_flag'] + post_contact)

    # Filter to only existing columns
    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols]

    # Report on post-contact feature availability
    print(f"\n  Post-contact feature coverage:")
    for feat in post_contact:
        if feat in out.columns:
            coverage = out[feat].notna().mean() * 100
            print(f"    {feat}: {coverage:.1f}% non-null")

    return out


def main(args):
    print("=" * 80)
    print("Extracting BIP Features (Pre-contact + Post-contact)")
    print("=" * 80)

    print("\n[1/2] Loading raw data...")
    df = read_input_parquet(args.input)
    print(f"  ✓ Loaded {len(df):,} rows")

    print("\n[2/2] Extracting BIP features...")
    features = extract_bip_features(df)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)

    print(f"\n✅ Features saved → {args.output}")
    print(f"   Shape: {features.shape[0]:,} rows × {features.shape[1]} columns")
    print(f"\n   Columns: {features.columns.tolist()}")

    # Check for missing values
    missing = features.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(f"\n   Missing values:")
        for col, count in missing.items():
            pct = count / len(features) * 100
            print(f"     {col}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_raw/statcast_full.parquet",
                    help="Path to raw Statcast data")
    ap.add_argument("--output", default="data_proc/features_bip.parquet",
                    help="Path to save BIP features")
    args = ap.parse_args()
    main(args)
