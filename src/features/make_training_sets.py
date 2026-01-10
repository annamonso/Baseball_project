#!/usr/bin/env python3
"""
make_training_sets.py
---------------------
Merges features with labels to create model-ready training datasets.

Usage:
    python -m src.features.make_training_sets \
        --features data_proc/features.parquet \
        --contact_labels data_proc/contact_labels.parquet \
        --bip_labels data_proc/labels.parquet \
        --output_dir data_proc
"""
import argparse
from pathlib import Path
import pandas as pd


def merge_contact_dataset(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge features with contact labels.

    Strategy:
    - Use positional merge (assumes same row order)
    - Validate row count matches
    - Keep only features + is_contact target
    """
    print(f"  Features shape: {features_df.shape}")
    print(f"  Labels shape:   {labels_df.shape}")

    if len(features_df) != len(labels_df):
        raise ValueError(
            f"Row count mismatch: features={len(features_df)}, labels={len(labels_df)}"
        )

    # Reset indices to ensure alignment
    features_reset = features_df.reset_index(drop=True)
    labels_reset = labels_df.reset_index(drop=True)

    # Extract only the target column
    if 'is_contact' not in labels_reset.columns:
        raise KeyError("'is_contact' column not found in contact labels")

    # Concatenate horizontally
    train = pd.concat([features_reset, labels_reset[['is_contact']]], axis=1)

    print(f"  ✓ Merged shape: {train.shape}")
    print(f"  ✓ Target distribution:\n{train['is_contact'].value_counts(normalize=True).round(3)}")

    return train


def merge_bip_dataset(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge features with BIP labels (balls in play only).

    Strategy:
    - Inner join on [game_date, pitcher, batter]
    - Keep only BIP rows (features for contact events)
    - Remove any post-contact columns from labels (hc_x, hc_y, spray_angle_deg, spray_distance_ft)
    """
    print(f"  Features shape: {features_df.shape}")
    print(f"  BIP labels shape: {labels_df.shape}")

    # Select only target columns from labels
    label_cols = ['game_date', 'pitcher', 'batter', 'outcome', 'sector_bin', 'ring_bin', 'spray_bin']
    label_cols = [c for c in label_cols if c in labels_df.columns]

    # Perform inner join
    merged = features_df.merge(
        labels_df[label_cols],
        on=['game_date', 'pitcher', 'batter'],
        how='inner'
    )

    # CRITICAL: Remove any post-contact columns that may have leaked through
    post_contact_cols = [
        'hc_x', 'hc_y', 'spray_angle_deg', 'spray_distance_ft',
        'launch_speed', 'launch_angle', 'hit_distance_sc',
        'events', 'description'
    ]
    drop_cols = [c for c in post_contact_cols if c in merged.columns]

    if drop_cols:
        print(f"  ⚠ Dropping post-contact columns: {drop_cols}")
        merged = merged.drop(columns=drop_cols)

    print(f"  ✓ Merged shape: {merged.shape}")
    print(f"  ✓ Outcome distribution:\n{merged['outcome'].value_counts(normalize=True).round(3)}")

    return merged


def main(args):
    print("=" * 80)
    print("Creating Training Datasets")
    print("=" * 80)

    # Load features
    print("\n[1/3] Loading features...")
    features = pd.read_parquet(args.features)
    print(f"  ✓ Loaded {len(features):,} rows")

    # ---- Contact Task ----
    if args.contact_labels:
        print("\n[2/3] Creating contact training set...")
        contact_labels = pd.read_parquet(args.contact_labels)
        train_contact = merge_contact_dataset(features, contact_labels)

        # Save
        output_contact = Path(args.output_dir) / "training_contact.parquet"
        output_contact.parent.mkdir(parents=True, exist_ok=True)
        train_contact.to_parquet(output_contact, index=False)
        print(f"  ✅ Saved → {output_contact}")
    else:
        print("\n[2/3] Skipping contact dataset (no contact_labels provided)")

    # ---- BIP Task ----
    if args.bip_labels:
        print("\n[3/3] Creating BIP training set...")
        bip_labels = pd.read_parquet(args.bip_labels)
        train_bip = merge_bip_dataset(features, bip_labels)

        # Save
        output_bip = Path(args.output_dir) / "training_bip.parquet"
        output_bip.parent.mkdir(parents=True, exist_ok=True)
        train_bip.to_parquet(output_bip, index=False)
        print(f"  ✅ Saved → {output_bip}")
    else:
        print("\n[3/3] Skipping BIP dataset (no bip_labels provided)")

    print("\n" + "=" * 80)
    print("✅ Training datasets created successfully")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="Path to features.parquet")
    ap.add_argument("--contact_labels", default=None,
                    help="Path to contact_labels.parquet (optional)")
    ap.add_argument("--bip_labels", default=None,
                    help="Path to BIP labels.parquet (optional)")
    ap.add_argument("--output_dir", default="data_proc",
                    help="Directory to save training datasets")
    args = ap.parse_args()

    # Validation
    if not args.contact_labels and not args.bip_labels:
        raise ValueError("Must provide at least one of --contact_labels or --bip_labels")

    main(args)
