#!/usr/bin/env python3
"""
make_labels_contact.py
----------------------
Builds a binary contact label over ALL pitches.

Usage:
    python -m src.features.make_labels_contact --input data_raw --output data_proc/contact_labels.parquet
"""
import argparse, glob
from pathlib import Path
import pandas as pd

CONTACT_DESCRIPTIONS = {
    "foul", "foul_tip", "foul_bunt", "hit_into_play",
    "swinging_strike_blocked", "foul_pitchout", "foul_tip_intentional_ball"
}
BIP_EVENTS = {
    "field_out", "single", "double", "triple", "home_run",
    "field_error", "force_out", "grounded_into_double_play",
    "fielders_choice", "fielders_choice_out", "double_play", "triple_play",
    "sac_fly", "sac_bunt"
}

def read_input_parquet(input_path: str) -> pd.DataFrame:
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

def build_is_contact(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["description"].astype("string") if "description" in df.columns else pd.Series([""], index=df.index, dtype="string")
    events = df["events"].astype("string") if "events" in df.columns else pd.Series([""], index=df.index, dtype="string")

    is_contact_desc = desc.str.lower().isin(CONTACT_DESCRIPTIONS)
    is_contact_events = events.str.lower().isin(BIP_EVENTS)
    is_contact = (is_contact_desc | is_contact_events).astype("int8")

    out = df.copy()
    out["is_contact"] = is_contact

    keep = [c for c in [
        "game_date","pitcher","batter","pitch_type","p_throws","stand",
        "balls","strikes","inning","release_speed","pfx_x","pfx_z",
        "events","description","is_contact"
    ] if c in out.columns]
    return out[keep]

def main(args):
    df = read_input_parquet(args.input)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    labels = build_is_contact(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.output, index=False)
    print(f"✅ contact labels → {args.output}")
    print(labels["is_contact"].value_counts(normalize=True).rename("frac"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_raw")
    ap.add_argument("--output", default="data_proc/contact_labels.parquet")
    args = ap.parse_args()
    main(args)
