#!/usr/bin/env python3
"""
make_labels_bip.py
------------------
Builds outcome + spray targets for BALLS IN PLAY only.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

BALL_IN_PLAY_EVENTS = {
    "field_out", "single", "double", "triple", "home_run",
    "field_error", "force_out", "grounded_into_double_play",
    "fielders_choice", "fielders_choice_out", "double_play", "triple_play",
    "sac_fly", "sac_bunt"
}

EVENT_MAP = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "field_error": "ROE",
    "field_out": "OUT",
    "grounded_into_double_play": "OUT",
    "force_out": "OUT",
    "fielders_choice": "OUT",
    "fielders_choice_out": "OUT",
    "double_play": "OUT",
    "triple_play": "OUT",
    "sac_fly": "OUT",
    "sac_bunt": "OUT",
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

def compute_spray(df: pd.DataFrame) -> pd.DataFrame:
    if not {"hc_x","hc_y"}.issubset(df.columns):
        raise KeyError("Required columns hc_x/hc_y not found.")
    x = df["hc_x"] - 125.42
    y = 198.27 - df["hc_y"]
    angle = np.degrees(np.arctan2(x, y))
    dist = np.sqrt(x**2 + y**2)
    out = df.copy()
    out["spray_angle_deg"] = angle
    out["spray_distance_ft"] = dist
    return out

def discretize_spray(df: pd.DataFrame, S: int, R: int):
    sector_edges = np.linspace(-45, 45, S+1)
    ring_edges = np.linspace(0, 400, R+1)
    sector_bin = np.digitize(df["spray_angle_deg"], sector_edges) - 1
    ring_bin = np.digitize(df["spray_distance_ft"], ring_edges) - 1
    df["sector_bin"] = sector_bin.clip(0, S-1)
    df["ring_bin"] = ring_bin.clip(0, R-1)
    df["spray_bin"] = df["sector_bin"].astype("int16").astype(str) + "_" + df["ring_bin"].astype("int16").astype(str)
    return df, {"sector_edges": sector_edges.tolist(), "ring_edges": ring_edges.tolist()}

def build_labels(df: pd.DataFrame, S: int, R: int):
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    events = df["events"].astype("string").str.lower()
    bip = df[events.isin(BALL_IN_PLAY_EVENTS)].copy()
    bip["outcome"] = bip["events"].str.lower().map(EVENT_MAP).fillna("OUT")
    bip = compute_spray(bip)
    bip, bins = discretize_spray(bip, S=S, R=R)
    keep = [c for c in [
        "game_date","pitcher","batter","pitch_type","p_throws","stand",
        "release_speed","pfx_x","pfx_z",
        "events","outcome",
        "hc_x","hc_y","spray_angle_deg","spray_distance_ft",
        "sector_bin","ring_bin","spray_bin"
    ] if c in bip.columns]
    return bip[keep], bins

def main(args):
    df = read_input_parquet(args.input)
    labels, bins = build_labels(df, S=args.S, R=args.R)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.output, index=False)
    print(f"✅ BIP labels → {args.output} ({len(labels):,} rows)")
    Path(args.bins).parent.mkdir(parents=True, exist_ok=True)
    with open(args.bins, "w", encoding="utf-8") as f:
        json.dump(bins, f, indent=2)
    print(f"✅ Bin metadata → {args.bins}")
    print('\\nOutcome distribution (fraction):')
    print(labels['outcome'].value_counts(normalize=True).round(3))
    print('\\nTop spray bins:')
    print(labels['spray_bin'].value_counts().head(10))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_raw")
    ap.add_argument("--output", default="data_proc/labels.parquet")
    ap.add_argument("--bins", default="data_proc/SxR_bins.json")
    ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--R", type=int, default=5)
    args = ap.parse_args()
    main(args)
