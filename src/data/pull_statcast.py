import argparse, os, json
import pandas as pd
from pybaseball import statcast, cache
from pathlib import Path
import yaml

from .columns import KEEP
from .utils import date_chunks

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def pull_range(start_dt: str, end_dt: str) -> pd.DataFrame:
    # data extraction 
    cache.enable()
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    return df

def run(config_path: str):
    cfg = load_cfg(config_path)
    raw_dir = cfg["raw_dir"]
    ensure_dirs(raw_dir)
    # split into weekly chunks, so requests don't time out 
    ranges = date_chunks(cfg["start_date"], cfg["end_date"], cfg["chunk_days"])
    parts = []
    for (s,e) in ranges:
        print(f"Pulling {s} → {e} ...")
        df = pull_range(s,e)
        if df is None or df.empty:
            print("  (empty)")
            continue

        # Keep only columns we care about (ignore missing)
        keep_cols = [c for c in KEEP if c in df.columns]
        df = df[keep_cols].copy()

        # Normalize dtypes
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

        # Save chunk
        outp = Path(raw_dir) / f"statcast_{s}_{e}.parquet"
        df.to_parquet(outp, index=False)
        parts.append(outp.as_posix())

    # Merge & save combined month file for convenience
    if parts:
        print("Concatenating chunks…")
        all_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        all_df.to_parquet(Path(raw_dir) / "statcast_full.parquet", index=False)
        print(f"Saved {len(all_df):,} rows → {raw_dir}/statcast_full.parquet")
    else:
        print("No data pulled.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)
