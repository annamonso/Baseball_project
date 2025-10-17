import pandas as pd

def null_report(df: pd.DataFrame) -> pd.DataFrame:
    rpt = (df.isna().mean().sort_values(ascending=False)
             .rename("null_frac").to_frame())
    return rpt

def value_counts_top(df: pd.DataFrame, col: str, n=20) -> pd.Series:
    return df[col].value_counts(dropna=False).head(n)

def basic_ranges(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["px","pz","release_speed","pfx_x","pfx_z","spin_rate_deprecated","inning","balls","strikes"]
    seen = [c for c in cols if c in df.columns]
    return df[seen].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).T
