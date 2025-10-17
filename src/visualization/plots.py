import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def pitch_type_outcome_bar(df: pd.DataFrame, outcome_col="events", top_n=10):
    d = df.dropna(subset=[outcome_col, "pitch_type"]).copy()

    pct = (d.groupby("pitch_type")[outcome_col]
             .value_counts(normalize=True)
             .rename("pct")
             .reset_index())   # -> columns: ['pitch_type', outcome_col, 'pct']

    top_types = d["pitch_type"].value_counts().head(top_n).index
    pct = pct[pct["pitch_type"].isin(top_types)]

    pivot = pct.pivot(index="pitch_type", columns=outcome_col, values="pct").fillna(0)
    pivot.plot(kind="bar", stacked=True, figsize=(10,5))
    plt.ylabel("Proportion")
    plt.title("Outcome distribution by pitch_type")
    plt.tight_layout()



def _plate_cols(df: pd.DataFrame):
    if {"px","pz"}.issubset(df.columns):
        return "px","pz"
    if {"plate_x","plate_z"}.issubset(df.columns):
        return "plate_x","plate_z"
    raise KeyError("Could not find plate location columns. Expected px/pz or plate_x/plate_z.")

def location_heatmap(df: pd.DataFrame, filter_col="events", filter_val="home_run", bins=40):
    # pick the correct plate-location column names
    xcol, ycol = _plate_cols(df)

    # filter to the requested event and valid coords
    d = df[(df[filter_col] == filter_val)].copy()
    d = d[[xcol, ycol]].dropna()

    if d.empty:
        print(f"No rows where {filter_col} == {filter_val} with valid {xcol}/{ycol}.")
        return

    # 2D histogram over the strike zone-ish bounds
    H, xedges, yedges = np.histogram2d(
        d[xcol], d[ycol],
        bins=bins,
        range=[[-1.8, 1.8], [1.0, 4.0]]
    )

    plt.imshow(
        H.T, origin="lower",
        extent=[-1.8, 1.8, 1.0, 4.0],
        aspect="auto"
    )
    plt.colorbar(label="Count")
    plt.xlabel(f"{xcol} (horizontal)")
    plt.ylabel(f"{ycol} (vertical)")
    plt.title(f"Pitch location heatmap for {filter_val}")
    plt.tight_layout()
