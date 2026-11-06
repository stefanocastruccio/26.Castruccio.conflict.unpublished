"""Read the original pipeline's partition files and write one clean Parquet table for
fit_acled.py.

Input: data/partitions/data_final_world_XX.csv (or .parquet), as written by
papersblindreview/26.Conflicts.unpublished/scripts/preprocessing.py.

What it does (see STUDENT_CODE_REVIEW.md):
  * removes duplicated (grid_id, week_start) rows created by the country joins; the first
    row is kept and the number removed is reported;
  * country id: ISO_A3 if present; otherwise the most frequent combination of national
    indicators in the cell (a proxy, flagged in the output);
  * outcomes Conflict / Unrest and 4-week rolling sums of past event counts per type,
    computed within cell (shift(1).rolling(4)), as in the original feature engineering;
  * climate 4-week rolling means of past weeks, as in the original.

Usage:  python load_partitions.py <partitions_dir> <out.parquet>
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

EVENTS = {"Conflict": ["Battles", "ExplosionsRemote_violence", "Violence_against_civilians"],
          "Unrest": ["Riots", "Protests"]}
CLIMATE = ["hi", "tmin", "tmax", "wbgt", "drought", "ppt", "pm25"]
NATIONAL = ["EPI_index", "GHG_pc_t", "MF_pc_t", "ECO_pc_gha", "v2x_libdem", "electricity_access"]


def read_parts(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet"))) or \
        sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no .parquet/.csv files in {folder}")
    read = pd.read_parquet if files[0].endswith(".parquet") else pd.read_csv
    return pd.concat([read(f) for f in files], ignore_index=True)


def main(folder, out):
    df = read_parts(folder)
    df["week_start"] = pd.to_datetime(df["week_start"])
    n0 = len(df)
    dup = df.duplicated(["grid_id", "week_start"])
    print(f"rows {n0:,}; duplicated (grid_id, week_start): {dup.sum():,} "
          f"({dup.mean():.1%}); cells {df.grid_id.nunique():,}; weeks {df.week_start.nunique()}")
    df = df[~dup].copy()

    if "ISO_A3" in df.columns:
        df["country"] = df["ISO_A3"]
        df["country_is_proxy"] = False
    else:
        cols = [c for c in NATIONAL if c in df.columns]
        key = df[cols].round(4).astype(str).agg("|".join, axis=1)
        cell_key = key.groupby(df["grid_id"]).agg(lambda s: s.mode().iat[0])
        df["country"] = pd.factorize(df["grid_id"].map(cell_key))[0]
        df["country_is_proxy"] = True
        print("ISO_A3 not found: country approximated by national-indicator profile "
              f"({df['country'].nunique()} groups)")

    df = df.sort_values(["grid_id", "week_start"])
    g = df.groupby("grid_id", sort=False)
    for out_name, cols in EVENTS.items():
        df[out_name] = (df[cols].sum(axis=1) > 0).astype("int8")
    for c in sum(EVENTS.values(), []):
        df[f"{c}_rollsum_4"] = g[c].transform(lambda s: s.shift(1).rolling(4).sum())
    for c in [c for c in CLIMATE if c in df.columns]:
        df[f"{c}_rollmean_4"] = g[c].transform(lambda s: s.shift(1).rolling(4).mean())
    df = df.drop(columns=sum(EVENTS.values(), []) + [c for c in CLIMATE if c in df.columns])
    if "BIOME_NAME" in df.columns:
        df = df[df["BIOME_NAME"].fillna("N/A") != "N/A"]
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
