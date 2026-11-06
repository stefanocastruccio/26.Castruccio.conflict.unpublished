"""Read the original pipeline's partition files and write one clean Parquet table
(one row per grid cell and week) for reproduce_xgb.py and fit_acled.py.

Input: data/partitions/data_final_world_XX.csv (or .parquet), as written by
papersblindreview/26.Conflicts.unpublished/scripts/preprocessing.py. The files split
the data by time, so every (grid_id, week_start) pair lives in a single file.

Repair of the duplicated rows (STUDENT_CODE_REVIEW.md, item 1). In the original
partitions a cell-week appears once per combination of the countries / IPC regions
its cell intersects (up to hundreds of copies), with national indicators mixed
across countries. Here:
  * each cell is assigned to the country containing its centroid (--countries,
    Natural Earth admin-0 shapefile, the file used by the original pipeline);
  * national indicators (EPI, GHG, MF, ECO, V-Dem) are re-attached per country-year
    from the cells that were *not* duplicated (interior cells, which have the
    correct values);
  * IPC phase populations are averaged over the copies;
  * GDP is re-apportioned by population within the centroid country when the
    original GDP table is given (--gdp-csv); otherwise the first copy is kept
    and flagged;
  * all other columns are identical across copies and the first copy is kept.
--keep-duplicates skips the repair (to reproduce the original analysis).

Outcomes Conflict / Unrest and the 4-week features are computed as in the
original feature_engineering (past 4-week rolling sums of event counts, past
4-week rolling means of climate), but strictly within cell.

Usage:
  python load_partitions.py PARTS_DIR OUT.parquet [--countries ne_10m_admin_0_countries.shp]
         [--gdp-csv GDP.csv] [--region LONMIN LONMAX LATMIN LATMAX] [--keep-duplicates]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

EVENTS = {"Conflict": ["Battles", "ExplosionsRemote_violence", "Violence_against_civilians"],
          "Unrest": ["Riots", "Protests"]}
EVENT_COLS = sum(EVENTS.values(), [])
CLIMATE = ["hi", "tmin", "tmax", "wbgt", "drought", "ppt", "pm25"]
NATIONAL = ["EPI_index", "GHG_pc_t", "MF_pc_t", "ECO_pc_gha", "v2x_libdem"]
PHASES = [f"phase_{i}" for i in range(1, 6)]
KEY = ["grid_id", "week_start"]


def read_one(path, region):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    if region is not None:
        x0, x1, y0, y1 = region
        df = df[df.lon.between(x0, x1) & df.lat.between(y0, y1)]
    num = df.select_dtypes("number").columns.difference(["grid_id"])
    df[num] = df[num].astype("float32")
    return df


def collapse(df):
    """One row per (grid_id, week_start): first copy, IPC phases averaged."""
    n = df.groupby(KEY)["grid_id"].transform("size")
    df["n_copies"] = n.astype("int16")
    phases = [c for c in PHASES if c in df.columns]
    first = df.groupby(KEY, sort=False).first()
    if phases:
        first[phases] = df.groupby(KEY, sort=False)[phases].mean()
    return first.reset_index()


def assign_countries(cells, shp):
    import geopandas as gpd
    world = gpd.read_file(shp)[["ISO_A3", "ADM0_A3", "geometry"]].to_crs(4326)
    iso = world["ISO_A3"].where(world["ISO_A3"] != "-99", world["ADM0_A3"])
    world = world.assign(ISO_A3=iso)
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.lon, cells.lat), crs=4326)
    j = gpd.sjoin(pts, world[["ISO_A3", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j["ISO_A3"].isna()
    if miss.any():   # centroid in the sea (coastal cell): nearest country
        near = gpd.sjoin_nearest(pts[miss].to_crs(6933), world.to_crs(6933)[["ISO_A3", "geometry"]])
        near = near[~near.index.duplicated()]
        j.loc[miss, "ISO_A3"] = near["ISO_A3"]
    return j["ISO_A3"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts"); ap.add_argument("out")
    ap.add_argument("--countries"); ap.add_argument("--gdp-csv")
    ap.add_argument("--region", nargs=4, type=float)
    ap.add_argument("--keep-duplicates", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.parts, "*.parquet"))) or \
        sorted(glob.glob(os.path.join(a.parts, "*.csv")))
    if not files:
        raise FileNotFoundError(a.parts)
    pieces, n_rows, n_pairs = [], 0, 0
    for f in files:
        df = read_one(f, a.region)
        n_rows += len(df)
        if a.keep_duplicates:
            df["n_copies"] = df.groupby(KEY)["grid_id"].transform("size").astype("int16")
        else:
            df = collapse(df)
        n_pairs += df[KEY].drop_duplicates().shape[0] if a.keep_duplicates else len(df)
        pieces.append(df)
        print(f"{os.path.basename(f)}: {len(df):,} rows kept", flush=True)
    df = pd.concat(pieces, ignore_index=True); del pieces
    print(f"input rows {n_rows:,}; distinct cell-weeks {n_pairs:,}; "
          f"duplicated share of rows {1 - n_pairs / n_rows:.1%}")

    df["year"] = df["week_start"].dt.year
    cells = df.groupby("grid_id", as_index=False)[["lon", "lat"]].first()
    if a.countries:
        cells["ISO_A3"] = assign_countries(cells, a.countries)
        df = df.merge(cells[["grid_id", "ISO_A3"]], on="grid_id", how="left")
        df["country_is_proxy"] = False
    else:
        print("WARNING: no --countries shapefile; country = national-indicator profile (proxy)")
        nat = [c for c in NATIONAL if c in df.columns]
        key = df[nat].round(4).astype(str).agg("|".join, axis=1)
        prof = key.groupby(df["grid_id"]).agg(lambda s: s.mode().iat[0])
        df["ISO_A3"] = df["grid_id"].map(prof).factorize()[0].astype(str)
        df["country_is_proxy"] = True

    if not a.keep_duplicates:
        # national indicators per country-year from interior (non-duplicated) cells
        nat = [c for c in NATIONAL if c in df.columns]
        interior = df[df.n_copies == 1]
        table = (interior.groupby(["ISO_A3", "year"])[nat]
                 .agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan))
        fill = df[["ISO_A3", "year"]].merge(table, left_on=["ISO_A3", "year"],
                                            right_index=True, how="left")
        for c in nat:   # countries with no interior cell keep their first copy
            df[c] = fill[c].to_numpy(dtype="float32", na_value=np.nan)
            df[c] = df[c].fillna(df.groupby(["ISO_A3", "year"])[c].transform("first"))
        if a.gdp_csv and "gdp" in df.columns:
            g = pd.read_csv(a.gdp_csv).drop(columns=["Indicator Name", "Indicator Code"], errors="ignore")
            g = g.melt(id_vars=[c for c in g.columns if not str(c).isdigit()], var_name="year",
                       value_name="gdp_nat").rename(columns={"Country Code": "ISO_A3"})
            g["year"] = g["year"].astype(int)
            pop_c = df.drop_duplicates(["grid_id", "year"]).groupby(["ISO_A3", "year"])["population"].sum()
            df = df.merge(g[["ISO_A3", "year", "gdp_nat"]], on=["ISO_A3", "year"], how="left")
            df["gdp"] = (df["gdp_nat"] * df["population"]
                         / df.set_index(["ISO_A3", "year"]).index.map(pop_c).to_numpy()).astype("float32")
            df = df.drop(columns="gdp_nat")
            df["gdp_flag"] = "reapportioned"
        else:
            df["gdp_flag"] = "first_copy"

    df = df.sort_values(["grid_id", "week_start"]).reset_index(drop=True)
    g = df.groupby("grid_id", sort=False)
    for out_name, cols in EVENTS.items():
        df[out_name] = (df[cols].sum(axis=1) > 0).astype("int8")
    for c in EVENT_COLS:
        df[f"{c}_rollsum_4"] = g[c].transform(lambda s: s.shift(1).rolling(4).sum()).astype("float32")
    for c in [c for c in CLIMATE if c in df.columns]:
        df[f"{c}_rollmean_4"] = g[c].transform(lambda s: s.shift(1).rolling(4).mean()).astype("float32")
    df = df.drop(columns=EVENT_COLS + ["year"], errors="ignore")
    df = df[df["BIOME_NAME"].fillna("N/A") != "N/A"]
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} rows, {df.grid_id.nunique():,} cells, "
          f"{df.week_start.nunique()} weeks, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
