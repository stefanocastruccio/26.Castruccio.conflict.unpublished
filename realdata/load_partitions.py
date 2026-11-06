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
         [--coverage-csv coverage.csv]

ACLED coverage (repaired mode only). ACLED added countries at different dates, so
cell-weeks before a country's coverage start are unobserved, not event free. The
coverage start of a country is taken from --coverage-csv (columns ISO_A3, start) if
given (preferably ACLED's official country coverage table). Otherwise it is estimated
from the data: a country is treated as entering ACLED late only if its first event (of
any type, including strategic developments) falls after 30 June 2018 and is followed
by at least 20 events in the next 52 weeks, the signature of coverage being switched
on; all other countries are covered from the first week. Taking the first event alone
would condition on the outcome and drop genuine zeros in quiet countries. Earlier
cell-weeks are dropped, and the 4-week event histories of the first four covered weeks
are set to missing.

A JSON report (OUT.report.json) records the counts behind every repair step.
"""
from __future__ import annotations

import argparse
import glob
import json
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


def assign_countries(cells, shp, report=None):
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
    if report is not None:
        report["cells_centroid_in_sea_nearest_country"] = int(miss.sum())
    return j["ISO_A3"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts"); ap.add_argument("out")
    ap.add_argument("--countries"); ap.add_argument("--gdp-csv")
    ap.add_argument("--region", nargs=4, type=float)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--coverage-csv")
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
    report = {"input_rows": int(n_rows), "distinct_cell_weeks": int(n_pairs),
              "duplicated_share_of_rows": float(1 - n_pairs / n_rows)}

    df["year"] = df["week_start"].dt.year
    cells = df.groupby("grid_id", as_index=False)[["lon", "lat"]].first()
    if a.countries:
        cells["ISO_A3"] = assign_countries(cells, a.countries, report)
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
        no_interior = fill[nat].isna().all(axis=1).to_numpy()
        df["nat_source"] = np.where(no_interior, "first_copy", "interior")
        # a country-year can lack interior values because the country has no interior
        # cell, or because the indicators are not yet published for that year (all NaN)
        has_int = set(interior["ISO_A3"].dropna().astype(str))
        report["countries_without_any_interior_cell"] = sorted(set(df["ISO_A3"].dropna().astype(str)) - has_int)
        report["share_rows_nat_from_interior_by_year"] = {
            int(y): float(v) for y, v in pd.Series(~no_interior).groupby(df["year"].to_numpy()).mean().items()}
        report["share_rows_nat_missing_by_year"] = {
            c: {int(y): float(v) for y, v in df[c].isna().groupby(df["year"]).mean().items()} for c in nat}
        report["cells_nat_from_first_copy_any_year"] = int(df.loc[no_interior, "grid_id"].nunique())
        for c in nat:   # countries with no interior cell keep their first copy
            df[c] = fill[c].to_numpy(dtype="float32", na_value=np.nan)
            df[c] = df[c].fillna(df.groupby(["ISO_A3", "year"])[c].transform("first"))
        if a.gdp_csv and "gdp" in df.columns:
            with open(a.gdp_csv, encoding="utf-8-sig") as fh:   # World Bank files start with metadata lines
                skip = next(i for i, line in enumerate(fh) if line.lstrip('"').startswith("Country Name"))
            g = pd.read_csv(a.gdp_csv, skiprows=skip, encoding="utf-8-sig")
            g = g.loc[:, ~g.columns.str.startswith("Unnamed")]
            g = g.drop(columns=["Indicator Name", "Indicator Code"], errors="ignore")
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
    if not a.keep_duplicates:
        allev = EVENT_COLS + [c for c in ["Strategic_developments"] if c in df.columns]
        if a.coverage_csv:
            cov = pd.read_csv(a.coverage_csv, parse_dates=["start"]).set_index("ISO_A3")["start"]
            report["coverage_source"] = os.path.basename(a.coverage_csv)
        else:
            n_ev = df[allev].sum(axis=1)
            first = df[n_ev > 0].groupby("ISO_A3")["week_start"].min()
            t0 = df["week_start"].min()
            cov = pd.Series(t0, index=df["ISO_A3"].unique())
            for iso, f in first.items():
                if f > pd.Timestamp("2018-06-30"):
                    nxt = n_ev[(df["ISO_A3"] == iso) & (df["week_start"] >= f)
                               & (df["week_start"] < f + pd.Timedelta(weeks=52))].sum()
                    if nxt >= 20:
                        cov[iso] = f
            report["coverage_source"] = ("estimated: late entry if first event after 2018-06-30 "
                                         "with >= 20 events in the following 52 weeks")
        start = df["ISO_A3"].map(cov)
        # a week is covered if it ends on or after the coverage start (weeks may start
        # before 1 January 2018); countries never covered are dropped
        week_end = df["week_start"] + pd.Timedelta(days=6)
        covered = week_end >= start
        warm = covered & (week_end < start + pd.Timedelta(weeks=4))
        lagcols = [f"{c}_rollsum_4" for c in EVENT_COLS]
        df.loc[warm, lagcols] = np.nan
        dropped = (~covered).groupby(df["ISO_A3"]).sum()
        report["cell_weeks_dropped_before_coverage"] = int((~covered).sum())
        report["countries_with_dropped_weeks"] = {k: int(v) for k, v in dropped[dropped > 0].items()}
        report["coverage_start"] = {k: str(pd.Timestamp(v).date()) for k, v in cov.items()}
        df = df[covered.to_numpy()]
    df = df.drop(columns=EVENT_COLS + ["year"], errors="ignore")
    df = df[df["BIOME_NAME"].fillna("N/A") != "N/A"]
    report.update(rows=int(len(df)), cells=int(df.grid_id.nunique()), weeks=int(df.week_start.nunique()),
                  gdp=str(df["gdp_flag"].iloc[0]) if "gdp_flag" in df else "not repaired")
    json.dump(report, open(a.out + ".report.json", "w"), indent=1)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} rows, {df.grid_id.nunique():,} cells, "
          f"{df.week_start.nunique()} weeks, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
