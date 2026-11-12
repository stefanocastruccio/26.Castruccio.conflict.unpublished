"""Task 40, part B: one covariate file set per variable for the global cell-week table, from the task-35 outputs.

  python realdata/table/join_cov.py --var tmax      # writes $WORK/table_global/cov/<var>/...
  python realdata/table/join_cov.py --var all

Each variable is stored at its natural resolution (assemble.py expands it onto the cell-weeks):
- weekly, keyed (grid_id, week_start), year=<y>.parquet: tmax, tmin, hi, wbgt, ppt, pm25 -> <name> (the week's value) and
  <name>_rollmean_4 (mean over weeks t-4..t-1, NaN unless all four exist); spi3 -> drought, drought_rollmean_4 (SPI-3 fills $CLEAN's
  drought slot); ipc -> phase_1..phase_5 (people in each IPC phase, national numbers spread by population), phase<k>_share,
  phase3p_share, ipc_carried.
  ppt and spi3 add ppt_pending_era5 / drought_pending_era5 = True for cells outside the CHIRPS domain (|lat| > 50) awaiting the ERA5 fill
  (their values are NaN until era5_precip.py merge has run; then rerun this script for ppt and spi3).
- yearly per cell, keyed (grid_id, year), all.parquet: pop -> population; malaria -> malaria (PfPR, 2000-2025).
- yearly per country, keyed (ISO_A3, year), all.parquet: national -> gdp, gdp_pc, electricity_access, v2x_libdem, GHG_pc_t, EPI_index,
  ECO_pc_gha and their flags (national_source_iso3, epi_extrapolated, epi_edition, eco_flag, eco_source_iso3, eco_pc_carried).
- static per cell, keyed grid_id, all.parquet: static -> river_length, lake_area, BIOME_NAME (taken from $CLEAN, one value per cell).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "covariates"))
import common as C  # noqa: E402

OUT = os.path.join(C.WORK, "table_global", "cov")
YEARS = range(1996, 2026)
# Cells whose task-35 values are physically impossible, masked (NaN) with a flag column <name>_masked:
# grid_id 14 (Fiji, 179.75W 16.75S, on the dateline): CHIRTS-ERA5 daily Tmax/Tmin/HI/WBGT follow an annual cycle from about -52 to
# +38 C in every year (both cache paths), i.e. not this tropical island's climate. Owner/coordinator informed (STATUS 2026-11-12).
MASK = {"tmax": [14], "tmin": [14], "hi": [14], "wbgt": [14]}
WEEKLY = {"tmax": "tmax", "tmin": "tmin", "hi": "hi", "wbgt": "wbgt", "ppt": "ppt", "pm25": "pm25", "spi3": "drought"}


def write(var, name, df):
    d = os.path.join(OUT, var); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, name); df.to_parquet(f + ".tmp", index=False); os.replace(f + ".tmp", f)


def weekly(var):
    name = WEEKLY[var]
    cl = C.cells(); outside = cl.grid_id[(cl.lat.abs() > 50)].to_numpy()
    prev = None
    for y in YEARS:
        f = os.path.join(C.GRID, var, f"{var}_{y}.parquet")
        if not os.path.exists(f):
            prev = None; continue
        d = pd.read_parquet(f).pivot(index="week_start", columns="grid_id", values="value").sort_index()
        full = d if prev is None else pd.concat([prev.iloc[-4:], d])
        roll = full.shift(1).rolling(4, min_periods=4).mean().loc[d.index]
        out = pd.DataFrame({"grid_id": np.tile(d.columns.to_numpy(), len(d)).astype("int32"),
                            "week_start": np.repeat(d.index.to_numpy(), d.shape[1]),
                            name: d.to_numpy().ravel().astype("float32"),
                            f"{name}_rollmean_4": roll.to_numpy().ravel().astype("float32")})
        if var in MASK:
            bad = out.grid_id.isin(MASK[var])
            out.loc[bad, [name, f"{name}_rollmean_4"]] = np.nan
            out[f"{name}_masked"] = bad
        if var in ("ppt", "spi3"):
            out[f"{name}_pending_era5"] = out.grid_id.isin(outside) & out[name].isna()
        write(var, f"year={y}.parquet", out)
        prev = d
        print(var, y, len(out), flush=True)


def ipc():
    for y in YEARS:
        f = os.path.join(C.GRID, "ipc", f"ipc_{y}.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        ren = {f"phase{k}": f"phase_{k}" for k in range(1, 6)}
        d = d.rename(columns=ren)[["grid_id", "week_start"] + list(ren.values()) +
                                  [f"phase{k}_share" for k in range(1, 6)] + ["phase3p_share", "ipc_carried"]]
        write("ipc", f"year={y}.parquet", d)
        print("ipc", y, len(d), flush=True)


def yearly_cell():
    p = pd.read_parquet(os.path.join(C.GRID, "pop", "pop_1990_2025.parquet")).rename(columns={"value": "population"})
    p["grid_id"] = p.grid_id.astype("int32"); p["year"] = p.year.astype("int16")
    write("pop", "all.parquet", p[["grid_id", "year", "population"]])
    m = pd.read_parquet(os.path.join(C.GRID, "malaria", "malaria_2000_2025.parquet")).rename(columns={"value": "malaria"})
    m["grid_id"] = m.grid_id.astype("int32"); m["year"] = m.year.astype("int16")
    write("malaria", "all.parquet", m[["grid_id", "year", "malaria"]])
    print("pop", len(p), "malaria", len(m))


def national():
    G = os.path.join(C.GRID, "national")
    n = pd.read_parquet(os.path.join(G, "national_1990_2025.parquet")).rename(columns={
        "iso3": "ISO_A3", "elec_access": "electricity_access", "libdem": "v2x_libdem", "ghg_pc": "GHG_pc_t",
        "source_iso3": "national_source_iso3"})
    e = pd.read_parquet(os.path.join(G, "epi_1990_2025.parquet")).rename(columns={"iso3": "ISO_A3", "epi": "EPI_index"})
    g = pd.read_parquet(os.path.join(G, "gfn_eco_1990_2025.parquet")).rename(columns={
        "iso3": "ISO_A3", "eco_pc_gha": "ECO_pc_gha", "flag": "eco_flag", "source_iso3": "eco_source_iso3"})
    d = n.merge(e, on=["ISO_A3", "year"], how="outer").merge(
        g[["ISO_A3", "year", "ECO_pc_gha", "eco_flag", "eco_source_iso3", "eco_pc_source", "eco_pc_carried"]], on=["ISO_A3", "year"], how="outer")
    d["year"] = d.year.astype("int16")
    write("national", "all.parquet", d)
    cl = C.cells(); print("national", d.shape, "cell countries without any national row:", sorted(set(cl.ISO_A3) - set(d.ISO_A3))[:30])


def static():
    import pyarrow.parquet as pq
    t = pq.read_table(C.CLEAN, columns=["grid_id", "river_length", "lake_area", "BIOME_NAME"]).to_pandas()
    s = t.drop_duplicates("grid_id").copy(); s["grid_id"] = s.grid_id.astype("int32")
    nun = t.groupby("grid_id")[["river_length", "lake_area"]].nunique().max()
    write("static", "all.parquet", s)
    print("static", len(s), "max distinct values per cell (should be 1):", nun.to_dict())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--var", required=True)
    v = ap.parse_args().var
    todo = list(WEEKLY) + ["ipc", "pop", "national", "static"] if v == "all" else [v]
    for var in todo:
        if var in WEEKLY:
            weekly(var)
        elif var == "ipc":
            ipc()
        elif var in ("pop", "malaria"):
            yearly_cell()
        elif var == "national":
            national()
        elif var == "static":
            static()
        else:
            raise SystemExit(f"unknown var {var}")


if __name__ == "__main__":
    main()
