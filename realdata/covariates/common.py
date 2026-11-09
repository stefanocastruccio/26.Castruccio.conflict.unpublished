"""Shared grid and week definitions for the 1990-2025 covariates (task 35).

Grid: the 0.5-degree cells of the clean ACLED table ($CLEAN: grid_id, lon, lat, ISO_A3), restricted to 40S-72N.
Cell centres sit at .25/.75 degrees; row/column indices on the global 0.5-degree lattice are
    col = floor((lon + 180) / 0.5),  row = floor((lat + 90) / 0.5)   (row 0 = 90S..89.5S).
Weeks: start on the weekday of week_start in $CLEAN (Tuesday), from the first such day in 1990 to the last in 2025;
a week covers week_start .. week_start + 6 days and belongs to the year of its week_start.
Outputs: $WORK/covariates/grid/<var>/<var>_<year>.parquet with columns grid_id (int32), week_start, value (float32).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

WORK = os.environ.get("WORK", "/groups/scastruc/conflict_spatial_work")
CLEAN = os.environ.get("CLEAN", os.path.join(WORK, "clean_r3.parquet"))
COV = os.path.join(WORK, "covariates")
RAW, GRID = os.path.join(COV, "raw"), os.path.join(COV, "grid")
LAT_MIN, LAT_MAX = -40.0, 72.0
YEARS = range(1990, 2026)


def cells(path=CLEAN):
    """One row per cell: grid_id (int32), lon, lat (cell centre), ISO_A3, row, col."""
    cache = os.path.join(COV, "cells.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    c = pd.read_parquet(path, columns=["grid_id", "lon", "lat", "ISO_A3"]).drop_duplicates("grid_id")
    c = c[(c.lat >= LAT_MIN) & (c.lat <= LAT_MAX)].copy()
    c["grid_id"] = c.grid_id.astype("int32")
    c["col"] = np.floor((c.lon.astype(float) + 180) / 0.5).astype("int32")
    c["row"] = np.floor((c.lat.astype(float) + 90) / 0.5).astype("int32")
    c["lon"] = -180 + 0.5 * c.col + 0.25
    c["lat"] = -90 + 0.5 * c.row + 0.25
    c = c.sort_values("grid_id").reset_index(drop=True)
    os.makedirs(COV, exist_ok=True)
    c.to_parquet(cache, index=False)
    return c


def weekday(path=CLEAN):
    w = pd.read_parquet(path, columns=["week_start"]).week_start.min()
    return pd.Timestamp(w).dayofweek


def weeks(year, wd=None):
    """Week starts in `year` (all on weekday wd)."""
    wd = weekday() if wd is None else wd
    d = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return d[d.dayofweek == wd]


def region_cells():
    """grid_id of the cells in the five study regions (grid/cell_region.parquet, from cell_region.py), or None."""
    f = os.path.join(GRID, "cell_region.parquet")
    if not os.path.exists(f):
        return None
    cr = pd.read_parquet(f)
    return cr.loc[cr.region.notna(), "grid_id"].astype("int32").to_numpy()


def write(var, year, grid_id, week_start, value):
    """Write one variable-year; only cells in the study regions are kept (task 35 scope)."""
    out = os.path.join(GRID, var); os.makedirs(out, exist_ok=True)
    df = pd.DataFrame({"grid_id": np.asarray(grid_id, "int32"), "week_start": pd.to_datetime(week_start),
                       "value": np.asarray(value, "float32")})
    keep = region_cells()
    if keep is not None:
        df = df[df.grid_id.isin(keep)]
    f = os.path.join(out, f"{var}_{year}.parquet")
    df.to_parquet(f + ".tmp", index=False); os.replace(f + ".tmp", f)
    return f


def to_half_degree(arr, lat, lon, how="mean", res=0.5):
    """Aggregate a (T, nlat, nlon) array on a regular lat/lon grid to the global (T, 360, 720) 0.5-degree lattice
    (rows/cols as in cells()), assigning each pixel to the cell that contains its centre (so grids offset from the
    lattice, e.g. GHSL, work too). 'mean' is the area-weighted (cos latitude) mean over the pixels with data; 'sum' adds
    them. NaN where no pixel of a cell has data or the cell is outside the grid."""
    arr = np.asarray(arr); lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if lat[0] > lat[-1]:
        arr, lat = arr[:, ::-1, :], lat[::-1]
    lon = ((lon + 180) % 360) - 180
    o = np.argsort(lon, kind="stable"); arr, lon = arr[:, :, o], lon[o]
    NR, NC = int(round(180 / res)), int(round(360 / res))   # target lattice (res = 0.5 by default)
    r = np.clip(np.floor((lat + 90) / res + 1e-9).astype(int), 0, NR - 1)
    c = np.clip(np.floor((lon + 180) / res + 1e-9).astype(int), 0, NC - 1)
    rb = np.r_[0, np.flatnonzero(np.diff(r)) + 1]; cb = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
    ru, cu = r[rb], c[cb]
    T = arr.shape[0]
    out = np.full((T, NR, NC), np.nan, "float32")
    w = np.cos(np.deg2rad(lat)).astype("float32")[:, None] if how == "mean" else None
    for t in range(T):
        a = np.asarray(arr[t], "float32"); ok = np.isfinite(a)
        x = np.where(ok, a, 0).astype("float64")
        if how == "mean":
            x = x * w; n = (ok * w).astype("float64")
        else:
            n = ok.astype("float64")
        s_ = np.add.reduceat(np.add.reduceat(x, cb, axis=1), rb, axis=0)
        n_ = np.add.reduceat(np.add.reduceat(n, cb, axis=1), rb, axis=0)
        v = np.where(n_ > 0, (s_ / np.where(n_ > 0, n_, 1)) if how == "mean" else s_, np.nan)
        out[t][np.ix_(ru, cu)] = v
    return out
