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


def write(var, year, grid_id, week_start, value):
    out = os.path.join(GRID, var); os.makedirs(out, exist_ok=True)
    df = pd.DataFrame({"grid_id": np.asarray(grid_id, "int32"), "week_start": pd.to_datetime(week_start),
                       "value": np.asarray(value, "float32")})
    f = os.path.join(out, f"{var}_{year}.parquet")
    df.to_parquet(f + ".tmp", index=False); os.replace(f + ".tmp", f)
    return f


def to_half_degree(arr, lat, lon, how="mean"):
    """Aggregate a (T, nlat, nlon) array on a regular grid aligned with the 0.5-degree lattice (pixel size 0.5/k)
    to the global (T, 360, 720) lattice (rows/cols as in cells()). 'mean' is the area-weighted (cos latitude) mean
    over the pixels with data; 'sum' adds them. NaN where no pixel of a cell has data or the cell is outside the grid."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if lat[0] > lat[-1]:
        arr, lat = arr[:, ::-1, :], lat[::-1]
    lon = ((lon + 180) % 360) - 180
    o = np.argsort(lon); arr, lon = arr[:, :, o], lon[o]
    d = abs(lat[1] - lat[0]); k = int(round(0.5 / d))
    r = np.floor((lat + 90) / 0.5).astype(int); c = np.floor((lon + 180) / 0.5).astype(int)
    r0, c0 = r[0], c[0]
    nr, nc = len(lat) // k, len(lon) // k
    assert (r[::k][:nr] == r0 + np.arange(nr)).all() and (c[::k][:nc] == c0 + np.arange(nc)).all(), "grid not aligned"
    a = np.asarray(arr[:, :nr * k, :nc * k], "float32")
    T = a.shape[0]
    a = a.reshape(T, nr, k, nc, k)
    ok = np.isfinite(a)
    if how == "mean":
        w = np.cos(np.deg2rad(lat[:nr * k])).reshape(nr, k)[None, :, :, None, None] * ok
        s = (np.where(ok, a, 0) * w).sum((2, 4)); ws = w.sum((2, 4))
        v = np.where(ws > 0, s / np.where(ws > 0, ws, 1), np.nan)
    else:
        s = np.where(ok, a, 0).sum((2, 4)); n = ok.sum((2, 4))
        v = np.where(n > 0, s, np.nan)
    out = np.full((T, 360, 720), np.nan, "float32")
    out[:, r0:r0 + nr, c0:c0 + nc] = v
    return out
