"""CHIRTS-ERA5 daily Tmax / Tmin (0.05 deg, 60S-70N, 1980-) -> weekly means per 0.5-degree cell, 1990-2025. Task 35, item 2.

  python realdata/covariates/chirts_weekly.py daily <year>     # batch (array task): read Tmax + Tmin of <year> for the BOX only, over HTTP
                                                               # (the netCDF files are HDF5-chunked: 19 days x 137 x 379 px, gzip, so
                                                               # only the chunks intersecting the box are transferred; nothing is stored
                                                               # raw), aggregate to daily 0.5-degree cell means, save the daily cache
  python realdata/covariates/chirts_weekly.py weekly           # after all years: weekly means from the daily caches (weeks may cross
                                                               # into the next year; 2026's first days come from 2026 if cached)

Daily cache: $WORK/covariates/daily/{tmax,tmin}_<year>.npz (dates, values (T, ncell) float32), about 80 MB per variable-year.
Weekly output: grid/{tmax,tmin}/{tmax,tmin}_<year>.parquet (grid_id, week_start, value in deg C); NaN if any day is missing
or the cell is outside the CHIRTS domain (north of 70N).
Source: https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/{tmax,tmin}/netcdf/daily/CHIRTS-ERA5.daily_{Tmax,Tmin}.<year>.nc
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

URL = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/{v}/netcdf/daily/CHIRTS-ERA5.daily_{V}.{y}.nc"
VARS = {"tmax": "Tmax", "tmin": "Tmin"}
RAWD, DAILY = os.path.join(C.RAW, "chirts"), os.path.join(C.COV, "daily")
CHUNK = 19   # days per read = the file's time chunk
# Box (lon0, lon1, lat0, lat1): the paper regions (ame = [-20, 65, -36, 45], which contains west_sahel), on 0.5-degree edges.
# Cells outside the box are NaN. Override with CHIRTS_BOX="lon0 lon1 lat0 lat1" (e.g. to extend later).
BOX = tuple(float(x) for x in os.environ.get("CHIRTS_BOX", "-20 65 -36 45").split())


def cache(v, y):
    return os.path.join(DAILY, f"{v}_{y}.npz")


def daily(y):
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    os.makedirs(RAWD, exist_ok=True); os.makedirs(DAILY, exist_ok=True)
    for v, V in VARS.items():
        if os.path.exists(cache(v, y)):
            print(y, v, "cached"); continue
        import fsspec
        t = time.time()
        fo = fsspec.open(URL.format(v=v, V=V, y=y), block_size=2 ** 22).open()
        with xr.open_dataset(fo, engine="h5netcdf") as ds:
            name = [k for k, x in ds.data_vars.items() if x.ndim == 3][0]
            x = ds[name]; tdim, ydim, xdim = x.dims
            lat, lon = ds[ydim].values, ds[xdim].values
            yi = np.where((lat > BOX[2]) & (lat < BOX[3]))[0]; xi = np.where((lon > BOX[0]) & (lon < BOX[1]))[0]
            x = x.isel({ydim: slice(yi.min(), yi.max() + 1), xdim: slice(xi.min(), xi.max() + 1)})
            la, lo = lat[yi.min():yi.max() + 1], lon[xi.min():xi.max() + 1]
            dates = pd.DatetimeIndex(ds[tdim].values).normalize()
            out = []
            for i in range(0, len(dates), CHUNK):
                for k in range(4):
                    try:
                        a = x.isel({tdim: slice(i, i + CHUNK)}).values.astype("float32"); break
                    except Exception as e:  # transient HTTP errors
                        print(f"  {y} {v} days {i}: {e.__class__.__name__}; retry {k + 1}", flush=True); time.sleep(30 * (k + 1))
                else:
                    raise RuntimeError(f"{y} {v}: read failed at day {i}")
                a[(a < -100) | (a > 100)] = np.nan
                out.append(C.to_half_degree(a, la, lo, "mean")[:, r, c])
        td = time.time() - t; size = 0.0
        vals = np.concatenate(out)
        np.savez(cache(v, y) + ".tmp.npz", dates=dates.values, values=vals)
        os.replace(cache(v, y) + ".tmp.npz", cache(v, y))
        inbox = np.isfinite(vals).any(0).mean()
        print(f"{y} {v}: box {BOX}, {len(dates)} days read in {td:.0f} s; cells with data {inbox:.3f} of all cells", flush=True)


def weekly():
    cl = C.cells(); wd = C.weekday()
    for v in VARS:
        for y in C.YEARS:
            parts = [np.load(cache(v, yy)) for yy in (y, y + 1) if os.path.exists(cache(v, yy))]
            if not parts or not os.path.exists(cache(v, y)):
                print(y, v, "no daily cache"); continue
            dates = pd.DatetimeIndex(np.concatenate([p["dates"] for p in parts]))
            vals = np.concatenate([p["values"] for p in parts])
            idx = pd.Series(np.arange(len(dates)), index=dates)
            W = C.weeks(y, wd); rows = []
            for w in W:
                days = pd.date_range(w, periods=7, freq="D")
                rows.append(vals[idx[days].to_numpy()].mean(0) if days.isin(dates).all()
                            else np.full(len(cl), np.nan, "float32"))
            m = np.stack(rows)
            C.write(v, y, np.tile(cl.grid_id, len(W)), np.repeat(W, len(cl)), m.ravel())
            print(f"{y} {v}: {len(W)} weeks, missing {np.isnan(m).mean():.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "daily":
        daily(int(sys.argv[2]))
    else:
        weekly()
