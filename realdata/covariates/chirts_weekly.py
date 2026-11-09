"""CHIRTS-ERA5 daily Tmax / Tmin (0.05 deg, 60S-70N, 1980-) -> weekly means per 0.5-degree cell, 1990-2025. Task 35, item 2.

  python realdata/covariates/chirts_weekly.py daily <year>     # batch (array task): download Tmax + Tmin of <year>, aggregate to daily
                                                               # 0.5-degree cell means (area-weighted), save daily cache, DELETE raw files
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
CHUNK = 16   # days per read (16 x 2600 x 7200 float32, about 1.2 GB)


def cache(v, y):
    return os.path.join(DAILY, f"{v}_{y}.npz")


def daily(y):
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    os.makedirs(RAWD, exist_ok=True); os.makedirs(DAILY, exist_ok=True)
    for v, V in VARS.items():
        if os.path.exists(cache(v, y)):
            print(y, v, "cached"); continue
        f = os.path.join(RAWD, f"CHIRTS-ERA5.daily_{V}.{y}.nc"); t = time.time()
        if not os.path.exists(f):
            subprocess.run(["curl", "-sfL", "--retry", "8", "--retry-delay", "30", "-C", "-", "-o", f + ".part",
                            URL.format(v=v, V=V, y=y)], check=True)
            os.replace(f + ".part", f)
        td = time.time() - t; size = os.path.getsize(f) / 1e9
        with xr.open_dataset(f) as ds:
            name = [k for k, x in ds.data_vars.items() if x.ndim == 3][0]
            x = ds[name]; tdim, ydim, xdim = x.dims
            lat, lon = ds[ydim].values, ds[xdim].values
            dates = pd.DatetimeIndex(ds[tdim].values).normalize()
            out = []
            for i in range(0, len(dates), CHUNK):
                a = x.isel({tdim: slice(i, i + CHUNK)}).values.astype("float32")
                a[(a < -100) | (a > 100)] = np.nan
                out.append(C.to_half_degree(a, lat, lon, "mean")[:, r, c])
        vals = np.concatenate(out)
        np.savez(cache(v, y) + ".tmp.npz", dates=dates.values, values=vals)
        os.replace(cache(v, y) + ".tmp.npz", cache(v, y))
        os.remove(f)
        print(f"{y} {v}: {size:.1f} GB downloaded in {td:.0f} s, {len(dates)} days, missing {np.isnan(vals).mean():.3f}, "
              f"total {time.time() - t:.0f} s; raw deleted", flush=True)


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
