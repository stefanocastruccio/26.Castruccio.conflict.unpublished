"""CHIRTS-ERA5 daily Tmax / Tmin (0.05 deg, 60S-70N, 1980-) -> weekly means per 0.5-degree cell, 1990-2025. Task 35, item 2.

  python realdata/covariates/chirts_weekly.py daily <year>     # batch (array task): download Tmax + Tmin of <year> (whole files, 7.2 GB
                                                               # each; reading only a box is rate-limited by the server, HTTP 429),
                                                               # aggregate to daily 0.5-degree cell means (all cells), save the daily
                                                               # cache, DELETE the raw files
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


def cache(v, y):
    return os.path.join(DAILY, f"{v}_{y}.npz")


def daily(y):
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    os.makedirs(RAWD, exist_ok=True); os.makedirs(DAILY, exist_ok=True)
    for v, V in VARS.items():
        if os.path.exists(cache(v, y)):
            print(y, v, "cached"); continue
        # Whole-file download (one request, resumable): partial range reads of the box are rate-limited by the server (HTTP 429).
        f = os.path.join(RAWD, f"CHIRTS-ERA5.daily_{V}.{y}.nc"); t = time.time()
        if not os.path.exists(f):
            for k in range(6):
                rc = subprocess.run(["curl", "-sfL", "--retry", "10", "--retry-delay", "60", "--retry-all-errors", "-C", "-",
                                     "-o", f + ".part", URL.format(v=v, V=V, y=y)]).returncode
                if rc == 0:
                    break
                print(f"  {y} {v}: curl rc {rc}; retry {k + 1}", flush=True); time.sleep(120)
            else:
                raise RuntimeError(f"{y} {v}: download failed")
            os.replace(f + ".part", f)
        tdl = time.time() - t; size = os.path.getsize(f) / 1e9
        with xr.open_dataset(f) as ds:
            name = [k for k, q in ds.data_vars.items() if q.ndim == 3][0]
            x = ds[name]; tdim, ydim, xdim = x.dims
            la, lo = ds[ydim].values, ds[xdim].values
            dates = pd.DatetimeIndex(ds[tdim].values).normalize()
            out = []
            for i in range(0, len(dates), CHUNK):
                a = x.isel({tdim: slice(i, i + CHUNK)}).values.astype("float32")
                a[(a < -100) | (a > 100)] = np.nan
                out.append(C.to_half_degree(a, la, lo, "mean")[:, r, c])
        os.remove(f)
        td = time.time() - t
        vals = np.concatenate(out)
        np.savez(cache(v, y) + ".tmp.npz", dates=dates.values, values=vals)
        os.replace(cache(v, y) + ".tmp.npz", cache(v, y))
        inbox = np.isfinite(vals).any(0).mean()
        print(f"{y} {v}: {size:.1f} GB downloaded in {tdl:.0f} s, processed in {td - tdl:.0f} s, {len(dates)} days; "
              f"cells with data {inbox:.3f}; raw deleted", flush=True)


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
