"""CHIRPS v2.0 daily precipitation (p25, 0.25 deg, 50S-50N) -> weekly totals and monthly totals per 0.5-degree cell.

  python realdata/covariates/chirps_weekly.py download [--years 1990 2026]   # login node (network): raw/chirps/*.nc
  python realdata/covariates/chirps_weekly.py process  [--years 1990 2025]   # batch job

Weekly total (mm) = sum over the 7 days of the week of the area-weighted mean of the 0.25-deg pixels in the cell;
NaN if any day is missing. A week belongs to the year of its week_start; its last days may fall in the next year,
so year y reads the files of y and y+1. Monthly totals (mm; for SPI-3) are written to grid/ppt_month/ppt_month_<y>.parquet
(grid_id, month, value). Cells outside the CHIRPS domain (north of 50N) are NaN.
Source: https://data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/netcdf/p25/chirps-v2.0.<year>.days_p25.nc
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

URL = "https://data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/netcdf/p25/chirps-v2.0.{y}.days_p25.nc"
RAWD = os.path.join(C.RAW, "chirps")


def raw(y):
    return os.path.join(RAWD, f"chirps-v2.0.{y}.days_p25.nc")


def download(years):
    os.makedirs(RAWD, exist_ok=True)
    for y in years:
        f = raw(y)
        if os.path.exists(f):
            print(y, "present"); continue
        t = time.time()
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", f + ".part", URL.format(y=y)], check=True)
        os.replace(f + ".part", f)
        print(y, f"{os.path.getsize(f) / 1e6:.0f} MB in {time.time() - t:.0f} s", flush=True)


def daily_half_degree(y):
    """(dates, array (T, 360, 720)) of daily 0.5-degree cell means for year y (NaN outside the domain)."""
    with xr.open_dataset(raw(y)) as ds:
        p = ds["precip"].values.astype("float32")
        p[p < 0] = np.nan
        a = C.to_half_degree(p, ds["latitude"].values, ds["longitude"].values, "mean")
        return pd.DatetimeIndex(ds["time"].values).normalize(), a


def process(years):
    cl = C.cells(); wd = C.weekday()
    r, c = cl.row.to_numpy(), cl.col.to_numpy()
    nxt = None
    for y in years:
        t0 = time.time()
        d0, a0 = nxt if nxt is not None and nxt[0][0].year == y else daily_half_degree(y)
        cell0 = a0[:, r, c]                                     # (T, ncell)
        if os.path.exists(raw(y + 1)):
            nxt = daily_half_degree(y + 1); d1, cell1 = nxt[0][:6], nxt[1][:6, r, c]
        else:
            nxt = None; d1, cell1 = pd.DatetimeIndex([]), np.empty((0, len(cl)), "float32")
        dates = d0.append(d1); x = np.concatenate([cell0, cell1])
        idx = pd.Series(np.arange(len(dates)), index=dates)
        W = C.weeks(y, wd)
        out = []
        for w in W:
            days = pd.date_range(w, periods=7, freq="D")
            if not days.isin(dates).all():
                out.append(np.full(len(cl), np.nan, "float32")); continue
            out.append(x[idx[days].to_numpy()].sum(0))           # NaN if any day missing
        v = np.stack(out)                                         # (nweek, ncell)
        C.write("ppt", y, np.tile(cl.grid_id, len(W)), np.repeat(W, len(cl)), v.ravel())
        m = pd.Series(d0.month)
        mon = np.stack([cell0[(m == k).to_numpy()].sum(0) for k in range(1, 13)])
        mo = os.path.join(C.GRID, "ppt_month"); os.makedirs(mo, exist_ok=True)
        pd.DataFrame({"grid_id": np.tile(cl.grid_id, 12).astype("int32"),
                      "month": np.repeat(pd.date_range(f"{y}-01-01", periods=12, freq="MS"), len(cl)),
                      "value": mon.ravel().astype("float32")}).to_parquet(os.path.join(mo, f"ppt_month_{y}.parquet"), index=False)
        print(f"{y}: {len(W)} weeks x {len(cl)} cells, missing {np.isnan(v).mean():.3f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "process"])
    ap.add_argument("--years", type=int, nargs=2, default=None)
    a = ap.parse_args()
    if a.step == "download":
        download(range(1990, 2027) if a.years is None else range(a.years[0], a.years[1] + 1))
    else:
        process(range(a.years[0], a.years[1] + 1) if a.years else C.YEARS)
