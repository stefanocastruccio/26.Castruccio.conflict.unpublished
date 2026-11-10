"""CHIRTS-ERA5 daily Tmax / Tmin (0.05 deg, 60S-70N, 1980-) -> weekly means per 0.5-degree cell, 1990-2025. Task 35, item 2.

  python realdata/covariates/chirts_weekly.py daily <year>     # batch (array task): download Tmax + Tmin of <year> (whole files, 7.2 GB
                                                               # each; reading only a box is rate-limited by the server, HTTP 429),
                                                               # aggregate to daily 0.5-degree cell means (all cells), save the daily
                                                               # cache, DELETE the raw files
  python realdata/covariates/chirts_weekly.py daily_tif <year> # heat index and WBGT (daily GeoTIFFs only; one day at a time)
  python realdata/covariates/chirts_weekly.py weekly [vars]    # after all years: weekly means from the daily caches (weeks may cross
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
TIFVARS = {"hi": "HI", "wbgt": "WBGT"}   # heat index and WBGT: only daily GeoTIFFs (about 27 MB per day)
TIFURL = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/{v}/tifs/daily/{y}/{V}.{y}.{m:02d}.{d:02d}.tif"
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
        np.savez(cache(v, y) + ".tmp.npz", dates=dates.values, values=vals, grid_id=cl.grid_id.to_numpy())
        os.replace(cache(v, y) + ".tmp.npz", cache(v, y))
        inbox = np.isfinite(vals).any(0).mean()
        print(f"{y} {v}: {size:.1f} GB downloaded in {tdl:.0f} s, processed in {td - tdl:.0f} s, {len(dates)} days; "
              f"cells with data {inbox:.3f}; raw deleted", flush=True)


def daily_tif(y, which=("hi", "wbgt")):
    """HI / WBGT of year y: one daily GeoTIFF at a time (download, aggregate to the 0.5-degree cells, delete)."""
    import rasterio
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    os.makedirs(RAWD, exist_ok=True); os.makedirs(DAILY, exist_ok=True)
    last = f"{y}-12-31" if y <= C.YEARS[-1] else f"{y}-01-07"   # the year after the last only feeds the final week
    days = pd.date_range(f"{y}-01-01", last, freq="D")
    for v in which:
        V = TIFVARS[v]
        if os.path.exists(cache(v, y)):
            print(y, v, "cached"); continue
        t = time.time(); vals = np.full((len(days), len(cl)), np.nan, "float32"); nbytes = 0; miss = 0
        for i, d in enumerate(days):
            f = os.path.join(RAWD, f"{V}.{y}.{d.month:02d}.{d.day:02d}.{os.getpid()}.tif")
            ok = False
            for k in range(6):
                rc = subprocess.run(["curl", "-sfL", "--retry", "5", "--retry-delay", "30", "--retry-all-errors", "-o", f,
                                     TIFURL.format(v=v, V=V, y=y, m=d.month, d=d.day)]).returncode
                if rc == 0:
                    ok = True; break
                time.sleep(60 * (k + 1))
            if not ok:
                miss += 1; print(f"  {y} {v} {d.date()}: download failed", flush=True); continue
            nbytes += os.path.getsize(f)
            with rasterio.open(f) as src:
                a = src.read(1).astype("float32"); tr = src.transform
            os.remove(f)
            a[(a < -100) | (a > 150)] = np.nan
            lat = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5); lon = tr.c + tr.a * (np.arange(a.shape[1]) + 0.5)
            vals[i] = C.to_half_degree(a[None], lat, lon, "mean")[0][r, c]
        np.savez(cache(v, y) + ".tmp.npz", dates=days.values, values=vals, grid_id=cl.grid_id.to_numpy())
        os.replace(cache(v, y) + ".tmp.npz", cache(v, y))
        print(f"{y} {v}: {len(days)} days, {nbytes / 1e9:.1f} GB downloaded and deleted, {miss} days missing, "
              f"cells with data {np.isfinite(vals).any(0).mean():.3f}, {time.time() - t:.0f} s", flush=True)


MONURL = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/{v}/netcdf/monthly/CHIRTS-ERA5.monthly_{V}.monthly.nc"
FILL = os.path.join(C.GRID, "chirts_monthly_fill.parquet")


def monthly_fill():
    """Monthly Tmax/Tmin means for the cells added on 2026-11-09 (60S-40S; not in cells_40S), whose daily caches for the
    years processed before that date lack them. One monthly file per variable (about 7.2 GB each), downloaded, used, deleted.
    Output: grid/chirts_monthly_fill.parquet (var, grid_id, month, value). weekly() uses it where a cell has no daily data."""
    cl = C.cells(); old = set(pd.read_parquet(os.path.join(C.COV, "cells_40S.parquet")).grid_id)
    new = cl[~cl.grid_id.isin(old)]; r, c = new.row.to_numpy(), new.col.to_numpy()
    os.makedirs(RAWD, exist_ok=True); rows = []
    for v, V in VARS.items():
        f = os.path.join(RAWD, f"CHIRTS-ERA5.monthly_{V}.monthly.nc"); t = time.time()
        if not os.path.exists(f):
            subprocess.run(["curl", "-sfL", "--retry", "10", "--retry-delay", "60", "--retry-all-errors", "-C", "-",
                            "-o", f + ".part", MONURL.format(v=v, V=V)], check=True)
            os.replace(f + ".part", f)
        with xr.open_dataset(f) as ds:
            name = [k for k, q in ds.data_vars.items() if q.ndim == 3][0]
            x = ds[name]; tdim, ydim, xdim = x.dims
            months = pd.DatetimeIndex(ds[tdim].values).to_period("M").to_timestamp()
            sel = np.where((months.year >= C.YEARS[0]) & (months.year <= C.YEARS[-1] + 1))[0]
            la, lo = ds[ydim].values, ds[xdim].values
            for i in range(0, len(sel), 12):
                idx = sel[i:i + 12]
                a = x.isel({tdim: idx}).values.astype("float32"); a[(a < -100) | (a > 100)] = np.nan
                vals = C.to_half_degree(a, la, lo, "mean")[:, r, c]
                for j, k in enumerate(idx):
                    rows.append(pd.DataFrame({"var": v, "grid_id": new.grid_id.to_numpy(), "month": months[k], "value": vals[j]}))
        os.remove(f)
        print(f"monthly fill {v}: {len(new)} cells, {len(sel)} months, {time.time() - t:.0f} s; raw deleted", flush=True)
    out = pd.concat(rows); out["value"] = out.value.astype("float32")
    out.to_parquet(FILL, index=False)
    print("cells with data:", round(out.groupby("grid_id").value.apply(lambda s: s.notna().any()).mean(), 3))


def weekly(which=None):
    global OLD_IDS
    cl = C.cells(); wd = C.weekday()
    OLD_IDS = pd.read_parquet(os.path.join(C.COV, "cells_40S.parquet")).grid_id.to_numpy()
    for v in (which or list(VARS) + list(TIFVARS)):
        if not any(os.path.exists(cache(v, y)) for y in C.YEARS):
            continue
        fl = os.path.join(C.GRID, v, f"{v}_monthly_filled_cells.csv")
        if os.path.exists(fl):
            os.remove(fl)
        for y in C.YEARS:
            parts = [np.load(cache(v, yy)) for yy in (y, y + 1) if os.path.exists(cache(v, yy))]
            if not parts or not os.path.exists(cache(v, y)):
                print(y, v, "no daily cache"); continue
            def aligned(p):  # caches written before 2026-11-09 23:15 hold the 40S-72N cells and no grid_id
                g = p["grid_id"] if "grid_id" in p.files else OLD_IDS
                pos = pd.Series(np.arange(len(g)), index=g).reindex(cl.grid_id).to_numpy()
                out = np.full((len(p["dates"]), len(cl)), np.nan, "float32"); ok = ~np.isnan(pos)
                out[:, ok] = p["values"][:, pos[ok].astype(int)]
                return out
            dates = pd.DatetimeIndex(np.concatenate([p["dates"] for p in parts]))
            vals = np.concatenate([aligned(p) for p in parts])
            idx = pd.Series(np.arange(len(dates)), index=dates)
            W = C.weeks(y, wd); rows = []
            for w in W:
                days = pd.date_range(w, periods=7, freq="D")
                rows.append(vals[idx[days].to_numpy()].mean(0) if days.isin(dates).all()
                            else np.full(len(cl), np.nan, "float32"))
            m = np.stack(rows)
            if v in VARS and os.path.exists(FILL):   # cells with no daily data this year: the month's mean (flagged)
                fill = pd.read_parquet(FILL); fill = fill[fill["var"] == v]
                nod = np.isnan(m).all(0) & cl.grid_id.isin(fill.grid_id.unique()).to_numpy()
                if nod.any():
                    F = fill.pivot(index="month", columns="grid_id", values="value")
                    ids = cl.grid_id.to_numpy()[nod]
                    m[:, nod] = F.reindex(W.to_period("M").to_timestamp())[ids].to_numpy()
                    fl = os.path.join(C.GRID, v, f"{v}_monthly_filled_cells.csv")
                    pd.DataFrame({"grid_id": ids, "year": y}).to_csv(fl, mode="a", header=not os.path.exists(fl), index=False)
            C.write(v, y, np.tile(cl.grid_id, len(W)), np.repeat(W, len(cl)), m.ravel())
            print(f"{y} {v}: {len(W)} weeks, missing {np.isnan(m).mean():.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "daily":
        daily(int(sys.argv[2]))
    elif sys.argv[1] == "monthly_fill":
        monthly_fill()
    elif sys.argv[1] == "daily_tif":
        daily_tif(int(sys.argv[2]))
    else:
        weekly(sys.argv[2:] or None)
