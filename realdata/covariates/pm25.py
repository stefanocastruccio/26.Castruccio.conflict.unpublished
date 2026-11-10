"""PM2.5: Washington University ACAG satellite-derived surface PM2.5, V6.GL.02.04, monthly, 0.1 deg (1998-2023; CC BY 4.0).
Replaces CAMS (which needs an account). Task 35 (all of Luca's covariates).

  python realdata/covariates/pm25.py download   # login node (network): raw/pm25/*.nc from the public Box share (~1.7 GB)
  python realdata/covariates/pm25.py process    # monthly area mean per 0.5-deg cell; weekly = the month containing week_start

Source: https://sites.wustl.edu/acag/datasets/surface-pm2-5/ -> Box share s7eiaxytjr9w1z7glat45cesitcemprv
(folder V6.GL.02.04-0p10 / GL / Monthly / <year> / V6GL02.04.CNNPM25.0p10.GL.<yyyymm>-<yyyymm>.nc).
Outputs (study-region cells only): grid/pm25_month/pm25_month.parquet (grid_id, month, value [ug/m3]) and
grid/pm25/pm25_<year>.parquet (weekly). 1990-1997 and 2024-2025 are not covered (missing).
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

SHARE = "s7eiaxytjr9w1z7glat45cesitcemprv"
BASE = f"https://wustl.box.com/s/{SHARE}"
MONTHLY = 327765383385          # V6.GL.02.04-0p10 / GL / Monthly
RAWD = os.path.join(C.RAW, "pm25")
_cj = http.cookiejar.CookieJar()
_op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cj))
_op.addheaders = [("User-Agent", "Mozilla/5.0 (research covariates)")]


def items(folder, page=1):
    h = _op.open(f"{BASE}/folder/{folder}?page={page}", timeout=120).read().decode()
    i = h.index('"items":['); j = h.index("[", i); depth = 0
    for k in range(j, len(h)):
        depth += h[k] == "["; depth -= h[k] == "]"
        if depth == 0:
            break
    pc = re.search(r'"pageCount":(\d+)', h)
    return json.loads(h[j:k + 1]), int(pc.group(1)) if pc else 1


def all_items(folder):
    out, p, n = [], 1, 1
    while p <= n:
        its, n = items(folder, p); out += its; p += 1
    return out


def download():
    os.makedirs(RAWD, exist_ok=True)
    _op.open(BASE, timeout=120).read()                       # session cookie
    years = [it for it in all_items(MONTHLY) if it["type"] == "folder"]
    print(len(years), "year folders:", sorted(y["name"] for y in years), flush=True)
    for y in sorted(years, key=lambda it: it["name"]):
        for f in [it for it in all_items(y["id"]) if it["type"] == "file" and it["name"].endswith(".nc")]:
            dst = os.path.join(RAWD, f["name"])
            if os.path.exists(dst) and os.path.getsize(dst) == f.get("itemSize"):
                continue
            url = f"https://wustl.box.com/index.php?rm=box_download_shared_file&shared_name={SHARE}&file_id=f_{f['id']}"
            for k in range(5):
                try:
                    data = _op.open(url, timeout=600).read()
                    if data[:4] == b"\x89HDF":
                        open(dst + ".part", "wb").write(data); os.replace(dst + ".part", dst); break
                    raise IOError("not a netCDF/HDF5 file")
                except Exception as e:
                    print(f"  {f['name']}: {e}; retry {k + 1}", flush=True); time.sleep(30 * (k + 1))
            time.sleep(1)
        print(y["name"], "done", flush=True)


def process():
    import xarray as xr
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    files = sorted(f for f in os.listdir(RAWD) if f.endswith(".nc"))
    months, vals = [], []
    for f in files:
        ym = re.search(r"\.(\d{6})-\d{6}\.nc$", f).group(1)
        with xr.open_dataset(os.path.join(RAWD, f)) as ds:
            name = [k for k, q in ds.data_vars.items() if q.ndim == 2][0]
            x = ds[name]; ydim, xdim = x.dims
            a = x.values.astype("float32")[None]
            a[(a < 0) | (a > 2000)] = np.nan
            vals.append(C.to_half_degree(a, ds[ydim].values, ds[xdim].values, "mean")[0][r, c])
        months.append(pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01"))
    M = pd.DataFrame(np.stack(vals), index=pd.DatetimeIndex(months), columns=cl.grid_id).sort_index()
    keep = C.region_cells() if os.environ.get("COV_REGIONS_ONLY") == "1" else cl.grid_id.to_numpy()
    od = os.path.join(C.GRID, "pm25_month"); os.makedirs(od, exist_ok=True)
    long = M.stack(future_stack=True).rename("value").reset_index().rename(columns={"level_0": "month", "index": "month"})
    long.columns = ["month", "grid_id", "value"]
    long[long.grid_id.isin(keep)].astype({"value": "float32"}).to_parquet(os.path.join(od, "pm25_month.parquet"), index=False)
    wd = C.weekday()
    for y in C.YEARS:
        W = C.weeks(y, wd); v = M.reindex(W.to_period("M").to_timestamp()).to_numpy()
        C.write("pm25", y, np.tile(cl.grid_id, len(W)), np.repeat(W, len(cl)), v.ravel())
    print(f"PM2.5: {len(M)} months {M.index.min().date()}..{M.index.max().date()}; in-region missing share "
          f"{np.isnan(M[keep].to_numpy()).mean():.3f}; mean {np.nanmean(M[keep].to_numpy()):.1f} ug/m3")


if __name__ == "__main__":
    {"download": download, "process": process}[sys.argv[1]]()
