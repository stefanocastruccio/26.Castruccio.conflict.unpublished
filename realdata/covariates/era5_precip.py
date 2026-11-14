"""ERA5 daily precipitation for the cells outside the CHIRPS domain (north of 50N; 50S-60S), task 35 (owner, 2026-11-11:
"ERA5 for the cells north of 50N"; CHIRPS v2.0 stops at 50N/50S).

  python realdata/covariates/era5_precip.py download [--years 1990 2026] [--workers 4]   # login node; needs ~/.cdsapirc
  python realdata/covariates/era5_precip.py process  [--years 1990 2026]                 # batch: daily/ppt_era5_<y>.npz
  python realdata/covariates/era5_precip.py merge                                        # batch: fill ppt, ppt_month, daily ppt
  then spi3.py (SPI-3 for the filled cells, same 1991-2020 baseline from ERA5 itself).

Source: Copernicus CDS "derived-era5-single-levels-daily-statistics" (ERA5 post-processed daily statistics, CC-BY,
doi:10.24381/cds.4991cf48), total_precipitation, daily_mean of the hourly values over the UTC day, 0.25 deg. Daily total
(mm) = daily mean (m per hour) x 24 x 1000. Bands: north 39.5N-72.5N (the 40N-50N part overlaps CHIRPS and is used only
for the comparison), south 60S-50S.
Aggregation to the 0.5-degree cells: ERA5 points sit on multiples of 0.25 deg, i.e. on the cell edges and centres; each
point stands for the 0.25 x 0.25 box around it, so a cell is the cos(lat)-weighted mean of its 3 x 3 points with weights
1/2, 1, 1/2 in each direction (exact area overlap).
Merge: cells where CHIRPS has no data (no finite daily value in 2000) take ERA5 for all years; grid/ppt_source.parquet
(grid_id, source in {chirps, era5_scaled, era5, none}) records the source. Weekly totals as for CHIRPS (NaN if any day is missing).
Adjustment (owner, 2026-11-14): warm-season level scaling only. ERA5 is about 15% wetter than CHIRPS where both exist and has more
light-rain days (era5_check.py). North of 50N, for May-October, ERA5 daily values are multiplied by f(month, 30-degree longitude
sector) = sum CHIRPS / sum ERA5 over the cells at 45-50N with both, 1991-2020 (clipped to [0.5, 2]; grid/ppt_era5_scaling.csv;
warm-season factors mostly 0.8-1.1). November-April is left as ERA5, because CHIRPS (satellite IR) underestimates cold-season and
solid precipitation (winter factors 0.5-0.8 there). The number of rain days is not changed: quantile mapping and LOCI calibrated at
45-50N were tested (2015) and impose that band's wet-day frequency on the north, removing 19-24% of northern May-October rain
(qm_setup kept for reference, not used). Source flags: era5_scaled (north of 50N), era5 (south of 50S, no overlap, unadjusted).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

DS = "derived-era5-single-levels-daily-statistics"
RAWD = os.path.join(C.RAW, "era5")
DAILY = os.path.join(C.COV, "daily")
BANDS = {"north": [72.5, -180, 39.5, 180], "south": [-50, -180, -60, 180]}   # N, W, S, E


def raw(band, y, m):
    return os.path.join(RAWD, f"tp_{band}_{y}{m:02d}.nc")


def request(band, y, m):
    import cdsapi
    f = raw(band, y, m)
    if os.path.exists(f):
        return f"{band} {y}-{m:02d} present"
    nd = pd.Period(f"{y}-{m:02d}").days_in_month
    if pd.Timestamp(f"{y}-{m:02d}-01") > pd.Timestamp.today() - pd.Timedelta(days=10):
        return f"{band} {y}-{m:02d} not yet available"
    t = time.time()
    c = cdsapi.Client(quiet=True, progress=False)
    tmp = f + ".part"
    c.retrieve(DS, {"product_type": "reanalysis", "variable": ["total_precipitation"], "year": str(y),
                    "month": [f"{m:02d}"], "day": [f"{d:02d}" for d in range(1, nd + 1)],
                    "daily_statistic": "daily_mean", "time_zone": "utc+00:00", "frequency": "1_hourly",
                    "area": BANDS[band]}, tmp)
    if zipfile.is_zipfile(tmp):                     # some CDS products deliver a zip with one netCDF
        with zipfile.ZipFile(tmp) as z:
            n = [x for x in z.namelist() if x.endswith(".nc")][0]
            with z.open(n) as src, open(tmp + ".nc", "wb") as dst:
                dst.write(src.read())
        os.replace(tmp + ".nc", tmp)
    os.replace(tmp, f)
    return f"{band} {y}-{m:02d} {os.path.getsize(f) / 1e6:.1f} MB in {time.time() - t:.0f} s"


def download(years, workers):
    os.makedirs(RAWD, exist_ok=True)
    # after 2025 only January is needed (the last 2025 week runs into the first days of 2026)
    jobs = [(b, y, m) for y in years for m in (range(1, 13) if y <= C.YEARS[-1] else [1]) for b in BANDS]
    with ThreadPoolExecutor(workers) as ex:
        for fut in [ex.submit(request, *j) for j in jobs]:
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # keep going; rerun to fill gaps
                print("FAILED", repr(e)[:300], flush=True)


def to_cells_era5(a, lat, lon):
    """(T, nlat, nlon) on the 0.25-deg ERA5 grid -> (T, 360, 720) on the 0.5-deg lattice, exact area weights."""
    a = np.asarray(a, "float64"); lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if lat[0] > lat[-1]:
        a, lat = a[:, ::-1], lat[::-1]
    lon = ((lon + 180) % 360) - 180
    keep = np.r_[True, np.diff(np.sort(lon)) > 1e-6]
    o = np.argsort(lon, kind="stable")[keep]; a, lon = a[:, :, o], lon[o]
    assert len(lon) == 1440 and abs(lon[0] + 180) < 1e-6, "expects the full 0.25-deg longitude circle"
    T = a.shape[0]
    ok = np.isfinite(a); x = np.where(ok, a, 0.0)
    wl = np.cos(np.deg2rad(lat))[None, :, None]
    x, n = x * wl, ok * wl
    # longitude: cell column c covers points 2c, 2c+1, 2c+2 (wrapping) with weights 1/2, 1, 1/2
    def lonagg(z):
        return 0.5 * z[:, :, 0::2] + z[:, :, 1::2] + 0.5 * np.roll(z, -2, axis=2)[:, :, 0::2]
    x, n = lonagg(x), lonagg(n)                                     # (T, nlat, 720)
    out = np.full((T, 360, 720), np.nan, "float32")
    i0 = np.round((lat - lat[0]) / 0.25).astype(int)
    assert np.all(i0 == np.arange(len(lat))), "expects a regular 0.25-deg latitude grid"
    first = int(np.ceil((lat[0] + 90) / 0.5 - 1e-9))               # first lattice row whose south edge is in the grid
    for R in range(first, 360):
        s = -90 + 0.5 * R
        j = int(round((s - lat[0]) / 0.25))
        if j + 2 >= len(lat):
            break
        xs = 0.5 * x[:, j] + x[:, j + 1] + 0.5 * x[:, j + 2]
        ns = 0.5 * n[:, j] + n[:, j + 1] + 0.5 * n[:, j + 2]
        out[:, R] = np.where(ns > 0, xs / np.where(ns > 0, ns, 1), np.nan)
    return out


def read_month(band, y, m):
    import xarray as xr
    with xr.open_dataset(raw(band, y, m)) as ds:
        v = [k for k in ds.data_vars if k in ("tp", "total_precipitation")][0]
        tdim = [d for d in ds[v].dims if d not in ("latitude", "longitude")][0]
        p = ds[v].transpose(tdim, "latitude", "longitude").values * 24 * 1000.0      # mm/day
        return (pd.DatetimeIndex(ds[tdim].values).normalize(), p, ds["latitude"].values, ds["longitude"].values)


def process(years):
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    os.makedirs(DAILY, exist_ok=True)
    for y in years:
        f = os.path.join(DAILY, f"ppt_era5_{y}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time(); dates = None; vals = None
        for band in BANDS:
            ds_, arrs = [], []
            for m in range(1, 13):
                if not os.path.exists(raw(band, y, m)):
                    continue
                d, p, la, lo = read_month(band, y, m)
                ds_.append(d); arrs.append(to_cells_era5(p, la, lo)[:, r, c])
            if not arrs:
                continue
            d = ds_[0].append(ds_[1:]) if len(ds_) > 1 else ds_[0]; a = np.concatenate(arrs)
            if dates is None:
                dates, vals = d, a
            else:
                assert dates.equals(d), "bands cover different days"
                vals = np.where(np.isfinite(vals), vals, a)
        if dates is None:
            print(y, "no ERA5 files"); continue
        np.savez(f + ".tmp.npz", dates=dates.values, values=vals.astype("float32"), grid_id=cl.grid_id.to_numpy())
        os.replace(f + ".tmp.npz", f)
        print(f"{y}: {len(dates)} days, cells with ERA5 {np.isfinite(vals).any(0).mean():.3f}, {time.time() - t0:.0f} s", flush=True)


def load_daily(name, y, grid_id):
    z = np.load(os.path.join(DAILY, f"{name}_{y}.npz"), allow_pickle=True)
    v = pd.DataFrame(z["values"], index=pd.DatetimeIndex(z["dates"]), columns=z["grid_id"])
    return v.reindex(columns=grid_id)


def scaling_factors(cl, years=range(1991, 2021)):
    """(12, 12) factors [calendar month, 30-degree longitude sector]: sum CHIRPS / sum ERA5 over cells 45-50N with both."""
    gid = cl.grid_id.to_numpy(); lat = cl.lat.to_numpy(); sec = ((cl.lon.to_numpy() + 180) // 30).astype(int)
    ov = (lat > 45) & (lat < 50); sc, se = np.zeros((12, 12)), np.zeros((12, 12)); used = []
    for y in years:
        if not os.path.exists(os.path.join(DAILY, f"ppt_era5_{y}.npz")):
            continue
        ch = load_daily("ppt", y, gid[ov]); er = load_daily("ppt_era5", y, gid[ov]).reindex(ch.index)
        a, b = ch.to_numpy(), er.to_numpy(); ok = np.isfinite(a) & np.isfinite(b)
        mon = ch.index.month.to_numpy() - 1
        for k in range(12):
            m = ok & (sec[ov][None, :] == k)
            np.add.at(sc, (mon, k), np.where(m, a, 0).sum(1)); np.add.at(se, (mon, k), np.where(m, b, 0).sum(1))
        used.append(y)
    f = np.clip(np.where(se > 0, sc / np.maximum(se, 1e-9), 1.0), 0.5, 2.0)
    pd.DataFrame(f, index=pd.Index(range(1, 13), name="month"), columns=[f"{-180 + 30 * k}E" for k in range(12)]).to_csv(
        os.path.join(C.GRID, "ppt_era5_scaling.csv"))
    print(f"ERA5 scaling factors from {len(used)} years ({used[:1]}..{used[-1:]}); range {f.min():.2f}-{f.max():.2f}, "
          f"median {np.median(f):.2f}", flush=True)
    return f


QM_MONTHS = (5, 6, 7, 8, 9, 10)
QM_P = np.r_[np.linspace(0.01, 0.99, 99), 0.995, 0.999]


def qm_setup(cl, sel, years=range(1991, 2021)):
    """Quantile-mapping tables: R[m, k, :] (CHIRPS/ERA5 quantile ratios at 45-50N per month and sector) and, for each ERA5 cell
    north of 50N, its own 1991-2020 ERA5 quantiles per month Qc[m][cell] (at QM_P)."""
    gid = cl.grid_id.to_numpy(); lat = cl.lat.to_numpy(); sec = ((cl.lon.to_numpy() + 180) // 30).astype(int)
    ov = gid[(lat > 45) & (lat < 50)]; secov = sec[(lat > 45) & (lat < 50)]
    yrs = [y for y in years if os.path.exists(os.path.join(DAILY, f"ppt_era5_{y}.npz"))]
    pe = {(m, k): [] for m in QM_MONTHS for k in range(12)}; pc = {(m, k): [] for m in QM_MONTHS for k in range(12)}
    nb = {m: [] for m in QM_MONTHS}
    for y in yrs:
        ch = load_daily("ppt", y, ov); er = load_daily("ppt_era5", y, ov).reindex(ch.index)
        en = load_daily("ppt_era5", y, sel)
        for m in QM_MONTHS:
            a = ch[ch.index.month == m].to_numpy(); b = er[er.index.month == m].to_numpy(); ok = np.isfinite(a) & np.isfinite(b)
            for k in range(12):
                kk = ok & (secov[None, :] == k)
                pc[(m, k)].append(a[kk]); pe[(m, k)].append(b[kk])
            nb[m].append(en[en.index.month == m].to_numpy("float32"))
    R = np.ones((13, 12, len(QM_P))); have = np.zeros((13, 12), bool)
    for (m, k), v in pe.items():
        e_ = np.concatenate(v) if v else np.array([]); c_ = np.concatenate(pc[(m, k)]) if pc[(m, k)] else np.array([])
        if len(e_) < 1000:
            continue
        QE, QC = np.quantile(e_, QM_P), np.quantile(c_, QM_P)
        R[m, k] = np.clip(np.where(QE > 0, QC / np.where(QE > 0, QE, 1), np.where(QC > 0, 1.0, 0.0)), 0, 3); have[m, k] = True
    Qc = {m: np.nanquantile(np.concatenate(nb[m]), QM_P, axis=0).T for m in QM_MONTHS}   # cells x len(QM_P)
    pd.DataFrame([dict(month=m, sector_lon=-180 + 30 * k, **{f"R_p{int(p * 1000):03d}": R[m, k, i] for i, p in enumerate(QM_P)
                  if p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}) for m in QM_MONTHS for k in range(12) if have[m, k]]).to_csv(
        os.path.join(C.GRID, "ppt_era5_qm_ratios.csv"), index=False)
    print(f"QM tables from {len(yrs)} years; sectors with overlap per month: {have[5].sum()}; "
          f"median R at p=0.5 {np.median(R[list(QM_MONTHS)][:, :, 49][have[list(QM_MONTHS)]]):.2f}", flush=True)
    return R, have, Qc


def merge():
    cl = C.cells(); gid = cl.grid_id.to_numpy(); wd = C.weekday()
    sf = os.path.join(C.GRID, "ppt_source.parquet")
    if os.path.exists(sf):   # rerun: the daily ppt cache already holds the filled cells, so keep the first classification
        src = pd.read_parquet(sf).set_index("grid_id").reindex(gid).reset_index()
        use = src.source.str.startswith("era5").to_numpy()
    else:
        ch = load_daily("ppt", 2000, gid)
        use = ~np.isfinite(ch.to_numpy()).any(0)                    # cells with no CHIRPS data
        e0 = load_daily("ppt_era5", 2000, gid)
        use &= np.isfinite(e0.to_numpy()).any(0)
        src = pd.DataFrame({"grid_id": gid.astype("int32"), "source": np.where(use, "era5", "chirps")})
        no = ~np.isfinite(ch.to_numpy()).any(0) & ~use
        src.loc[no, "source"] = "none"
    north = use & (cl.lat.to_numpy() > 50)
    src.loc[north, "source"] = "era5_scaled"; src.loc[use & ~north, "source"] = "era5"
    src.to_parquet(sf, index=False)
    print("cells by source:", src.source.value_counts().to_dict(), flush=True)
    F = scaling_factors(cl)   # (12 months, 12 sectors)
    sel = gid[use]
    secsel = ((cl.lon.to_numpy()[use] + 180) // 30).astype(int); nsel = north[use]
    warm = np.isin(np.arange(1, 13), QM_MONTHS)

    def scale(df):   # df: days x sel cells (ERA5, mm/day) -> May-October scaled north of 50N; November-April unchanged
        mo = df.index.month.to_numpy() - 1
        fac = np.where(warm[mo][:, None] & nsel[None, :], F[mo][:, secsel], 1.0)
        return df * fac
    comp = []
    for y in C.YEARS:
        e = load_daily("ppt_era5", y, gid)
        nx = os.path.join(DAILY, f"ppt_era5_{y + 1}.npz")
        if os.path.exists(nx):
            e = pd.concat([e, load_daily("ppt_era5", y + 1, gid).iloc[:6]])
        E = scale(e[sel])
        W = C.weeks(y, wd); rows = []
        for w in W:
            days = pd.date_range(w, periods=7, freq="D")
            rows.append(E.reindex(days).sum(0, min_count=7).to_numpy() if days.isin(E.index).all()
                        else np.full(len(sel), np.nan))
        wk = pd.DataFrame(np.stack(rows), index=W, columns=sel)
        # weekly ppt: replace the ERA5 cells
        f = os.path.join(C.GRID, "ppt", f"ppt_{y}.parquet")
        p = pd.read_parquet(f)
        long = wk.stack(future_stack=True).rename("v").reset_index(); long.columns = ["week_start", "grid_id", "v"]
        p = p.merge(long, on=["grid_id", "week_start"], how="left")
        p["value"] = np.where(p.grid_id.isin(sel), p.v, p.value).astype("float32"); p = p.drop(columns="v")
        C.write("ppt", y, p.grid_id, p.week_start, p.value)
        # monthly totals
        fm = os.path.join(C.GRID, "ppt_month", f"ppt_month_{y}.parquet")
        pm = pd.read_parquet(fm)
        mon = E[E.index.year == y].groupby(E[E.index.year == y].index.month).sum(min_count=1)
        ml = mon.stack(future_stack=True).rename("v").reset_index(); ml.columns = ["m", "grid_id", "v"]
        ml["month"] = pd.to_datetime(dict(year=y, month=ml.m, day=1)); ml = ml.drop(columns="m")
        pm = pm.merge(ml, on=["grid_id", "month"], how="left")
        pm["value"] = np.where(pm.grid_id.isin(sel), pm.v, pm.value).astype("float32")
        pm.drop(columns="v").to_parquet(fm + ".tmp", index=False); os.replace(fm + ".tmp", fm)
        # daily ppt cache: fill the ERA5 cells (CHIRPS values untouched)
        fd = os.path.join(DAILY, f"ppt_{y}.npz")
        z = dict(np.load(fd, allow_pickle=True)); d = pd.DatetimeIndex(z["dates"])
        col = np.isin(z["grid_id"], sel)
        ev = scale(load_daily("ppt_era5", y, sel).reindex(d)).reindex(columns=z["grid_id"][col]).to_numpy("float32")
        z["values"][:, col] = ev
        np.savez(fd + ".tmp.npz", **z); os.replace(fd + ".tmp.npz", fd)
        # comparison where both exist (40N-50N), weekly totals
        if y >= 2000:
            ov = gid[(cl.lat.to_numpy() > 40) & (cl.lat.to_numpy() < 50)]
            chd = load_daily("ppt", y, gid)[ov]; ed = e[ov]
            a = chd.resample("W").sum(min_count=7); b = ed.resample("W").sum(min_count=7).reindex(a.index)
            ok = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
            comp.append((a.to_numpy()[ok], b.to_numpy()[ok]))
        print(y, "merged", flush=True)
    if comp:
        a = np.concatenate([x[0] for x in comp]); b = np.concatenate([x[1] for x in comp])
        res = {"n": int(len(a)), "pearson": float(np.corrcoef(a, b)[0, 1]),
               "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
               "mean_chirps": float(a.mean()), "mean_era5": float(b.mean())}
        print("CHIRPS vs ERA5 weekly totals, 40N-50N, 2000-2025:", res)
        pd.Series(res).to_json(os.path.join(C.GRID, "ppt_era5_vs_chirps.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "process", "merge"])
    ap.add_argument("--years", type=int, nargs=2, default=[1990, 2026])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    yrs = range(a.years[0], a.years[1] + 1)
    if a.step == "download":
        download(yrs, a.workers)
    elif a.step == "process":
        process(yrs)
    else:
        merge()
