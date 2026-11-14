"""Preliminary consistency check of the ERA5 daily precipitation downloads (task 35, ERA5 fill), on whole years already downloaded.

  python realdata/covariates/era5_check.py 2000 2015

Per year: ERA5 daily totals (mm/day = daily mean of hourly tp x 24 x 1000) aggregated to the 0.5-degree cells (era5_precip.py),
then
 1. plausibility: share of negative values, daily maximum, annual totals by band and for some countries north of 50N;
 2. overlap with CHIRPS (cells 40N-50N, where both exist): daily and weekly (7-day sums) Pearson / Spearman correlation,
    mean annual totals, ratio ERA5/CHIRPS, and the share of wet days (> 1 mm) in each.
Prints a summary; nothing is written except results_real/era5_check_<years>.json (aggregates only).
"""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C, era5_precip as E  # noqa: E402

cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy(); gid = cl.grid_id.to_numpy()
out = {}
for y in map(int, sys.argv[1:]):
    vals, dates = None, None
    for band in E.BANDS:
        arrs, ds_ = [], []
        for m in range(1, 13):
            d, p, la, lo = E.read_month(band, y, m)
            ds_.append(d); arrs.append(E.to_cells_era5(p, la, lo)[:, r, c])
        a = np.concatenate(arrs); d = ds_[0].append(ds_[1:])
        vals = a if vals is None else np.where(np.isfinite(vals), vals, a); dates = d
    era = pd.DataFrame(vals, index=dates, columns=gid)
    has = era.notna().any().to_numpy()
    res = {"days": int(len(dates)), "cells_with_era5": int(has.sum()),
           "share_negative": float((era.to_numpy() < 0).mean()), "daily_max_mm": float(np.nanmax(era.to_numpy())),
           "daily_min_mm": float(np.nanmin(era.to_numpy()))}
    ann = era.sum(min_count=300)
    lat = cl.lat.to_numpy()
    res["annual_mean_mm_by_band"] = {b: float(np.nanmean(ann.to_numpy()[k])) for b, k in
                                     (("50-60N", (lat > 50) & (lat <= 60)), ("60-72N", lat > 60), ("40-50N", (lat > 40) & (lat <= 50)),
                                      ("60-50S", lat < -50))}
    res["annual_mean_mm_by_country"] = {iso: float(np.nanmean(ann.to_numpy()[(cl.ISO_A3 == iso).to_numpy() & (lat > 50)]))
                                        for iso in ("GBR", "IRL", "NOR", "SWE", "FIN", "POL", "DEU", "BLR", "RUS", "CAN", "UKR", "KAZ")}
    # overlap with CHIRPS
    z = np.load(os.path.join(C.COV, "daily", f"ppt_{y}.npz"), allow_pickle=True)
    ch = pd.DataFrame(z["values"], index=pd.DatetimeIndex(z["dates"]), columns=z["grid_id"]).reindex(columns=gid)
    ov = gid[(lat > 40) & (lat < 50)]
    a = era.reindex(ch.index)[ov]; b = ch[ov]
    ok = a.notna() & b.notna()
    av, bv = a.to_numpy()[ok.to_numpy()], b.to_numpy()[ok.to_numpy()]
    wa, wb = a.rolling(7).sum().iloc[6::7], b.rolling(7).sum().iloc[6::7]
    okw = wa.notna() & wb.notna(); wav, wbv = wa.to_numpy()[okw.to_numpy()], wb.to_numpy()[okw.to_numpy()]
    res["overlap_40_50N"] = {
        "cells": int(ok.any().sum()), "daily_pearson": float(np.corrcoef(av, bv)[0, 1]),
        "daily_spearman": float(pd.Series(av).corr(pd.Series(bv), method="spearman")),
        "weekly_pearson": float(np.corrcoef(wav, wbv)[0, 1]), "weekly_spearman": float(pd.Series(wav).corr(pd.Series(wbv), method="spearman")),
        "annual_mean_mm_era5": float(np.nanmean(a.sum(min_count=300))), "annual_mean_mm_chirps": float(np.nanmean(b.sum(min_count=300))),
        "wet_day_share_era5": float((av > 1).mean()), "wet_day_share_chirps": float((bv > 1).mean())}
    res["overlap_40_50N"]["ratio_era5_over_chirps"] = res["overlap_40_50N"]["annual_mean_mm_era5"] / res["overlap_40_50N"]["annual_mean_mm_chirps"]
    out[y] = res
    print(y, json.dumps(res, indent=1), flush=True)
od = os.path.join(os.path.dirname(__file__), "..", "..", "results_real"); os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, f"era5_check_{'_'.join(sys.argv[1:])}.json"), "w"), indent=1)
