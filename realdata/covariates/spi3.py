"""SPI-3 (Standardized Precipitation Index, 3-month) per 0.5-degree cell from CHIRPS monthly totals; task 35, item 1
(replaces the GDO drought index).

  python realdata/covariates/spi3.py      # after chirps_weekly.py process

3-month totals P3(m) = P(m-2) + P(m-1) + P(m). Per cell and calendar month, a gamma distribution is fitted to the
non-zero P3 of the baseline 1991-2020 (Thom's maximum-likelihood approximation), with q = share of zeros
(mixed distribution H(x) = q + (1 - q) G(x)); SPI = Phi^-1(H(P3)), clipped to [-3.5, 3.5]. Cells with fewer than 20
non-zero baseline values or no CHIRPS data are NaN. The first two months of 1990 have no SPI-3 (no December 1989).
Outputs: grid/spi3_month/spi3_month.parquet (grid_id, month, value) and grid/spi3/spi3_<year>.parquet (weekly: the SPI-3
of the calendar month containing week_start).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import special, stats

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

BASE = (1991, 2020)


def main():
    cl = C.cells()
    mo = pd.concat([pd.read_parquet(os.path.join(C.GRID, "ppt_month", f"ppt_month_{y}.parquet")) for y in C.YEARS])
    P = mo.pivot(index="month", columns="grid_id", values="value").reindex(columns=cl.grid_id).sort_index()
    months = P.index; X = P.to_numpy("float64")                       # (nmonth, ncell)
    P3 = np.full_like(X, np.nan); P3[2:] = X[2:] + X[1:-1] + X[:-2]
    spi = np.full_like(X, np.nan)
    inb = (months.year >= BASE[0]) & (months.year <= BASE[1])
    for m in range(1, 13):
        sel = months.month == m
        b = P3[sel & inb]                                              # (nyear, ncell)
        pos = np.where(b > 0, b, np.nan)
        n = np.isfinite(pos).sum(0); nall = np.isfinite(b).sum(0)
        q = np.where(nall > 0, (b == 0).sum(0) / np.maximum(nall, 1), np.nan)
        mean = np.nanmean(pos, 0); A = np.log(mean) - np.nanmean(np.log(pos), 0)
        alpha = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A); beta = mean / alpha
        x = P3[sel]
        G = special.gammainc(alpha[None, :], np.maximum(x, 0) / beta[None, :])
        H = q[None, :] + (1 - q[None, :]) * np.where(x > 0, G, 0.0)
        z = stats.norm.ppf(np.clip(H, 1e-10, 1 - 1e-10))
        ok = (n >= 20) & np.isfinite(alpha) & (A > 0)
        spi[sel] = np.where(ok[None, :] & np.isfinite(x), np.clip(z, -3.5, 3.5), np.nan)
    os.makedirs(os.path.join(C.GRID, "spi3_month"), exist_ok=True)
    pd.DataFrame({"grid_id": np.tile(cl.grid_id.to_numpy(), len(months)), "month": np.repeat(months, len(cl)),
                  "value": spi.ravel().astype("float32")}).to_parquet(os.path.join(C.GRID, "spi3_month", "spi3_month.parquet"), index=False)
    S = pd.DataFrame(spi, index=months, columns=cl.grid_id)
    wd = C.weekday()
    for y in C.YEARS:
        W = C.weeks(y, wd); mstart = W.to_period("M").to_timestamp()
        v = S.reindex(mstart).to_numpy()
        C.write("spi3", y, np.tile(cl.grid_id, len(W)), np.repeat(W, len(cl)), v.ravel())
    ins = spi[(months.year >= 1991)]
    print(f"SPI-3: {len(months)} months x {len(cl)} cells; missing {np.isnan(ins).mean():.3f} (1991-2025); "
          f"baseline mean {np.nanmean(spi[inb]):.3f} sd {np.nanstd(spi[inb]):.3f} (should be about 0 and 1)")


if __name__ == "__main__":
    main()
