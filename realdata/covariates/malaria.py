"""Malaria: Malaria Atlas Project PfPR 2-10 (annual, 2000-2025) per 0.5-degree cell-year; task 35, item 6.

  bash realdata/covariates/map_download.sh     # login node (network): raw/map/pfpr_<y>.tif (2.5 arcmin, 40S-72N)
  python realdata/covariates/malaria.py        # batch (needs grid/pop from ghsl_pop.py and raw/ghsl)

Cell value = population-weighted mean of the PfPR pixels in the cell, with weights = GHSL population of the year
(epochs interpolated as in ghsl_pop.py: log-linear between the 5-year epochs) binned to the 2.5-arcmin PfPR grid;
cells whose PfPR pixels have no population get the area mean. PfPR is defined only in the modelled (endemic) area;
outside it the cell is NaN in `value` and 0 in `value_fill0` (MAP models only areas at risk). Years 2026 onward: none
needed. Before 2000: missing (the model's missingness indicator handles it); 2025 is MAP's last year.
Output: grid/malaria/malaria_2000_2025.parquet (grid_id, year, value, value_fill0, n_pixels).
"""
from __future__ import annotations

import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import ghsl_pop as G  # noqa: E402

RES = 2.5 / 60
MAPD = os.path.join(C.RAW, "map")


def pop_on_map_grid(e):
    """GHSL population of epoch e summed onto the global 2.5-arcmin lattice, (180/RES, 360/RES)."""
    import rasterio
    n = G.NAME.format(e=e); z = os.path.join(G.RAWD, n + ".zip")
    tif = [m for m in zipfile.ZipFile(z).namelist() if m.endswith(".tif")][0]
    with rasterio.open(f"zip://{z}!{tif}") as src:
        a = src.read(1).astype("float32"); tr = src.transform
    a[a < 0] = np.nan
    lat = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5); lon = tr.c + tr.a * (np.arange(a.shape[1]) + 0.5)
    return C.to_half_degree(a[None], lat, lon, "sum", res=RES)[0]


def main():
    import rasterio
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    epochs = {}
    rows = []
    for y in range(2000, 2026):
        f = os.path.join(MAPD, f"pfpr_{y}.tif")
        if not os.path.exists(f):
            print(y, "missing raw"); continue
        t = time.time()
        e0 = max(e for e in G.EPOCHS if e <= y); e1 = min([e for e in G.EPOCHS if e >= y] or [e0])
        for e in (e0, e1):
            if e not in epochs:
                epochs[e] = np.nan_to_num(pop_on_map_grid(e)); print("  pop epoch", e, "binned", flush=True)
        for e in list(epochs):
            if e < e0:
                del epochs[e]
        a0, a1 = epochs[e0], epochs[e1]
        if e1 == e0:
            pop = a0
        else:
            fr = (y - e0) / (e1 - e0); both = (a0 > 0) & (a1 > 0)
            pop = np.where(both, np.exp(np.log(np.where(both, a0, 1)) * (1 - fr) + np.log(np.where(both, a1, 1)) * fr),
                           a0 * (1 - fr) + a1 * fr)
        with rasterio.open(f) as s:
            p = s.read(1).astype("float32"); tr = s.transform
        p[(p < 0) | (p > 1)] = np.nan
        lat = tr.f + tr.e * (np.arange(p.shape[0]) + 0.5); lon = tr.c + tr.a * (np.arange(p.shape[1]) + 0.5)
        # population on the same rows/cols as the PfPR raster (both on the global 2.5' lattice)
        R0 = int(round((90 - tr.f) / RES)); C0 = int(round((tr.c + 180) / RES))
        w = pop[R0:R0 + p.shape[0], C0:C0 + p.shape[1]].astype("float32")
        ok = np.isfinite(p)
        num = C.to_half_degree(np.where(ok, p * w, np.nan)[None], lat, lon, "sum")[0]
        den = C.to_half_degree(np.where(ok, w, np.nan)[None], lat, lon, "sum")[0]
        area = C.to_half_degree(p[None], lat, lon, "mean")[0]
        npx = C.to_half_degree(ok.astype("float32")[None], lat, lon, "sum")[0]
        v = np.where(den > 0, num / np.where(den > 0, den, 1), area)
        vv, nn = v[r, c], np.nan_to_num(npx[r, c])
        rows.append(pd.DataFrame({"grid_id": cl.grid_id, "year": y, "value": vv.astype("float32"),
                                  "value_fill0": np.where(np.isfinite(vv), vv, 0).astype("float32"), "n_pixels": nn.astype("int32")}))
        print(f"{y}: cells with PfPR {np.isfinite(vv).mean():.3f}, mean (where defined) {np.nanmean(vv):.4f}, {time.time() - t:.0f} s",
              flush=True)
    out = os.path.join(C.GRID, "malaria"); os.makedirs(out, exist_ok=True)
    pd.concat(rows).to_parquet(os.path.join(out, "malaria_2000_2025.parquet"), index=False)
    print("written")


if __name__ == "__main__":
    main()
