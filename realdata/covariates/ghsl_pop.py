"""GHSL GHS-POP R2023A (30 arcsec, EPSG:4326) population, epochs 1990-2025, summed to the 0.5-degree cells; yearly values by
interpolation between the 5-year epochs (log-linear where both epochs are > 0, linear otherwise). Task 35, item 3.

  python realdata/covariates/ghsl_pop.py download    # login node (network): raw/ghsl/*.zip, about 3.8 GB
  python realdata/covariates/ghsl_pop.py process     # batch job: grid/pop/pop_epochs.parquet and pop_1990_2025.parquet

Output pop_1990_2025.parquet: grid_id (int32), year, value (float32, persons in the cell).
Source: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

EPOCHS = list(range(1990, 2030, 5))
NAME = "GHS_POP_E{e}_GLOBE_R2023A_4326_30ss"
URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/{n}/V1-0/{n}_V1_0.zip"
RAWD = os.path.join(C.RAW, "ghsl")


def download():
    os.makedirs(RAWD, exist_ok=True)
    for e in EPOCHS:
        n = NAME.format(e=e); f = os.path.join(RAWD, n + ".zip")
        if os.path.exists(f):
            print(e, "present"); continue
        t = time.time()
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", f + ".part", URL.format(n=n)], check=True)
        os.replace(f + ".part", f)
        print(e, f"{os.path.getsize(f) / 1e6:.0f} MB in {time.time() - t:.0f} s", flush=True)


def epoch_sum(e):
    import rasterio
    n = NAME.format(e=e); z = os.path.join(RAWD, n + ".zip")
    tif = [m for m in zipfile.ZipFile(z).namelist() if m.endswith(".tif")][0]
    with rasterio.open(f"zip://{z}!{tif}") as src:
        a = src.read(1).astype("float32"); nod = src.nodata; tr = src.transform
        h, w = a.shape
    if nod is not None:
        a[a == nod] = np.nan
    a[a < 0] = np.nan
    lat = tr.f + tr.e * (np.arange(h) + 0.5); lon = tr.c + tr.a * (np.arange(w) + 0.5)
    # pad rows so that the raster starts and ends on 0.5-degree lattice rows (GHSL may not span the full 90S-90N)
    k = int(round(0.5 / abs(tr.e)))
    return C.to_half_degree(a[None], lat, lon, "sum")[0]


def process():
    cl = C.cells(); r, c = cl.row.to_numpy(), cl.col.to_numpy()
    out = os.path.join(C.GRID, "pop"); os.makedirs(out, exist_ok=True)
    E = {}
    for e in EPOCHS:
        t = time.time(); g = epoch_sum(e); E[e] = g[r, c]
        print(e, f"total in cells {np.nansum(E[e]) / 1e9:.3f} bn, NaN cells {np.isnan(E[e]).mean():.3f}, {time.time() - t:.0f} s",
              flush=True)
    pd.DataFrame({"grid_id": np.tile(cl.grid_id, len(EPOCHS)), "epoch": np.repeat(EPOCHS, len(cl)),
                  "value": np.concatenate([E[e] for e in EPOCHS]).astype("float32")}).to_parquet(
        os.path.join(out, "pop_epochs.parquet"), index=False)
    rows = []
    for y in C.YEARS:
        e0 = max(e for e in EPOCHS if e <= y); e1 = min([e for e in EPOCHS if e >= y] or [e0])
        a, b = np.nan_to_num(E[e0]), np.nan_to_num(E[e1])
        if e1 == e0:
            v = a
        else:
            f = (y - e0) / (e1 - e0)
            both = (a > 0) & (b > 0)
            v = np.where(both, np.exp(np.log(np.where(both, a, 1)) * (1 - f) + np.log(np.where(both, b, 1)) * f),
                         a * (1 - f) + b * f)
        rows.append(v.astype("float32"))
    pd.DataFrame({"grid_id": np.tile(cl.grid_id, len(C.YEARS)), "year": np.repeat(list(C.YEARS), len(cl)),
                  "value": np.concatenate(rows)}).to_parquet(os.path.join(out, "pop_1990_2025.parquet"), index=False)
    print("written", os.path.join(out, "pop_1990_2025.parquet"))


if __name__ == "__main__":
    {"download": download, "process": process}[sys.argv[1]]()
