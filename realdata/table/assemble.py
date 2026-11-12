"""Task 40, part C: the model-ready global table = CORE (A) joined with whichever covariate files (B) exist.

  python realdata/table/assemble.py [--years 1997 2025]
  -> $WORK/table_global/clean_global/year=<y>.parquet

Weekly covariates join on (grid_id, week_start); yearly ones on (grid_id, year of week_start); national ones on (ISO_A3, year);
static ones on grid_id. Every expected covariate column whose file does not exist yet is added as NaN, so the schema is stable.
Adding or replacing a covariate: join_cov.py --var <var>, then this script (minutes).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "covariates"))
import common as C  # noqa: E402

T = os.path.join(C.WORK, "table_global")
EXPECTED = {
    "tmax": ["tmax", "tmax_rollmean_4"], "tmin": ["tmin", "tmin_rollmean_4"], "hi": ["hi", "hi_rollmean_4"],
    "wbgt": ["wbgt", "wbgt_rollmean_4"], "ppt": ["ppt", "ppt_rollmean_4", "ppt_pending_era5"],
    "spi3": ["drought", "drought_rollmean_4", "drought_pending_era5"], "pm25": ["pm25", "pm25_rollmean_4"],
    "ipc": [f"phase_{k}" for k in range(1, 6)] + [f"phase{k}_share" for k in range(1, 6)] + ["phase3p_share", "ipc_carried"],
    "pop": ["population"], "malaria": ["malaria"],
    "national": ["gdp", "gdp_pc", "electricity_access", "v2x_libdem", "GHG_pc_t", "EPI_index", "ECO_pc_gha"],
    "static": ["river_length", "lake_area", "BIOME_NAME"],
}
ALLYEAR = {}


def load_all(var):
    if var not in ALLYEAR:
        f = os.path.join(T, "cov", var, "all.parquet")
        ALLYEAR[var] = pd.read_parquet(f) if os.path.exists(f) else None
    return ALLYEAR[var]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--years", type=int, nargs=2)
    a = ap.parse_args()
    ys = sorted(int(p.split("=")[1]) for p in glob.glob(os.path.join(T, "core", "year=*")))
    if a.years:
        ys = [y for y in ys if a.years[0] <= y <= a.years[1]]
    od = os.path.join(T, "clean_global"); os.makedirs(od, exist_ok=True)
    for y in ys:
        df = pd.read_parquet(os.path.join(T, "core", f"year={y}", "part.parquet"))
        df["year"] = df.week_start.dt.year.astype("int16")
        present = []
        for var, cols in EXPECTED.items():
            f = os.path.join(T, "cov", var, f"year={y}.parquet")
            if os.path.exists(f):
                df = df.merge(pd.read_parquet(f), on=["grid_id", "week_start"], how="left"); present.append(var)
            elif (d := load_all(var)) is not None:
                key = [k for k in ("grid_id", "ISO_A3", "year") if k in d.columns]
                df = df.merge(d, on=key, how="left"); present.append(var)
            for c in cols:
                if c not in df:
                    df[c] = np.nan
        df.to_parquet(os.path.join(od, f"year={y}.parquet"), index=False)
        print(y, df.shape, "covariates:", present, flush=True)


if __name__ == "__main__":
    main()
