"""Checks of the 1990-2025 covariates (task 35): coverage per variable, missing shares by region, and agreement with $CLEAN.

  python realdata/covariates/check.py [vars...]      # default: ppt tmax tmin hi wbgt pm25 spi3

For each weekly variable: number of cells and weeks, missing share overall, by region (cell_region.parquet; cells outside
the five regions are 'other'), and for 2018-2022 the correlation across cell-weeks with the corresponding $CLEAN column
(ppt, tmax, tmin, hi, wbgt, pm25; spi3 against CLEAN's drought). Writes grid/check_summary.json and prints a table.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

CLEANCOL = {"ppt": "ppt", "tmax": "tmax", "tmin": "tmin", "hi": "hi", "wbgt": "wbgt", "pm25": "pm25", "spi3": "drought"}


def main(vars_):
    reg = pd.read_parquet(os.path.join(C.GRID, "cell_region.parquet"))[["grid_id", "region"]]
    cl = C.cells()[["grid_id"]].merge(reg, how="left"); cl["region"] = cl.region.fillna("other")
    cols = [CLEANCOL[v] for v in vars_ if CLEANCOL[v]]
    clean = pd.read_parquet(C.CLEAN, columns=["grid_id", "week_start"] + sorted(set(cols)))
    clean["grid_id"] = clean.grid_id.astype("int32"); clean["week_start"] = pd.to_datetime(clean.week_start)
    clean = clean[(clean.week_start.dt.year >= 2018) & (clean.week_start.dt.year <= 2022)].drop_duplicates(["grid_id", "week_start"])
    out = {}
    for v in vars_:
        fs = sorted(glob.glob(os.path.join(C.GRID, v, f"{v}_*.parquet")))
        if not fs:
            print(v, "no files"); continue
        d = pd.concat([pd.read_parquet(f) for f in fs]).merge(cl, on="grid_id", how="left")
        r = {"files": len(fs), "years": [int(fs[0][-12:-8]), int(fs[-1][-12:-8])], "cells": int(d.grid_id.nunique()),
             "weeks": int(d.week_start.nunique()), "missing": round(float(d.value.isna().mean()), 4),
             "missing_by_region": d.groupby("region").value.apply(lambda s: round(float(s.isna().mean()), 4)).to_dict(),
             "missing_2000_2025": round(float(d[d.week_start.dt.year >= 2000].value.isna().mean()), 4)}
        m = d[(d.week_start.dt.year >= 2018) & (d.week_start.dt.year <= 2022)].merge(clean[["grid_id", "week_start", CLEANCOL[v]]],
                                                                                       on=["grid_id", "week_start"]).dropna()
        if len(m):
            r["clean_2018_2022"] = {"n": int(len(m)), "pearson": round(float(m.value.corr(m[CLEANCOL[v]])), 4),
                                    "spearman": round(float(m.value.corr(m[CLEANCOL[v]], method="spearman")), 4),
                                    "mean_ours": round(float(m.value.mean()), 3), "mean_clean": round(float(m[CLEANCOL[v]].mean()), 3)}
        out[v] = r
        print(v, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(C.GRID, "check_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ppt", "tmax", "tmin", "hi", "wbgt", "pm25", "spi3"])
