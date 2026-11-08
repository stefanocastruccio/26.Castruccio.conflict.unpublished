"""Where does the joint model predict conflict and unrest together better?

Reads the per-cell-week predictions written by fit_bivariate.py (test_pred*.parquet) and
decomposes the gain in joint log score of the joint model over the product of univariate
models, by country and by sliding lon/lat windows.

Region selection rule (used with the --validation run, i.e. on 2022 only, so the
2023-2024 test period never enters the choice): among windows of WIN x WIN degrees moved
in steps of STEP degrees with at least MIN_CELLS cells, take the window with the largest
total improvement in joint log score (sum over its cell-weeks) against the product of the
univariate static models (the stronger univariate baseline in both regions).

Usage:  python realdata/joint_gain_map.py PRED.parquet OUT_PREFIX [--model biv] [--base uni_static]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sc import diagnostics as dg  # noqa: E402

WIN, STEP = 10.0, 5.0
MIN_CELLS = int(os.environ.get("GAIN_MIN_CELLS", 50))


def logb(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.where(y == 1, np.log(p), np.log1p(-p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("out")
    ap.add_argument("--model", default="biv"); ap.add_argument("--base", default="uni_static")
    a = ap.parse_args()
    d = pd.read_parquet(a.pred)
    yc, yu = d["y_conflict"].to_numpy(), d["y_unrest"].to_numpy()
    base = logb(yc, d[f"p_conflict_{a.base}"].to_numpy()) + logb(yu, d[f"p_unrest_{a.base}"].to_numpy())
    # gain > 0: the joint model is better (difference in log predictive density per cell-week)
    d["gain"] = d[f"logj_{a.model}"].astype(float).to_numpy() - base
    d["gain_dep"] = (d[f"logj_{a.model}"] - d[f"logi_{a.model}"]).astype(float)   # dependence part only
    d["both"] = yc * yu
    out = {"n": int(len(d)), "mean_gain": float(d.gain.mean()), "mean_gain_dependence": float(d.gain_dep.mean())}

    g = d.groupby("iso").agg(rows=("gain", "size"), cells=("grid_id", "nunique"), gain_sum=("gain", "sum"),
                             gain_mean=("gain", "mean"), dep_mean=("gain_dep", "mean"),
                             conflict=("y_conflict", "mean"), unrest=("y_unrest", "mean"), both=("both", "mean"))
    g = g.sort_values("gain_sum", ascending=False)
    g.to_csv(a.out + "_by_country.csv")

    rows = []
    for x0 in np.arange(np.floor(d.lon.min() / STEP) * STEP, d.lon.max(), STEP):
        for y0 in np.arange(np.floor(d.lat.min() / STEP) * STEP, d.lat.max(), STEP):
            w = d[d.lon.between(x0, x0 + WIN) & d.lat.between(y0, y0 + WIN)]
            nc = w.grid_id.nunique()
            if nc < MIN_CELLS:
                continue
            m, ses, _ = dg.clustered_diff(w.gain.to_numpy(), np.zeros(len(w)), w.grid_id.to_numpy(), w.iso.to_numpy())
            rows.append(dict(lon0=x0, lon1=x0 + WIN, lat0=y0, lat1=y0 + WIN, cells=nc, rows=len(w),
                             gain_sum=float(w.gain.sum()), gain_mean=m, se_country=ses[1],
                             dep_mean=float(w.gain_dep.mean()), both=float(w.both.mean()),
                             countries=",".join(w.iso.value_counts().index[:6])))
    W = pd.DataFrame(rows, columns=["lon0", "lon1", "lat0", "lat1", "cells", "rows", "gain_sum", "gain_mean",
                                    "se_country", "dep_mean", "both", "countries"]).sort_values("gain_sum", ascending=False)
    W.to_csv(a.out + "_windows.csv", index=False)
    out["selected_window"] = W.iloc[0].to_dict() if len(W) else None
    out["top_windows"] = W.head(5).to_dict("records")
    out["top_countries"] = g.head(10).reset_index().to_dict("records")
    json.dump(out, open(a.out + "_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
