"""Like-for-like comparison on the same test rows: corrected XGBoost (the original
model without the duplication and test-set early stopping, probabilities
prior-corrected for the class weight) vs the spatially cross-fitted XGBoost vs the
hybrid model (cross-fitted XGBoost + country effects + barrier Matern field) vs
persistence.

Usage: python compare.py --xgb XGB_DIR/pred_corrected_Conflict.parquet
                         --hybrid SPATIAL_DIR/conflict/test_pred.parquet --out compare_conflict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sc import diagnostics as dg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xgb", required=True); ap.add_argument("--hybrid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    x = pd.read_parquet(a.xgb)
    h = pd.read_parquet(a.hybrid)
    for d in (x, h):
        d["week_start"] = pd.to_datetime(d["week_start"])
    m = h.merge(x[["grid_id", "week_start", "y", "logit_cal"]], on=["grid_id", "week_start"],
                suffixes=("", "_xgb"))
    assert (m["y"] == m["y_xgb"]).all(), "outcomes differ between the two tables"
    y = m["y"].to_numpy()
    rng = np.random.default_rng(0)
    preds = {"xgb_corrected": expit(m["logit_cal"].to_numpy()),
             "xgb_crossfit": m["p_xgb_crossfit"].to_numpy(),
             "hybrid": m["p_hybrid"].to_numpy()}
    out = {"n_rows": int(len(m)), "n_cells": int(m.grid_id.nunique()),
           "rows_hybrid_only": int(len(h) - len(m)), "prevalence": float(y.mean())}
    for k, p in preds.items():
        out[k] = dg.scores(y, p)
    out["persistence_auc_pr"] = dg.scores(
        y, m["lag_total"].fillna(0).to_numpy() + 1e-6 * rng.random(len(m)))["auc_pr"]
    # paired differences in mean log score (negative = hybrid better), with a cell-block SE
    ll = lambda p: -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    diff = pd.Series(ll(preds["hybrid"]) - ll(preds["xgb_corrected"])).groupby(m["grid_id"].to_numpy())
    cell_sum, cell_n = diff.sum(), diff.size()
    mean = cell_sum.sum() / cell_n.sum()
    resid = cell_sum - mean * cell_n
    se = np.sqrt((resid ** 2).sum()) / cell_n.sum()
    out["logscore_diff_hybrid_minus_xgb_corrected"] = {"mean": float(mean), "se_cell_clustered": float(se)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
