"""Like-for-like comparison of test-period predictions on the rows common to all inputs.

Each --pred is NAME=FILE:COLUMN; COLUMN is a probability, or 'logit_cal' / 'logit'
(converted with expit). The persistence baseline (ranking by past 4-week event count)
is taken from --lag FILE (a test_pred.parquet with a lag_total column). Paired
differences in mean log score and Brier score are reported against --ref, with
standard errors clustered by cell (cells are the units of dependence over time).

Usage:
  python compare.py --pred xgb=XGB/pred_corrected_Conflict.parquet:logit_cal \
                    --pred xgb_te=XGB/pred_corrected_te_Conflict.parquet:logit_cal \
                    --pred hybrid_field=SP/conflict/test_pred.parquet:p_field \
                    --pred hybrid_iid=SP/conflict/test_pred.parquet:p_iid \
                    --lag SP/conflict/test_pred.parquet --ref xgb --out compare_conflict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sc import diagnostics as dg  # noqa: E402

KEY = ["grid_id", "week_start"]


def load(spec):
    name, rest = spec.split("=", 1)
    path, col = rest.rsplit(":", 1)
    d = pd.read_parquet(path, columns=KEY + ["y", col])
    d["week_start"] = pd.to_datetime(d["week_start"])
    p = expit(d[col].to_numpy()) if col.startswith("logit") else d[col].to_numpy()
    return name, d[KEY + ["y"]].assign(**{name: p})


def clustered_diff(loss_a, loss_b, cells):
    """Mean of loss_a - loss_b and its cell-clustered standard error."""
    g = pd.Series(loss_a - loss_b).groupby(cells)
    s, n = g.sum(), g.size()
    mean = s.sum() / n.sum()
    return float(mean), float(np.sqrt(((s - mean * n) ** 2).sum()) / n.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--lag"); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names, m = [], None
    for spec in a.pred:
        name, d = load(spec)
        names.append(name)
        if m is None:
            m = d
        else:
            m = m.merge(d, on=KEY, suffixes=("", "_dup"))
            assert (m["y"] == m["y_dup"]).all(), f"outcomes differ for {name}"
            m = m.drop(columns="y_dup")
    if a.lag:
        lag = pd.read_parquet(a.lag, columns=KEY + ["lag_total"])
        lag["week_start"] = pd.to_datetime(lag["week_start"])
        m = m.merge(lag, on=KEY)
    y = m["y"].to_numpy(); cells = m["grid_id"].to_numpy()
    out = {"n_rows": int(len(m)), "n_cells": int(m.grid_id.nunique()), "prevalence": float(y.mean()),
           "reference": a.ref, "models": {}}
    clip = lambda p: np.clip(p, 1e-12, 1 - 1e-12)
    ll = lambda p: -(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p)))
    br = lambda p: (y - p) ** 2
    for n in names:
        p = m[n].to_numpy()
        r = dg.scores(y, p)
        if n != a.ref:
            pr = m[a.ref].to_numpy()
            r["logscore_diff_vs_ref"] = dict(zip(["mean", "se_cell"], clustered_diff(ll(p), ll(pr), cells)))
            r["brier_diff_vs_ref"] = dict(zip(["mean", "se_cell"], clustered_diff(br(p), br(pr), cells)))
        out["models"][n] = r
    if a.lag:
        out["persistence_auc_pr"] = float(average_precision_score(y, m["lag_total"].to_numpy()))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
