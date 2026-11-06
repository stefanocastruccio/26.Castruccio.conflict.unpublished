"""Like-for-like comparison of test-period predictions on the rows common to all inputs.

Each --pred is NAME=FILE:COLUMN; COLUMN is a probability, or 'logit_cal' / 'logit'
(converted with expit). The persistence baseline (ranking by past 4-week event count,
ties handled by average_precision_score) is taken from --lag FILE (a test_pred.parquet
of fit_acled.py with lag_total, events_52w, fold, lon, lat and ISO_A3 columns), on the
same rows. Paired differences in mean log score and Brier score are reported against
--ref, with cluster-robust standard errors clustered by cell (se_cell) and by country
(se_country), with the numbers of clusters.

Reported for every model:
  * scores on all common rows, by test year, and on the onset rows (cell-weeks with no
    event of the outcome in the cell over the previous 52 weeks, events_52w == 0) and
    the continuation rows (the others);
  * TEST-period residual diagnostics: Moran's I of the standardised cell residuals
    (999 permutations; p is bounded below by 1/1000), the neighbour residual
    correlation within countries and across borders (across-border pairs also split
    by whether the cross-fitting folds of the two cells agree), and a great-circle
    semivariogram (written to OUT with suffix _variogram.csv).

Usage:
  python compare.py --pred xgb=XGB/pred_corrected_Conflict.parquet:logit \\
                    --pred xgb_te=XGB/pred_corrected_te_Conflict.parquet:logit \\
                    --pred hybrid_field=SP/conflict/test_pred.parquet:p_field \\
                    --pred hybrid_iid=SP/conflict/test_pred.parquet:p_iid \\
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
    """Mean of loss_a - loss_b and its cell-clustered standard error (kept for callers)."""
    mean, se, _ = dg.clustered_diff(loss_a, loss_b, cells)
    return mean, se[0]


def subset_scores(y, P, names, ref, sel, clusters):
    """Scores of every model on the rows `sel`, with paired differences against ref."""
    out = {"n_rows": int(sel.sum()), "prevalence": float(y[sel].mean()) if sel.any() else None,
           "models": {}}
    if not sel.any() or y[sel].min() == y[sel].max():
        return out
    ys = y[sel]
    clip = lambda p: np.clip(p, 1e-12, 1 - 1e-12)
    ll = lambda p: -(ys * np.log(clip(p)) + (1 - ys) * np.log(1 - clip(p)))
    br = lambda p: (ys - p) ** 2
    cl = [np.asarray(c)[sel] for c in clusters.values()]
    for n in names:
        p = P[n][sel]
        r = dg.scores(ys, p)
        if n != ref:
            pr = P[ref][sel]
            for key, f in [("logscore_diff_vs_ref", ll), ("brier_diff_vs_ref", br)]:
                mean, ses, ns = dg.clustered_diff(f(p), f(pr), *cl)
                r[key] = {"mean": mean, **{f"se_{k}": s for k, s in zip(clusters, ses)},
                          **{f"n_clusters_{k}": g for k, g in zip(clusters, ns)}}
        out["models"][n] = r
    return out


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
        cols = KEY + ["lag_total", "lon", "lat", "ISO_A3"]
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(a.lag).schema.names)
        cols += [c for c in ("events_52w", "fold") if c in have]
        lag = pd.read_parquet(a.lag, columns=cols)
        lag["week_start"] = pd.to_datetime(lag["week_start"])
        m = m.merge(lag, on=KEY)
    y = m["y"].to_numpy(); cells = m["grid_id"].to_numpy()
    P = {n: m[n].to_numpy() for n in names}
    clusters = {"cell": cells}
    if "ISO_A3" in m:
        clusters["country"] = m["ISO_A3"].astype(str).to_numpy()
    out = {"n_rows": int(len(m)), "n_cells": int(m.grid_id.nunique()), "prevalence": float(y.mean()),
           "reference": a.ref, "period": "test (2023-2024)",
           "se_note": "cluster-robust SEs of the mean paired difference, small-sample factor G/(G-1)",
           "models": {}}
    allrows = subset_scores(y, P, names, a.ref, np.ones(len(m), bool), clusters)
    out["models"] = allrows["models"]
    if "country" in clusters:
        out["n_countries"] = int(pd.unique(clusters["country"]).size)
    if a.lag:
        lagv = m["lag_total"].to_numpy()
        out["persistence_auc_pr"] = float(average_precision_score(y, lagv))
        years = m["week_start"].dt.year.to_numpy()
        out["by_test_year"] = {}
        for t in sorted(np.unique(years)):
            sel = years == t
            r = subset_scores(y, P, names, a.ref, sel, clusters)
            if 0 < y[sel].sum() < sel.sum():
                r["persistence_auc_pr"] = float(average_precision_score(y[sel], lagv[sel]))
            out["by_test_year"][int(t)] = r
        if "events_52w" in m:
            onset = m["events_52w"].to_numpy() == 0
            for nm, sel in [("onset", onset), ("continuation", ~onset)]:
                r = subset_scores(y, P, names, a.ref, sel, clusters)
                r["definition"] = ("no event of the outcome in the cell in the previous 52 weeks"
                                   if nm == "onset" else "at least one event in the previous 52 weeks")
                if 0 < y[sel].sum() < sel.sum():
                    r["persistence_auc_pr"] = float(average_precision_score(y[sel], lagv[sel]))
                out[nm] = r
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fit_acled import build_lattice
        L, cell, observed = build_lattice(m["lon"].to_numpy(), m["lat"].to_numpy(), buffer=0)
        m["cell"] = cell
        country = m.groupby("cell")["ISO_A3"].first().reindex(range(L.n)).to_numpy()
        fold = (m.groupby("cell")["fold"].first().reindex(range(L.n)).to_numpy()
                if "fold" in m else None)
        vgs = []
        for n in names:
            r = dg.cell_pearson(m, m[n].to_numpy())
            rf = np.full(L.n, np.nan); rf[r.index] = r.to_numpy()
            ok = ~np.isnan(rf)
            adj = L.adjacency[ok][:, ok]
            out["models"][n]["residual_moran"] = dg.morans_i(rf[ok], adj, n_perm=999)
            out["models"][n]["residual_border"] = dg.border_contrast(
                rf[ok], adj, country[ok], fold=None if fold is None else fold[ok])
            vgs.append(dg.variogram(rf[ok], L.lon[ok], L.lat[ok], np.arange(0, 3000, 100)).assign(model=n))
        pd.concat(vgs).to_csv(os.path.splitext(a.out)[0] + "_variogram.csv", index=False)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_test_year",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
