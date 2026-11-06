"""Reproduce the XGBoost model of the original analysis (scripts/model_xgb.py) and a
corrected version, on the table written by load_partitions.py.

--mode original   table built with --keep-duplicates; early stopping on the 2023-2024
                  evaluation set; 5M training rows sampled with replacement. Because
                  duplicated cell-weeks are contiguous after sorting, the original
                  row-based shift(1).rolling(4) lags include same-week copies of the
                  outcome for duplicated cells (target leakage); load_partitions
                  reproduces that behaviour in --keep-duplicates mode.
--mode corrected  deduplicated table; number of rounds chosen by early stopping on
                  2022 (training on 2018-2021), then refitted on 2018-2022; training
                  rows sampled without replacement.

Both modes use the original features and hyperparameters, including
scale_pos_weight = sqrt(n0/n1) and the global calendar-month precipitation anomaly.
Outputs (to --out): metrics_<mode>_<outcome>.json and
pred_<mode>_<outcome>.parquet (test-period grid_id, week_start, y, logit).

Usage: python reproduce_xgb.py TABLE.parquet --mode corrected --outcome Conflict
       [--device cuda] [--nthread 4] [--n-train 5000000] [--out results_real]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sc import diagnostics as dg  # noqa: E402

PARAMS = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
              max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=30, reg_lambda=5.0, seed=42)
NON_FEATURES = {"grid_id", "week_start", "lon", "lat", "BIOME_NAME", "Conflict", "Unrest",
                "Strategic_developments", "n_copies", "ISO_A3", "country_is_proxy", "gdp_flag",
                "cell_id", "t"}
SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
LAGS = {"Conflict": ["Battles_rollsum_4", "ExplosionsRemote_violence_rollsum_4",
                     "Violence_against_civilians_rollsum_4"],
        "Unrest": ["Riots_rollsum_4", "Protests_rollsum_4"]}


def features(df):
    df["month"] = df.week_start.dt.month.astype("int8")
    s = df["month"].map(SEASON)
    df["season"] = np.where(df.lat < 0, (s + 2) % 4, s).astype("int8")   # hemisphere shift
    df["equatorial"] = (df.lat.abs() < 30).astype("int8")
    df["biome"] = df["BIOME_NAME"].astype("category").cat.codes.astype("int8")
    df = df.dropna(subset=[c for c in ["hi", "tmin", "tmax", "wbgt", "ppt"] if c in df.columns])
    ph = [c for c in df.columns if c.startswith("phase_")]
    df[ph] = df[ph].fillna(0.0)
    feats = [c for c in df.select_dtypes("number").columns if c not in NON_FEATURES]
    return df, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--mode", choices=["original", "corrected"], required=True)
    ap.add_argument("--outcome", choices=["Conflict", "Unrest"], required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--nthread", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=5_000_000); ap.add_argument("--out", default="results_real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1)

    df, feats = features(pd.read_parquet(a.table))
    y = a.outcome
    yr = df.week_start.dt.year
    train, test = df[yr <= 2022], df[yr > 2022]
    ppt = [c for c in feats if c.startswith("ppt")]
    month_mean = train.groupby("month")[ppt].mean()

    def X(d):
        x = d[feats].copy()
        x[ppt] = x[ppt].to_numpy() - month_mean.loc[d["month"].to_numpy(), ppt].to_numpy()
        return x

    params = dict(PARAMS, device=a.device, nthread=a.nthread)

    def sample(d, n, replace):
        idx = rng.choice(len(d), size=min(n, len(d)), replace=replace)
        return d.iloc[idx]

    if a.mode == "original":
        tr = sample(train, a.n_train, replace=True)
        params["scale_pos_weight"] = float(np.sqrt((len(tr) - tr[y].sum()) / tr[y].sum()))
        dtr = xgb.DMatrix(X(tr), label=tr[y])
        dte = xgb.DMatrix(X(test), label=test[y])
        model = xgb.train(params, dtr, 2000, evals=[(dtr, "train"), (dte, "val")],
                          early_stopping_rounds=100, verbose_eval=100)
        rounds = model.best_iteration + 1
    else:
        fit = sample(train[train.week_start.dt.year <= 2021], a.n_train, replace=False)
        val = train[train.week_start.dt.year == 2022]
        params["scale_pos_weight"] = float(np.sqrt((len(fit) - fit[y].sum()) / fit[y].sum()))
        m0 = xgb.train(params, xgb.DMatrix(X(fit), label=fit[y]), 2000,
                       evals=[(xgb.DMatrix(X(val), label=val[y]), "val2022")],
                       early_stopping_rounds=100, verbose_eval=100)
        rounds = m0.best_iteration + 1
        tr = sample(train, a.n_train, replace=False)
        params["scale_pos_weight"] = float(np.sqrt((len(tr) - tr[y].sum()) / tr[y].sum()))
        model = xgb.train(params, xgb.DMatrix(X(tr), label=tr[y]), rounds)
        dte = xgb.DMatrix(X(test), label=test[y])

    logit = model.predict(dte, output_margin=True, iteration_range=(0, rounds))
    p = 1 / (1 + np.exp(-logit))
    p_cal = 1 / (1 + np.exp(-dg.prior_correction(logit, params["scale_pos_weight"])))
    yt = test[y].to_numpy()
    lag = test[LAGS[y]].sum(axis=1).fillna(0).to_numpy()
    m = {"mode": a.mode, "outcome": y, "rounds": int(rounds),
         "n_train_rows": int(len(tr)), "n_test_rows": int(len(test)),
         "n_test_cells": int(test.grid_id.nunique()), "test_prevalence": float(yt.mean()),
         "auc_pr": float(average_precision_score(yt, p)), "auc_roc": float(roc_auc_score(yt, p)),
         "precision@0.5": float(precision_score(yt, p >= 0.5)),
         "recall@0.5": float(recall_score(yt, p >= 0.5)), "f1@0.5": float(f1_score(yt, p >= 0.5)),
         "calibrated": dg.scores(yt, p_cal),
         "persistence_auc_pr": float(average_precision_score(yt, lag + 1e-6 * rng.random(len(lag)))),
         "features": feats}
    if "n_copies" in test.columns:
        dup = test["n_copies"].to_numpy() > 1
        if dup.any() and (~dup).any():
            m["auc_pr_interior_cells"] = float(average_precision_score(yt[~dup], p[~dup]))
            m["auc_pr_border_cells"] = float(average_precision_score(yt[dup], p[dup]))
    imp = model.get_score(importance_type="total_gain")
    m["top_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])
    json.dump(m, open(os.path.join(a.out, f"metrics_{a.mode}_{y}.json"), "w"), indent=1)
    pd.DataFrame({"grid_id": test.grid_id.to_numpy(), "week_start": test.week_start.to_numpy(),
                  "y": yt.astype("int8"), "logit": logit.astype("float32"),
                  "logit_cal": dg.prior_correction(logit, params["scale_pos_weight"]).astype("float32")}
                 ).to_parquet(os.path.join(a.out, f"pred_{a.mode}_{y}.parquet"), index=False)
    print(json.dumps({k: v for k, v in m.items() if k not in ("features",)}, indent=1))


if __name__ == "__main__":
    main()
