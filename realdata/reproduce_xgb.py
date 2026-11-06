"""Reproduce the XGBoost model of the original analysis (scripts/model_xgb.py), apply the
same procedure to the repaired table, and fit a corrected version.

--mode original   table built with --keep-duplicates (orig.parquet); the published
                  procedure: positives weighted by scale_pos_weight = sqrt(n0/n1), early
                  stopping on the 2023-2024 evaluation set, 5M training rows sampled with
                  replacement, all numeric columns as features (current-week climate,
                  per-cell GDP, IPC phases with missing set to 0). Because duplicated
                  cell-weeks are contiguous after sorting, the original row-based
                  shift(1).rolling(4) lags include same-week copies of the outcome for
                  duplicated cells (target leakage); load_partitions reproduces that
                  behaviour in --keep-duplicates mode. Scores are also reported on the
                  test rows restricted to one row (the first copy) per cell-week, and,
                  with --restrict-to CLEAN.parquet, on the cell-weeks of the repaired
                  test set.
--mode dedup      the ORIGINAL procedure unchanged (class weight, early stopping on the
                  test years, sampling with replacement, the same features), applied to
                  the repaired table (clean.parquet). Isolates the effect of the
                  duplication.
--mode corrected  repaired table; no class weight; features use only information
                  available before the week (the current-week raw climate variables and
                  the per-cell GDP are dropped; climate enters through the 4-week means
                  of the previous weeks; national GDP per capita as log(gdp_pc); IPC
                  phases keep their missing values, which the trees handle natively);
                  the number of trees is chosen by early stopping on 2022 after fitting
                  2018-2021, then the model is refitted on 2018-2022 with that number of
                  trees; training rows are a subsample of --n-train rows drawn without
                  replacement.
All modes use the same rows (cell-weeks with the current-week climate observed), the
hyperparameters of the original analysis and the global calendar-month precipitation
anomaly (ppt and its 4-week mean minus the 2018-2022 mean of the calendar month).
All models are trained on the whole world.

Outputs (to --out): metrics_<tag>_<outcome>.json and pred_<tag>_<outcome>.parquet
(test-period grid_id, week_start, y, logit, logit_cal, events_52w). logit_cal is the
prior-corrected logit, logit - log(scale_pos_weight); it equals logit when no class
weight is used (corrected mode). events_52w is the number of weeks with an event of the
outcome in the cell over the previous 52 weeks (repaired table only).

--target-encode   (corrected mode) adds a location-aware feature, the cell's event rate
                  over the training years, leaving out the row's own year. For the
                  early-stopping fit the rate is computed from 2018-2021 only (2022
                  outcomes, the validation set, are never used); validation rows use all
                  of 2018-2021. For the final refit it is computed leave-one-year-out over
                  2018-2022, and test rows use all of 2018-2022. Output tag corrected_te.

Usage: python reproduce_xgb.py TABLE.parquet --mode corrected --outcome Conflict
       [--target-encode] [--device cuda] [--nthread 4] [--n-train 5000000] [--out results_real]
       [--restrict-to CLEAN.parquet]   (original mode)
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
from sc.boosting import cell_rate_feature  # noqa: E402

PARAMS = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
              max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=30, reg_lambda=5.0, seed=42)
NON_FEATURES = {"grid_id", "week_start", "lon", "lat", "BIOME_NAME", "Conflict", "Unrest",
                "Strategic_developments", "n_copies", "ISO_A3", "country_is_proxy", "gdp_flag",
                "cell_id", "t", "events_52w", "cell_rate_train"}
# columns added by the repaired loader, not part of the original feature set
NEW_COLUMNS = {"gdp_pc", "log_gdp_pc"}
# current-week raw climate: not known before the week, so not used by the corrected model
CURRENT_WEEK = ["hi", "tmin", "tmax", "wbgt", "drought", "ppt", "pm25"]
ROW_FILTER = ["hi", "tmin", "tmax", "wbgt", "ppt"]   # rows dropped by the original analysis
SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
LAGS = {"Conflict": ["Battles_rollsum_4", "ExplosionsRemote_violence_rollsum_4",
                     "Violence_against_civilians_rollsum_4"],
        "Unrest": ["Riots_rollsum_4", "Protests_rollsum_4"]}
KEY = ["grid_id", "week_start"]


def features(df, mode="corrected"):
    """Feature columns of each mode. The rows kept are the same in every mode.
    Adds columns to df (no copy, to save memory on the full table)."""
    df["month"] = df.week_start.dt.month.astype("int8")
    s = df["month"].map(SEASON)
    df["season"] = np.where(df.lat < 0, (s + 2) % 4, s).astype("int8")   # hemisphere shift
    df["equatorial"] = (df.lat.abs() < 30).astype("int8")
    df["biome"] = df["BIOME_NAME"].astype("category").cat.codes.astype("int8")
    df = df.dropna(subset=[c for c in ROW_FILTER if c in df.columns])
    ph = [c for c in df.columns if c.startswith("phase_")]
    num = [c for c in df.select_dtypes("number").columns if c not in NON_FEATURES]
    if mode in ("original", "dedup"):
        df[ph] = df[ph].fillna(0.0)          # as in the original analysis
        feats = [c for c in num if c not in NEW_COLUMNS]
    elif mode == "corrected":
        if "gdp_pc" in df.columns:
            df["log_gdp_pc"] = np.log(df["gdp_pc"].where(df["gdp_pc"] > 0)).astype("float32")
        num = [c for c in df.select_dtypes("number").columns if c not in NON_FEATURES]
        feats = [c for c in num if c not in CURRENT_WEEK and c not in ("gdp", "gdp_pc")]
    else:
        raise ValueError(mode)
    return df, feats


def history_52w(df, ycol):
    """Number of weeks with an event in the cell over the previous 52 weeks (time-based
    window [t - 364 days, t)), from one row per cell-week. Aligned with df.index."""
    d = df[KEY + [ycol]].sort_values(KEY, kind="mergesort")
    d[ycol] = d[ycol].astype(float)
    s = d.set_index("week_start").groupby("grid_id", sort=True)[ycol].rolling("364D", closed="left").sum()
    # groupby(sort=True) on a frame sorted by (grid_id, week_start) keeps the row order
    return pd.Series(np.nan_to_num(s.to_numpy(), nan=0.0), index=d.index).reindex(df.index)


def _scores(yt, p, lag):
    out = {"n_rows": int(len(yt)), "prevalence": float(yt.mean())}
    if 0 < yt.sum() < len(yt):
        out.update(auc_pr=float(average_precision_score(yt, p)), auc_roc=float(roc_auc_score(yt, p)),
                   persistence_auc_pr=float(average_precision_score(yt, lag)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--mode", choices=["original", "dedup", "corrected"], required=True)
    ap.add_argument("--outcome", choices=["Conflict", "Unrest"], required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--nthread", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=5_000_000); ap.add_argument("--out", default="results_real")
    ap.add_argument("--target-encode", action="store_true")
    ap.add_argument("--restrict-to", help="(original mode) repaired table: also score on its test cell-weeks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1)
    if a.target_encode and a.mode != "corrected":
        raise SystemExit("--target-encode is only for --mode corrected")

    y = a.outcome
    raw = pd.read_parquet(a.table)
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    has_dups = raw.duplicated(KEY).any()
    if not has_dups:
        raw["events_52w"] = history_52w(raw, y).astype("float32")
    df, feats = features(raw, a.mode); del raw
    yr = df.week_start.dt.year
    te_es = te_fin = None
    if a.target_encode:
        cell, yrs, yy = df["grid_id"].to_numpy(), yr.to_numpy(), df[y].to_numpy(float)
        # early-stopping fit: 2018-2021 only; 2022 (validation) rows use all of 2018-2021
        te_es = pd.Series(cell_rate_feature(cell, yrs, yy, (yrs <= 2021)), index=df.index)
        # final refit: leave-one-year-out over 2018-2022; test rows use all of 2018-2022
        te_fin = pd.Series(cell_rate_feature(cell, yrs, yy, (yrs <= 2022)), index=df.index)
        feats = feats + ["cell_rate_train"]
    train, test = df[yr <= 2022], df[yr > 2022]
    ppt = [c for c in feats if c.startswith("ppt")]
    month_mean = train.groupby("month")[ppt].mean()

    def X(d, te=None):
        x = d[[c for c in feats if c != "cell_rate_train"]].copy()
        if ppt:
            x[ppt] = x[ppt].to_numpy() - month_mean.loc[d["month"].to_numpy(), ppt].to_numpy()
        if te is not None:
            x["cell_rate_train"] = te.loc[d.index].to_numpy()
        return x

    params = dict(PARAMS, device=a.device, nthread=a.nthread)

    def sample(d, n, replace):
        idx = rng.choice(len(d), size=min(n, len(d)), replace=replace)
        return d.iloc[idx]

    if a.mode in ("original", "dedup"):
        tr = sample(train, a.n_train, replace=True)
        spw = float(np.sqrt((len(tr) - tr[y].sum()) / tr[y].sum()))
        params["scale_pos_weight"] = spw
        dtr = xgb.DMatrix(X(tr), label=tr[y])
        dte = xgb.DMatrix(X(test), label=test[y])
        model = xgb.train(params, dtr, 2000, evals=[(dtr, "train"), (dte, "val")],
                          early_stopping_rounds=100, verbose_eval=100)
        rounds = model.best_iteration + 1
        setup = {"class_weight": "scale_pos_weight = sqrt(n0/n1)", "scale_pos_weight": spw,
                 "early_stopping": "on the test years 2023-2024", "sampling": "with replacement"}
    else:
        spw = 1.0                                   # no class weight
        fit = sample(train[train.week_start.dt.year <= 2021], a.n_train, replace=False)
        val = train[train.week_start.dt.year == 2022]
        m0 = xgb.train(params, xgb.DMatrix(X(fit, te_es), label=fit[y]), 2000,
                       evals=[(xgb.DMatrix(X(val, te_es), label=val[y]), "val2022")],
                       early_stopping_rounds=100, verbose_eval=100)
        rounds = m0.best_iteration + 1
        tr = sample(train, a.n_train, replace=False)
        model = xgb.train(params, xgb.DMatrix(X(tr, te_fin), label=tr[y]), rounds)
        dte = xgb.DMatrix(X(test, te_fin), label=test[y])
        setup = {"class_weight": None, "scale_pos_weight": None,
                 "early_stopping": "on 2022 after fitting 2018-2021, then refit on 2018-2022",
                 "sampling": "without replacement",
                 "n_fit_rows_early_stopping": int(len(fit)), "n_val_rows_2022": int(len(val))}
        if a.target_encode:
            setup["target_encoding"] = ("early-stopping fit: leave-one-year-out within 2018-2021, "
                                        "2022 rows use 2018-2021; refit: leave-one-year-out within "
                                        "2018-2022, test rows use 2018-2022")

    logit = model.predict(dte, output_margin=True, iteration_range=(0, rounds))
    logit_cal = dg.prior_correction(logit, spw)
    p = 1 / (1 + np.exp(-logit))
    p_cal = 1 / (1 + np.exp(-logit_cal))
    yt = test[y].to_numpy()
    lag = test[LAGS[y]].sum(axis=1).fillna(0).to_numpy()
    m = {"mode": a.mode, "outcome": y, "rounds": int(rounds), **setup,
         "training_region": "global", "n_train_rows": int(len(tr)),
         "n_train_rows_available": int(len(train)), "n_test_rows": int(len(test)),
         "n_test_cells": int(test.grid_id.nunique()), "test_prevalence": float(yt.mean()),
         "auc_pr": float(average_precision_score(yt, p)), "auc_roc": float(roc_auc_score(yt, p)),
         "precision@0.5": float(precision_score(yt, p >= 0.5)),
         "recall@0.5": float(recall_score(yt, p >= 0.5)), "f1@0.5": float(f1_score(yt, p >= 0.5)),
         "calibrated": dg.scores(yt, p_cal),
         # ties are handled by average_precision_score (no jitter)
         "persistence_auc_pr": float(average_precision_score(yt, lag)),
         "ppt_centred_by_global_calendar_month_mean": bool(ppt), "features": feats}
    if "n_copies" in test.columns:
        dup = test["n_copies"].to_numpy() > 1
        if dup.any() and (~dup).any():
            m["auc_pr_interior_cells"] = float(average_precision_score(yt[~dup], p[~dup]))
            m["auc_pr_border_cells"] = float(average_precision_score(yt[dup], p[dup]))
    if a.mode == "original":
        # one row (the first copy) per cell-week, so that every cell-week counts once
        one = ~test.duplicated(KEY).to_numpy()
        m["one_row_per_cell_week"] = _scores(yt[one], p[one], lag[one])
        if a.restrict_to:
            cols = KEY + [c for c in ROW_FILTER]
            c = pd.read_parquet(a.restrict_to, columns=cols)
            c["week_start"] = pd.to_datetime(c["week_start"])
            c = c[(c.week_start.dt.year > 2022)].dropna(subset=ROW_FILTER)[KEY]
            keep = one & test[KEY].merge(c.assign(_in=True), on=KEY, how="left")["_in"].fillna(False).to_numpy(bool)
            m["one_row_per_cell_week_repaired_test_set"] = _scores(yt[keep], p[keep], lag[keep])
    if "events_52w" in test.columns:
        on = test["events_52w"].to_numpy() == 0
        m["onset_rows"] = _scores(yt[on], p[on], lag[on])
        m["continuation_rows"] = _scores(yt[~on], p[~on], lag[~on])
    m["by_test_year"] = {int(t): _scores(yt[test.week_start.dt.year.to_numpy() == t],
                                         p[test.week_start.dt.year.to_numpy() == t],
                                         lag[test.week_start.dt.year.to_numpy() == t])
                         for t in sorted(test.week_start.dt.year.unique())}
    imp = model.get_score(importance_type="total_gain")
    m["top_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])
    tag = f"{a.mode}{'_te' if a.target_encode else ''}"
    m["mode"] = tag
    json.dump(m, open(os.path.join(a.out, f"metrics_{tag}_{y}.json"), "w"), indent=1)
    pred = {"grid_id": test.grid_id.to_numpy(), "week_start": test.week_start.to_numpy(),
            "y": yt.astype("int8"), "logit": logit.astype("float32"),
            "logit_cal": logit_cal.astype("float32")}
    if "events_52w" in test.columns:
        pred["events_52w"] = test["events_52w"].to_numpy(dtype="float32")
    pd.DataFrame(pred).to_parquet(os.path.join(a.out, f"pred_{tag}_{y}.parquet"), index=False)
    print(json.dumps({k: v for k, v in m.items() if k not in ("features",)}, indent=1))


if __name__ == "__main__":
    main()
