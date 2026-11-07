"""Is the residual risk noise, or a spatial structure that moves? Three experiments on
the hybrid model of fit_acled.py (same region, rows, features, offset and lattice).

  --mode oracle     Fit the latent models (field: u + Matern field; iid: u + iid cells)
                    to the TEST period 2023-2024 itself, with the out-of-sample offset of
                    fit_acled.py. In-sample, so an upper bound: how much of the test-period
                    residual structure a spatial field can absorb when it knows where the
                    events are. Reports log marginal likelihoods (field vs iid: is the
                    test-period structure spatially smooth?), the fitted hyperparameters,
                    the correlation between this field and the 2018-2022 field of
                    fit_acled.py (did the hotspots move?), and in-sample residual Moran's I
                    and variograms (offset alone vs offset + field).
  --mode rolling    Retrain everything with 2023 added: offset cross-fitted by country with
                    early stopping on 2023 (fitting 2018-2022), refitted on 2018-2023;
                    latent models fitted on 2018-2023; scores on 2024. Writes
                    test_pred_2024.parquet with p_*_roll columns for compare.py, to be put
                    next to the 2024 rows of the static fit.
  --mode update     Quarterly updating, as an early-warning system would be run. The
                    hyperparameters of three latent models (field, iid, space-time with
                    --step weeks) are fitted once on 2018-2022; then for each quarter of
                    2023-2024 the latent effects are recomputed (posterior mode with the
                    hyperparameters fixed) from ALL rows before the start of the quarter,
                    and the quarter is predicted. Writes test_pred_update.parquet
                    (p_field_upd, p_iid_upd, p_st_upd) and summary_update.json.
  --mode spacetime  Offset + intercept + country effects + iid cell effects + AR(1) x Matern
                    space-time field v on the observed land cells (on a --st-res degree grid,
                    1 degree by default, to keep the latent size manageable), with time steps of
                    --step weeks covering 2018-2024. Fitted on 2018-2022; the test weeks are
                    forecast through the AR(1) dynamics of v (no test data enter the fit).
                    Writes test_pred_st.parquet (p_st) for compare.py and summary_st.json.

Usage:
  python fit_dynamic.py clean.parquet --outcome conflict --region africa_me --land SHP \
      --mode oracle --offset WORK/spatial/conflict/offset.npy --static-field WORK/spatial/conflict/field.csv \
      --out WORK/dynamic/conflict
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit

from sc import design as de, diagnostics as dg
from sc.boosting import PAPER_PARAMS
from sc.gmrf import FixedBlock, IIDBlock, MaternBlock, SpaceTimeAR1Block
from sc.laplace import LatentGaussianModel, prob_from_eta
from fit_acled import REGIONS, YCOL, build_lattice, sea_cells
from reproduce_xgb import features, history_52w, LAGS


def prepare(a):
    df = pd.read_parquet(a.table)
    df["week_start"] = pd.to_datetime(df["week_start"])
    if a.region:
        x0, x1, y0, y1 = REGIONS[a.region]
        df = df[df.lon.between(x0, x1) & df.lat.between(y0, y1)].copy()
    ycol = YCOL[a.outcome]
    df["events_52w"] = history_52w(df, ycol).astype("float32")
    df, feats = features(df, "corrected")
    df = df.reset_index(drop=True)
    df["y"] = df[ycol].to_numpy().astype(int)
    df["lag_total"] = df[LAGS[ycol]].sum(axis=1).fillna(0)
    df["year"] = df["week_start"].dt.year
    return df, feats


def design_X(df, feats, train_mask):
    ppt = [c for c in feats if c.startswith("ppt")]
    month_mean = df[train_mask].groupby("month")[ppt].mean()
    X = df[feats].copy()
    X[ppt] = X[ppt].to_numpy() - month_mean.loc[df["month"].to_numpy(), ppt].to_numpy()
    return X.to_numpy(np.float32)


def crossfit(df, X, y, country, last_train_year, folds=5, seed=0, nthread=4):
    """Country-fold cross-fitted offset: early stopping on last_train_year after fitting
    the earlier years, then refit on all years up to last_train_year."""
    rng = np.random.default_rng(seed)
    cs = np.unique(country)
    fold_of = dict(zip(cs, rng.permutation(len(cs)) % min(folds, len(cs))))
    fold = np.array([fold_of[c] for c in country])
    year = df["year"].to_numpy()
    params = dict(PAPER_PARAMS, nthread=nthread, seed=seed)
    off, rounds = np.zeros(len(df)), []
    for k in np.unique(fold):
        out = fold != k
        fit, val = out & (year < last_train_year), out & (year == last_train_year)
        m0 = xgb.train(params, xgb.DMatrix(X[fit], label=y[fit]), 2000,
                       evals=[(xgb.DMatrix(X[val], label=y[val]), "val")],
                       early_stopping_rounds=100, verbose_eval=False)
        r = m0.best_iteration + 1
        tr = out & (year <= last_train_year)
        m = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr]), r)
        off[fold == k] = np.ravel(m.predict(xgb.DMatrix(X[fold == k]), output_margin=True))
        rounds.append(int(r))
    return off, fold, rounds


def latent_specs(L, nC, barrier, d_fit, d_pred):
    ones = lambda d: np.ones((len(d), 1))
    Dc = lambda d: de.index_design(d["country_id"], nC)
    Dw = lambda d: de.index_design(d["cell"], L.n)
    return {
        "field": ([FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country"),
                   MaternBlock(L, name="field", range_init=300.0, barrier=barrier)],
                  [ones(d_fit), Dc(d_fit), Dw(d_fit)], [ones(d_pred), Dc(d_pred), Dw(d_pred)]),
        "iid": ([FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country"),
                 IIDBlock(L.n, name="cell", sd_init=0.5)],
                [ones(d_fit), Dc(d_fit), Dw(d_fit)], [ones(d_pred), Dc(d_pred), Dw(d_pred)]),
    }


def residual_diag(L, d, p, out, tag):
    r = dg.cell_pearson(d, p)
    full = np.full(L.n, np.nan); full[r.index] = r.to_numpy()
    ok = ~np.isnan(full)
    dg.variogram(full[ok], L.lon[ok], L.lat[ok], np.arange(0, 3000, 100)).assign(model=tag).to_csv(
        os.path.join(out, f"variogram_{tag}.csv"), index=False)
    return dg.morans_i(full[ok], L.adjacency[ok][:, ok], n_perm=199)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--outcome", default="conflict")
    ap.add_argument("--region", default="africa_me"); ap.add_argument("--land", required=True)
    ap.add_argument("--mode", choices=["oracle", "rolling", "spacetime", "update"], required=True)
    ap.add_argument("--offset", help="offset.npy written by fit_acled.py (oracle, spacetime)")
    ap.add_argument("--static-field", help="field.csv written by fit_acled.py (oracle)")
    ap.add_argument("--step", type=int, default=13, help="weeks per time step of v (spacetime)")
    ap.add_argument("--st-res", type=float, default=1.0,
                    help="grid spacing (degrees) of the space-time field v; each 0.5-degree cell "
                         "uses the v of the coarser cell containing it (spacetime)")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nthread", type=int, default=4); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, feats = prepare(a)
    y = df["y"].to_numpy()
    L, cell, observed = build_lattice(df.lon.to_numpy(), df.lat.to_numpy())
    df["cell"] = cell
    df["country_id"] = pd.factorize(df["ISO_A3"])[0]
    nC = int(df["country_id"].max() + 1)
    barrier = sea_cells(L, a.land)
    res = {"outcome": a.outcome, "region": a.region, "mode": a.mode, "rows": int(len(df)),
           "cells": int(observed.sum())}
    print(f"{a.mode}: {len(df):,} rows, {observed.sum():,} cells", flush=True)

    def load_offset():
        off = np.load(a.offset)
        assert len(off) == len(df), "offset.npy does not match the rows; rerun fit_acled.py"
        return off

    if a.mode == "oracle":
        off = load_offset()
        te = (df["year"] >= 2023).to_numpy()
        d = df[te]
        res["offset_alone"] = {"scores": dg.scores(d["y"].to_numpy(), expit(off[te])),
                               "moran_in_sample": residual_diag(L, d, expit(off[te]), a.out, "offset")}
        res["models"] = {}
        for name, (blocks, D, _) in latent_specs(L, nC, barrier, d, d).items():
            m = LatentGaussianModel(d["y"].to_numpy(), blocks, D, offset=off[te]).fit()
            eta, sd = m.predict_eta(D, off[te], n_samples=50)
            p = prob_from_eta(eta, sd)
            r = {"hyper": m.hyper(), "log_ml": float(m.log_ml), "seconds": float(m.fit_seconds),
                 "scores_in_sample": dg.scores(d["y"].to_numpy(), p),
                 "moran_in_sample": residual_diag(L, d, p, a.out, name)}
            if name == "field":
                w_test = m.block_mean(2)
                pd.DataFrame({"lon": L.lon, "lat": L.lat, "observed": observed,
                              "field_test": w_test}).to_csv(os.path.join(a.out, "field_test.csv"), index=False)
                if a.static_field:
                    f0 = pd.read_csv(a.static_field)
                    ok = observed & f0["observed"].to_numpy()
                    r["corr_with_2018_2022_field"] = float(np.corrcoef(w_test[ok], f0["field"].to_numpy()[ok])[0, 1])
            res["models"][name] = r
            print(name, json.dumps({k: v for k, v in r.items() if k != "hyper"}, default=float), flush=True)
        res["log_ml_field_minus_iid"] = res["models"]["field"]["log_ml"] - res["models"]["iid"]["log_ml"]
        json.dump(res, open(os.path.join(a.out, "summary_oracle.json"), "w"), indent=1, default=float)

    elif a.mode == "rolling":
        tr = (df["year"] <= 2023).to_numpy(); te = (df["year"] == 2024).to_numpy()
        X = design_X(df, feats, tr)
        off, fold, rounds = crossfit(df, X, y, df["ISO_A3"].to_numpy(), 2023, a.folds, a.seed, a.nthread)
        d_tr, d_te = df[tr], df[te]
        res.update(offset_rounds_per_fold=rounds, train_years="2018-2023", test_year=2024,
                   xgb_crossfit_roll=dg.scores(d_te["y"].to_numpy(), expit(off[te])), models={})
        preds = {"p_xgb_crossfit_roll": expit(off[te]).astype("float32")}
        for name, (blocks, D_tr, D_te) in latent_specs(L, nC, barrier, d_tr, d_te).items():
            m = LatentGaussianModel(d_tr["y"].to_numpy(), blocks, D_tr, offset=off[tr]).fit()
            eta, sd = m.predict_eta(D_te, off[te], n_samples=100)
            p = prob_from_eta(eta, sd)
            res["models"][name] = {"scores": dg.scores(d_te["y"].to_numpy(), p), "hyper": m.hyper(),
                                   "log_ml": float(m.log_ml), "seconds": float(m.fit_seconds)}
            preds[f"p_{name}_roll"] = p.astype("float32")
            print(name, json.dumps(res["models"][name]["scores"]), flush=True)
        json.dump(res, open(os.path.join(a.out, "summary_rolling.json"), "w"), indent=1, default=float)
        pd.DataFrame({"grid_id": d_te["grid_id"].to_numpy(), "week_start": d_te["week_start"].to_numpy(),
                      "y": d_te["y"].to_numpy().astype("int8"), **preds}).to_parquet(
            os.path.join(a.out, "test_pred_2024.parquet"), index=False)

    elif a.mode == "update":
        off = load_offset()
        tr = (df["year"] <= 2022).to_numpy()
        wk = ((df["week_start"] - df["week_start"].min()).dt.days // 7).to_numpy()
        df["tstep"] = wk // a.step
        T = int(df["tstep"].max() + 1)
        r = a.st_res
        lon_v = np.floor(df.lon.to_numpy() / r) * r + r / 2
        lat_v = np.floor(df.lat.to_numpy() / r) * r + r / 2
        Lv, cell_v, _ = build_lattice(lon_v, lat_v, res=r, buffer=0)
        df["cell_v"] = cell_v

        def designs(name, d):
            base = [np.ones((len(d), 1)), de.index_design(d["country_id"], nC)]
            if name == "field":
                return base + [de.index_design(d["cell"], L.n)]
            if name == "iid":
                return base + [de.index_design(d["cell"], L.n)]
            return base + [de.index_design(d["cell"], L.n),
                           de.spacetime_design(d["cell_v"].to_numpy(), d["tstep"].to_numpy(), Lv.n, T)]

        def blocks(name):
            base = [FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country")]
            if name == "field":
                return base + [MaternBlock(L, name="field", range_init=300.0, barrier=barrier)]
            if name == "iid":
                return base + [IIDBlock(L.n, name="cell", sd_init=0.5)]
            return base + [IIDBlock(L.n, name="cell", sd_init=0.5),
                           SpaceTimeAR1Block(Lv, T, name="st", range_init=300.0, sd_init=0.5, rho_init=0.8)]

        qstarts = pd.date_range("2023-01-01", "2025-01-01", freq="QS")
        te = (df["year"] >= 2023).to_numpy()
        d_te = df[te]
        preds = {}
        res["models"] = {}
        for name in ["field", "iid", "st"]:
            d_tr = df[tr]
            m = LatentGaussianModel(d_tr["y"].to_numpy(), blocks(name), designs(name, d_tr), offset=off[tr]).fit()
            theta, x = m.theta, m.x_hat
            p = np.full(te.sum(), np.nan)
            for q0, q1 in zip(qstarts[:-1], qstarts[1:]):
                cond = (df["week_start"] < q0).to_numpy()
                d_c = df[cond]
                mq = LatentGaussianModel(d_c["y"].to_numpy(), blocks(name), designs(name, d_c), offset=off[cond])
                mq.theta = theta
                mq.x_hat, mq.f_hat, mq.lu = mq.mode(theta, x0=x)
                x = mq.x_hat
                inq = ((d_te["week_start"] >= q0) & (d_te["week_start"] < q1)).to_numpy()
                dq = d_te[inq]
                eta, sd = mq.predict_eta(designs(name, dq), off[te][inq], n_samples=30)
                p[inq] = prob_from_eta(eta, sd)
                print(name, q0.date(), "conditioning rows", int(cond.sum()), flush=True)
            res["models"][name] = {"hyper": m.hyper(), "log_ml_2018_2022": float(m.log_ml),
                                   "scores": dg.scores(d_te["y"].to_numpy(), p)}
            preds[f"p_{name}_upd"] = p.astype("float32")
            print(name, json.dumps(res["models"][name]["scores"]), flush=True)
        res.update(step_weeks=a.step, st_res_deg=a.st_res, quarters=[str(q.date()) for q in qstarts[:-1]])
        json.dump(res, open(os.path.join(a.out, "summary_update.json"), "w"), indent=1, default=float)
        pd.DataFrame({"grid_id": d_te["grid_id"].to_numpy(), "week_start": d_te["week_start"].to_numpy(),
                      "y": d_te["y"].to_numpy().astype("int8"), **preds}).to_parquet(
            os.path.join(a.out, "test_pred_update.parquet"), index=False)

    else:  # spacetime
        off = load_offset()
        tr = (df["year"] <= 2022).to_numpy(); te = ~tr
        wk = ((df["week_start"] - df["week_start"].min()).dt.days // 7).to_numpy()
        df["tstep"] = wk // a.step
        T = int(df["tstep"].max() + 1)
        r = a.st_res
        lon_v = np.floor(df.lon.to_numpy() / r) * r + r / 2
        lat_v = np.floor(df.lat.to_numpy() / r) * r + r / 2
        Lv, cell_v, _ = build_lattice(lon_v, lat_v, res=r, buffer=0)
        d_tr, d_te = df[tr], df[te]
        cv_tr, cv_te = cell_v[tr], cell_v[te]

        def D(d, cv):
            return [np.ones((len(d), 1)), de.index_design(d["country_id"], nC),
                    de.index_design(d["cell"], L.n), de.spacetime_design(cv, d["tstep"].to_numpy(), Lv.n, T)]
        blocks = [FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country"),
                  IIDBlock(L.n, name="cell", sd_init=0.5),
                  SpaceTimeAR1Block(Lv, T, name="st", range_init=300.0, sd_init=0.5, rho_init=0.8)]
        m = LatentGaussianModel(d_tr["y"].to_numpy(), blocks, D(d_tr, cv_tr), offset=off[tr]).fit()
        eta, sd = m.predict_eta(D(d_te, cv_te), off[te], n_samples=100)
        p = prob_from_eta(eta, sd)
        res.update(step_weeks=a.step, st_res_deg=a.st_res, st_cells=int(Lv.n), time_steps=T, train_steps=int(d_tr["tstep"].max() + 1),
                   latent_size=int(sum(b.size for b in blocks)),
                   scores=dg.scores(d_te["y"].to_numpy(), p), hyper=m.hyper(),
                   log_ml=float(m.log_ml), seconds=float(m.fit_seconds))
        print(json.dumps(res, default=float), flush=True)
        json.dump(res, open(os.path.join(a.out, f"summary_st{a.step}_r{a.st_res:g}.json"), "w"), indent=1, default=float)
        pd.DataFrame({"grid_id": d_te["grid_id"].to_numpy(), "week_start": d_te["week_start"].to_numpy(),
                      "y": d_te["y"].to_numpy().astype("int8"), f"p_st{a.step}_r{a.st_res:g}": p.astype("float32")}).to_parquet(
            os.path.join(a.out, f"test_pred_st{a.step}_r{a.st_res:g}.parquet"), index=False)


if __name__ == "__main__":
    main()
