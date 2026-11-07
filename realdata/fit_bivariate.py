"""Joint (bivariate) latent Gaussian model of conflict and unrest: escalation or shared
episodes? (paper Sec. 3.x and 5.x; simulation S5).

For outcome j in {conflict, unrest}, cell s and week t,

  logit P(Y_jst = 1) = x_jst' beta_j + delta_j L_own + kappa_j L_other
                       + kappa_nb_j L_other_nb + delta_nb_j L_own_nb
                       + u_j,c(s) + e_j(s) + w_j(s) + lambda_j v(s, t),

with outcome-specific fixed effects (the covariates of fit_glmm.prepare, including the
missingness indicators), country effects u_j, iid cell effects e_j, static barrier
Matern fields w_j, and ONE space-time field v (AR(1) x Matern on a --st-res degree
grid, --st-step-week steps) shared by both outcomes, with loading 1 for conflict and
lambda for unrest (CopyBlock). L_other_nb is the mean of the other outcome's 4-week
history over the rook neighbours of the cell (spillover across cells).

kappa_conflict (unrest history -> conflict) is the escalation effect. It is compared
with univariate conflict models:
  uni_static : conflict model with u, e, w (no space-time field)
  uni_st     : uni_static + conflict's own space-time field
  biv        : the joint model above
Fitted on 2018-2022; test-period (2023-2024) predictions for both outcomes are written
to test_pred_biv.parquet (p_conflict_<model>, p_unrest_biv) for compare.py.

Usage: python fit_bivariate.py clean.parquet --region west_sahel --land SHP --out DIR
       [--st-step 13] [--st-res 1.0] [--models uni_static,uni_st,biv]
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
import scipy.sparse as sp

from sc import design as de, diagnostics as dg
from sc.gmrf import CopyBlock, FixedBlock, IIDBlock, MaternBlock, SpaceTimeAR1Block
from sc.laplace import LatentGaussianModel, prob_from_eta
from fit_acled import REGIONS, build_lattice, sea_cells
from fit_glmm import prepare

REGIONS = dict(REGIONS, west_sahel=(-18, 16, 4, 25), central_sahel=(-6, 16, 9, 20))
OUTS = ("conflict", "unrest")


def neighbour_mean(df, L, col):
    """Mean of `col` over the rook neighbours (with data) of each cell, week by week."""
    A = L.adjacency.tocsr()
    wk = pd.factorize(df["week_start"])[0]
    out = np.full(len(df), np.nan)
    for w in np.unique(wk):
        i = np.flatnonzero(wk == w)
        v = np.zeros(L.n); has = np.zeros(L.n)
        v[df["cell"].to_numpy()[i]] = df[col].to_numpy()[i]; has[df["cell"].to_numpy()[i]] = 1
        s, k = A @ v, A @ has
        out[i] = (s / np.maximum(k, 1))[df["cell"].to_numpy()[i]]
    return out


def stack(Dc, Du):
    """Rows of conflict on top of rows of unrest; Dc / Du may be None (zero block)."""
    Dc = sp.csr_matrix(Dc) if Dc is not None else None
    Du = sp.csr_matrix(Du) if Du is not None else None
    if Dc is None:
        Dc = sp.csr_matrix((Du.shape[0], Du.shape[1]))
    if Du is None:
        Du = sp.csr_matrix((Dc.shape[0], Dc.shape[1]))
    return sp.vstack([Dc, Du], format="csr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--region", default="west_sahel")
    ap.add_argument("--land", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--st-step", type=int, default=13); ap.add_argument("--st-res", type=float, default=1.0)
    ap.add_argument("--models", default="uni_static,uni_st,biv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw = pd.read_parquet(a.table)
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    x0, x1, y0, y1 = REGIONS[a.region]
    raw = raw[raw.lon.between(x0, x1) & raw.lat.between(y0, y1)].copy()
    dfs, covs = {}, {}
    for o in OUTS:
        dfo, cov, _ = prepare(raw, o)
        dfs[o], covs[o] = dfo.reset_index(drop=True), cov
    key = ["grid_id", "week_start"]
    assert dfs["conflict"][key].equals(dfs["unrest"][key]), "prepare() rows differ between outcomes"
    base = dfs["conflict"]
    L, cell, observed = build_lattice(base.lon.to_numpy(), base.lat.to_numpy())
    barrier = sea_cells(L, a.land)
    for o in OUTS:
        dfs[o]["cell"] = cell
        dfs[o]["country_id"] = pd.factorize(base["ISO_A3"])[0]
    for o in OUTS:                     # neighbour (spillover) histories, standardised on training rows
        dd = dfs[o]
        for c in ("lag_own", "lag_other"):
            v = neighbour_mean(dd, L, c)
            mu, sd = v[dd.train].mean(), v[dd.train].std()
            dd[c + "_nb"] = (v - mu) / (sd if sd > 0 else 1)
        covs[o] = covs[o] + ["lag_own_nb", "lag_other_nb"]
    nC = int(base["country_id"].max() + 1)
    tr = base["train"].to_numpy(); te = ~tr
    r = a.st_res
    lon_v = np.floor(base.lon.to_numpy() / r) * r + r / 2
    lat_v = np.floor(base.lat.to_numpy() / r) * r + r / 2
    Lv, cell_v, _ = build_lattice(lon_v, lat_v, res=r, buffer=0)
    wk = ((base["week_start"] - base["week_start"].min()).dt.days // 7).to_numpy()
    tstep = wk // a.st_step
    T = int(tstep.max() + 1)
    print(f"{a.region}: {len(base):,} rows per outcome, {observed.sum():,} cells, {nC} countries, "
          f"v on {Lv.n:,} cells x {T} steps", flush=True)

    def parts(o, rows):
        dd = dfs[o][rows]
        X, names = de.fixed_design(dd, covs[o])
        return dict(X=X, names=[f"{o}:{n}" for n in names], C=de.index_design(dd["country_id"], nC),
                    E=de.index_design(dd["cell"], L.n), W=de.index_design(dd["cell"], L.n),
                    V=de.spacetime_design(cell_v[rows], tstep[rows], Lv.n, T), y=dd["y"].to_numpy())

    def uni_blocks(o, st):
        b = [FixedBlock(len(parts_tr[o]["names"]), parts_tr[o]["names"]), IIDBlock(nC, name=f"country_{o}"),
             IIDBlock(L.n, name=f"cell_{o}", sd_init=0.5),
             MaternBlock(L, name=f"field_{o}", range_init=300.0, barrier=barrier)]
        if st:
            b.append(SpaceTimeAR1Block(Lv, T, name=f"st_{o}", sd_init=0.5, rho_init=0.8))
        return b

    def uni_designs(P, st):
        D = [P["X"], P["C"], P["E"], P["W"]]
        return D + [P["V"]] if st else D

    def biv_blocks():
        names = parts_tr["conflict"]["names"] + parts_tr["unrest"]["names"]
        return [FixedBlock(len(names), names),
                IIDBlock(nC, name="country_conflict"), IIDBlock(nC, name="country_unrest"),
                IIDBlock(L.n, name="cell_conflict", sd_init=0.5), IIDBlock(L.n, name="cell_unrest", sd_init=0.5),
                MaternBlock(L, name="field_conflict", range_init=300.0, barrier=barrier),
                MaternBlock(L, name="field_unrest", range_init=100.0, barrier=barrier),
                CopyBlock(SpaceTimeAR1Block(Lv, T, name="st", sd_init=0.5, rho_init=0.8), name="shared_st")]

    def biv_designs(Pc, Pu):
        Xc, Xu = sp.csr_matrix(Pc["X"]), sp.csr_matrix(Pu["X"])
        X = sp.bmat([[Xc, None], [None, Xu]], format="csr")
        return [X, stack(Pc["C"], None), stack(None, Pu["C"]), stack(Pc["E"], None), stack(None, Pu["E"]),
                stack(Pc["W"], None), stack(None, Pu["W"]),
                sp.bmat([[Pc["V"], None], [None, Pu["V"]]], format="csr")]

    parts_tr = {o: parts(o, tr) for o in OUTS}
    parts_te = {o: parts(o, te) for o in OUTS}
    res = {"region": a.region, "box": list(REGIONS[a.region]), "rows_per_outcome": int(len(base)),
           "cells": int(observed.sum()), "countries": nC, "st_res_deg": r, "st_step_weeks": a.st_step,
           "st_cells": int(Lv.n), "time_steps": T, "models": {}}
    preds = {"grid_id": base.loc[te, "grid_id"].to_numpy(), "week_start": base.loc[te, "week_start"].to_numpy(),
             "y_conflict": parts_te["conflict"]["y"].astype("int8"), "y_unrest": parts_te["unrest"]["y"].astype("int8"),
             "events_52w_conflict": np.nan * np.zeros(te.sum())}
    keys = ["conflict:lag_own", "conflict:lag_other", "conflict:lag_own_nb", "conflict:lag_other_nb",
            "unrest:lag_own", "unrest:lag_other", "unrest:lag_own_nb", "unrest:lag_other_nb"]
    for model in a.models.split(","):
        if model in ("uni_static", "uni_st"):
            st = model == "uni_st"
            P = parts_tr["conflict"]
            m = LatentGaussianModel(P["y"], uni_blocks("conflict", st), uni_designs(P, st)).fit()
            eta, sd = m.predict_eta(uni_designs(parts_te["conflict"], st), None, n_samples=50)
            preds[f"p_conflict_{model}"] = prob_from_eta(eta, sd).astype("float32")
        else:
            y = np.r_[parts_tr["conflict"]["y"], parts_tr["unrest"]["y"]]
            m = LatentGaussianModel(y, biv_blocks(), biv_designs(parts_tr["conflict"], parts_tr["unrest"])).fit()
            Dte = biv_designs(parts_te["conflict"], parts_te["unrest"])
            eta, sd = m.predict_eta(Dte, None, n_samples=50)
            p = prob_from_eta(eta, sd); nte = int(te.sum())
            preds["p_conflict_biv"] = p[:nte].astype("float32"); preds["p_unrest_biv"] = p[nte:].astype("float32")
        fs = m.fixed_summary()
        res["models"][model] = {"hyper": m.hyper(), "log_ml": float(m.log_ml), "seconds": float(m.fit_seconds),
                                "lag_coefficients": {k: list(map(float, fs[k])) for k in keys if k in fs},
                                "fixed": {k: list(map(float, v)) for k, v in fs.items()}}
        sc = {"conflict": dg.scores(parts_te["conflict"]["y"], preds[f"p_conflict_{model}"])}
        if model == "biv":
            sc["unrest"] = dg.scores(parts_te["unrest"]["y"], preds["p_unrest_biv"])
        res["models"][model]["test_scores"] = sc
        print(model, json.dumps({k: res["models"][model][k] for k in ("hyper", "lag_coefficients", "test_scores")},
                                default=float), flush=True)
        json.dump(res, open(os.path.join(a.out, "summary_biv.json"), "w"), indent=1, default=float)
    pd.DataFrame(preds).drop(columns=["events_52w_conflict"]).to_parquet(
        os.path.join(a.out, "test_pred_biv.parquet"), index=False)


if __name__ == "__main__":
    main()
