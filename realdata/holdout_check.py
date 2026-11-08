"""Does the shared field carry information across outcomes? Country hold-out experiment.

Conflict is treated as unobserved in a group of countries (all weeks, 2018-2024), while
their unrest stays observed. The countries are split into K folds; for each fold, every
model is re-conditioned at its saved hyperparameters (--fixed-theta: one Laplace mode, no
re-optimisation) on the training rows minus the fold's conflict rows, and conflict is
predicted in the fold's countries.

Since conflict is unobserved there, the covariates built from conflict history (conflict:
lag_own, lag_own_nb, h_init; unrest: lag_other, lag_other_nb) are set to 0, their
training mean, for every row of the fold's countries in every model. The univariate
conflict models keep everything else, including the unrest history (lag_other) of the
held-out cells, so any gain of the joint models over them comes from the latent
structure (the shared field v reading the observed unrest), not from access to unrest.

Scores (conflict, held-out countries, by period: 2018-2022 'past' = unreported area,
2023-2024 'test' = forecast): log score, Brier, AUC-PR per model; for the joint models
also P(conflict | same-week unrest) (nowcast). Differences against uni_static and
uni_st, with country-clustered SEs.

Usage: python realdata/holdout_check.py clean.parquet --region africa_me --land SHP
         --fixed-theta results_real/bivariate/africa_me/summary_biv.json
         [--fixed-theta2 .../summary_biv2.json] --st-step 26 --folds 5 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc import design as de, diagnostics as dg
from sc.gmrf import CopyBlock, FixedBlock, IIDBlock, MaternBlock, SpaceTimeAR1Block
from sc.laplace import LatentGaussianModel
from fit_bivariate import (REGIONS, OUTS, neighbour_mean, stack, theta_from_hyper, mc_prob, joint_eval, _logb)
from fit_acled import build_lattice, sea_cells
from fit_glmm import prepare

# conflict-history columns per outcome (unknown where conflict is unobserved)
CONFLICT_HIST = {"conflict": ["lag_own", "h_init"], "unrest": ["lag_other"]}


def collinear(X, names, tol=1e-9):
    """Names of the columns of X that are (numerically) linear combinations of earlier
    columns, by pivoted Cholesky of the standardised Gram matrix (the intercept is kept)."""
    X = X.toarray() if sp.issparse(X) else np.asarray(X, float)
    G = np.zeros((X.shape[1], X.shape[1]))
    for a in range(0, X.shape[0], 500_000):
        G += X[a:a + 500_000].T @ X[a:a + 500_000]
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None)); C = G / np.outer(d, d)
    keep, drop = [], []
    for j in range(C.shape[0]):
        idx = keep + [j]
        ev = np.linalg.eigvalsh(C[np.ix_(idx, idx)])
        (keep if ev[0] > tol * len(idx) and np.diag(G)[j] > 0 else drop).append(j)
    return [names[j] for j in drop if names[j].split(":", 1)[1] != "intercept"]


def condition(m, hyper):
    t0 = time.time()
    m.theta = theta_from_hyper(m.blocks, hyper)
    m.x_hat, m.f_hat, m.lu = m.mode(m.theta)
    m.fit_seconds = time.time() - t0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--region", default="west_sahel")
    ap.add_argument("--box", type=float, nargs=4, default=None)
    ap.add_argument("--land", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--st-step", type=int, default=13); ap.add_argument("--st-res", type=float, default=1.0)
    ap.add_argument("--fixed-theta", required=True, help="summary with uni_static, uni_st, biv hyperparameters")
    ap.add_argument("--fixed-theta2", default=None, help="summary with biv2 hyperparameters (optional)")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-events", type=int, default=20,
                    help="countries with fewer conflict events (2018-2024) are never held out")
    ap.add_argument("--only-fold", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    saved = json.load(open(a.fixed_theta))["models"]
    if a.fixed_theta2:
        saved.update(json.load(open(a.fixed_theta2))["models"])
    models = [m for m in ("uni_static", "uni_st", "biv", "biv2") if m in saved]
    print("models:", models, flush=True)

    raw = pd.read_parquet(a.table)
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    x0, x1, y0, y1 = a.box if a.box else REGIONS[a.region]
    raw = raw[raw.lon.between(x0, x1) & raw.lat.between(y0, y1)].copy()
    dfs0, covs = {}, {}
    for o in OUTS:
        dfo, cov, _ = prepare(raw, o)
        dfs0[o], covs[o] = dfo.reset_index(drop=True), cov
    for o in OUTS:
        tr_o = dfs0[o]["train"].to_numpy()
        dead = [c for c in covs[o] if c in dfs0[o] and not np.any(dfs0[o].loc[tr_o, c].to_numpy() != 0)]
        covs[o] = [c for c in covs[o] if c not in dead]
        covs[o] = covs[o] + ["lag_own_nb", "lag_other_nb"]
    base = dfs0["conflict"]
    L, cell, observed = build_lattice(base.lon.to_numpy(), base.lat.to_numpy())
    barrier = sea_cells(L, a.land)
    iso = base["ISO_A3"].astype(str).to_numpy()
    country_id = pd.factorize(base["ISO_A3"])[0]
    nC = int(country_id.max() + 1)
    tr = base["train"].to_numpy()
    r = a.st_res
    lon_v = np.floor(base.lon.to_numpy() / r) * r + r / 2
    lat_v = np.floor(base.lat.to_numpy() / r) * r + r / 2
    Lv, cell_v, _ = build_lattice(lon_v, lat_v, res=r, buffer=0)
    wk = ((base["week_start"] - base["week_start"].min()).dt.days // 7).to_numpy()
    tstep = wk // a.st_step
    T = int(tstep.max() + 1)

    # folds: countries with enough conflict events, shuffled, split into K groups
    ev = pd.Series(dfs0["conflict"]["y"].to_numpy()).groupby(iso).sum()
    elig = sorted(ev[ev >= a.min_events].index)
    rng = np.random.default_rng(a.seed)
    perm = list(rng.permutation(elig))
    folds = [sorted(perm[k::a.folds]) for k in range(a.folds)]
    print("folds:", folds, flush=True)

    def blocks(model, names_c, names_u):
        if model in ("uni_static", "uni_st"):
            b = [FixedBlock(len(names_c), names_c), IIDBlock(nC, name="country_conflict"),
                 IIDBlock(L.n, name="cell_conflict", sd_init=0.5),
                 MaternBlock(L, name="field_conflict", range_init=300.0, barrier=barrier)]
            if model == "uni_st":
                b.append(SpaceTimeAR1Block(Lv, T, name="st_conflict", sd_init=0.5, rho_init=0.8))
            return b
        names = names_c + names_u
        b = [FixedBlock(len(names), names),
             IIDBlock(nC, name="country_conflict"), IIDBlock(nC, name="country_unrest"),
             IIDBlock(L.n, name="cell_conflict", sd_init=0.5), IIDBlock(L.n, name="cell_unrest", sd_init=0.5),
             MaternBlock(L, name="field_conflict", range_init=300.0, barrier=barrier),
             MaternBlock(L, name="field_unrest", range_init=100.0, barrier=barrier),
             CopyBlock(SpaceTimeAR1Block(Lv, T, name="st", sd_init=0.5, rho_init=0.8), name="shared_st")]
        if model == "biv2":
            b.append(SpaceTimeAR1Block(Lv, T, name="st_unrest", sd_init=0.3, rho_init=0.8))
        return b

    def designs(model, Pc, Pu):
        if model in ("uni_static", "uni_st"):
            D = [Pc["X"], Pc["C"], Pc["E"], Pc["W"]]
            return D + [Pc["V"]] if model == "uni_st" else D
        nc, nu = len(Pc["y"]), len(Pu["y"])
        X = sp.bmat([[sp.csr_matrix(Pc["X"]), None], [None, sp.csr_matrix(Pu["X"])]], format="csr")
        D = [X, stack(Pc["C"], None, nu=nu), stack(None, Pu["C"], nc=nc),
             stack(Pc["E"], None, nu=nu), stack(None, Pu["E"], nc=nc),
             stack(Pc["W"], None, nu=nu), stack(None, Pu["W"], nc=nc),
             sp.bmat([[Pc["V"], None], [None, Pu["V"]]], format="csr")]
        if model == "biv2":
            D.append(stack(None, Pu["V"], nc=nc))
        return D

    rows_out, summary = [], {"region": a.region, "folds": folds, "models": models, "per_fold": []}
    for k, fold in enumerate(folds):
        if a.only_fold is not None and k != a.only_fold:
            continue
        t0 = time.time()
        hold = np.isin(iso, fold)
        dfs = {}
        for o in OUTS:   # conflict history unknown in the held-out countries
            d_ = dfs0[o].copy()
            for c in CONFLICT_HIST[o]:
                if c in d_:
                    d_.loc[hold, c] = 0.0
            d_["cell"] = cell; d_["country_id"] = country_id
            for c in ("lag_own", "lag_other"):
                v = neighbour_mean(d_, L, c)
                mu, sd = v[tr].mean(), v[tr].std()
                d_[c + "_nb"] = (v - mu) / (sd if sd > 0 else 1)
            dfs[o] = d_
        # lag_own_nb of conflict and lag_other_nb of unrest are conflict-history spillovers:
        # their neighbour means already use the imputed values of held-out cells

        cv = {o: list(covs[o]) for o in OUTS}

        def parts(o, rows):
            dd = dfs[o][rows]
            X, names = de.fixed_design(dd, cv[o])
            return dict(X=X, names=[f"{o}:{n}" for n in names], C=de.index_design(dd["country_id"], nC),
                        E=de.index_design(dd["cell"], L.n), W=de.index_design(dd["cell"], L.n),
                        V=de.spacetime_design(cell_v[rows], tstep[rows], Lv.n, T), y=dd["y"].to_numpy())
        fit_c = tr & ~hold          # conflict likelihood: training rows outside the fold
        fit_u = tr                  # unrest likelihood: all training rows
        Pc, Pu = parts("conflict", fit_c), parts("unrest", fit_u)
        # without the fold's countries some country-level covariates can become collinear on
        # the conflict rows (e.g. a national indicator equal to a combination of others): drop them
        drop = collinear(Pc["X"], Pc["names"])
        if drop:
            print(f"fold {k}: dropping collinear conflict covariates {drop}", flush=True)
            cv["conflict"] = [c for c in cv["conflict"] if f"conflict:{c}" not in drop]
            Pc = parts("conflict", fit_c)
        Hc, Hu = parts("conflict", hold), parts("unrest", hold)   # held-out rows, all weeks
        yc, yu = Hc["y"], Hu["y"]
        out = {"grid_id": base.loc[hold, "grid_id"].to_numpy(), "week_start": base.loc[hold, "week_start"].to_numpy(),
               "iso": iso[hold], "period": np.where(tr[hold], "past", "test"), "fold": k,
               "y_conflict": yc.astype("int8"), "y_unrest": yu.astype("int8")}
        info = {"fold": k, "countries": fold, "rows": int(hold.sum()), "conflict_events": int(yc.sum())}
        for model in models:
            if model in ("uni_static", "uni_st"):
                m = condition(LatentGaussianModel(Pc["y"], blocks(model, Pc["names"], None), designs(model, Pc, None)),
                              saved[model]["hyper"])
                out[f"p_{model}"] = mc_prob(m, designs(model, Hc, None)).astype("float32")
            else:
                y = np.r_[Pc["y"], Pu["y"]]
                m = condition(LatentGaussianModel(y, blocks(model, Pc["names"], Pu["names"]), designs(model, Pc, Pu)),
                              saved[model]["hyper"])
                je = joint_eval(m, designs(model, Hc, Hu), yc, yu, seed=1)
                out[f"p_{model}"] = je["pc"].astype("float32")
                out[f"pcg_{model}"] = je["pc_given"].astype("float32")
            info[f"seconds_{model}"] = float(m.fit_seconds)
            print(f"fold {k} {model}: {m.fit_seconds:.0f} s", flush=True)
        info["seconds"] = time.time() - t0
        summary["per_fold"].append(info)
        rows_out.append(pd.DataFrame(out))
        pd.concat(rows_out).to_parquet(os.path.join(a.out, "holdout_pred.parquet"), index=False)

    summary["scores"] = score(pd.concat(rows_out))
    json.dump(summary, open(os.path.join(a.out, "holdout_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({p: {k: v for k, v in summary["scores"][p].items() if k != "by_country"}
                      for p in summary["scores"]}, indent=1, default=float))


def score(d):
    from sklearn.metrics import average_precision_score as aps
    res = {}
    for per in ("past", "test", "all"):
        s = d if per == "all" else d[d.period == per]
        y = s.y_conflict.to_numpy(); cl = s.grid_id.to_numpy(); ct = s.iso.to_numpy()
        R = {"rows": int(len(s)), "events": int(y.sum())}
        cols = [c for c in s.columns if c.startswith("p_") or c.startswith("pcg_")]
        for c in cols:
            p = s[c].to_numpy().astype(float)
            R[c] = {"log_score": float(-np.mean(_logb(y, p))), "brier": float(np.mean((p - y) ** 2)),
                    "auc_pr": float(aps(y, p)) if y.sum() else None}
        for c in cols:
            for b in ("p_uni_static", "p_uni_st"):
                if c == b or b not in s:
                    continue
                md, ses, _ = dg.clustered_diff(-_logb(y, s[c].to_numpy().astype(float)),
                                               -_logb(y, s[b].to_numpy().astype(float)), cl, ct)
                # negative = c better (lower log score) than the baseline
                R[f"{c}_minus_{b}"] = {"mean": md, "se_cell": ses[0], "se_country": ses[1]}
        by = {}
        for cc, g in s.groupby("iso"):
            yy = g.y_conflict.to_numpy()
            by[cc] = {"events": int(yy.sum()), **{c: float(-np.mean(_logb(yy, g[c].to_numpy().astype(float))))
                                                  for c in cols}}
        R["by_country"] = by
        res[per] = R
    return res


def merge(dirs, out):
    """Pool the per-fold predictions of an array job (one --only-fold per task) and score."""
    d = pd.concat([pd.read_parquet(os.path.join(x, "holdout_pred.parquet")) for x in dirs])
    folds = [json.load(open(os.path.join(x, "holdout_summary.json")))["per_fold"] for x in dirs]
    res = {"per_fold": [f for ff in folds for f in ff], "scores": score(d)}
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({p: {k: v for k, v in res["scores"][p].items() if k != "by_country"}
                      for p in res["scores"]}, indent=1, default=float))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":   # --merge OUT.json DIR [DIR ...]
        merge(sys.argv[3:], sys.argv[2])
    else:
        main()
