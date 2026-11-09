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
from fit_bivariate import (REGIONS, region_rows, OUTS, neighbour_mean, stack, theta_from_hyper, theta0_from_init, subsample,
                           mc_prob, joint_eval, _logb)
from fit_acled import build_lattice, sea_cells
from fit_glmm import prepare

# history columns that are unknown where the TARGET outcome is unobserved, per outcome
UNKNOWN_HIST = {"conflict": {"conflict": ["lag_own", "h_init"], "unrest": ["lag_other"]},
                "unrest": {"unrest": ["lag_own", "h_init"], "conflict": ["lag_other"]}}


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


def refit(make, P_by_outcome, designs_fn, hyper_init, r, seed):
    """Re-estimate the hyperparameters on this fold's data (all events + a fraction r of the
    non-events, offset -log r; warm start from hyper_init), then the latent posterior on all
    of the fold's fitting rows at those hyperparameters."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    S = {o: subsample(P, r, rng) for o, P in P_by_outcome.items()}
    ys = np.concatenate([S[o]["y"] for o in S]); off = np.concatenate([S[o]["off"] for o in S])
    blocks = make()
    th0 = theta0_from_init(blocks, hyper_init) if hyper_init else None
    ms = LatentGaussianModel(ys, blocks, designs_fn(S), offset=off).fit(theta0=th0)
    yf = np.concatenate([P_by_outcome[o]["y"] for o in P_by_outcome])
    m = LatentGaussianModel(yf, make(), designs_fn(P_by_outcome))
    m.theta = ms.theta
    m.x_hat, m.f_hat, m.lu = m.mode(m.theta)
    m.fit_seconds = time.time() - t0
    m.hyper_fold = m.hyper()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--region", default="west_sahel")
    ap.add_argument("--box", type=float, nargs=4, default=None)
    ap.add_argument("--land", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--st-step", type=int, default=13); ap.add_argument("--st-res", type=float, default=1.0)
    ap.add_argument("--fixed-theta", required=True,
                    help="summary with the uni_static(_u), uni_st(_u), biv hyperparameters (starting values with --refit)")
    ap.add_argument("--fixed-theta2", default=None, help="summary with biv2 hyperparameters (optional)")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-events", type=int, default=20,
                    help="countries with fewer conflict events (2018-2024) are never held out")
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--target", choices=("conflict", "unrest"), default="conflict",
                    help="the outcome treated as unobserved in the held-out countries")
    ap.add_argument("--refit", action="store_true",
                    help="re-estimate the hyperparameters in every fold (no information from the held-out "
                         "countries' target), with --zero-frac subsampling and a warm start from the saved values")
    ap.add_argument("--zero-frac", type=float, default=0.25)
    a = ap.parse_args()
    tg = a.target; ot = "unrest" if tg == "conflict" else "conflict"
    os.makedirs(a.out, exist_ok=True)
    saved = json.load(open(a.fixed_theta))["models"]
    if a.fixed_theta2:
        saved.update(json.load(open(a.fixed_theta2))["models"])
    sfx = "" if tg == "conflict" else "_u"         # saved univariate unrest models: uni_static_u, uni_st_u
    models = [m for m in ("uni_static", "uni_st", "biv", "biv2")
              if a.refit or (m + sfx if m.startswith("uni") else m) in saved]
    hyper_of = lambda m: (saved.get(m + sfx if m.startswith("uni") else m) or {}).get("hyper")   # noqa: E731
    print("models:", models, flush=True)

    raw = pd.read_parquet(a.table)
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    x0, x1, y0, y1 = a.box if a.box else REGIONS[a.region]
    raw = region_rows(raw, a.region, (x0, x1, y0, y1), a.land)
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

    # folds: countries with enough target events, shuffled, split into K groups
    ev = pd.Series(dfs0[tg]["y"].to_numpy()).groupby(iso).sum()
    elig = sorted(ev[ev >= a.min_events].index)
    rng = np.random.default_rng(a.seed)
    perm = list(rng.permutation(elig))
    folds = [sorted(perm[k::a.folds]) for k in range(a.folds)]
    print("folds:", folds, flush=True)

    def blocks(model, names_c, names_u):
        if model in ("uni_static", "uni_st"):    # univariate model of the target (names_c = its names)
            b = [FixedBlock(len(names_c), names_c), IIDBlock(nC, name=f"country_{tg}"),
                 IIDBlock(L.n, name=f"cell_{tg}", sd_init=0.5),
                 MaternBlock(L, name=f"field_{tg}", range_init=300.0, barrier=barrier)]
            if model == "uni_st":
                b.append(SpaceTimeAR1Block(Lv, T, name=f"st_{tg}", sd_init=0.5, rho_init=0.8))
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
        for o in OUTS:   # target history unknown in the held-out countries
            d_ = dfs0[o].copy()
            for c in UNKNOWN_HIST[tg][o]:
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
        fit_rows = {tg: tr & ~hold, ot: tr}   # target: training rows outside the fold; other: all
        P = {o: parts(o, fit_rows[o]) for o in OUTS}
        # without the fold's countries some country-level covariates can become collinear on
        # the target's rows (e.g. a national indicator equal to a combination of others): drop them
        drop = collinear(P[tg]["X"], P[tg]["names"])
        if drop:
            print(f"fold {k}: dropping collinear {tg} covariates {drop}", flush=True)
            cv[tg] = [c for c in cv[tg] if f"{tg}:{c}" not in drop]
            P[tg] = parts(tg, fit_rows[tg])
        Pc, Pu = P["conflict"], P["unrest"]
        Hc, Hu = parts("conflict", hold), parts("unrest", hold)   # held-out rows, all weeks
        H = {"conflict": Hc, "unrest": Hu}
        yc, yu = Hc["y"], Hu["y"]
        out = {"grid_id": base.loc[hold, "grid_id"].to_numpy(), "week_start": base.loc[hold, "week_start"].to_numpy(),
               "iso": iso[hold], "period": np.where(tr[hold], "past", "test"), "fold": k,
               "y_conflict": yc.astype("int8"), "y_unrest": yu.astype("int8")}
        out["y_target"] = H[tg]["y"].astype("int8")
        info = {"fold": k, "countries": fold, "rows": int(hold.sum()), "target_events": int(H[tg]["y"].sum())}
        for model in models:
            if model in ("uni_static", "uni_st"):
                mk = lambda: blocks(model, P[tg]["names"], None)                         # noqa: E731
                if a.refit:
                    m = refit(mk, {tg: P[tg]}, lambda S: designs(model, S[tg], None), hyper_of(model),
                              a.zero_frac, 100 + k)
                else:
                    m = condition(LatentGaussianModel(P[tg]["y"], mk(), designs(model, P[tg], None)), hyper_of(model))
                out[f"p_{model}"] = mc_prob(m, designs(model, H[tg], None)).astype("float32")
            else:
                mk = lambda: blocks(model, Pc["names"], Pu["names"])                     # noqa: E731
                if a.refit:
                    m = refit(mk, {"conflict": Pc, "unrest": Pu},
                              lambda S: designs(model, S["conflict"], S["unrest"]), hyper_of(model), a.zero_frac, 100 + k)
                else:
                    m = condition(LatentGaussianModel(np.r_[Pc["y"], Pu["y"]], mk(), designs(model, Pc, Pu)),
                                  hyper_of(model))
                je = joint_eval(m, designs(model, Hc, Hu), yc, yu, seed=1)
                out[f"p_{model}"] = (je["pc"] if tg == "conflict" else je["pu"]).astype("float32")
                if tg == "conflict":
                    out[f"pcg_{model}"] = je["pc_given"].astype("float32")
            if hasattr(m, "hyper_fold"):
                info[f"hyper_{model}"] = m.hyper_fold
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
        y = s.y_target.to_numpy(); cl = s.grid_id.to_numpy(); ct = s.iso.to_numpy()
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
            yy = g.y_target.to_numpy()
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
