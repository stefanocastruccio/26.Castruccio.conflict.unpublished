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
from sc.gmrf import CopyBlock, FixedBlock, IIDBlock, MaternBlock, RegionCopyBlock, SpaceTimeAR1Block
from sc.laplace import LatentGaussianModel, prob_from_eta
from fit_acled import REGIONS, build_lattice, sea_cells
from fit_glmm import prepare

REGIONS = dict(REGIONS, west_sahel=(-18, 16, 4, 25), central_sahel=(-6, 16, 9, 20))
# "ame": the africa_me box without its European countries (2026-11-09). They enter the box only
# through its corner, ACLED covers most of them only from 2020, and their dense, well-reported
# protest with almost no conflict is unrest-only variation that the joint model must absorb.
EUROPE = ["ALB", "AND", "BGR", "BIH", "ESP", "FRA", "GRC", "HRV", "ITA", "KOS", "MCO", "MKD", "MLT",
          "MNE", "PRT", "ROU", "RUS", "SMR", "SRB", "VAT", "SVN", "HUN", "MDA", "UKR"]
REGIONS["ame"] = REGIONS["africa_me"]
EXCLUDE = {"ame": EUROPE}


def region_rows(raw, region, box, shp):
    """Rows of `raw` in the box, minus the cells whose nearest country (by polygon, from the
    Natural Earth file `shp`, ADM0_A3 codes; the table's ISO_A3 is -99 for some countries)
    is excluded for this region."""
    x0, x1, y0, y1 = box
    raw = raw[raw.lon.between(x0, x1) & raw.lat.between(y0, y1)]
    if region not in EXCLUDE:
        return raw.copy()
    import geopandas as gpd
    land = gpd.read_file(shp).to_crs(4326)
    code = "ADM0_A3" if "ADM0_A3" in land.columns else "ISO_A3"
    cells = raw[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.lon, cells.lat), crs=4326)
    import warnings
    with warnings.catch_warnings():   # nearest in degrees: only decides coastal cells
        warnings.simplefilter("ignore", UserWarning)
        near = gpd.sjoin_nearest(pts, land[[code, "geometry"]], how="left")
    near = near[~near.index.duplicated()]
    drop = cells[near[code].isin(EXCLUDE[region]).to_numpy()]
    key = pd.MultiIndex.from_frame(raw[["lon", "lat"]])
    out = raw[~key.isin(pd.MultiIndex.from_frame(drop[["lon", "lat"]]))].copy()
    print(f"{region}: excluded {len(drop):,} cells ({len(raw) - len(out):,} rows) in "
          f"{sorted(set(near[code][near[code].isin(EXCLUDE[region])]))}", flush=True)
    return out
OUTS = ("conflict", "unrest")
# Regions for spatially varying escalation and loadings (prototype on "ame"; the global model
# uses the same assignment by country). Every other country is Africa.
MIDDLE_EAST = ["TUR", "SYR", "LBN", "ISR", "PSE", "JOR", "IRQ", "IRN", "SAU", "YEM", "OMN", "ARE",
               "QAT", "BHR", "KWT", "CYP", "CYN"]
SOUTH_ASIA = ["AFG", "PAK", "IND", "NPL", "BTN", "BGD", "LKA", "MDV", "KAS", "KAB"]
GROUPS = ("africa", "middle_east", "south_asia")
JOINT = ("biv", "biv2", "biv3", "biv3_nolam", "biv2_rl", "biv2_rlk", "biv2_rlf")


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


def stack(Dc, Du, nc=None, nu=None):
    """Rows of conflict on top of rows of unrest; Dc / Du may be None (zero block with nc /
    nu rows; by default the other outcome's row count)."""
    Dc = sp.csr_matrix(Dc) if Dc is not None else None
    Du = sp.csr_matrix(Du) if Du is not None else None
    if Dc is None:
        Dc = sp.csr_matrix((Du.shape[0] if nc is None else nc, Du.shape[1]))
    if Du is None:
        Du = sp.csr_matrix((Dc.shape[0] if nu is None else nu, Dc.shape[1]))
    return sp.vstack([Dc, Du], format="csr")


def theta_from_hyper(blocks, hyper):
    """Internal hyperparameter vector from a saved hyper() dictionary (inverse of describe)."""
    from sc.gmrf import CopyBlock, IIDBlock, MaternBlock, RegionCopyBlock, SpaceTimeAR1Block
    th = []
    for b in blocks:
        if isinstance(b, (CopyBlock, RegionCopyBlock)):
            base, nm = b.base, b.name
        else:
            base, nm = b, b.name
        if isinstance(base, IIDBlock):
            th += [np.log(hyper[f"{nm}.sd"])]
        elif isinstance(base, SpaceTimeAR1Block):
            th += [np.log(hyper[f"{nm}.range_km"]), np.log(hyper[f"{nm}.sd"]), np.arctanh(hyper[f"{nm}.rho"])]
        elif isinstance(base, MaternBlock):
            th += [np.log(hyper[f"{nm}.range_km"]), np.log(hyper[f"{nm}.sd"])]
        if isinstance(b, CopyBlock):
            th += [hyper[f"{nm}.loading"]]
        elif isinstance(b, RegionCopyBlock):   # region loadings; fall back to a single saved loading
            th += [hyper.get(f"{nm}.loading[{g}]", hyper.get(f"{nm}.loading")) for g in b.group_names]
            if any(v is None for v in th[-b.G:]):
                raise KeyError(f"{nm}.loading")
    return np.array(th, float)


def theta0_from_init(blocks, hyper):
    """Starting values: saved hyperparameters for every block named in `hyper` (e.g. the
    fitted biv model when fitting biv2), the generic defaults for the others."""
    th = []
    for b in blocks:
        try:
            th.append(theta_from_hyper([b], hyper))
        except KeyError:
            th.append(np.asarray(b.theta0(), float))
    th = np.concatenate(th)
    lo, hi = zip(*[bd for b in blocks for bd in b.bounds()])
    return np.clip(th, np.array(lo) + 1e-6, np.array(hi) - 1e-6)


def fit_or_condition(m, saved, model, init=None):
    """Full empirical-Bayes fit, or (with --fixed-theta) the posterior at the saved
    hyperparameters of the same model: one Laplace mode instead of the optimisation.
    With `init` (a saved summary, --init-theta), the fit is warm-started from the
    hyperparameters of the joint model `biv` in that summary."""
    if saved is None or model not in saved.get("models", {}):
        src = next((k for k in (model, "biv3", "biv2", "biv") if init is not None and k in init.get("models", {})), None)
        if src is not None:
            th0 = theta0_from_init(m.blocks, init["models"][src]["hyper"])
            print(f"warm start for {model} from saved {src}: {np.round(th0, 3)}", flush=True)
            return m.fit(theta0=th0)
        return m.fit()
    import time
    t0 = time.time()
    m.theta = theta_from_hyper(m.blocks, saved["models"][model]["hyper"])
    m.x_hat, m.f_hat, m.lu = m.mode(m.theta)
    m.log_ml = m.log_marginal(m.theta)
    m.x_hat, m.f_hat, m.lu = m.mode(m.theta, x0=m._x)
    m.fit_seconds = time.time() - t0
    return m


def subsample(P, r, rng):
    """All events and a fraction r of the non-events of one outcome's training parts; the
    offset -log r makes the logistic likelihood of the retained rows exact (case-control
    sampling on the outcome: logit P(y=1 | kept) = eta - log r)."""
    keep = (P["y"] == 1) | (rng.random(len(P["y"])) < r)
    Q = {k: (v[keep] if k != "names" else v) for k, v in P.items()}
    Q["off"] = np.full(int(keep.sum()), -np.log(r))
    return Q


def fit_subsampled(make, y_sub, D_sub, off_sub, y_full, D_full, model, saved, init):
    """--zero-frac: hyperparameters from the case-control subsample, latent posterior on
    all training rows at those hyperparameters (one Laplace mode)."""
    import time
    t0 = time.time()
    ms = fit_or_condition(LatentGaussianModel(y_sub, make(), D_sub, offset=off_sub), saved, model, init)
    t_sub = time.time() - t0
    m = LatentGaussianModel(y_full, make(), D_full)
    m.theta = ms.theta
    m.mode(m.theta)
    m.log_ml = m.log_marginal(m.theta)
    m.x_hat, m.f_hat, m.lu = m.mode(m.theta, x0=m._x)
    m.fit_seconds = time.time() - t0
    m.sub_model = ms
    m.sub_info = {"rows": int(len(y_sub)), "rows_full": int(len(y_full)), "seconds_hyper": float(t_sub),
                  "log_ml_subsample": float(ms.log_ml),
                  "n_evals": int(ms.n_evals), "nit": int(getattr(getattr(ms, "opt", None), "nit", -1))}
    print(f"{model}: subsample {len(y_sub):,}/{len(y_full):,} rows, hyperparameters in {t_sub:.0f} s, "
          f"total {m.fit_seconds:.0f} s", flush=True)
    return m


def calibration(yc, yu, pc, pu, pboth):
    """Observed / predicted ratios, overall and in the top 1% and 10% of predicted
    co-occurrence (the hotspots, where the joint model under-predicted)."""
    both = yc * yu
    out = {"obs_conflict": float(yc.mean()), "pred_conflict": float(pc.mean()),
           "obs_unrest": float(yu.mean()), "pred_unrest": float(pu.mean()),
           "obs_both": float(both.mean()), "pred_both": float(pboth.mean())}
    for q in (0.90, 0.99):
        top = pboth >= np.quantile(pboth, q)
        out[f"top{int(round(100 * (1 - q)))}"] = {
            "obs_both": float(both[top].mean()), "pred_both": float(pboth[top].mean()),
            "obs_conflict": float(yc[top].mean()), "pred_conflict": float(pc[top].mean()),
            "obs_unrest": float(yu[top].mean()), "pred_unrest": float(pu[top].mean())}
    return out


def _logb(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.where(y == 1, np.log(p), np.log1p(-p))


def mc_prob(m, D, n_samples=200, seed=1, chunk=100_000):
    """Marginal predictive probability E[expit(eta)] by Monte Carlo over the same posterior
    draws of the latent vector that joint_eval uses, so that univariate and joint models are
    scored with the same approximation (the probit-matching formula of prob_from_eta is
    not accurate enough for rare events to compare log scores at the 1e-3 level)."""
    A = sp.hstack([sp.csr_matrix(x) for x in D], format="csr")
    S = m.sample(n_samples, np.random.default_rng(seed))
    out = np.empty(A.shape[0])
    for a in range(0, A.shape[0], chunk):
        b = min(a + chunk, A.shape[0])
        out[a:b] = (1 / (1 + np.exp(-(A[a:b] @ S)))).mean(1)
    return out


def joint_eval(m, Dte, yc, yu, n_samples=200, seed=0, chunk=100_000):
    """Joint and conditional predictive quantities of a joint model on the test rows, from
    posterior samples of the latent vector (so conflict and unrest share the same draw of
    every latent effect, including the shared field v).

    Returns per-row arrays:
      logj      log p(yc, yu)                    (joint predictive, dependence kept)
      logi      log p(yc) + log p(yu)            (same model, dependence ignored)
      pc, pu    marginal predictive probabilities
      pc_given  P(yc = 1 | yu observed)           (conditional, 'nowcast' given unrest)
      pboth     P(yc = 1, yu = 1)"""
    A = sp.hstack([sp.csr_matrix(D) for D in Dte], format="csr")
    n = len(yc)
    Ac, Au = A[:n], A[n:]
    S = m.sample(n_samples, np.random.default_rng(seed))
    out = {k: np.empty(n) for k in ("logj", "logi", "pc", "pu", "pc_given", "pboth")}
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        pc = 1 / (1 + np.exp(-(Ac[a:b] @ S))); pu = 1 / (1 + np.exp(-(Au[a:b] @ S)))
        bc = np.where(yc[a:b, None] == 1, pc, 1 - pc); bu = np.where(yu[a:b, None] == 1, pu, 1 - pu)
        mc, mu = pc.mean(1), pu.mean(1)
        out["logj"][a:b] = np.log(np.clip((bc * bu).mean(1), 1e-300, None))
        out["logi"][a:b] = _logb(yc[a:b], mc) + _logb(yu[a:b], mu)
        out["pc"][a:b], out["pu"][a:b] = mc, mu
        out["pc_given"][a:b] = (pc * bu).mean(1) / np.clip(bu.mean(1), 1e-300, None)
        out["pboth"][a:b] = (pc * pu).mean(1)
    return out


def export_fields(m, model, L, observed, Lv, T, a, week0):
    """Posterior means of the latent fields of a joint model, for maps (Section 5):
    fields_<model>_static.csv.gz  (0.5-degree cells: static fields, cell effects, shared static field)
    fields_<model>_st.csv.gz      (st-res cells x time steps: shared and unrest-specific space-time fields)
    Model output only (no event data)."""
    k = {b.name: i for i, b in enumerate(m.blocks)}
    st = pd.DataFrame({"lon": L.lon[observed], "lat": L.lat[observed]})
    for nm in ("field_conflict", "field_unrest", "cell_conflict", "cell_unrest",
               "kappa_field_conflict", "kappa_field_unrest"):
        if nm in k:
            st[nm] = m.block_mean(k[nm])[observed]
    if "shared_field" in k:
        st["shared_field"] = m.block_mean(k["shared_field"])[:L.n][observed]
    st.to_csv(os.path.join(a.out, f"fields_{model}_static.csv.gz"), index=False, float_format="%.4f")
    n = Lv.n
    if "shared_st" not in k:
        return
    v = m.block_mean(k["shared_st"])[:n * T].reshape(T, n)
    tt = pd.DataFrame({"lon": np.tile(Lv.lon, T), "lat": np.tile(Lv.lat, T), "step": np.repeat(np.arange(T), n),
                       "step_start": np.repeat(week0 + pd.to_timedelta(7 * a.st_step * np.arange(T), unit="D"), n),
                       "shared_st": v.ravel()})
    if "st_unrest" in k:
        tt["st_unrest"] = m.block_mean(k["st_unrest"]).reshape(T, n).ravel()
    tt.to_csv(os.path.join(a.out, f"fields_{model}_st.csv.gz"), index=False, float_format="%.4f")


def export_fitted(m, model, base, tr, tstep, a):
    """Plug-in fitted probabilities expit(eta at the posterior mode) on the TRAINING rows of both
    outcomes, for the maps of fitted intensity (paper/make_fitted_map.py). Written to --out only
    (it is row-level and sits next to the event data on the cluster), never to the repository."""
    from scipy.special import expit
    eta = np.asarray(m.offset + m.A @ m.x_hat).ravel()
    n = int(tr.sum())
    d = base.loc[tr, ["grid_id", "lon", "lat", "week_start"]].reset_index(drop=True)
    d["step"] = tstep[tr]
    d["p_conflict"] = expit(eta[:n]).astype("float32"); d["p_unrest"] = expit(eta[n:2 * n]).astype("float32")
    d.to_parquet(os.path.join(a.out, f"fitted_{model}.parquet"), index=False)


def semisynth(a, saved, dfs, covs, base, tr, wk, L, cell, barrier, Lv, T, parts, biv3_blocks, biv3_designs):
    """One replicate of the semi-synthetic check (see --semisynth). The truth is the fitted biv3 at its saved
    hyperparameters: fixed effects and latent effects at their posterior mode on all training rows, the
    own and cross histories recomputed from the simulated outcomes, neighbour terms set to zero."""
    import time
    from scipy.optimize import minimize
    from scipy.special import expit
    t0 = time.time()
    rng = np.random.default_rng(1000 + a.sim_seed)
    if saved is None or "biv3" not in saved.get("models", {}):
        sys.exit("--semisynth needs --fixed-theta with a saved biv3")
    P = {o: parts(o, tr) for o in OUTS}
    y = np.r_[P["conflict"]["y"], P["unrest"]["y"]]
    m = LatentGaussianModel(y, biv3_blocks(), biv3_designs(P["conflict"], P["unrest"]))
    m.theta = theta_from_hyper(m.blocks, saved["models"]["biv3"]["hyper"])
    m.x_hat, _, _ = m.mode(m.theta)
    names = m.blocks[0].names
    ix = {k: names.index(k) for k in ("conflict:lag_own", "conflict:lag_other", "conflict:lag_own_nb", "conflict:lag_other_nb",
                                     "unrest:lag_own", "unrest:lag_other", "unrest:lag_own_nb", "unrest:lag_other_nb")}
    x0 = m.x_hat.copy(); xs = m.slices[0]
    coef = {k: float(m.x_hat[xs][i]) for k, i in ix.items()}
    for i in ix.values():
        x0[xs.start + i] = 0.0
    eta0 = np.asarray(m.A @ x0).ravel()
    n = len(P["conflict"]["y"]); eta0c, eta0u = eta0[:n], eta0[n:]
    kappa = coef["conflict:lag_other"]
    gsd = abs(kappa) if a.sim_gamma_sd < 0 else a.sim_gamma_sd
    gamma = np.zeros(L.n)
    if gsd > 0:
        gb = MaternBlock(L, name="g", barrier=barrier)
        gamma = gb.sample(np.log([500.0, gsd]), rng).ravel()
    # week-by-week simulation on the training rows (same order in both outcomes)
    c_tr = cell[tr]; w_tr = wk[tr]; w0 = w_tr.min(); W = int(w_tr.max() - w0 + 1)
    Yc = np.zeros((L.n, W + 4)); Yu = np.zeros((L.n, W + 4))     # 4 empty weeks of burn-in at the start
    yc = np.zeros(n, int); yu = np.zeros(n, int); hc = np.zeros(n); hu = np.zeros(n)
    order = np.argsort(w_tr, kind="stable")
    bounds_w = np.searchsorted(w_tr[order], np.arange(w0, w0 + W + 1))
    for k in range(W):
        i = order[bounds_w[k]:bounds_w[k + 1]]
        if len(i) == 0:
            continue
        c = c_tr[i]; col = k + 4
        hc[i] = np.log1p(Yc[c, col - 4:col].sum(1)); hu[i] = np.log1p(Yu[c, col - 4:col].sum(1))
        pc = expit(eta0c[i] + coef["conflict:lag_own"] * hc[i] + (kappa + gamma[c]) * hu[i])
        pu = expit(eta0u[i] + coef["unrest:lag_own"] * hu[i] + coef["unrest:lag_other"] * hc[i])
        yc[i] = rng.random(len(i)) < pc; yu[i] = rng.random(len(i)) < pu
        Yc[c, col] = yc[i]; Yu[c, col] = yu[i]
    t_sim = time.time() - t0
    # replace outcomes and histories on the training rows, recompute the neighbour terms
    for o, yy, own, oth in (("conflict", yc, hc, hu), ("unrest", yu, hu, hc)):
        dd = dfs[o]
        # cast to the columns' dtypes (pandas >= 3 refuses to downcast float64 into a float32 column)
        dd.loc[tr, "y"] = yy.astype(dd["y"].dtype); dd.loc[tr, "lag_own"] = own.astype(dd["lag_own"].dtype)
        dd.loc[tr, "lag_other"] = oth.astype(dd["lag_other"].dtype)
        for cc in ("lag_own", "lag_other"):
            v = neighbour_mean(dd, L, cc)
            mu, sd = v[dd.train].mean(), v[dd.train].std()
            dd[cc + "_nb"] = (v - mu) / (sd if sd > 0 else 1)
    P = {o: parts(o, tr) for o in OUTS}
    rng_s = np.random.default_rng(a.zero_seed + a.sim_seed)
    S = {o: subsample(P[o], a.zero_frac or 1.0, rng_s) for o in OUTS}

    def blocks():
        return biv3_blocks() + [
            MaternBlock(L, name="kappa_field_conflict", range_init=800.0, sd_init=0.05, pc_sd=(0.2, 0.05), barrier=barrier),
            MaternBlock(L, name="kappa_field_unrest", range_init=800.0, sd_init=0.05, pc_sd=(0.2, 0.05), barrier=barrier)]

    def designs(Pc, Pu):
        return biv3_designs(Pc, Pu) + [stack(Pc["K"], None, nu=len(Pu["y"])), stack(None, Pu["K"], nc=len(Pc["y"]))]

    ms = LatentGaussianModel(np.r_[S["conflict"]["y"], S["unrest"]["y"]], blocks(), designs(S["conflict"], S["unrest"]),
                             offset=np.r_[S["conflict"]["off"], S["unrest"]["off"]] if a.zero_frac else None)
    th = theta0_from_init(ms.blocks, saved["models"]["biv3"]["hyper"])
    nb = sum(b.n_theta for b in ms.blocks[:-2]); free = np.arange(nb, ms.n_theta)
    bds = [bd for b in ms.blocks for bd in b.bounds()]

    def nlp(z):
        t = th.copy(); t[free] = z
        try:
            v = -(ms.log_marginal(t, require_converged=True) + ms.log_prior(t))
        except Exception:
            return 1e100
        return v if np.isfinite(v) else 1e100
    opt = minimize(nlp, th[free], method="L-BFGS-B", bounds=[bds[i] for i in free], options=dict(eps=1e-4, maxiter=60))
    th[free] = opt.x
    t_fit = time.time() - t0 - t_sim
    # posterior of kappa(s) on all training rows at the fitted hyperparameters
    mf = LatentGaussianModel(np.r_[P["conflict"]["y"], P["unrest"]["y"]], blocks(), designs(P["conflict"], P["unrest"]))
    mf.theta = th
    mf.x_hat, mf.f_hat, mf.lu = mf.mode(th)   # keep the factorization: sample() solves with it
    kb = [b.name for b in mf.blocks].index("kappa_field_conflict")
    k0 = mf.blocks[0].names.index("conflict:lag_other")
    Sx = mf.sample(100, np.random.default_rng(a.sim_seed))
    draws = Sx[mf.slices[0]][k0][None, :] + Sx[mf.slices[kb]]           # kappa_0 + gamma(s), cells x 100
    hasu = np.zeros(L.n, bool); hasu[c_tr[yu == 1]] = True             # cells with simulated unrest
    km = draws.mean(1)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=1)
    k0hat = float(mf.x_hat[mf.slices[0]][k0])
    q10, q90 = np.quantile(km[hasu], [0.1, 0.9])
    tru = kappa + gamma
    out = {"seed": a.sim_seed, "gamma_sd_true": gsd, "kappa_true": kappa, "region": a.region,
           "kappa0_hat": k0hat, "kappa0_se": float(np.sqrt(np.var(Sx[mf.slices[0]][k0]))),
           "hyper_gamma": {b.name: b.describe(t) for b, t in zip(mf.blocks, mf._split(th)) if b.name.startswith("kappa_field")},
           "spread_q90_q10": float(q90 - q10), "spread_true_q90_q10": float(np.subtract(*np.quantile(tru[hasu], [0.9, 0.1]))),
           "share_cells_interval_excl_kappa0": float(np.mean(((lo > k0hat) | (hi < k0hat))[hasu])),
           "share_cells_sign_opposite": float(np.mean((np.sign(km) != np.sign(k0hat))[hasu])),
           "corr_with_truth": float(np.corrcoef(km[hasu], tru[hasu])[0, 1]) if gsd > 0 else None,
           "rmse": float(np.sqrt(np.mean((km - tru)[hasu] ** 2))), "cover95": float(np.mean(((lo <= tru) & (tru <= hi))[hasu])),
           "cells_with_unrest": int(hasu.sum()), "prev_conflict": float(yc.mean()), "prev_unrest": float(yu.mean()),
           "truth_coef": coef, "opt_nit": int(opt.nit), "opt_success": bool(opt.success),
           "seconds": {"simulate": t_sim, "fit": t_fit, "total": time.time() - t0}}
    print("semisynth", json.dumps(out, default=float), flush=True)
    if a.sim_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.sim_out)), exist_ok=True)
        json.dump(out, open(a.sim_out, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--region", default="west_sahel")
    ap.add_argument("--land", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--st-step", type=int, default=13); ap.add_argument("--st-res", type=float, default=1.0)
    ap.add_argument("--models", default="uni_static,uni_st,biv")   # also: biv2
    ap.add_argument("--summary-name", default="summary_biv.json")
    ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("LON0", "LON1", "LAT0", "LAT1"),
                    help="custom region (overrides --region's box; --region then only names it)")
    ap.add_argument("--validation", action="store_true",
                    help="model selection split: fit on 2018-2021, score 2022; 2023-2024 rows are dropped")
    ap.add_argument("--init-theta", default=None,
                    help="summary JSON whose fitted 'biv' hyperparameters warm-start the fit (e.g. for biv2)")
    ap.add_argument("--fixed-theta", default=None,
                    help="summary json of an earlier run: reuse its hyperparameters for the models it "
                         "contains (no optimisation); other models are fitted")
    ap.add_argument("--zero-frac", type=float, default=None,
                    help="speed-up: estimate the hyperparameters on all training events and a random "
                         "fraction r of the training non-events of each outcome, with offset -log r "
                         "(exact for the logistic likelihood); the latent posterior is then computed "
                         "on all training rows at those hyperparameters")
    ap.add_argument("--zero-seed", type=int, default=0)
    ap.add_argument("--semisynth", action="store_true",
                    help="semi-synthetic check of a spatially varying kappa (needs --fixed-theta with a biv3 summary): "
                         "regenerate both outcomes on the real training rows from the fitted biv3, week by week, with "
                         "kappa_U->C(s) = kappa + gamma(s), refit biv3 + gamma fields with only the gamma hyperparameters "
                         "free (others at the truth), and report the fitted spread of kappa(s)")
    ap.add_argument("--sim-seed", type=int, default=1)
    ap.add_argument("--sim-gamma-sd", type=float, default=0.0,
                    help="SD of the planted gamma field (Matern, 500 km); 0 = constant kappa (null); "
                         "negative = |fitted kappa_U->C|")
    ap.add_argument("--sim-out", default=None, help="JSON file for one replicate's results")
    ap.add_argument("--hessian-only", action="store_true",
                    help="with --fixed-theta: only the hyperparameter intervals (no predictions, no exports)")
    ap.add_argument("--hessian", action="store_true",
                    help="approximate 95%% intervals for the hyperparameters from the Hessian of the "
                         "Laplace log posterior of theta at the estimate (on the subsample with --zero-frac)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw = pd.read_parquet(a.table)
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    x0, x1, y0, y1 = a.box if a.box else REGIONS[a.region]
    raw = region_rows(raw, a.region, (x0, x1, y0, y1), a.land)
    dfs, covs = {}, {}
    for o in OUTS:
        dfo, cov, _ = prepare(raw, o)
        dfs[o], covs[o] = dfo.reset_index(drop=True), cov
    if a.validation:   # 2023-2024 never enter: train 2018-2021, score 2022
        keep = dfs["conflict"]["train"].to_numpy().copy()
        for o in OUTS:
            d_ = dfs[o][keep].reset_index(drop=True)
            d_["train"] = (d_["week_start"] < pd.Timestamp("2022-01-01")).to_numpy()
            # the validation year takes the last training year's effect (as the test years
            # take 2022's in the main fit); year_2022 would otherwise be unidentified
            if "year_2022" in d_:
                v22 = d_["year_2022"].to_numpy() == 1
                if "year_2021" in d_:
                    d_.loc[v22, "year_2021"] = 1
                d_["year_2022"] = 0
            dfs[o] = d_
    for o in OUTS:   # guard: drop fixed effects that are constant zero on the training rows
        tr_o = dfs[o]["train"].to_numpy()
        dead = [c for c in covs[o] if c in dfs[o] and not np.any(dfs[o].loc[tr_o, c].to_numpy() != 0)]
        if dead:
            print(f"{o}: dropping covariates that are zero on all training rows: {dead}", flush=True)
            covs[o] = [c for c in covs[o] if c not in dead]
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
    iso_rows = base["ISO_A3"].astype(str).to_numpy()
    grp_row = np.select([np.isin(iso_rows, MIDDLE_EAST), np.isin(iso_rows, SOUTH_ASIA)], [1, 2], 0)
    present = sorted(set(grp_row.tolist()))
    remap = {g: i for i, g in enumerate(present)}
    grp_row = np.array([remap[g] for g in grp_row])
    grp_names = [GROUPS[g] for g in present]
    gv = pd.DataFrame({"c": cell_v, "g": grp_row}).groupby("c")["g"].agg(lambda x: np.bincount(x).argmax())
    grp_v = np.zeros(Lv.n, int); grp_v[gv.index.to_numpy()] = gv.to_numpy()
    print(f"regions {grp_names}: rows {np.bincount(grp_row).tolist()}, st cells {np.bincount(grp_v, minlength=len(present)).tolist()}",
          flush=True)
    if "biv2_rlk" in a.models:   # region-specific short-run cross effects (fixed): baseline + deviations
        for o in OUTS:
            for gi in range(1, len(grp_names)):
                c = f"lag_other_x_{grp_names[gi]}"
                dfs[o][c] = dfs[o]["lag_other"].to_numpy() * (grp_row == gi)
                covs[o] = covs[o] + [c]
    print(f"{a.region}: {len(base):,} rows per outcome, {observed.sum():,} cells, {nC} countries, "
          f"v on {Lv.n:,} cells x {T} steps", flush=True)

    def parts(o, rows):
        dd = dfs[o][rows]
        X, names = de.fixed_design(dd, covs[o])
        return dict(X=X, names=[f"{o}:{n}" for n in names], C=de.index_design(dd["country_id"], nC),
                    E=de.index_design(dd["cell"], L.n), W=de.index_design(dd["cell"], L.n),
                    V=de.spacetime_design(cell_v[rows], tstep[rows], Lv.n, T), y=dd["y"].to_numpy(),
                    K=de.index_design(dd["cell"], L.n, values=dd["lag_other"].to_numpy(float)))

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

    def biv2_blocks():
        return biv_blocks() + [SpaceTimeAR1Block(Lv, T, name="st_unrest", sd_init=0.3, rho_init=0.8)]

    def biv3_blocks():   # biv2 + a shared STATIC field (persistent shared hotspots do not decay in forecasts)
        return biv2_blocks() + [CopyBlock(MaternBlock(L, name="sfield", range_init=300.0, barrier=barrier),
                                          name="shared_field", lam_init=0.5)]

    def biv2_rl_blocks():   # biv2 with one loading of the shared field per region
        b = biv_blocks()[:-1]
        b.append(RegionCopyBlock(SpaceTimeAR1Block(Lv, T, name="st", sd_init=0.5, rho_init=0.8),
                                 np.tile(grp_v, T), grp_names, name="shared_st", lam_init=0.5))
        return b + [SpaceTimeAR1Block(Lv, T, name="st_unrest", sd_init=0.3, rho_init=0.8)]

    def biv2_rlf_blocks():  # + spatially varying short-run cross effects: kappa(s) = kappa_0 + gamma(s)
        return biv2_rl_blocks() + [
            MaternBlock(L, name="kappa_field_conflict", range_init=800.0, sd_init=0.05, pc_sd=(0.2, 0.05), barrier=barrier),
            MaternBlock(L, name="kappa_field_unrest", range_init=800.0, sd_init=0.05, pc_sd=(0.2, 0.05), barrier=barrier)]

    def biv2_rlf_designs(Pc, Pu):
        return biv2_designs(Pc, Pu) + [stack(Pc["K"], None, nu=len(Pu["y"])), stack(None, Pu["K"], nc=len(Pc["y"]))]

    def biv3_nolam_blocks():   # channels 1 + 2 only: biv3 with lambda = 0 (no shared space-time field)
        b = biv_blocks()[:-1]
        return b + [SpaceTimeAR1Block(Lv, T, name="st_conflict", sd_init=0.5, rho_init=0.8),
                    SpaceTimeAR1Block(Lv, T, name="st_unrest", sd_init=0.3, rho_init=0.8),
                    CopyBlock(MaternBlock(L, name="sfield", range_init=300.0, barrier=barrier),
                              name="shared_field", lam_init=0.5)]

    def biv3_nolam_designs(Pc, Pu):
        D = biv_designs(Pc, Pu)[:-1]
        nc, nu = len(Pc["y"]), len(Pu["y"])
        return D + [stack(Pc["V"], None, nu=nu), stack(None, Pu["V"], nc=nc),
                    sp.bmat([[Pc["W"], None], [None, Pu["W"]]], format="csr")]

    def biv3_designs(Pc, Pu):
        return biv2_designs(Pc, Pu) + [sp.bmat([[Pc["W"], None], [None, Pu["W"]]], format="csr")]

    def biv2_designs(Pc, Pu):
        return biv_designs(Pc, Pu) + [stack(None, Pu["V"], nc=len(Pc["y"]))]

    def biv_designs(Pc, Pu):
        Xc, Xu = sp.csr_matrix(Pc["X"]), sp.csr_matrix(Pu["X"])
        X = sp.bmat([[Xc, None], [None, Xu]], format="csr")
        nc, nu = len(Pc["y"]), len(Pu["y"])
        return [X, stack(Pc["C"], None, nu=nu), stack(None, Pu["C"], nc=nc),
                stack(Pc["E"], None, nu=nu), stack(None, Pu["E"], nc=nc),
                stack(Pc["W"], None, nu=nu), stack(None, Pu["W"], nc=nc),
                sp.bmat([[Pc["V"], None], [None, Pu["V"]]], format="csr")]

    saved = json.load(open(a.fixed_theta)) if a.fixed_theta else None
    init = json.load(open(a.init_theta)) if a.init_theta else None
    parts_tr = {o: parts(o, tr) for o in OUTS}   # before --semisynth: the *_blocks() closures read parts_tr[o]["names"]
    if a.semisynth:
        semisynth(a, saved, dfs, covs, base, tr, wk, L, cell, barrier, Lv, T, parts, biv3_blocks, biv3_designs)
        return
    parts_te = {o: parts(o, te) for o in OUTS}
    if a.zero_frac:
        rng_s = np.random.default_rng(a.zero_seed)
        parts_sub = {o: subsample(parts_tr[o], a.zero_frac, rng_s) for o in OUTS}
    res = {"region": a.region, "rows_per_outcome": int(len(base)),
           "box": [x0, x1, y0, y1], "validation": bool(a.validation), "cells": int(observed.sum()), "countries": nC, "st_res_deg": r, "st_step_weeks": a.st_step,
           "st_cells": int(Lv.n), "time_steps": T, "zero_frac": a.zero_frac, "models": {}}
    preds = {"grid_id": base.loc[te, "grid_id"].to_numpy(), "week_start": base.loc[te, "week_start"].to_numpy(),
             "iso": base.loc[te, "ISO_A3"].astype(str).to_numpy(),
             "lon": base.loc[te, "lon"].to_numpy(), "lat": base.loc[te, "lat"].to_numpy(),
             "y_conflict": parts_te["conflict"]["y"].astype("int8"), "y_unrest": parts_te["unrest"]["y"].astype("int8"),
             "events_52w_conflict": np.nan * np.zeros(te.sum())}
    keys = ["conflict:lag_own", "conflict:lag_other", "conflict:lag_own_nb", "conflict:lag_other_nb",
            "unrest:lag_own", "unrest:lag_other", "unrest:lag_own_nb", "unrest:lag_other_nb"]
    for model in a.models.split(","):
        if model in ("uni_static", "uni_st", "uni_static_u", "uni_st_u"):
            st = model in ("uni_st", "uni_st_u")
            o = "unrest" if model.endswith("_u") else "conflict"
            P = parts_tr[o]
            if a.zero_frac:
                Ps = parts_sub[o]
                m = fit_subsampled(lambda: uni_blocks(o, st), Ps["y"], uni_designs(Ps, st), Ps["off"],
                                   P["y"], uni_designs(P, st), model, saved, None)
            else:
                m = fit_or_condition(LatentGaussianModel(P["y"], uni_blocks(o, st), uni_designs(P, st)), saved, model)
            Dt = uni_designs(parts_te[o], st)
            eta, sd = m.predict_eta(Dt, None, n_samples=50)
            preds[f"p_{o}_{model.replace('_u', '')}_probit"] = prob_from_eta(eta, sd).astype("float32")
            preds[f"p_{o}_{model.replace('_u', '')}"] = mc_prob(m, Dt).astype("float32")
        else:   # biv: shared v only; biv2: shared v + unrest-specific space-time field
            bl, dz = {"biv": (biv_blocks, biv_designs), "biv2": (biv2_blocks, biv2_designs),
                      "biv3": (biv3_blocks, biv3_designs), "biv3_nolam": (biv3_nolam_blocks, biv3_nolam_designs),
                      "biv2_rl": (biv2_rl_blocks, biv2_designs),
                      "biv2_rlk": (biv2_rl_blocks, biv2_designs), "biv2_rlf": (biv2_rlf_blocks, biv2_rlf_designs)}[model]
            y = np.r_[parts_tr["conflict"]["y"], parts_tr["unrest"]["y"]]
            if a.zero_frac:
                Sc, Su = parts_sub["conflict"], parts_sub["unrest"]
                m = fit_subsampled(bl, np.r_[Sc["y"], Su["y"]], dz(Sc, Su), np.r_[Sc["off"], Su["off"]],
                                   y, dz(parts_tr["conflict"], parts_tr["unrest"]), model, saved, init)
            else:
                m = fit_or_condition(LatentGaussianModel(y, bl(), dz(parts_tr["conflict"], parts_tr["unrest"])), saved, model, init)
            if a.hessian_only:
                import time
                t0 = time.time()
                mh = getattr(m, "sub_model", m)
                H, se = mh.hessian_theta()
                lo_, hi_, k_ = {}, {}, 0
                for b, t in zip(mh.blocks, mh._split(mh.theta)):
                    n_ = len(t)
                    lo_.update(b.describe(t - 1.96 * se[k_:k_ + n_])); hi_.update(b.describe(t + 1.96 * se[k_:k_ + n_]))
                    k_ += n_
                res["models"][model] = {"hyper": m.hyper(), "hyper_ci95": {nm: [lo_[nm], hi_[nm]] for nm in lo_},
                                        "theta_se": list(map(float, se)), "hessian_eig_min": float(np.linalg.eigvalsh(H).min()),
                                        "hessian_rows": int(len(mh.y)), "hessian_seconds": time.time() - t0,
                                        "lag_coefficients": {k: list(map(float, v)) for k, v in m.fixed_summary().items() if k in keys}}
                print(model, json.dumps(res["models"][model], default=float), flush=True)
                json.dump(res, open(os.path.join(a.out, a.summary_name), "w"), indent=1, default=float)
                continue
            Dte = dz(parts_te["conflict"], parts_te["unrest"])
            eta, sd = m.predict_eta(Dte, None, n_samples=50)
            p = prob_from_eta(eta, sd); nte = int(te.sum())
            preds[f"p_conflict_{model}_probit"] = p[:nte].astype("float32")
            preds[f"p_unrest_{model}_probit"] = p[nte:].astype("float32")
            je = joint_eval(m, Dte, parts_te["conflict"]["y"], parts_te["unrest"]["y"], seed=1)
            preds[f"p_conflict_{model}"] = je["pc"].astype("float32"); preds[f"p_unrest_{model}"] = je["pu"].astype("float32")
            for k in ("logj", "logi", "pc_given", "pboth"):
                preds[f"{k}_{model}"] = je[k].astype("float32")
        fs = m.fixed_summary()
        res["models"][model] = {"hyper": m.hyper(), "log_ml": float(m.log_ml), "seconds": float(m.fit_seconds),
                                "hyper_source": "fixed from " + a.fixed_theta
                                if saved is not None and model in saved.get("models", {}) else "fitted",
                                "lag_coefficients": {k: list(map(float, fs[k])) for k in keys if k in fs},
                                "fixed": {k: list(map(float, v)) for k, v in fs.items()}}
        if a.hessian and model in JOINT:
            import time
            t0 = time.time()
            mh = getattr(m, "sub_model", m)
            H, se = mh.hessian_theta()
            lo_, hi_, k_ = {}, {}, 0
            for b, t in zip(mh.blocks, mh._split(mh.theta)):
                n_ = len(t)
                lo_.update(b.describe(t - 1.96 * se[k_:k_ + n_])); hi_.update(b.describe(t + 1.96 * se[k_:k_ + n_]))
                k_ += n_
            res["models"][model]["hyper_ci95"] = {nm: [lo_[nm], hi_[nm]] for nm in lo_}
            res["models"][model]["theta_se"] = list(map(float, se))
            res["models"][model]["hessian_seconds"] = time.time() - t0
            print(model, "hyper_ci95", json.dumps(res["models"][model]["hyper_ci95"]), flush=True)
        sc = {}
        if f"p_conflict_{model}" in preds:
            sc["conflict"] = dg.scores(parts_te["conflict"]["y"], preds[f"p_conflict_{model}"])
        mu_name = model.replace("_u", "")
        if model.endswith("_u") and f"p_unrest_{mu_name}" in preds:
            sc["unrest"] = dg.scores(parts_te["unrest"]["y"], preds[f"p_unrest_{mu_name}"])
        if model in JOINT:
            sc["unrest"] = dg.scores(parts_te["unrest"]["y"], preds[f"p_unrest_{model}"])
        res["models"][model]["test_scores"] = sc
        if hasattr(m, "sub_info"):
            res["models"][model]["subsample"] = dict(m.sub_info, zero_frac=a.zero_frac)
        if model in JOINT:
            export_fields(m, model, L, observed, Lv, T, a, base["week_start"].min())
            export_fitted(m, model, base, tr, tstep, a)
        print(model, json.dumps({k: res["models"][model][k] for k in ("hyper", "lag_coefficients", "test_scores")},
                                default=float), flush=True)
        json.dump(res, open(os.path.join(a.out, a.summary_name), "w"), indent=1, default=float)
    if a.hessian_only:
        return
    # joint and conditional prediction (Section 5.x): per cell-week, both outcomes together
    yc, yu = parts_te["conflict"]["y"], parts_te["unrest"]["y"]
    cell_te = base.loc[te, "grid_id"].to_numpy(); ctry_te = base.loc[te, "ISO_A3"].astype(str).to_numpy()
    from sklearn.metrics import average_precision_score as aps
    J = {"n_cell_weeks": int(len(yc)), "share_both": float(np.mean(yc * yu)),
         "share_conflict_given_unrest": float(yc[yu == 1].mean()) if yu.sum() else None,
         "share_conflict_given_no_unrest": float(yc[yu == 0].mean())}
    def jls(logp):   # mean joint log score (negative log-likelihood per cell-week)
        return float(-np.mean(logp))
    for uni in ("uni_static", "uni_st"):
        if f"p_conflict_{uni}" in preds and f"p_unrest_{uni}" in preds:
            preds[f"logi_{uni}"] = (_logb(yc, preds[f"p_conflict_{uni}"]) + _logb(yu, preds[f"p_unrest_{uni}"])).astype("float32")
            J[f"joint_log_score_{uni}_product"] = jls(preds[f"logi_{uni}"])
    for bm in ("uni_static", "uni_st"):    # calibration of co-occurrence (product of the univariate forecasts)
        if f"p_conflict_{bm}" in preds and f"p_unrest_{bm}" in preds:
            pc, pu = preds[f"p_conflict_{bm}"].astype(float), preds[f"p_unrest_{bm}"].astype(float)
            J[f"{bm}_calibration"] = calibration(yc, yu, pc, pu, pc * pu)
    for bm in JOINT:
        if f"logj_{bm}" in preds:
            J[f"{bm}_calibration"] = calibration(yc, yu, preds[f"p_conflict_{bm}"].astype(float),
                                                 preds[f"p_unrest_{bm}"].astype(float), preds[f"pboth_{bm}"].astype(float))
    for bm in JOINT:
        if f"logj_{bm}" not in preds:
            continue
        lj, li = preds[f"logj_{bm}"].astype(float), preds[f"logi_{bm}"].astype(float)
        J[f"joint_log_score_{bm}"] = jls(lj)
        J[f"joint_log_score_{bm}_independent"] = jls(li)
        mdiff, ses, ns = dg.clustered_diff(-lj, -li, cell_te, ctry_te)
        J[f"{bm}_joint_minus_independent"] = {"mean": mdiff, "se_cell": ses[0], "se_country": ses[1]}
        for uni in ("uni_static", "uni_st"):
            if f"logi_{uni}" in preds:
                mdiff, ses, ns = dg.clustered_diff(-lj, -preds[f"logi_{uni}"].astype(float), cell_te, ctry_te)
                J[f"{bm}_joint_minus_{uni}_product"] = {"mean": mdiff, "se_cell": ses[0], "se_country": ses[1]}
        pcg = preds[f"pc_given_{bm}"].astype(float); pc = preds[f"p_conflict_{bm}"].astype(float)
        J[f"{bm}_conditional"] = {
            "log_score_conflict_given_unrest": float(-np.mean(_logb(yc, pcg))),
            "log_score_conflict_marginal": float(-np.mean(_logb(yc, pc))),
            "auc_pr_conflict_given_unrest": float(aps(yc, pcg)), "auc_pr_conflict_marginal": float(aps(yc, pc))}
        pb = preds[f"pboth_{bm}"].astype(float)
        J[f"{bm}_cooccurrence"] = {"auc_pr_joint": float(aps(yc * yu, pb)),
                                   "auc_pr_independent": float(aps(yc * yu, pc * preds[f"p_unrest_{bm}"].astype(float)))}
    res["joint_prediction"] = J
    print("joint_prediction", json.dumps(J, default=float), flush=True)
    json.dump(res, open(os.path.join(a.out, a.summary_name), "w"), indent=1, default=float)
    pd.DataFrame(preds).drop(columns=["events_52w_conflict"]).to_parquet(
        os.path.join(a.out, a.summary_name.replace("summary", "test_pred").replace(".json", ".parquet")), index=False)


if __name__ == "__main__":
    main()
