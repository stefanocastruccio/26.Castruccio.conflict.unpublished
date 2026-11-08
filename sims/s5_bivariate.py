"""S5 -- Does unrest escalate into violence, or do the two share the same episodes?

Two binary outcomes per cell-week, conflict (c) and unrest (u), generated jointly:

  logit P(c_st) = b_c + d_c Lc_st + k Lu_st + s(s) + w_c(s) + e(s,t)
  logit P(u_st) = b_u + d_u Lu_st         + l_s s(s) + w_u(s) + l e(s,t)

where L = log(1 + events of that outcome in the previous 4 weeks), s is a static field
shared by both outcomes (sd 0.6, range 400 km), w_c and w_u are outcome-specific static
fields, and e is a SHARED space-time episode field: AR(1) in time (0.95 per week) of
Matern innovations (range 300 km, marginal sd 0.8). Every field is drawn from the exact
continuous Matern covariance. The escalation effect is k (unrest history -> conflict):
  no_escalation : k = 0      escalation : k = 0.4
Shared episodes raise both outcomes at once, so recent unrest predicts conflict even
when k = 0.

Estimators of k (and d_c), fitted on the training weeks:
  glm       conflict ~ Lc + Lu
  uni_field + static Matern field
  uni_st    + static field + space-time AR(1) x Matern field (4-week steps)
  biv_st    joint model of both outcomes: outcome-specific intercepts and lags, outcome-
            specific static fields, and a SHARED space-time field entering unrest with
            loading l (CopyBlock); also reports the estimated loading.
Also: one-quarter-ahead forecasts of conflict on the test weeks (scores, onset rows).

Domain: 16 x 14 cells of 0.5 degrees (224 cells), 104 training + 26 test weeks.
"""
from common import *  # noqa: F401,F403

import scipy.sparse as sp
from scipy.special import expit
from sklearn.metrics import average_precision_score

from sc import diagnostics as dg
from sc.gmrf import CopyBlock, SphereLattice
from sc.laplace import prob_from_eta
from sc.simulate import matern_gp

CKPT = os.environ.get("S5_CKPT", "s5v4_ckpt.jsonl")
N_REPS = int(os.environ.get("N_REPS", 20))
# (kappa, sd of an UNREST-ONLY space-time episode field). The *_uST scenarios add unrest
# episodes that do not touch conflict: a joint model with a single shared space-time field
# must then absorb them into v, which can bias kappa towards zero (review round 3).
SCEN = {"no_escalation": (0.0, 0.0), "escalation": (0.4, 0.0),
        "no_escalation_uST": (0.0, 0.6), "escalation_uST": (0.4, 0.6)}
NX, NY, BURN, T_TR, T_TE, STEP = 16, 14, 12, 104, 26, 4
PAR = dict(b_c=-3.9, b_u=-3.6, d_c=0.3, d_u=0.3, l_s=0.8, l=0.8, phi=0.95, e_sd=0.8)


def lattice():
    return SphereLattice(10.25 + 0.5 * np.arange(NX), -1.75 + 0.5 * np.arange(NY))


def simulate5(kappa, seed, u_sd=0.0):
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    s = matern_gp(L, 400.0, 0.6, rng).ravel()
    wc = matern_gp(L, 300.0, 0.5, rng).ravel()
    wu = matern_gp(L, 150.0, 0.5, rng).ravel()
    Ttot = BURN + T_TR + T_TE
    e = np.zeros((Ttot, n)); e[0] = matern_gp(L, 300.0, PAR["e_sd"], rng).ravel()
    for t in range(1, Ttot):
        e[t] = PAR["phi"] * e[t - 1] + np.sqrt(1 - PAR["phi"] ** 2) * matern_gp(L, 300.0, PAR["e_sd"], rng).ravel()
    eu = np.zeros((Ttot, n))
    if u_sd > 0:   # unrest-only episodes: AR(1) (same phi) of Matern innovations, range 200 km
        rng_u = np.random.default_rng(seed + 777)
        eu[0] = matern_gp(L, 200.0, u_sd, rng_u).ravel()
        for t in range(1, Ttot):
            eu[t] = PAR["phi"] * eu[t - 1] + np.sqrt(1 - PAR["phi"] ** 2) * matern_gp(L, 200.0, u_sd, rng_u).ravel()
    yc = np.zeros((Ttot, n), int); yu = np.zeros((Ttot, n), int)
    for t in range(Ttot):
        lc = np.log1p(yc[max(0, t - 4):t].sum(0)); lu = np.log1p(yu[max(0, t - 4):t].sum(0))
        pc = expit(PAR["b_c"] + PAR["d_c"] * lc + kappa * lu + s + wc + e[t])
        pu = expit(PAR["b_u"] + PAR["d_u"] * lu + PAR["l_s"] * s + wu + PAR["l"] * e[t] + eu[t])
        yc[t] = rng.random(n) < pc; yu[t] = rng.random(n) < pu
    rows = []
    for t in range(BURN + 4, Ttot):
        lc = np.log1p(yc[t - 4:t].sum(0)); lu = np.log1p(yu[t - 4:t].sum(0))
        ev52 = yc[max(0, t - 52):t].sum(0)
        rows.append(pd.DataFrame({"cell": np.arange(n), "week": t - BURN, "yc": yc[t], "yu": yu[t],
                                  "Lc": lc, "Lu": lu, "ev52_c": ev52}))
    df = pd.concat(rows, ignore_index=True)
    df["train"] = df["week"] < T_TR
    df["tstep"] = df["week"] // STEP
    return L, df


def fit_uni(L, d, kind, T, outcome="c"):
    """Univariate model for conflict (outcome='c') or unrest ('u'); the fixed effects are
    always named Lc and Lu (own and other history)."""
    n = L.n
    X = np.c_[np.ones(len(d)), d["Lc"], d["Lu"]]
    blocks, D = [FixedBlock(3, ["int", "Lc", "Lu"])], [X]
    if kind in ("uni_field", "uni_st"):
        blocks.append(MaternBlock(L, name="field")); D.append(de.index_design(d["cell"], n))
    if kind == "uni_st":
        blocks.append(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8))
        D.append(de.spacetime_design(d["cell"], d["tstep"], n, T))
    return LatentGaussianModel(d["y" + outcome].to_numpy(), blocks, D).fit(), blocks


def uni_test_designs(L, d, kind, T):
    D = [np.c_[np.ones(len(d)), d["Lc"], d["Lu"]]]
    if kind in ("uni_field", "uni_st"):
        D.append(de.index_design(d["cell"], L.n))
    if kind == "uni_st":
        D.append(de.spacetime_design(d["cell"], d["tstep"], L.n, T))
    return D


def _logb(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.where(y == 1, np.log(p), np.log1p(-p))


def mc_prob(m, D, n_samples=200, seed=0):
    """E[expit(eta)] by Monte Carlo over posterior latent draws: the same approximation as
    joint_eval, so univariate and joint models are scored alike."""
    A = sp.hstack([sp.csr_matrix(x) for x in D], format="csr")
    S = m.sample(n_samples, np.random.default_rng(seed))
    return (1 / (1 + np.exp(-(A @ S)))).mean(1)


def joint_eval(m, Dte_full, yc, yu, n_samples=200, seed=0):
    """Joint/conditional predictive quantities from posterior samples (both outcomes share
    each latent draw). Dte_full: designs of the stacked test rows (conflict, then unrest)."""
    A = sp.hstack([sp.csr_matrix(D) for D in Dte_full], format="csr")
    n = len(yc)
    S = m.sample(n_samples, np.random.default_rng(seed))
    pc = 1 / (1 + np.exp(-(A[:n] @ S))); pu = 1 / (1 + np.exp(-(A[n:] @ S)))
    bc = np.where(yc[:, None] == 1, pc, 1 - pc); bu = np.where(yu[:, None] == 1, pu, 1 - pu)
    mc, mu = pc.mean(1), pu.mean(1)
    return dict(logj=np.log(np.clip((bc * bu).mean(1), 1e-300, None)),
                logi=_logb(yc, mc) + _logb(yu, mu), pc=mc, pu=mu,
                pc_given=(pc * bu).mean(1) / np.clip(bu.mean(1), 1e-300, None),
                pboth=(pc * pu).mean(1))


def biv_designs(L, d, T):
    """Stacked rows: conflict rows first, then unrest rows."""
    n, m = L.n, len(d)
    z = np.zeros(m); one = np.ones(m)
    X = np.r_[np.c_[one, d["Lc"], d["Lu"], z, z], np.c_[z, z, z, one, d["Lu"]]]
    Dw = de.index_design(d["cell"], n)
    Zw = sp.csr_matrix((m, n))
    Dst = de.spacetime_design(d["cell"], d["tstep"], n, T)
    Zst = sp.csr_matrix((m, n * T))
    return [X, sp.vstack([Dw, Zw]), sp.vstack([Zw, Dw]),
            sp.bmat([[Dst, Zst], [Zst, Dst]], format="csr")]


def fit_biv2(L, d, T):
    """Joint model with the shared space-time field AND an unrest-specific space-time field."""
    blocks = [FixedBlock(5, ["int_c", "Lc", "Lu", "int_u", "Lu_u"]),
              MaternBlock(L, name="field_c"), MaternBlock(L, name="field_u"),
              CopyBlock(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8), name="shared_st"),
              SpaceTimeAR1Block(L, T, name="st_u", sd_init=0.3, rho_init=0.8)]
    y = np.r_[d["yc"].to_numpy(), d["yu"].to_numpy()]
    return LatentGaussianModel(y, blocks, biv2_designs(L, d, T)).fit(), blocks


def biv2_designs(L, d, T):
    D = biv_designs(L, d, T)
    m = len(d)
    Dst = de.spacetime_design(d["cell"], d["tstep"], L.n, T)
    return D + [sp.vstack([sp.csr_matrix((m, L.n * T)), Dst], format="csr")]


def fit_biv(L, d, T):
    blocks = [FixedBlock(5, ["int_c", "Lc", "Lu", "int_u", "Lu_u"]),
              MaternBlock(L, name="field_c"), MaternBlock(L, name="field_u"),
              CopyBlock(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8), name="shared_st")]
    y = np.r_[d["yc"].to_numpy(), d["yu"].to_numpy()]
    return LatentGaussianModel(y, blocks, biv_designs(L, d, T)).fit(), blocks


def one(arg):
    scen, seed = arg
    kappa, u_sd = SCEN[scen]
    L, df = simulate5(kappa, seed, u_sd)
    T = int(df["tstep"].max() + 1)
    d_tr, d_te = df[df.train], df[~df.train]
    out = []
    yte = d_te["yc"].to_numpy(); yu_te = d_te["yu"].to_numpy(); onset = (d_te["ev52_c"] == 0).to_numpy()
    both = yte * yu_te
    # univariate unrest model (for the product of univariate predictions)
    mu_uni, _ = fit_uni(L, d_tr, "uni_st", T, outcome="u")
    pu_uni = mc_prob(mu_uni, uni_test_designs(L, d_te, "uni_st", T), seed=seed)
    pc_uni = None
    for kind in ["glm", "uni_field", "uni_st", "biv_st", "biv_st2"]:
        if kind in ("biv_st", "biv_st2"):
            m, _ = (fit_biv if kind == "biv_st" else fit_biv2)(L, d_tr, T)
            n_te = len(d_te)
            Dfull = (biv_designs if kind == "biv_st" else biv2_designs)(L, d_te, T)
            Dte = [D[:n_te] for D in Dfull]                     # conflict rows only
        else:
            m, _ = fit_uni(L, d_tr, kind, T)
            Dte = uni_test_designs(L, d_te, kind, T)
        fs = m.fixed_summary()
        est, se = fs["Lu"]
        p = mc_prob(m, Dte, seed=seed)   # same Monte Carlo approximation for every model
        sc = dg.scores(yte, p)
        r = dict(scenario=scen, seed=seed, model=kind, truth=kappa, est=float(est), se=float(se),
                 d_c=float(fs["Lc"][0]), log_ml=float(m.log_ml), seconds=float(m.fit_seconds),
                 log_score=sc["log_score"], auc_pr=sc["auc_pr"],
                 onset_auc_pr=float(average_precision_score(yte[onset], p[onset])) if yte[onset].sum() else np.nan,
                 prevalence_c=float(df["yc"].mean()), prevalence_u=float(df["yu"].mean()))
        if kind == "uni_st":
            pc_uni = p
            # joint prediction by two independent univariate space-time models
            r["joint_log_score"] = float(-np.mean(_logb(yte, p) + _logb(yu_te, pu_uni)))
            r["cond_log_score"] = float(-np.mean(_logb(yte, p)))      # no conditioning possible
            r["cond_auc_pr"] = float(average_precision_score(yte, p))
            r["both_auc_pr"] = float(average_precision_score(both, p * pu_uni)) if both.sum() else np.nan
        if kind in ("biv_st", "biv_st2"):
            h = m.hyper()
            r["loading"] = h.get("shared_st.loading"); r["st_rho"] = h.get("shared_st.rho")
            r["st_sd"] = h.get("shared_st.sd"); r["st_range"] = h.get("shared_st.range_km")
            je = joint_eval(m, Dfull, yte, yu_te, seed=seed)
            r["joint_log_score"] = float(-np.mean(je["logj"]))
            r["joint_log_score_indep"] = float(-np.mean(je["logi"]))
            r["cond_log_score"] = float(-np.mean(_logb(yte, je["pc_given"])))
            r["cond_auc_pr"] = float(average_precision_score(yte, je["pc_given"]))
            r["both_auc_pr"] = float(average_precision_score(both, je["pboth"])) if both.sum() else np.nan
            r["both_auc_pr_indep"] = (float(average_precision_score(both, je["pc"] * je["pu"]))
                                      if both.sum() else np.nan)
        out.append(r)
        print(scen, seed, kind, round(est, 3), round(se, 3), round(m.fit_seconds, 1), flush=True)
    return out


if __name__ == "__main__":
    args = [(s, seed) for s in SCEN for seed in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    if os.environ.get("SHARD"):
        sys.exit(0)
    res.to_csv(os.path.join(RESULTS, "s5_raw.csv"), index=False)
    res["covers"] = (res.est - 1.96 * res.se <= res.truth) & (res.truth <= res.est + 1.96 * res.se)
    res["rejects"] = (res.est - 1.96 * res.se > 0) | (res.est + 1.96 * res.se < 0)
    tab = (res.groupby(["scenario", "model"])
           .agg(truth=("truth", "first"), mean_est=("est", "mean"), emp_sd=("est", "std"),
                mean_se=("se", "mean"), coverage95=("covers", "mean"), reject_zero=("rejects", "mean"),
                d_c=("d_c", "mean"), log_score=("log_score", "mean"), auc_pr=("auc_pr", "mean"),
                onset_auc_pr=("onset_auc_pr", "mean"), loading=("loading", "mean"),
                seconds=("seconds", "mean"), n=("est", "size")).reset_index())
    tab["mc_se_coverage"] = np.sqrt(tab.coverage95 * (1 - tab.coverage95) / tab.n)
    tab.to_csv(os.path.join(RESULTS, "s5_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    # joint and conditional prediction: paired (by replicate) differences against two
    # independent univariate space-time models (uni_st for conflict x uni_st for unrest)
    cols = ["joint_log_score", "joint_log_score_indep", "cond_log_score", "cond_auc_pr",
            "both_auc_pr", "both_auc_pr_indep"]
    pr = res[res.model.isin(["uni_st", "biv_st", "biv_st2"])].reindex(columns=["scenario", "seed", "model"] + cols)
    ref = pr[pr.model == "uni_st"].set_index(["scenario", "seed"])
    rows = []
    for (sc_, mdl), g in pr.groupby(["scenario", "model"]):
        g = g.set_index(["scenario", "seed"])
        row = {"scenario": sc_, "model": mdl, "n": len(g)}
        for c in cols:
            row[c] = g[c].mean()
        for c in ["joint_log_score", "cond_log_score", "cond_auc_pr", "both_auc_pr"]:
            dd = (g[c] - ref.loc[g.index, c]).dropna()
            row[f"{c}_minus_uni"] = dd.mean(); row[f"{c}_minus_uni_se"] = dd.std() / np.sqrt(max(len(dd), 1))
        if mdl != "uni_st":
            dd = (g["joint_log_score"] - g["joint_log_score_indep"]).dropna()
            row["joint_minus_indep"] = dd.mean(); row["joint_minus_indep_se"] = dd.std() / np.sqrt(max(len(dd), 1))
        rows.append(row)
    pt = pd.DataFrame(rows)
    pt.to_csv(os.path.join(RESULTS, "s5_prediction.csv"), index=False)
    print(pt.round(4).to_string(index=False))
