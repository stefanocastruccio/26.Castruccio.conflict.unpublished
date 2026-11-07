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

CKPT = os.environ.get("S5_CKPT", "s5v2_ckpt.jsonl")
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


def fit_uni(L, d, kind, T):
    n = L.n
    X = np.c_[np.ones(len(d)), d["Lc"], d["Lu"]]
    blocks, D = [FixedBlock(3, ["int", "Lc", "Lu"])], [X]
    if kind in ("uni_field", "uni_st"):
        blocks.append(MaternBlock(L, name="field")); D.append(de.index_design(d["cell"], n))
    if kind == "uni_st":
        blocks.append(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8))
        D.append(de.spacetime_design(d["cell"], d["tstep"], n, T))
    return LatentGaussianModel(d["yc"].to_numpy(), blocks, D).fit(), blocks


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
    yte = d_te["yc"].to_numpy(); onset = (d_te["ev52_c"] == 0).to_numpy()
    for kind in ["glm", "uni_field", "uni_st", "biv_st", "biv_st2"]:
        if kind in ("biv_st", "biv_st2"):
            m, _ = (fit_biv if kind == "biv_st" else fit_biv2)(L, d_tr, T)
            n_te = len(d_te)
            Dte = (biv_designs if kind == "biv_st" else biv2_designs)(L, d_te, T)
            Dte = [D[:n_te] for D in Dte]                       # conflict rows only
        else:
            m, _ = fit_uni(L, d_tr, kind, T)
            Dte = [np.c_[np.ones(len(d_te)), d_te["Lc"], d_te["Lu"]]]
            if kind in ("uni_field", "uni_st"):
                Dte.append(de.index_design(d_te["cell"], L.n))
            if kind == "uni_st":
                Dte.append(de.spacetime_design(d_te["cell"], d_te["tstep"], L.n, T))
        fs = m.fixed_summary()
        est, se = fs["Lu"]
        eta, sd = m.predict_eta(Dte, None, n_samples=50)
        p = prob_from_eta(eta, sd)
        sc = dg.scores(yte, p)
        r = dict(scenario=scen, seed=seed, model=kind, truth=kappa, est=float(est), se=float(se),
                 d_c=float(fs["Lc"][0]), log_ml=float(m.log_ml), seconds=float(m.fit_seconds),
                 log_score=sc["log_score"], auc_pr=sc["auc_pr"],
                 onset_auc_pr=float(average_precision_score(yte[onset], p[onset])) if yte[onset].sum() else np.nan,
                 prevalence_c=float(df["yc"].mean()), prevalence_u=float(df["yu"].mean()))
        if kind in ("biv_st", "biv_st2"):
            h = m.hyper()
            r["loading"] = h.get("shared_st.loading"); r["st_rho"] = h.get("shared_st.rho")
            r["st_sd"] = h.get("shared_st.sd"); r["st_range"] = h.get("shared_st.range_km")
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
