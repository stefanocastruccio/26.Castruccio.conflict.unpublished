"""S9 -- Toy check of the conflict split: can a model with three binary outcomes (repression R,
battles B, unrest U) recover escalation effects psi that differ by type of conflict, and does the
two-outcome model (conflict = R or B) hide them when they have opposite signs?

  logit P(R_st) = b_R + d h_R + (kappa_R + psi_R' z_st) h_U + beta' x + w_R(s) + lam_R e(s,t)
  logit P(B_st) = b_B + d h_B + (kappa_B + psi_B' z_st) h_U + beta' x + w_B(s) +       e(s,t)
  logit P(U_st) = b_U + d h_U + k_RU h_R + k_BU h_B      + beta' x + w_U(s) + lam_U e(s,t) + eu(s,t)

h = log(1 + events of that outcome in the previous 4 weeks); z = (governance, drought), standardised
as in add_psi_terms (realdata/fit_bivariate.py), which builds the interaction columns; e is one
shared episode field with a loading per outcome (MultiCopyBlock). Same lattice, conditions and
countries as S8.

Scenarios (kappa_R = kappa_B = 0.3):
  null      psi = 0 for both types
  opposite  psi_R,gov = +0.3, psi_B,gov = -0.3 (governance raises repression after unrest and lowers
            battles: the effects cancel when R and B are merged); psi_drought = 0.2 for battles only
Fitted per replicate:
  tri  the three-outcome model with psi per type
  biv  the two-outcome model, conflict = R or B, with one psi
Reported: estimate, SE, coverage and rejection of 0 for every psi.
"""
from common import *  # noqa: F401,F403

import scipy.sparse as sp
from scipy.special import expit

from sc.gmrf import CopyBlock, MultiCopyBlock
from sc.simulate import matern_gp

from s8_psi import NX, NY, BURN, T_TR, STEP, add_psi_terms, ar_field, lattice, std

CKPT = os.environ.get("S9_CKPT", "s9_ckpt.jsonl")
N_REPS = int(os.environ.get("N_REPS", 10))
PAR = dict(b_R=-3.3, b_B=-3.3, b_U=-2.8, d=0.3, k_RU=0.15, k_BU=0.15, kappa_R=0.3, kappa_B=0.3,
           lam_R=0.8, lam_U=0.8, e_sd=0.7, u_sd=0.5, phi=0.95)
BETA = dict(gov=-0.2, drought=0.1)
TERMS = ["gov", "drought"]
PSI = {"null": {"R": dict(gov=0.0, drought=0.0), "B": dict(gov=0.0, drought=0.0)},
       "opposite": {"R": dict(gov=0.3, drought=0.0), "B": dict(gov=-0.3, drought=0.2)}}


def simulate9(scen, seed):
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    ix, iy = np.arange(n) % NX, np.arange(n) // NX
    country = (ix // 5) + 4 * (iy // 3)
    Ttot = BURN + T_TR
    year = np.arange(Ttot) // 52
    gov = rng.normal(size=(year.max() + 1, 16))[year][:, country]
    drought = ar_field(L, Ttot, rng, 400.0, 1.0, 0.9)
    w = {o: matern_gp(L, 300.0, 0.5, rng).ravel() for o in "RBU"}
    e = ar_field(L, Ttot, rng, 300.0, PAR["e_sd"], PAR["phi"])
    eu = ar_field(L, Ttot, rng, 200.0, PAR["u_sd"], PAR["phi"])
    z = {"gov": std(gov), "drought": std(drought)}
    ps = PSI[scen]
    kap = {o: PAR[f"kappa_{o}"] + sum(ps[o][t] * z[t] for t in TERMS) for o in "RB"}
    xb = BETA["gov"] * gov + BETA["drought"] * drought
    Y = {o: np.zeros((Ttot, n), int) for o in "RBU"}
    for t in range(Ttot):
        h = {o: np.log1p(Y[o][max(0, t - 4):t].sum(0)) for o in "RBU"}
        pR = expit(PAR["b_R"] + PAR["d"] * h["R"] + kap["R"][t] * h["U"] + xb[t] + w["R"] + PAR["lam_R"] * e[t])
        pB = expit(PAR["b_B"] + PAR["d"] * h["B"] + kap["B"][t] * h["U"] + xb[t] + w["B"] + e[t])
        pU = expit(PAR["b_U"] + PAR["d"] * h["U"] + PAR["k_RU"] * h["R"] + PAR["k_BU"] * h["B"] + xb[t] + w["U"]
                   + PAR["lam_U"] * e[t] + eu[t])
        for o, p in zip("RBU", (pR, pB, pU)):
            Y[o][t] = rng.random(n) < p
    rows = []
    for t in range(BURN + 4, Ttot):
        C = Y["R"] | Y["B"]
        rows.append(pd.DataFrame({
            "cell": np.arange(n), "week": t - BURN, "yR": Y["R"][t], "yB": Y["B"][t], "yU": Y["U"][t], "yC": C[t],
            "hR": np.log1p(Y["R"][t - 4:t].sum(0)), "hB": np.log1p(Y["B"][t - 4:t].sum(0)),
            "hU": np.log1p(Y["U"][t - 4:t].sum(0)), "hC": np.log1p(C[t - 4:t].sum(0)),
            "v2x_libdem": gov[t], "drought_z": -drought[t], "tmax_z": 0.0, "log_ipc3p": 0.0, "log_pop": 0.0}))
    df = pd.concat(rows, ignore_index=True)
    df["tstep"] = df["week"] // STEP; df["train"] = True
    return L, df


def outcome_frame(d, y, own, others):
    dd = d[["cell", "tstep", "train", "v2x_libdem", "drought_z", "tmax_z", "log_ipc3p", "log_pop"]].copy()
    dd["y"] = d[y].to_numpy(); dd["lag_own"] = d[own].to_numpy()
    cols = ["lag_own", "v2x_libdem", "drought_z"]
    for i, oth in enumerate(others):
        c = "lag_other" if i == 0 else f"lag_{oth}"
        dd[c] = d[oth].to_numpy(); cols.append(c)
    return dd, cols


def fit(d, L, T, kind):
    """kind 'tri': outcomes R, B, U; 'biv': outcomes C (= R or B), U."""
    if kind == "tri":
        spec = [("repr", "yR", "hR", ["hU"]), ("battle", "yB", "hB", ["hU"]), ("unrest", "yU", "hU", ["hR", "hB"])]
    else:
        spec = [("conflict", "yC", "hC", ["hU"]), ("unrest", "yU", "hU", ["hC"])]
    dfs, covs = {}, {}
    for o, y, own, oth in spec:
        dfs[o], covs[o] = outcome_frame(d, y, own, oth)
    conf = {o: dfs[o] for o in dfs if o != "unrest"}
    cconf = {o: covs[o] for o in conf}
    add_psi_terms(conf, cconf, TERMS, "v2x_libdem")
    covs.update(cconf)
    n, m, K = L.n, len(d), len(spec)
    Xs, names = [], []
    for o, *_ in spec:
        X, nm = de.fixed_design(dfs[o], covs[o]); Xs.append(sp.csr_matrix(X)); names += [f"{o}:{v}" for v in nm]
    X = sp.block_diag(Xs, format="csr")
    Dw = de.index_design(d["cell"], n); Dst = de.spacetime_design(d["cell"], d["tstep"], n, T)
    Zw, Zst = sp.csr_matrix((m, n)), sp.csr_matrix((m, n * T))

    def rowblock(k, D, Z):     # D on outcome k's rows, zeros elsewhere
        return sp.vstack([D if i == k else Z for i in range(K)], format="csr")
    blocks = [FixedBlock(X.shape[1], names)]
    designs = [X]
    for k, (o, *_) in enumerate(spec):
        blocks.append(MaternBlock(L, name=f"field_{o}")); designs.append(rowblock(k, Dw, Zw))
    base = SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8)
    # the shared episode field enters the conflict outcome with weight 1 (battles in 'tri'), the others with loadings
    order = [1, 0, 2] if kind == "tri" else [0, 1]
    if kind == "tri":
        blocks.append(MultiCopyBlock(base, k=2, name="shared_st", lam_init=0.8, copy_names=["repr", "unrest"]))
    else:
        blocks.append(CopyBlock(base, name="shared_st", lam_init=0.8))
    designs.append(sp.bmat([[Dst if i == order[j] else Zst for j in range(K)] for i in range(K)], format="csr"))
    blocks.append(SpaceTimeAR1Block(L, T, name="st_u", sd_init=0.3, rho_init=0.8)); designs.append(rowblock(K - 1, Dst, Zst))
    y = np.concatenate([dfs[o]["y"].to_numpy() for o, *_ in spec])
    return LatentGaussianModel(y, blocks, designs).fit()


def one(arg):
    scen, seed = arg
    L, d = simulate9(scen, seed)
    T = int(d["tstep"].max() + 1)
    r = dict(scenario=scen, seed=seed, prev_R=float(d.yR.mean()), prev_B=float(d.yB.mean()), prev_U=float(d.yU.mean()))
    truth = {"repr": PSI[scen]["R"], "battle": PSI[scen]["B"],
             "conflict": {t: np.nan for t in TERMS}}           # merged: no single true value
    for kind, outs in (("tri", ["repr", "battle"]), ("biv", ["conflict"])):
        m = fit(d, L, T, kind)
        fs = m.fixed_summary(); r[f"{kind}_seconds"] = float(m.fit_seconds)
        for o in outs:
            for t in TERMS:
                est, se = map(float, fs[f"{o}:lag_other_x_{t}"])
                k = f"{kind}_{o}_{t}"
                r[k] = est; r[k + "_se"] = se
                r[k + "_excl0"] = bool(abs(est) > 1.96 * se)
                if not np.isnan(truth[o][t]):
                    r[k + "_cover"] = bool(abs(est - truth[o][t]) <= 1.96 * se)
    print(scen, seed, {k: round(v, 3) for k, v in r.items() if isinstance(v, float) and not k.endswith("_se")}, flush=True)
    return [r]


if __name__ == "__main__":
    args = [(s, seed) for s in PSI for seed in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    res.to_csv(os.path.join(RESULTS, "s9_raw.csv"), index=False)
    out = []
    for scen, g in res.groupby("scenario"):
        for kind, o in (("tri", "repr"), ("tri", "battle"), ("biv", "conflict")):
            for t in TERMS:
                k = f"{kind}_{o}_{t}"
                tr = PSI[scen]["R" if o == "repr" else "B"][t] if o != "conflict" else np.nan
                out.append(dict(scenario=scen, model=kind, outcome=o, term=t, truth=tr, mean_est=g[k].mean(),
                                sd_est=g[k].std(), mean_se=g[k + "_se"].mean(),
                                cover95=g[k + "_cover"].mean() if k + "_cover" in g else np.nan,
                                share_excl0=g[k + "_excl0"].mean(), n=len(g)))
    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(RESULTS, "s9_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
