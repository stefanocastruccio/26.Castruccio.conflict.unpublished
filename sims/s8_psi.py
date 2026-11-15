"""S8 -- Toy check of the escalation interactions psi (realdata/fit_bivariate.py --psi).

kappa_{U->C}(s,t) = kappa + sum_m psi_m z_m,st, built by the real add_psi_terms(), so the code that
will run on the CRC is the code tested here. Two binary outcomes per cell-week:

  logit P(c_st) = b_c + d_c Lc + (kappa + psi' z_st) Lu + beta' x_st + w_c(s) + e(s,t)
  logit P(u_st) = b_u + d_u Lu + k_cu Lc               + beta' x_st + w_u(s) + lam e(s,t) + eu(s,t)

L = log(1 + events of that outcome in the previous 4 weeks). Conditions on a 28 x 16 lattice of
0.5-degree cells split into 16 "countries" (4 x 4 blocks of 7 x 4 cells), 104 weeks:
  governance  country-year, N(0,1) per country and year (v2x_libdem column)
  drought     -SPI anomaly: smooth AR(1) field (range 400 km, 0.9 per week)
  heat        tmax anomaly: AR(1) field (range 300 km, 0.7 per week)
  IPC 3+      country-month, half the countries in crisis in a given month
  population  static Matern field (remote = -log population)
Each condition also has a direct effect on both outcomes (beta, fixed). The shared space-time field
e is the confounder that the model must separate from escalation.

Scenarios: null (all psi = 0) and planted (psi_gov 0.15, psi_gov2 -0.10, psi_drought 0.10,
psi_drought_ipc 0.10; heat and remote 0). C->U has no psi in either scenario. kappa = 0.3, larger than
the real 0.06, because the toy has about 100 times fewer cell-weeks than Africa-Middle East: this is a
check of the code and of calibration, not a power study.

Fitted: the joint model (biv2 structure) with all six psi in both directions. Reported per psi:
estimate, SE, coverage of the truth, and whether the 95% interval excludes 0.
"""
from common import *  # noqa: F401,F403

import scipy.sparse as sp
from scipy.special import expit

from sc.gmrf import CopyBlock, SphereLattice
from sc.simulate import matern_gp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realdata"))
from fit_bivariate import PSI_TERMS, add_psi_terms  # noqa: E402

CKPT = os.environ.get("S8_CKPT", "s8_ckpt.jsonl")
N_REPS = int(os.environ.get("N_REPS", 10))
NX, NY, BURN, T_TR, STEP = 28, 16, 12, 104, 4
PAR = dict(b_c=-3.6, b_u=-3.4, d_c=0.3, d_u=0.3, k_cu=0.2, kappa=0.3, lam=0.8, e_sd=0.7, u_sd=0.5, phi=0.95)
BETA = dict(gov=-0.2, drought=0.1, heat=0.05, ipc=0.15, remote=-0.2)
PSI = {"null": dict(gov=0.0, gov2=0.0, drought=0.0, heat=0.0, drought_ipc=0.0, remote=0.0),
       "planted": dict(gov=0.15, gov2=-0.10, drought=0.10, heat=0.0, drought_ipc=0.10, remote=0.0)}
TERMS = list(PSI_TERMS)


def lattice():
    return SphereLattice(10.25 + 0.5 * np.arange(NX), -1.75 + 0.5 * np.arange(NY))


def ar_field(L, T, rng, rng_km, sd, phi):
    x = np.zeros((T, L.n)); x[0] = matern_gp(L, rng_km, sd, rng).ravel()
    for t in range(1, T):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * matern_gp(L, rng_km, sd, rng).ravel()
    return x


def std(v):
    return (v - v.mean()) / v.std()


def simulate8(scen, seed):
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    ix, iy = np.arange(n) % NX, np.arange(n) // NX          # SphereLattice orders lon fastest
    country = (ix // 7) + 4 * (iy // 4)
    Ttot = BURN + T_TR
    year = np.arange(Ttot) // 52
    gov = rng.normal(size=(year.max() + 1, 16))[year][:, country]               # T x n
    drought = ar_field(L, Ttot, rng, 400.0, 1.0, 0.9)
    heat = ar_field(L, Ttot, rng, 300.0, 1.0, 0.7)
    month = np.arange(Ttot) // 4
    ipc = (rng.random((month.max() + 1, 16)) < 0.5).astype(float)[month][:, country]
    logpop = matern_gp(L, 300.0, 1.0, rng).ravel()
    wc = matern_gp(L, 300.0, 0.5, rng).ravel(); wu = matern_gp(L, 200.0, 0.5, rng).ravel()
    e = ar_field(L, Ttot, rng, 300.0, PAR["e_sd"], PAR["phi"])
    eu = ar_field(L, Ttot, rng, 200.0, PAR["u_sd"], PAR["phi"])
    # the conditions exactly as add_psi_terms standardises them (all rows are training rows)
    z = {"gov": std(gov), "drought": std(drought), "heat": std(heat), "remote": std(-np.broadcast_to(logpop, (Ttot, n)))}
    z["gov2"] = std(std(gov) ** 2)
    z["drought_ipc"] = std(drought * ipc)
    ps = PSI[scen]
    kap = PAR["kappa"] + sum(ps[t] * z[t] for t in TERMS)                       # T x n
    x_eff = (BETA["gov"] * gov + BETA["drought"] * drought + BETA["heat"] * heat + BETA["ipc"] * ipc
             + BETA["remote"] * (-logpop))
    yc = np.zeros((Ttot, n), int); yu = np.zeros((Ttot, n), int)
    for t in range(Ttot):
        lc = np.log1p(yc[max(0, t - 4):t].sum(0)); lu = np.log1p(yu[max(0, t - 4):t].sum(0))
        pc = expit(PAR["b_c"] + PAR["d_c"] * lc + kap[t] * lu + x_eff[t] + wc + e[t])
        pu = expit(PAR["b_u"] + PAR["d_u"] * lu + PAR["k_cu"] * lc + x_eff[t] + wu + PAR["lam"] * e[t] + eu[t])
        yc[t] = rng.random(n) < pc; yu[t] = rng.random(n) < pu
    rows = []
    for t in range(BURN + 4, Ttot):
        rows.append(pd.DataFrame({
            "cell": np.arange(n), "week": t - BURN, "yc": yc[t], "yu": yu[t],
            "Lc": np.log1p(yc[t - 4:t].sum(0)), "Lu": np.log1p(yu[t - 4:t].sum(0)),
            "v2x_libdem": gov[t], "drought_z": -drought[t], "tmax_z": heat[t],      # drought_z is an SPI anomaly
            "log_ipc3p": ipc[t], "log_pop": logpop}))
    df = pd.concat(rows, ignore_index=True)
    df["tstep"] = df["week"] // STEP; df["train"] = True
    # rebuild z on the kept rows to compare like with like (truth uses the full panel; the
    # difference is a burn-in of 16 weeks out of 120 and only shifts centring slightly)
    return L, df


def build(d, L, T):
    dfs, covs = {}, {}
    base = ["lag_own", "lag_other", "v2x_libdem", "drought_z", "tmax_z", "log_ipc3p", "log_pop"]
    for o, (yo, own, oth) in {"conflict": ("yc", "Lc", "Lu"), "unrest": ("yu", "Lu", "Lc")}.items():
        dd = d[["cell", "tstep", "train", "v2x_libdem", "drought_z", "tmax_z", "log_ipc3p", "log_pop"]].copy()
        dd["y"] = d[yo].to_numpy(); dd["lag_own"] = d[own].to_numpy(); dd["lag_other"] = d[oth].to_numpy()
        dfs[o], covs[o] = dd, list(base)
    add_psi_terms(dfs, covs, TERMS, "v2x_libdem")
    n, m = L.n, len(d)
    Xs, names = [], []
    for o in ("conflict", "unrest"):
        X, nm = de.fixed_design(dfs[o], covs[o]); Xs.append(sp.csr_matrix(X)); names += [f"{o}:{v}" for v in nm]
    X = sp.block_diag(Xs, format="csr")
    Dw = de.index_design(d["cell"], n); Zw = sp.csr_matrix((m, n))
    Dst = de.spacetime_design(d["cell"], d["tstep"], n, T); Zst = sp.csr_matrix((m, n * T))
    blocks = [FixedBlock(X.shape[1], names), MaternBlock(L, name="field_c"), MaternBlock(L, name="field_u"),
              CopyBlock(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8), name="shared_st"),
              SpaceTimeAR1Block(L, T, name="st_u", sd_init=0.3, rho_init=0.8)]
    designs = [X, sp.vstack([Dw, Zw], format="csr"), sp.vstack([Zw, Dw], format="csr"),
               sp.bmat([[Dst, Zst], [Zst, Dst]], format="csr"), sp.vstack([Zst, Dst], format="csr")]
    y = np.r_[dfs["conflict"]["y"].to_numpy(), dfs["unrest"]["y"].to_numpy()]
    return y, blocks, designs


def one(arg):
    scen, seed = arg
    L, d = simulate8(scen, seed)
    T = int(d["tstep"].max() + 1)
    y, blocks, designs = build(d, L, T)
    m = LatentGaussianModel(y, blocks, designs).fit()
    fs = m.fixed_summary()
    r = dict(scenario=scen, seed=seed, seconds=float(m.fit_seconds), prev_c=float(d.yc.mean()), prev_u=float(d.yu.mean()),
             kappa=float(fs["conflict:lag_other"][0]), kappa_se=float(fs["conflict:lag_other"][1]))
    for o, truth in (("conflict", PSI[scen]), ("unrest", PSI["null"])):
        for t in TERMS:
            est, se = map(float, fs[f"{o}:lag_other_x_{t}"])
            k = f"{o[0]}_{t}"
            r[k] = est; r[k + "_se"] = se
            r[k + "_cover"] = bool(abs(est - truth[t]) <= 1.96 * se)
            r[k + "_excl0"] = bool(abs(est) > 1.96 * se)
    print(scen, seed, {k: round(v, 3) for k, v in r.items() if k.startswith("c_") and not k.endswith(("_cover", "_excl0"))},
          round(r["seconds"], 1), flush=True)
    return [r]


if __name__ == "__main__":
    args = [(s, seed) for s in PSI for seed in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    res.to_csv(os.path.join(RESULTS, "s8_raw.csv"), index=False)
    out = []
    for scen, g in res.groupby("scenario"):
        for o, truth in (("c", PSI[scen]), ("u", PSI["null"])):
            for t in TERMS:
                k = f"{o}_{t}"
                out.append(dict(scenario=scen, direction="U->C" if o == "c" else "C->U", term=t, truth=truth[t],
                                mean_est=g[k].mean(), sd_est=g[k].std(), mean_se=g[k + "_se"].mean(),
                                cover95=g[k + "_cover"].mean(), share_excl0=g[k + "_excl0"].mean(), n=len(g)))
    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(RESULTS, "s8_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
