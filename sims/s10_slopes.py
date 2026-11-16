"""S10 -- Toy check of the country random slopes (realdata/fit_bivariate.py, fit_country_slopes / model biv3_cs):
kappa_c = kappa + u_c, u_c ~ N(0, tau^2) over 16 countries, in the unrest -> conflict direction only. The code under test
is fit_country_slopes itself: the base model (biv2 structure, standing in for biv3) is fitted first, its hyperparameters
saved, then only the two slope SDs are optimised, as on the CRC.

Scenarios: tau = 0 (null) and tau = 0.25 (planted; kappa = 0.3). Reported: tau_conflict and tau_unrest with intervals,
correlation of the fitted u_c with the truth, coverage of u_c, kappa.
"""
from common import *  # noqa: F401,F403

import types

import scipy.sparse as sp
from scipy.special import expit

from sc.gmrf import CopyBlock
from sc.simulate import matern_gp

from s8_psi import NX, NY, BURN, T_TR, STEP, ar_field, lattice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realdata"))
from fit_bivariate import fit_country_slopes, stack  # noqa: E402

CKPT = os.environ.get("S10_CKPT", "s10_ckpt.jsonl")
N_REPS = int(os.environ.get("N_REPS", 5))
PAR = dict(b_c=-3.0, b_u=-2.8, d=0.3, k_cu=0.2, kappa=0.3, lam=0.8, e_sd=0.7, u_sd=0.5, phi=0.95)
TAU = {"null": 0.0, "planted": 0.25}


def simulate10(scen, seed):
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    ix, iy = np.arange(n) % NX, np.arange(n) // NX
    country = (ix // 5) + 4 * (iy // 3)
    u = rng.normal(0, TAU[scen], 16)
    Ttot = BURN + T_TR
    wc = matern_gp(L, 300.0, 0.5, rng).ravel(); wu = matern_gp(L, 200.0, 0.5, rng).ravel()
    e = ar_field(L, Ttot, rng, 300.0, PAR["e_sd"], PAR["phi"]); eu = ar_field(L, Ttot, rng, 200.0, PAR["u_sd"], PAR["phi"])
    kap = PAR["kappa"] + u[country]
    yc = np.zeros((Ttot, n), int); yu = np.zeros((Ttot, n), int)
    for t in range(Ttot):
        lc = np.log1p(yc[max(0, t - 4):t].sum(0)); lu = np.log1p(yu[max(0, t - 4):t].sum(0))
        yc[t] = rng.random(n) < expit(PAR["b_c"] + PAR["d"] * lc + kap * lu + wc + e[t])
        yu[t] = rng.random(n) < expit(PAR["b_u"] + PAR["d"] * lu + PAR["k_cu"] * lc + wu + PAR["lam"] * e[t] + eu[t])
    rows = []
    for t in range(BURN + 4, Ttot):
        rows.append(pd.DataFrame({"cell": np.arange(n), "country": country, "week": t - BURN, "yc": yc[t], "yu": yu[t],
                                  "Lc": np.log1p(yc[t - 4:t].sum(0)), "Lu": np.log1p(yu[t - 4:t].sum(0))}))
    d = pd.concat(rows, ignore_index=True); d["tstep"] = d["week"] // STEP
    return L, d, u


def one(arg):
    scen, seed = arg
    L, d, u = simulate10(scen, seed)
    n, T, nC = L.n, int(d.tstep.max() + 1), 16
    P = {}
    for o, (y, own, oth) in {"conflict": ("yc", "Lc", "Lu"), "unrest": ("yu", "Lu", "Lc")}.items():
        X = np.c_[np.ones(len(d)), d[own], d[oth]]
        P[o] = dict(X=X, names=[f"{o}:intercept", f"{o}:lag_own", f"{o}:lag_other"], y=d[y].to_numpy(),
                    W=de.index_design(d["cell"], n), V=de.spacetime_design(d["cell"], d["tstep"], n, T),
                    S=de.index_design(d["country"], nC, values=d[oth].to_numpy(float)))

    def base_blocks():
        names = P["conflict"]["names"] + P["unrest"]["names"]
        return [FixedBlock(len(names), names), MaternBlock(L, name="field_conflict"), MaternBlock(L, name="field_unrest"),
                CopyBlock(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8), name="shared_st"),
                SpaceTimeAR1Block(L, T, name="st_unrest", sd_init=0.3, rho_init=0.8)]

    def base_designs(Pc, Pu):
        nc, nu = len(Pc["y"]), len(Pu["y"])
        return [sp.block_diag([sp.csr_matrix(Pc["X"]), sp.csr_matrix(Pu["X"])], format="csr"),
                stack(Pc["W"], None, nu=nu), stack(None, Pu["W"], nc=nc),
                sp.bmat([[Pc["V"], None], [None, Pu["V"]]], format="csr"), stack(None, Pu["V"], nc=nc)]

    m0 = LatentGaussianModel(np.r_[P["conflict"]["y"], P["unrest"]["y"]], base_blocks(),
                             base_designs(P["conflict"], P["unrest"])).fit()
    saved = {"models": {"biv3": {"hyper": m0.hyper()}}}
    a = types.SimpleNamespace(zero_frac=None, zero_seed=0, fixed_theta="toy")
    base = pd.DataFrame({"ISO_A3": [f"C{c:02d}" for c in d["country"]], "train": True})
    # fit_country_slopes names countries by order of appearance (pd.factorize), which is 0..15 here
    keys = ["conflict:lag_other", "unrest:lag_other"]
    out = fit_country_slopes(a, saved, base_blocks, base_designs, P, nC, base, keys)
    iso = pd.factorize(base["ISO_A3"])[1]
    uh = np.array([out["countries"][c]["conflict"]["u"] for c in iso]); us = np.array([out["countries"][c]["conflict"]["sd"] for c in iso])
    ut = u[[int(c[1:]) for c in iso]]
    tc, tu = out["tau"]["kslope_conflict"], out["tau"]["kslope_unrest"]
    r = dict(scenario=scen, seed=seed, tau_true=TAU[scen], tau_c=tc["est"], tau_c_lo=tc["lo"], tau_c_hi=tc["hi"],
             tau_u=tu["est"], tau_u_lo=tu["lo"], tau_u_hi=tu["hi"],
             kappa=out["fixed_ci95"]["conflict:lag_other"]["est"], kappa_se=out["fixed_ci95"]["conflict:lag_other"]["se"],
             u_corr=float(np.corrcoef(uh, ut)[0, 1]) if TAU[scen] > 0 else np.nan,
             u_cover=float(np.mean(np.abs(uh - ut) <= 1.96 * us)), seconds=float(out["seconds"]) + float(m0.fit_seconds))
    print(scen, seed, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    return [r]


if __name__ == "__main__":
    args = [(s, seed) for s in TAU for seed in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    res.to_csv(os.path.join(RESULTS, "s10_raw.csv"), index=False)
    print(res.groupby("scenario").mean(numeric_only=True).round(3).T.to_string())
