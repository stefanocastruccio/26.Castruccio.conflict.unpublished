"""S10 -- Toy check of the country effects on escalation (realdata/fit_bivariate.py, fit_country_slopes, model biv3_cs):

  kappa_c(s,t) = kappa + u_c + v_c * drought_st     (unrest -> conflict; conflict -> unrest has none)

16 countries on a 4 x 4 grid of 5 x 3 cells (S8's lattice and conditions). The code under test is fit_country_slopes
itself, with terms u and drought, both BYM2 on the country adjacency, and the boundary test of each tau: the base model
(biv2 structure, standing in for biv3) is fitted first, its hyperparameters saved, then only the country effects are
optimised, as on the CRC.

Scenarios:
  null     u = 0, v = 0 (no country differences)
  planted  u_c = 0.2 in the western half, -0.2 in the eastern half (spatially structured);
           drought sensitivity v_c = +0.3 in the 4 north-western countries (amplifies), -0.3 in the 4 south-eastern
           (suppresses), 0 in the other 8 (no effect); global psi_drought = 0.
Reported: tau and phi with intervals; the boundary p-values; for each country the class (amplifies / suppresses /
no_effect / inconclusive) against the truth: false flags (a true 0 classed as amplifies/suppresses), detections, and
wrong signs.
"""
from common import *  # noqa: F401,F403

import types

import scipy.sparse as sp
from scipy.special import expit

from sc.gmrf import CopyBlock
from sc.simulate import matern_gp

from s8_psi import NX, NY, BURN, T_TR, STEP, ar_field, lattice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realdata"))
from fit_bivariate import add_psi_terms, fit_country_slopes, stack  # noqa: E402

CKPT = os.environ.get("S10_CKPT", "s10b_ckpt.jsonl")
N_REPS = int(os.environ.get("N_REPS", 5))
PAR = dict(b_c=-3.0, b_u=-2.8, d=0.3, k_cu=0.2, kappa=0.3, lam=0.8, e_sd=0.7, u_sd=0.5, phi=0.95, beta_dr=0.1)
CX, CY = np.arange(16) % 4, np.arange(16) // 4                 # country column (west -> east), row (south -> north)
TRUTH = {"null": dict(u=np.zeros(16), v=np.zeros(16)),
         "planted": dict(u=np.where(CX < 2, 0.2, -0.2),
                         v=np.where((CX < 2) & (CY >= 2), 0.3, np.where((CX >= 2) & (CY < 2), -0.3, 0.0)))}


def simulate10(scen, seed):
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    ix, iy = np.arange(n) % NX, np.arange(n) // NX
    country = (ix // 5) + 4 * (iy // 3)
    tr = TRUTH[scen]
    Ttot = BURN + T_TR
    drought = ar_field(L, Ttot, rng, 400.0, 1.0, 0.9)
    dz = (drought - drought.mean()) / drought.std()
    wc = matern_gp(L, 300.0, 0.5, rng).ravel(); wu = matern_gp(L, 200.0, 0.5, rng).ravel()
    e = ar_field(L, Ttot, rng, 300.0, PAR["e_sd"], PAR["phi"]); eu = ar_field(L, Ttot, rng, 200.0, PAR["u_sd"], PAR["phi"])
    kap = PAR["kappa"] + tr["u"][country] + tr["v"][country] * dz          # T x n
    yc = np.zeros((Ttot, n), int); yu = np.zeros((Ttot, n), int)
    for t in range(Ttot):
        lc = np.log1p(yc[max(0, t - 4):t].sum(0)); lu = np.log1p(yu[max(0, t - 4):t].sum(0))
        yc[t] = rng.random(n) < expit(PAR["b_c"] + PAR["d"] * lc + kap[t] * lu + PAR["beta_dr"] * drought[t] + wc + e[t])
        yu[t] = rng.random(n) < expit(PAR["b_u"] + PAR["d"] * lu + PAR["k_cu"] * lc + PAR["beta_dr"] * drought[t] + wu
                                      + PAR["lam"] * e[t] + eu[t])
    rows = []
    for t in range(BURN + 4, Ttot):
        rows.append(pd.DataFrame({"cell": np.arange(n), "country_id": country, "week": t - BURN, "yc": yc[t], "yu": yu[t],
                                  "Lc": np.log1p(yc[t - 4:t].sum(0)), "Lu": np.log1p(yu[t - 4:t].sum(0)),
                                  "drought_z": -drought[t], "lon": L.lon, "lat": L.lat}))
    d = pd.concat(rows, ignore_index=True); d["tstep"] = d["week"] // STEP; d["train"] = True
    return L, d


def one(arg):
    scen, seed = arg
    L, d = simulate10(scen, seed)
    n, T, nC = L.n, int(d.tstep.max() + 1), 16
    dfs, covs = {}, {}
    for o, (y, own, oth) in {"conflict": ("yc", "Lc", "Lu"), "unrest": ("yu", "Lu", "Lc")}.items():
        dd = d[["cell", "country_id", "tstep", "train", "drought_z"]].copy()
        dd["y"] = d[y].to_numpy(); dd["lag_own"] = d[own].to_numpy(); dd["lag_other"] = d[oth].to_numpy()
        dfs[o], covs[o] = dd, ["lag_own", "lag_other", "drought_z"]
    add_psi_terms(dfs, covs, ["drought"], "none")
    P = {}
    for o in dfs:
        X, nm = de.fixed_design(dfs[o], covs[o])
        P[o] = dict(X=X, names=[f"{o}:{v}" for v in nm], y=dfs[o]["y"].to_numpy(), W=de.index_design(d["cell"], n),
                    V=de.spacetime_design(d["cell"], d["tstep"], n, T), C=de.index_design(d["country_id"], nC))

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
    base = pd.DataFrame({"ISO_A3": [f"C{c:02d}" for c in d["country_id"]], "country_id": d["country_id"],
                         "lon": d["lon"], "lat": d["lat"], "train": True})
    out = fit_country_slopes(a, saved, base_blocks, base_designs, P, nC, base, ["conflict:lag_other", "unrest:lag_other"],
                             terms=("u", "drought"), bym2=("u", "drought"), tau_test=True, n_draws=400)
    iso = pd.factorize(base["ISO_A3"])[1]
    cid = np.array([int(c[1:]) for c in iso])
    tr = TRUTH[scen]
    r = dict(scenario=scen, seed=seed, seconds=float(out["seconds"]) + float(m0.fit_seconds), adjacency_edges=out.get("adjacency_edges"))
    for o in ("conflict", "unrest"):
        for t in ("u", "drought"):
            nm = f"k_{o}_{t}"
            r[f"{o}_{t}_tau"] = out["random"][nm][f"{nm}.sd"]["est"]
            r[f"{o}_{t}_phi"] = out["random"][nm][f"{nm}.phi"]["est"]
            r[f"{o}_{t}_p"] = out["tau_test"][nm]["p_boundary"]
    cls = np.array([out["countries"][c]["conflict_drought"]["class"] for c in iso])
    v = tr["v"][cid]
    flagged = np.isin(cls, ["amplifies", "suppresses"])
    r["drought_false_flags"] = int(np.sum(flagged & (v == 0)))
    r["drought_detected"] = int(np.sum(flagged & (v != 0) & (np.sign(v) == np.where(cls == "amplifies", 1, -1))))
    r["drought_wrong_sign"] = int(np.sum(flagged & (v != 0) & (np.sign(v) != np.where(cls == "amplifies", 1, -1))))
    r["drought_no_effect_correct"] = int(np.sum((cls == "no_effect") & (v == 0)))
    r["drought_no_effect_wrong"] = int(np.sum((cls == "no_effect") & (v != 0)))
    r["drought_true_nonzero"] = int(np.sum(v != 0)); r["drought_true_zero"] = int(np.sum(v == 0))
    ucls = np.array([out["countries"][c]["conflict_u"]["class"] for c in iso])
    r["u_flagged"] = int(np.sum(np.isin(ucls, ["amplifies", "suppresses"])))
    print(scen, seed, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
    return [r]


if __name__ == "__main__":
    args = [(s, seed) for s in TRUTH for seed in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    res.to_csv(os.path.join(RESULTS, "s10_raw.csv"), index=False)
    print(res.groupby("scenario").mean(numeric_only=True).round(3).T.to_string())
