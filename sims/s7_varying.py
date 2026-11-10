"""S7 -- Can spatially varying escalation kappa(s) and region-specific loadings lambda_r be
recovered, and do the varying models stay quiet (no spurious variation) when both are constant?

Two binary outcomes per cell-week, conflict (c) and unrest (u):

  logit P(c_st) = b_c + d_c Lc_st + kappa(s) Lu_st + s(s) + w_c(s) + e(s,t)
  logit P(u_st) = b_u + d_u Lu_st + l_s s(s) + w_u(s) + lambda_{r(s)} e(s,t) + eu(s,t)

L = log(1 + events of that outcome in the previous 4 weeks); s, w_c, w_u static Matern fields;
e a shared space-time episode field (AR(1), 0.95 per week, of Matern innovations, range 300 km,
sd 0.8) and eu an unrest-only one (sd 0.6, range 200 km). The domain (28 x 16 cells of 0.5 deg)
is split into two regions, west and east. Scenarios:
  constant : kappa = 0.3 everywhere, lambda = 0.8 in both regions
  varying  : kappa(s) = 0.3 + Matern field (sd 0.3, range 500 km); lambda = 0.3 (west), 1.2 (east)
Fields are drawn from the exact continuous Matern covariance (no inverse crime).

Fitted (4-week steps for the space-time fields):
  biv2      joint model, shared field with one loading + unrest-only field (the paper's extended model)
  biv2_rl   + one loading per region (RegionCopyBlock)
  biv2_rlf  + kappa(s) = kappa_0 + gamma(s), gamma a Matern field (spatially varying coefficient)
Reported: loadings with Hessian-based 95% intervals and their difference; the kappa field's sd and
range; RMSE, mean width and pointwise coverage of kappa(s); log marginal likelihoods.
"""
from common import *  # noqa: F401,F403

import scipy.sparse as sp
from scipy.special import expit

from sc.gmrf import RegionCopyBlock, SphereLattice
from sc.simulate import matern_gp

CKPT = os.environ.get("S7_CKPT", "s7_ckpt.jsonl")
N_REPS = int(os.environ.get("N_REPS", 20))
NX, NY, BURN, T_TR, STEP = 28, 16, 12, 104, 4
PAR = dict(b_c=-3.9, b_u=-3.6, d_c=0.3, d_u=0.3, l_s=0.8, phi=0.95, e_sd=0.8, u_sd=0.6, kappa0=0.3)
SCEN = {"constant": dict(kappa_sd=0.0, lam=(0.8, 0.8)),
        "varying": dict(kappa_sd=0.3, lam=(0.3, 1.2))}
GROUPS = ["west", "east"]


def lattice():
    return SphereLattice(10.25 + 0.5 * np.arange(NX), -1.75 + 0.5 * np.arange(NY))


def ar_field(L, T, rng, rng_km, sd, phi):
    x = np.zeros((T, L.n)); x[0] = matern_gp(L, rng_km, sd, rng).ravel()
    for t in range(1, T):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * matern_gp(L, rng_km, sd, rng).ravel()
    return x


def simulate7(scen, seed):
    cfg = SCEN[scen]
    rng = np.random.default_rng(seed)
    L = lattice(); n = L.n
    grp = (L.lon >= L.lon.min() + 0.5 * NX / 2).astype(int)          # 0 west, 1 east
    kappa = PAR["kappa0"] + (matern_gp(L, 500.0, cfg["kappa_sd"], rng).ravel() if cfg["kappa_sd"] > 0 else 0.0)
    kappa = np.broadcast_to(kappa, (n,)).copy()
    lam = np.asarray(cfg["lam"])[grp]
    s = matern_gp(L, 400.0, 0.6, rng).ravel()
    wc = matern_gp(L, 300.0, 0.5, rng).ravel(); wu = matern_gp(L, 150.0, 0.5, rng).ravel()
    Ttot = BURN + T_TR
    e = ar_field(L, Ttot, rng, 300.0, PAR["e_sd"], PAR["phi"])
    eu = ar_field(L, Ttot, np.random.default_rng(seed + 777), 200.0, PAR["u_sd"], PAR["phi"])
    yc = np.zeros((Ttot, n), int); yu = np.zeros((Ttot, n), int)
    for t in range(Ttot):
        lc = np.log1p(yc[max(0, t - 4):t].sum(0)); lu = np.log1p(yu[max(0, t - 4):t].sum(0))
        pc = expit(PAR["b_c"] + PAR["d_c"] * lc + kappa * lu + s + wc + e[t])
        pu = expit(PAR["b_u"] + PAR["d_u"] * lu + PAR["l_s"] * s + wu + lam * e[t] + eu[t])
        yc[t] = rng.random(n) < pc; yu[t] = rng.random(n) < pu
    rows = []
    for t in range(BURN + 4, Ttot):
        rows.append(pd.DataFrame({"cell": np.arange(n), "week": t - BURN, "yc": yc[t], "yu": yu[t],
                                  "Lc": np.log1p(yc[t - 4:t].sum(0)), "Lu": np.log1p(yu[t - 4:t].sum(0))}))
    df = pd.concat(rows, ignore_index=True)
    df["tstep"] = df["week"] // STEP
    return L, grp, kappa, df


def designs(L, d, T, kfield):
    n, m = L.n, len(d)
    z = np.zeros(m); one = np.ones(m)
    X = np.r_[np.c_[one, d["Lc"], d["Lu"], z, z], np.c_[z, z, z, one, d["Lu"]]]
    Dw = de.index_design(d["cell"], n); Zw = sp.csr_matrix((m, n))
    Dst = de.spacetime_design(d["cell"], d["tstep"], n, T); Zst = sp.csr_matrix((m, n * T))
    D = [X, sp.vstack([Dw, Zw]), sp.vstack([Zw, Dw]), sp.bmat([[Dst, Zst], [Zst, Dst]], format="csr"),
         sp.vstack([Zst, Dst], format="csr")]
    if kfield:
        D.append(sp.vstack([de.index_design(d["cell"], n, d["Lu"].to_numpy(float)), Zw], format="csr"))
    return D


def blocks(L, T, grp, kind):
    b = [FixedBlock(5, ["int_c", "Lc", "Lu", "int_u", "Lu_u"]),
         MaternBlock(L, name="field_c"), MaternBlock(L, name="field_u")]
    st = SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8)
    if kind == "biv2":
        from sc.gmrf import CopyBlock
        b.append(CopyBlock(st, name="shared_st"))
    else:
        b.append(RegionCopyBlock(st, np.tile(grp, T), GROUPS, name="shared_st", lam_init=0.8))
    b.append(SpaceTimeAR1Block(L, T, name="st_u", sd_init=0.3, rho_init=0.8))
    if kind == "biv2_rlf":
        b.append(MaternBlock(L, name="kappa_field", range_init=800.0, sd_init=0.05, pc_sd=(0.2, 0.05)))
    return b


def theta_pos(m, block_name):
    k = 0
    for b in m.blocks:
        if b.name == block_name:
            return k
        k += b.n_theta
    raise KeyError(block_name)


def one(arg):
    scen, seed = arg
    L, grp, kappa, d = simulate7(scen, seed)
    T = int(d["tstep"].max() + 1)
    y = np.r_[d["yc"].to_numpy(), d["yu"].to_numpy()]
    lam_true = SCEN[scen]["lam"]
    out = []
    for kind in ["biv2", "biv2_rl", "biv2_rlf"]:
        m = LatentGaussianModel(y, blocks(L, T, grp, kind), designs(L, d, T, kind == "biv2_rlf")).fit()
        fs = m.fixed_summary(); h = m.hyper()
        r = dict(scenario=scen, seed=seed, model=kind, log_ml=float(m.log_ml), seconds=float(m.fit_seconds),
                 kappa0_est=float(fs["Lu"][0]), kappa0_se=float(fs["Lu"][1]),
                 prevalence_c=float(d["yc"].mean()), prevalence_u=float(d["yu"].mean()))
        if kind == "biv2":
            r["lam"] = h["shared_st.loading"]
        else:
            H, se = m.hessian_theta(n_jobs=1)
            p0 = theta_pos(m, "shared_st") + m.blocks[[b.name for b in m.blocks].index("shared_st")].base.n_theta
            th = np.asarray(m.theta)
            try:
                V = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                V = np.full(H.shape, np.nan)
            for j, g in enumerate(GROUPS):
                r[f"lam_{g}"] = float(th[p0 + j]); r[f"lam_{g}_se"] = float(np.sqrt(V[p0 + j, p0 + j]))
                r[f"lam_{g}_true"] = lam_true[j]
                r[f"lam_{g}_covers"] = bool(abs(th[p0 + j] - lam_true[j]) <= 1.96 * r[f"lam_{g}_se"])
            dvar = V[p0, p0] + V[p0 + 1, p0 + 1] - 2 * V[p0, p0 + 1]
            r["lam_diff"] = float(th[p0 + 1] - th[p0]); r["lam_diff_se"] = float(np.sqrt(dvar))
            r["lam_diff_rejects0"] = bool(abs(r["lam_diff"]) > 1.96 * r["lam_diff_se"])
            r["hessian_eig_min"] = float(np.linalg.eigvalsh(H).min())
        if kind == "biv2_rlf":
            r["kappa_field_sd"] = h["kappa_field.sd"]; r["kappa_field_range"] = h["kappa_field.range_km"]
            pk = theta_pos(m, "kappa_field")
            r["kappa_field_logsd_se"] = float(np.sqrt(V[pk + 1, pk + 1]))
            S = m.sample(200, np.random.default_rng(seed))
            ib = [b.name for b in m.blocks].index("kappa_field")
            draws = S[m.slices[0]][2][None, :] + S[m.slices[ib]]           # kappa_0 + gamma(s), n x 200
            lo, hi = np.quantile(draws, [0.025, 0.975], axis=1)
            mean = draws.mean(1)
            r["kappa_rmse"] = float(np.sqrt(np.mean((mean - kappa) ** 2)))
            r["kappa_rmse_const"] = float(np.sqrt(np.mean((r["kappa0_est"] - kappa) ** 2)))
            r["kappa_cover"] = float(np.mean((lo <= kappa) & (kappa <= hi)))
            r["kappa_width"] = float(np.mean(hi - lo))
            r["kappa_cells_excl_kappa0"] = float(np.mean((lo > r["kappa0_est"]) | (hi < r["kappa0_est"])))
        out.append(r)
        print(scen, seed, kind, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                 if k.startswith(("lam", "kappa"))}, round(m.fit_seconds, 1), flush=True)
    return out


if __name__ == "__main__":
    args = [(s, seed) for s in SCEN for seed in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    if os.environ.get("SHARD"):
        sys.exit(0)
    res.to_csv(os.path.join(RESULTS, "s7_raw.csv"), index=False)
    base = res[res.model == "biv2"].set_index(["scenario", "seed"])["log_ml"]
    res["dlogml_vs_biv2"] = res["log_ml"].to_numpy() - base.loc[list(zip(res.scenario, res.seed))].to_numpy()
    agg = {c: "mean" for c in res.columns if c not in ("scenario", "model", "seed") and res[c].dtype != object}
    tab = res.groupby(["scenario", "model"]).agg(agg).reset_index()
    tab["n"] = res.groupby(["scenario", "model"]).size().to_numpy()
    tab.to_csv(os.path.join(RESULTS, "s7_summary.csv"), index=False)
    print(tab.round(3).T.to_string())
