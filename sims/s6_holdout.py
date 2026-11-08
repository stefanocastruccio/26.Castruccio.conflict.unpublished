"""S6 -- Prediction where one outcome is unobserved (spatial hold-out).

Data from the S5 generator (sims/s5_bivariate.py): conflict and unrest share a static
field and space-time episodes; scenarios with and without escalation (kappa) and with and
without unrest-only episodes.

In every replicate a random block of HOLD x HOLD cells is chosen and ONE outcome (the
'target': conflict or unrest) is treated as unobserved there in every week; the other
outcome stays observed everywhere. All models are refitted (hyperparameters included) on
what remains, and the target is predicted in the block, both in the training weeks
('past': an unreported area) and in the test weeks ('test': a forecast).

Since the target is unobserved in the block, its own history there is unknown: the
target's history covariate (Lc for conflict, Lu for unrest) is set to its mean over the
observed rows, in the block, for every model. The univariate models of the target keep
the other outcome's history (Lu or Lc), so they see the same data as the joint models:
any gain of the joint models comes from the latent structure linking the two outcomes.

Models of the target:  uni_field (static field), uni_st (+ space-time field),
                       biv_st (joint, shared space-time field), biv_st2 (+ unrest-only field)
Scores in the block: log score and AUC-PR; paired differences across replicates.
"""
from common import *  # noqa: F401,F403

import scipy.sparse as sp
from sklearn.metrics import average_precision_score

from sc.gmrf import CopyBlock
from s5_bivariate import simulate5, mc_prob, _logb, SCEN, T_TR

CKPT = os.environ.get("S6_CKPT", "s6_ckpt.jsonl")
N_REPS = int(os.environ.get("N_REPS", 20))
HOLD = int(os.environ.get("S6_HOLD", 6))          # 6 x 6 of the 16 x 14 cells (16%)
TARGETS = ("c", "u")


def block_mask(L, seed):
    rng = np.random.default_rng(10_000 + seed)
    nx, ny = len(L.lon_c), len(L.lat_c)
    i0, j0 = rng.integers(0, nx - HOLD + 1), rng.integers(0, ny - HOLD + 1)
    return (L.col >= i0) & (L.col < i0 + HOLD) & (L.row >= j0) & (L.row < j0 + HOLD)


def uni(L, dfit, dpred, kind, T, target):
    other = "u" if target == "c" else "c"
    X = lambda d: np.c_[np.ones(len(d)), d["L" + target], d["L" + other]]   # noqa: E731
    blocks = [FixedBlock(3, ["int", "own", "other"]), MaternBlock(L, name="field")]
    D = lambda d: [X(d), de.index_design(d["cell"], L.n)]                    # noqa: E731
    if kind == "uni_st":
        blocks.append(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8))
        D0 = D
        D = lambda d: D0(d) + [de.spacetime_design(d["cell"], d["tstep"], L.n, T)]   # noqa: E731
    m = LatentGaussianModel(dfit["y" + target].to_numpy(), blocks, D(dfit)).fit()
    return m, mc_prob(m, D(dpred), seed=1)


def biv(L, dc, du, dpred, T, target, two):
    """Joint model; conflict rows dc, unrest rows du (different sets). Returns the predicted
    probability of the target on the rows of dpred."""
    n = L.n

    def designs(dc_, du_):
        mc, mu = len(dc_), len(du_)
        zc, zu = np.zeros(mc), np.zeros(mu)
        X = np.r_[np.c_[np.ones(mc), dc_["Lc"], dc_["Lu"], zc, zc],
                  np.c_[zu, zu, zu, np.ones(mu), du_["Lu"]]]
        Wc, Wu = de.index_design(dc_["cell"], n), de.index_design(du_["cell"], n)
        Sc = de.spacetime_design(dc_["cell"], dc_["tstep"], n, T)
        Su = de.spacetime_design(du_["cell"], du_["tstep"], n, T)
        D = [X, sp.vstack([Wc, sp.csr_matrix((mu, n))]), sp.vstack([sp.csr_matrix((mc, n)), Wu]),
             sp.bmat([[Sc, None], [None, Su]], format="csr")]
        if two:
            D.append(sp.vstack([sp.csr_matrix((mc, n * T)), Su], format="csr"))
        return D

    blocks = [FixedBlock(5, ["int_c", "Lc", "Lu", "int_u", "Lu_u"]),
              MaternBlock(L, name="field_c"), MaternBlock(L, name="field_u"),
              CopyBlock(SpaceTimeAR1Block(L, T, name="st", sd_init=0.5, rho_init=0.8), name="shared_st")]
    if two:
        blocks.append(SpaceTimeAR1Block(L, T, name="st_u", sd_init=0.3, rho_init=0.8))
    y = np.r_[dc["yc"].to_numpy(), du["yu"].to_numpy()]
    m = LatentGaussianModel(y, blocks, designs(dc, du)).fit()
    Dp = designs(dpred, dpred)
    p = mc_prob(m, Dp, seed=1)
    return m, (p[:len(dpred)] if target == "c" else p[len(dpred):])


def one(arg):
    scen, seed, target = arg
    kappa, u_sd = SCEN[scen]
    L, df = simulate5(kappa, seed, u_sd)
    T = int(df["tstep"].max() + 1)
    hold_cell = block_mask(L, seed)
    hold = hold_cell[df["cell"].to_numpy()]
    own = "L" + target
    d = df.copy()
    d.loc[hold, own] = d.loc[~hold & d.train, own].mean()     # target's history unknown in the block
    fit_target = d[d.train & ~hold]
    fit_other = d[d.train]
    dpred = d[hold]
    y = dpred["y" + target].to_numpy(); per = np.where(dpred.train, "past", "test")
    out = []
    for kind in ("uni_field", "uni_st", "biv_st", "biv_st2"):
        if kind.startswith("uni"):
            m, p = uni(L, fit_target, dpred, kind, T, target)
            lam = None
        else:
            dc, du = (fit_target, fit_other) if target == "c" else (fit_other, fit_target)
            m, p = biv(L, dc, du, dpred, T, target, kind == "biv_st2")
            lam = m.hyper().get("shared_st.loading")
        for pp in ("past", "test"):
            s = per == pp
            out.append(dict(scenario=scen, seed=seed, target=target, model=kind, period=pp,
                            rows=int(s.sum()), events=int(y[s].sum()),
                            log_score=float(-np.mean(_logb(y[s], p[s]))),
                            auc_pr=float(average_precision_score(y[s], p[s])) if y[s].sum() else np.nan,
                            loading=lam, seconds=float(m.fit_seconds)))
        print(scen, seed, target, kind, round(m.fit_seconds, 1), flush=True)
    return out


if __name__ == "__main__":
    scens = os.environ.get("S6_SCEN", "no_escalation,escalation,escalation_uST").split(",")
    args = [(s, seed, t) for s in scens for seed in range(1, N_REPS + 1) for t in TARGETS]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    if os.environ.get("SHARD"):
        sys.exit(0)
    res.to_csv(os.path.join(RESULTS, "s6_raw.csv"), index=False)
    key = ["scenario", "seed", "target", "period"]
    rows = []
    for base in ("uni_field", "uni_st"):
        ref = res[res.model == base].set_index(key)
        for (sc_, tg, pp, mdl), g in res.groupby(["scenario", "target", "period", "model"]):
            if mdl == base:
                continue
            g = g.set_index(key)
            dl = g["log_score"] - ref.loc[g.index, "log_score"]
            da = g["auc_pr"] - ref.loc[g.index, "auc_pr"]
            rows.append(dict(scenario=sc_, target=tg, period=pp, model=mdl, base=base, n=len(g),
                             log_score=g["log_score"].mean(), d_log=dl.mean(), d_log_se=dl.std() / np.sqrt(len(dl)),
                             d_aucpr=da.mean(), d_aucpr_se=da.std() / np.sqrt(len(da)),
                             loading=g["loading"].mean()))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RESULTS, "s6_summary.csv"), index=False)
    print(tab.round(4).to_string(index=False))
