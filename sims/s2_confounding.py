"""S2 -- National indicators and spatial confounding.

The paper's sustainability findings (high EPI / ecological footprint => much lower
predicted conflict) come from covariates that are constant within a country. If such
an indicator tracks unobserved country-level risk (state capacity, reporting), its
apparent effect is confounded. Here Z0 is correlated (nat_conf) with the country
mean of the latent risk w + u, and its true effect beta_nat is 0 or -0.3.

Estimators of the Z0 effect:
  glm                 logistic regression, no latent effects
  country             + iid country effects
  field               + Matern field
  field+country       + both
  spatial+            field+country with Z0 residualised on the 10 smoothest
                      eigenvectors of the lattice Laplacian (Dupont et al. 2022)
  xgb_shap            slope of XGBoost SHAP(Z0) on Z0. As in the original procedure,
                      the model is fitted to a fixed 30% subsample of the training
                      rows and the reported value (est) is the median over B_BOOT
                      spatial block-bootstrap refits (2.5-deg blocks), which also give
                      a 95% percentile interval and an IQR. Also stored:
                        est_sub   the same model fitted once to the 30% subsample
                        est_full  the same procedure fitted once to ALL training rows
"""
from common import *  # noqa: F401,F403

from sc import boosting as bo
from sc.simulate import SimConfig, simulate

CKPT = "s2v2_ckpt.jsonl"   # v2: adds est_full (all training rows); old est_full is est_sub
N_REPS = int(os.environ.get("N_REPS", 30))
B_BOOT = int(os.environ.get("B_BOOT", 40))
BLOCK_DEG = float(os.environ.get("BLOCK_DEG", 2.5))   # 48 blocks on the 20 x 15 degree domain
SCENARIOS = {"A_conf_null": dict(nat_conf=0.8, beta_nat=0.0),
             "B_noconf_null": dict(nat_conf=0.0, beta_nat=0.0),
             "C_conf_effect": dict(nat_conf=0.8, beta_nat=-0.3)}
COLS = ["pop", "log_lag4", "temp_anom", "prec_anom", "Z0"]


def one(arg):
    scen, seed = arg
    cfg = SimConfig(seed=seed, n_countries=25, **SCENARIOS[scen])
    sim = simulate(cfg)
    df = sim.df
    d = df[df["train"]].copy()
    rows = []

    def record(name, est, se):
        rows.append(dict(scenario=scen, seed=seed, model=name, est=est, se=se,
                         lo=est - 1.96 * se, hi=est + 1.96 * se))

    for name, kw in [("glm", {}), ("country", dict(country=True)),
                     ("field", dict(field=True)), ("field+country", dict(field=True, country=True))]:
        m = glmm(d, sim, COLS, **kw)
        est, se = m.fixed_summary()["Z0"]
        record(name, est, se)

    # spatial+: residualise the covariate on low-frequency spatial basis
    E = sim.lattice.smooth_basis(10)
    z_cell = sim.countries["Z0"].to_numpy()[sim.cells["country"].to_numpy()]
    beta = np.linalg.lstsq(E, z_cell, rcond=None)[0]
    z_res = z_cell - E @ beta
    d["Z0"] = z_res[d["cell"].to_numpy()]
    m = glmm(d, sim, COLS, field=True, country=True)
    est, se = m.fixed_summary()["Z0"]
    record("spatial+", est, se)

    # XGBoost + SHAP slope, block bootstrap as in the paper
    d = df[df["train"]]
    rng = np.random.default_rng(seed)
    sub = rng.random(len(d)) < 0.3                     # fixed training subsample
    dsub = d[sub]
    X, y = dsub[XGB_FEATURES].to_numpy(float), dsub["y"].to_numpy()
    j = XGB_FEATURES.index("Z0")
    eval_idx = rng.choice(len(d), 30000, replace=False)
    Xe = d[XGB_FEATURES].to_numpy(float)[eval_idx]
    slope = lambda mod: bo.shap_slope_by_group(bo.shap_values(mod, Xe)[:, j], Xe[:, j],
                                               np.zeros(len(Xe)))[0.0]
    # as in the original analysis: sqrt(n0/n1) class weight; trees by early stopping
    # on the last 26 training weeks (the class weight only shifts the intercept,
    # so SHAP values of Z0 are unaffected)
    val = (dsub["week"] >= sim.cfg.T_train - 26).to_numpy()
    m0, spw = bo.fit_xgb_like_original(X, y, val, seed=seed)
    rounds = m0.num_boosted_rounds()
    point = slope(m0)
    # the same procedure on all training rows (no subsample, no bootstrap); uses no
    # random numbers from rng, so the bootstrap draws below are unchanged
    val_all = (d["week"] >= sim.cfg.T_train - 26).to_numpy()
    m_all, _ = bo.fit_xgb_like_original(d[XGB_FEATURES].to_numpy(float), d["y"].to_numpy(),
                                        val_all, seed=seed)
    point_all = slope(m_all)
    blk = bo.spatial_blocks(dsub["lon"], dsub["lat"], BLOCK_DEG)
    ub = np.unique(blk)
    boots = []
    for b in range(B_BOOT):
        cnt = np.bincount(rng.choice(len(ub), len(ub)), minlength=len(ub))
        wts = cnt[np.searchsorted(ub, blk)].astype(float)
        keep = wts > 0
        mod = bo.fit_xgb(X[keep], y[keep], weight=wts[keep], seed=seed + b, n_rounds=rounds,
                         params={"scale_pos_weight": spw})
        boots.append(slope(mod))
    boots = np.array(boots)
    # as in the paper, the reported value is the median over bootstrap replicates
    rows.append(dict(scenario=scen, seed=seed, model="xgb_shap", est=float(np.median(boots)),
                     est_sub=point, est_full=point_all,
                     se=boots.std(), lo=np.quantile(boots, 0.025), hi=np.quantile(boots, 0.975),
                     iqr_lo=np.quantile(boots, 0.25), iqr_hi=np.quantile(boots, 0.75),
                     n_blocks=int(len(ub)), n_boot=int(B_BOOT)))
    return rows


if __name__ == "__main__":
    args = [(s, seed) for s in SCENARIOS for seed in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    if os.environ.get("SHARD"):
        sys.exit(0)
    res["truth"] = res["scenario"].map({k: v["beta_nat"] for k, v in SCENARIOS.items()})
    res.to_csv(os.path.join(RESULTS, "s2_raw.csv"), index=False)
    res["covers"] = (res["lo"] <= res["truth"]) & (res["truth"] <= res["hi"])
    res["rejects_zero"] = (res["lo"] > 0) | (res["hi"] < 0)
    res["iqr_excludes_zero"] = (res["iqr_lo"] > 0) | (res["iqr_hi"] < 0)
    order = ["glm", "country", "field", "field+country", "spatial+", "xgb_shap"]
    tab = (res.groupby(["scenario", "model"])
           .agg(mean_est=("est", "mean"), emp_sd=("est", "std"), mean_se=("se", "mean"),
                coverage95=("covers", "mean"), reject_zero=("rejects_zero", "mean"),
                iqr_excl_zero=("iqr_excludes_zero", "mean"),
                mean_est_sub=("est_sub", "mean"), mean_est_full=("est_full", "mean"),
                emp_sd_full=("est_full", "std"), n_reps=("est", "size"))
           .reset_index())
    # Monte Carlo standard errors (binomial for proportions, emp. SD / sqrt(R) for means)
    R = tab["n_reps"]
    tab["mc_se_mean"] = tab["emp_sd"] / np.sqrt(R)
    for c in ["coverage95", "reject_zero", "iqr_excl_zero"]:
        tab[f"mc_se_{c}"] = np.sqrt(tab[c] * (1 - tab[c]) / R)
    tab["model"] = pd.Categorical(tab["model"], order)
    tab = tab.sort_values(["scenario", "model"])
    tab.to_csv(os.path.join(RESULTS, "s2_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
