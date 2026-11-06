"""S4 -- Biome-specific climate effects: spatially varying coefficients vs SHAP-by-biome.

Truth: the temperature-anomaly effect is beta(s) = gamma_{biome(s)} + eta(s), with
gamma = (-0.3, -0.1, 0.1, 0.3) from cold to tropical biomes and eta a Matern
field (sd 0.15, range 600 km), plus the usual residual field, country effects
and lags.

Estimators:
  biome_nofield   fixed biome-specific slopes, no latent effects
  biome_field     fixed biome-specific slopes + field + country
  svc             biome_field + SVC field eta(s) on temp_anom
  xgb             per-biome slope of SHAP(temp) on temp (point estimate from the
                  full model, interval from the paper's 5-deg block bootstrap)
  xgb_median      the paper's Fig. 3 statistic: median SHAP(temp) within biome
"""
from common import *  # noqa: F401,F403

from sc import boosting as bo
from sc.simulate import SimConfig, simulate

N_REPS = int(os.environ.get("N_REPS", 30))
B_BOOT = int(os.environ.get("B_BOOT", 10))
GAMMA = (-0.3, -0.1, 0.1, 0.3)
BASE = ["pop", "log_lag4", "prec_anom"]
FACT = [("biome", None), ("biome", "temp_anom")]


def one(seed):
    sim = simulate(SimConfig(seed=seed, svc_sd=0.15, gamma_temp=GAMMA, field_sd=0.8))
    df = sim.df
    d = df[df["train"]]
    rows, extra = [], {"seed": seed}
    names = [f"temp_anom:biome[{b}]" for b in range(4)]
    # realised biome-average effect gamma_b + mean of eta(s) over the biome's cells:
    # the estimand of fixed biome slopes; gamma_b is the estimand of the SVC hyper-mean
    realized = sim.cells.groupby("biome")["beta_temp"].mean().to_dict()

    def rec(model, b, est, se=None, lo=None, hi=None):
        if se is not None:
            lo, hi = est - 1.96 * se, est + 1.96 * se
        rows.append(dict(seed=seed, model=model, biome=b, truth=GAMMA[b],
                         truth_realized=realized[b], est=est, lo=lo, hi=hi))

    for model, kw in [("biome_nofield", {}), ("biome_field", dict(field=True, country=True)),
                      ("svc", dict(field=True, country=True, svc_col="temp_anom"))]:
        m = glmm(d, sim, BASE, factors=FACT, **kw)
        fs = m.fixed_summary()
        for b in range(4):
            rec(model, b, *fs[names[b]])
        if model == "svc":
            k = block_index(m, "svc")
            biome = sim.cells["biome"].to_numpy()
            bhat = np.array([fs[names[b]][0] for b in biome]) + m.block_mean(k)
            truth = sim.cells["beta_temp"].to_numpy()
            extra["svc_corr_beta_s"] = float(np.corrcoef(bhat, truth)[0, 1])
            extra["svc_rmse_beta_s"] = float(np.sqrt(np.mean((bhat - truth) ** 2)))
            extra["svc_sd_hat"] = m.hyper()["svc.sd"]
            extra["svc_range_hat"] = m.hyper()["svc.range_km"]
            # biome-only fit for comparison of beta(s) recovery
            gam = np.array([fs[names[b]][0] for b in biome])
            extra["biome_only_rmse_beta_s"] = float(np.sqrt(np.mean((gam - truth) ** 2)))

    # XGBoost: SHAP slope per biome + block bootstrap; and the paper's median-SHAP statistic
    rng = np.random.default_rng(seed)
    X = d[XGB_FEATURES].to_numpy(float); y = d["y"].to_numpy()
    j = XGB_FEATURES.index("temp_anom")
    ev = rng.choice(len(d), 40000, replace=False)
    Xe, be = X[ev], d["biome"].to_numpy()[ev]

    def stats(mod):
        S = bo.shap_values(mod, Xe)[:, j]
        sl = bo.shap_slope_by_group(S, Xe[:, j], be)
        med = {b: float(np.median(S[be == b])) for b in range(4)}
        return sl, med

    sub = rng.random(len(d)) < 0.3
    blk = bo.spatial_blocks(d["lon"], d["lat"], 5.0)
    ub = np.unique(blk)
    boots_sl, boots_med = [], []
    for b in range(B_BOOT):
        cnt = np.bincount(rng.choice(len(ub), len(ub)), minlength=len(ub))
        wts = cnt[np.searchsorted(ub, blk)].astype(float) * sub
        keep = wts > 0
        s_, m_ = stats(bo.fit_xgb(X[keep], y[keep], weight=wts[keep], seed=seed + b))
        boots_sl.append([s_[k] for k in range(4)]); boots_med.append([m_[k] for k in range(4)])
    boots_sl, boots_med = np.array(boots_sl), np.array(boots_med)
    # as in the paper: median over bootstrap replicates; 95% band for slopes,
    # interquartile band (the paper's Fig. 3 bars) for the median-SHAP statistic
    for b in range(4):
        rec("xgb_slope", b, float(np.median(boots_sl[:, b])), lo=np.quantile(boots_sl[:, b], .025),
            hi=np.quantile(boots_sl[:, b], .975))
        rec("xgb_median", b, float(np.median(boots_med[:, b])), lo=np.quantile(boots_med[:, b], .25),
            hi=np.quantile(boots_med[:, b], .75))
    return rows, extra


if __name__ == "__main__":
    out = run_parallel(one, range(1, N_REPS + 1), checkpoint=os.path.join(RESULTS, "s4_ckpt.jsonl"))
    res = pd.DataFrame([r for rows, _ in out for r in rows])
    extra = pd.DataFrame([e for _, e in out])
    res.to_csv(os.path.join(RESULTS, "s4_raw.csv"), index=False)
    extra.to_csv(os.path.join(RESULTS, "s4_extra.csv"), index=False)
    res["covers"] = (res["lo"] <= res["truth"]) & (res["truth"] <= res["hi"])
    res["covers_realized"] = ((res["lo"] <= res["truth_realized"])
                              & (res["truth_realized"] <= res["hi"]))
    res["sign_ok"] = np.sign(res["est"]) == np.sign(res["truth"])
    res["excl_zero_right_sign"] = ((res["lo"] > 0) & (res["truth"] > 0)) | ((res["hi"] < 0) & (res["truth"] < 0))
    tab = (res.groupby(["model", "biome"])
           .agg(truth=("truth", "first"), mean_realized=("truth_realized", "mean"),
                mean_est=("est", "mean"), emp_sd=("est", "std"),
                coverage95=("covers", "mean"), coverage95_realized=("covers_realized", "mean"), sign_correct=("sign_ok", "mean"),
                detects=("excl_zero_right_sign", "mean")).reset_index())
    tab.to_csv(os.path.join(RESULTS, "s4_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    print(extra.describe().loc[["mean", "std"]].T.round(3).to_string())
