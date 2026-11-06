"""S4 -- Biome-specific climate effects: spatially varying coefficients vs SHAP-by-biome.

Truth: the temperature-anomaly effect is beta(s) = gamma_{biome(s)} + eta(s), with
gamma = (-0.3, -0.1, 0.1, 0.3) from cold to tropical biomes and eta a Matern
field (sd 0.15, range 600 km), plus the usual residual field, country effects
and lags.

Estimators:
  biome_nofield   fixed biome-specific slopes, no latent effects
  biome_field     fixed biome-specific slopes + field + country
  svc             biome_field + SVC field eta(s) on temp_anom
  xgb_slope       per-biome slope of SHAP(temp_anom) on temp_anom
  xgb_slope_full  the same with the interaction values credited to temp_anom
  xgb_median_raw  the original Fig. 3 statistic: median SHAP(temp_raw) within biome
For the three XGBoost statistics, as in the original procedure, `est` is the median
over B_BOOT spatial block-bootstrap refits (2.5-deg blocks) on a fixed 30% subsample
of the training rows, with 95% percentile bands (IQR bands for xgb_median_raw).
`est_full` is the same statistic from one fit to ALL training rows (the models used
to choose the number of trees), with no subsample and no bootstrap.

For f = g(b) z with b and z independent, the interventional SHAP value of z is
(1/2) (E g(B) + g(b)) (z - E z), so the ordinary SHAP slope in biome b targets
(gbar + gamma_b) / 2 (column shap_target), which equals gamma_b / 2 only when the
cell-weighted mean gbar of gamma is zero. The interaction-credited slope targets
gamma_b in either case.

S4_GAMMA=asym uses gamma = (0, 0.1, 0.2, 0.4) (gbar about 0.175), to show the
difference; its outputs carry the suffix _asym.
"""
from common import *  # noqa: F401,F403

from sc import boosting as bo
from sc.simulate import SimConfig, simulate

N_REPS = int(os.environ.get("N_REPS", 30))
B_BOOT = int(os.environ.get("B_BOOT", 20))
S4_GAMMA = os.environ.get("S4_GAMMA", "default")
GAMMA = {"default": (-0.3, -0.1, 0.1, 0.3), "asym": (0.0, 0.1, 0.2, 0.4)}[S4_GAMMA]
SUFFIX = "" if S4_GAMMA == "default" else f"_{S4_GAMMA}"
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

    gbar = float(np.mean(np.asarray(GAMMA)[sim.cells["biome"].to_numpy()]))

    def rec(model, b, est, se=None, lo=None, hi=None, est_full=np.nan):
        if se is not None:
            lo, hi = est - 1.96 * se, est + 1.96 * se
        rows.append(dict(seed=seed, model=model, biome=b, truth=GAMMA[b],
                         truth_realized=realized[b], est=est, lo=lo, hi=hi,
                         est_full=est_full, shap_target=(gbar + GAMMA[b]) / 2))

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

    # XGBoost, two feature sets (both with the original's sqrt class weight and early
    # stopping on the last 26 training weeks):
    #   anomaly features: per-biome slope of the ordinary SHAP value of temp_anom (for a
    #     pure biome x temperature interaction it targets (gbar + gamma_b) / 2, see the
    #     docstring) and of the interaction-credited contribution (targets gamma_b);
    #   raw temperature (as in the original analysis): median SHAP value of temp_raw
    #     within biome, the original's Fig. 3 statistic.
    rng = np.random.default_rng(seed)
    y = d["y"].to_numpy()
    feats_raw = [("temp_raw" if f == "temp_anom" else f) for f in XGB_FEATURES]
    XA, XR = d[XGB_FEATURES].to_numpy(float), d[feats_raw].to_numpy(float)
    j = XGB_FEATURES.index("temp_anom")
    val = (d["week"] >= sim.cfg.T_train - 26).to_numpy()
    ev = rng.choice(len(d), 20000, replace=False)
    be = d["biome"].to_numpy()[ev]
    mA, spwA = bo.fit_xgb_like_original(XA, y, val, seed=seed)
    mR, spwR = bo.fit_xgb_like_original(XR, y, val, seed=seed)
    rA, rR = mA.num_boosted_rounds(), mR.num_boosted_rounds()

    def stats(modA, modR):
        S = bo.shap_values(modA, XA[ev])[:, j]
        F = bo.shap_full_credit(modA, XA[ev], j)
        SR = bo.shap_values(modR, XR[ev])[:, j]
        sl = bo.shap_slope_by_group(S, XA[ev, j], be)
        fl = bo.shap_slope_by_group(F, XA[ev, j], be)
        med = {b: float(np.median(SR[be == b])) for b in range(4)}
        return [sl[k] for k in range(4)], [fl[k] for k in range(4)], [med[k] for k in range(4)]

    full = stats(mA, mR)   # one fit to all training rows; draws nothing from rng
    sub = rng.random(len(d)) < 0.3
    blk = bo.spatial_blocks(d["lon"], d["lat"], 2.5)
    ub = np.unique(blk)
    boots = []
    for b in range(B_BOOT):
        cnt = np.bincount(rng.choice(len(ub), len(ub)), minlength=len(ub))
        wts = cnt[np.searchsorted(ub, blk)].astype(float) * sub
        keep = wts > 0
        bA = bo.fit_xgb(XA[keep], y[keep], weight=wts[keep], seed=seed + b, n_rounds=rA,
                        params={"scale_pos_weight": spwA})
        bR = bo.fit_xgb(XR[keep], y[keep], weight=wts[keep], seed=seed + b, n_rounds=rR,
                        params={"scale_pos_weight": spwR})
        boots.append(stats(bA, bR))
    boots = np.array(boots)            # (B, 3 statistics, 4 biomes)
    # medians over bootstrap replicates as in the original; 95% bands for slopes,
    # interquartile bands (the original's Fig. 3 bars) for the median-SHAP statistic
    for b in range(4):
        for k, name in enumerate(["xgb_slope", "xgb_slope_full"]):
            v = boots[:, k, b]
            rec(name, b, float(np.median(v)), lo=np.quantile(v, .025), hi=np.quantile(v, .975),
                est_full=float(full[k][b]))
        v = boots[:, 2, b]
        rec("xgb_median_raw", b, float(np.median(v)), lo=np.quantile(v, .25), hi=np.quantile(v, .75),
            est_full=float(full[2][b]))
    return rows, extra


if __name__ == "__main__":
    out = run_parallel(one, range(1, N_REPS + 1), checkpoint=os.path.join(RESULTS, f"s4v2{SUFFIX}_ckpt.jsonl"))
    if os.environ.get("SHARD"):
        sys.exit(0)
    res = pd.DataFrame([r for rows, _ in out for r in rows])
    extra = pd.DataFrame([e for _, e in out])
    res.to_csv(os.path.join(RESULTS, f"s4_raw{SUFFIX}.csv"), index=False)
    extra.to_csv(os.path.join(RESULTS, f"s4_extra{SUFFIX}.csv"), index=False)
    res["covers"] = (res["lo"] <= res["truth"]) & (res["truth"] <= res["hi"])
    res["covers_realized"] = ((res["lo"] <= res["truth_realized"])
                              & (res["truth_realized"] <= res["hi"]))
    res["sign_ok"] = np.sign(res["est"]) == np.sign(res["truth"])
    res["excl_zero_right_sign"] = ((res["lo"] > 0) & (res["truth"] > 0)) | ((res["hi"] < 0) & (res["truth"] < 0))
    tab = (res.groupby(["model", "biome"])
           .agg(truth=("truth", "first"), mean_realized=("truth_realized", "mean"),
                mean_est=("est", "mean"), emp_sd=("est", "std"),
                coverage95=("covers", "mean"), coverage95_realized=("covers_realized", "mean"), sign_correct=("sign_ok", "mean"),
                detects=("excl_zero_right_sign", "mean"),
                shap_target=("shap_target", "mean"), mean_est_full=("est_full", "mean"),
                n_reps=("est", "size")).reset_index())
    # Monte Carlo standard errors (binomial for proportions, emp. SD / sqrt(R) for means)
    R = tab["n_reps"]
    tab["mc_se_mean"] = tab["emp_sd"] / np.sqrt(R)
    for c in ["coverage95", "coverage95_realized", "detects"]:
        tab[f"mc_se_{c}"] = np.sqrt(tab[c] * (1 - tab[c]) / R)
    tab.to_csv(os.path.join(RESULTS, f"s4_summary{SUFFIX}.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    print(extra.describe().loc[["mean", "std"]].T.round(3).to_string())
