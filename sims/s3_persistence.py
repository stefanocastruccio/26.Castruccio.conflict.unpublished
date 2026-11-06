"""S3 -- 'Recent history dominates': state dependence or persistent heterogeneity?

The original analysis finds that 4-week event counts are the most important
predictors. Autocorrelation in Y arises both when events truly beget events (state
dependence, delta > 0) and when some places are persistently riskier (latent
heterogeneity w(s) and a slowly varying space-time field v(s,t)) with no contagion.

Truths (224 cells, 104 training weeks; fields from the exact continuous Matern process):
  state_dep           delta = 1.0, weak field (sd 0.3)
  heterogeneity       delta = 0,   strong field (sd 1.3) + AR(1) space-time field (sd 0.7, phi 0.9)
  mixed               delta = 0.5, field (sd 1.0) + AR(1) space-time field (sd 0.5, phi 0.9)
  heterogeneity_slow  as heterogeneity, with phi = 0.97 per week
  mixed_slow          as mixed, with phi = 0.97 per week

Estimators of delta (coefficient of log1p(4-week count)):
  lag_only        logistic regression with lags
  lag+field       + static Matern field
  lag+field+st    + AR(1) x Matern space-time field (4-week time steps)
  lag+field+st1   + AR(1) x Matern space-time field at weekly steps (ST_WEEKLY=1;
                  matches the data-generating process; slower)
  lag+field+st8   + AR(1) x Matern space-time field at 8-week steps (the resolution
                  used in the application)
and the XGBoost share of mean |SHAP| attributed to the lag feature (unweighted
XGBoost, 300 trees; not the original procedure).

Parts (S3_PART), each with its own checkpoint so earlier results are reused:
  base   all fits except the 8-week one, for every truth (s3_ckpt.jsonl; the keys of
         the original three truths are unchanged)
  d8     the 8-week fit only, for every truth (s3_d8_ckpt.jsonl)
  all    both (default); the merge run uses this and writes the summaries
"""
from common import *  # noqa: F401,F403

from sc import boosting as bo
from sc.simulate import SimConfig, simulate

CKPT = "s3_ckpt.jsonl"
N_REPS = int(os.environ.get("N_REPS", 20))
FIT_ST = os.environ.get("FIT_ST", "1") == "1"
TRUTHS = {"state_dep": dict(delta=1.0, field_sd=0.3, st_sd=0.0),
          "heterogeneity": dict(delta=0.0, field_sd=1.3, st_sd=0.7, st_rho=0.9),
          "mixed": dict(delta=0.5, field_sd=1.0, st_sd=0.5, st_rho=0.9),
          "heterogeneity_slow": dict(delta=0.0, field_sd=1.3, st_sd=0.7, st_rho=0.97),
          "mixed_slow": dict(delta=0.5, field_sd=1.0, st_sd=0.5, st_rho=0.97)}
S3_PART = os.environ.get("S3_PART", "all")
ST_WEEKLY = os.environ.get("ST_WEEKLY", "0") == "1"
COLS = ["pop", "log_lag4", "temp_anom", "prec_anom"]
T_TRAIN = 104
ST_STEP = 4   # space-time field resolved at 4-week steps (keeps the sparse LU cheap)
ST_STEP8 = 8  # resolution of the space-time field in the application


def one(arg):
    # arg is (truth, seed) for the base part and (truth, seed, "d8") for the 8-week fit
    truth, seed = arg[0], arg[1]
    only_d8 = len(arg) > 2 and arg[2] == "d8"
    cfg = SimConfig(seed=seed, nx=16, ny=14, n_countries=6, T_train=T_TRAIN, T_test=26,
                    beta0=-4.0, **TRUTHS[truth])
    sim = simulate(cfg)
    d0 = sim.df[sim.df["train"]].copy()
    d = d0.copy()
    d["week"] = d["week"] // ST_STEP          # only used by the space-time block
    rows = []
    if only_d8:
        d8 = d0.copy()
        d8["week"] = d8["week"] // ST_STEP8
        m = glmm(d8, sim, COLS, field=True, spacetime=True, T=-(-T_TRAIN // ST_STEP8))
        est, se = m.fixed_summary()["log_lag4"]
        return [dict(truth=truth, seed=seed, model="lag+field+st8", delta=TRUTHS[truth]["delta"],
                     est=est, se=se, log_ml=m.log_ml, seconds=m.fit_seconds,
                     prevalence=float(d0["y"].mean()))]
    specs = [("lag_only", {}), ("lag+field", dict(field=True))]
    if FIT_ST:
        specs.append(("lag+field+st", dict(field=True, spacetime=True, T=T_TRAIN // ST_STEP)))
    if ST_WEEKLY:
        specs.append(("lag+field+st1", dict(field=True, spacetime=True, T=T_TRAIN)))
    for name, kw in specs:
        m = glmm(d0 if name == "lag+field+st1" else d, sim, COLS, **kw)
        est, se = m.fixed_summary()["log_lag4"]
        rows.append(dict(truth=truth, seed=seed, model=name, delta=TRUTHS[truth]["delta"],
                         est=est, se=se, log_ml=m.log_ml, seconds=m.fit_seconds,
                         prevalence=float(d["y"].mean())))
    X = d[XGB_FEATURES].to_numpy(float)
    mod = bo.fit_xgb(X, d["y"].to_numpy(), seed=seed)
    S = np.abs(bo.shap_values(mod, X[::5])[:, :-1]).mean(0)
    rows.append(dict(truth=truth, seed=seed, model="xgb_lag_shap_share",
                     delta=TRUTHS[truth]["delta"],
                     est=float(S[XGB_FEATURES.index("lag4")] / S.sum()), se=np.nan,
                     prevalence=float(d["y"].mean())))
    return rows


if __name__ == "__main__":
    out = []
    if S3_PART in ("base", "all"):
        args = [(t, s) for t in TRUTHS for s in range(1, N_REPS + 1)]
        out += run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT))
    if S3_PART in ("d8", "all"):
        args = [(t, s, "d8") for t in TRUTHS for s in range(1, N_REPS + 1)]
        out += run_parallel(one, args, checkpoint=os.path.join(RESULTS, "s3_d8_ckpt.jsonl"))
    if os.environ.get("SHARD") or S3_PART != "all":
        sys.exit(0)
    res = pd.DataFrame([r for rows in out for r in rows])
    res.to_csv(os.path.join(RESULTS, "s3_raw.csv"), index=False)
    res["covers"] = (res["est"] - 1.96 * res["se"] <= res["delta"]) & (res["delta"] <= res["est"] + 1.96 * res["se"])
    tab = (res.groupby(["truth", "model"])
           .agg(true_delta=("delta", "first"), mean_est=("est", "mean"), emp_sd=("est", "std"),
                mean_se=("se", "mean"), coverage95=("covers", "mean"),
                mean_seconds=("seconds", "mean"), prevalence=("prevalence", "mean"),
                n_reps=("est", "size"))
           .reset_index())
    # Monte Carlo standard errors: of the mean estimate (emp. SD / sqrt(R)), of the
    # coverage (binomial), and the ratio emp. SD / mean SE with the approximate
    # relative SE 1 / sqrt(2 (R - 1)) of an empirical SD
    R = tab["n_reps"]
    tab["mc_se_mean"] = tab["emp_sd"] / np.sqrt(R)
    tab["mc_se_coverage"] = np.sqrt(tab["coverage95"] * (1 - tab["coverage95"]) / R)
    tab["sd_se_ratio"] = tab["emp_sd"] / tab["mean_se"]
    tab["mc_se_sd_se_ratio"] = tab["sd_se_ratio"] / np.sqrt(2 * (R - 1))
    tab.loc[tab["model"] == "xgb_lag_shap_share", ["true_delta", "coverage95", "mean_se", "mc_se_coverage",
                                                 "sd_se_ratio", "mc_se_sd_se_ratio"]] = np.nan
    tab.to_csv(os.path.join(RESULTS, "s3_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
