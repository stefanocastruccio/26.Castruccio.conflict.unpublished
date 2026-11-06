"""S3 -- 'Recent history dominates': state dependence or persistent heterogeneity?

The paper finds that 4-week event counts are by far the most important predictors.
That pattern arises both when events truly beget events (state dependence, delta > 0)
and when some places are simply persistently riskier (latent heterogeneity w(s) and
a slowly varying space-time field v(s,t)) with no contagion at all.

Truths (224 cells, 104 training weeks; fields from the exact continuous Matern process):
  state_dep      delta = 1.0, weak field (sd 0.3)
  heterogeneity  delta = 0,   strong field (sd 1.3) + AR(1) space-time field (sd 0.7, rho 0.9)
  mixed          delta = 0.5, field (sd 1.0) + AR(1) space-time field (sd 0.5, rho 0.9)

Estimators of delta (coefficient of log1p(4-week count)):
  lag_only        logistic regression with lags
  lag+field       + static Matern field
  lag+field+st    + AR(1) x Matern space-time field (4-week time steps)
  lag+field+st1   + AR(1) x Matern space-time field at weekly steps (ST_WEEKLY=1;
                  matches the data-generating process; slower)
and the XGBoost share of mean |SHAP| attributed to the lag feature.
"""
from common import *  # noqa: F401,F403

from sc import boosting as bo
from sc.simulate import SimConfig, simulate

CKPT = "s3_ckpt.jsonl"
N_REPS = int(os.environ.get("N_REPS", 20))
FIT_ST = os.environ.get("FIT_ST", "1") == "1"
TRUTHS = {"state_dep": dict(delta=1.0, field_sd=0.3, st_sd=0.0),
          "heterogeneity": dict(delta=0.0, field_sd=1.3, st_sd=0.7, st_rho=0.9),
          "mixed": dict(delta=0.5, field_sd=1.0, st_sd=0.5, st_rho=0.9)}
ST_WEEKLY = os.environ.get("ST_WEEKLY", "0") == "1"
COLS = ["pop", "log_lag4", "temp_anom", "prec_anom"]
T_TRAIN = 104
ST_STEP = 4   # space-time field resolved at 4-week steps (keeps the sparse LU cheap)


def one(arg):
    truth, seed = arg
    cfg = SimConfig(seed=seed, nx=16, ny=14, n_countries=6, T_train=T_TRAIN, T_test=26,
                    beta0=-4.0, **TRUTHS[truth])
    sim = simulate(cfg)
    d0 = sim.df[sim.df["train"]].copy()
    d = d0.copy()
    d["week"] = d["week"] // ST_STEP          # only used by the space-time block
    rows = []
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
    args = [(t, s) for t in TRUTHS for s in range(1, N_REPS + 1)]
    res = pd.DataFrame([r for rows in run_parallel(one, args, checkpoint=os.path.join(RESULTS, CKPT)) for r in rows])
    if os.environ.get("SHARD"):
        sys.exit(0)
    res.to_csv(os.path.join(RESULTS, "s3_raw.csv"), index=False)
    res["covers"] = (res["est"] - 1.96 * res["se"] <= res["delta"]) & (res["delta"] <= res["est"] + 1.96 * res["se"])
    tab = (res.groupby(["truth", "model"])
           .agg(true_delta=("delta", "first"), mean_est=("est", "mean"), emp_sd=("est", "std"),
                mean_se=("se", "mean"), coverage95=("covers", "mean"),
                mean_seconds=("seconds", "mean"), prevalence=("prevalence", "mean"))
           .reset_index())
    tab.loc[tab["model"] == "xgb_lag_shap_share", ["true_delta", "coverage95", "mean_se"]] = np.nan
    tab.to_csv(os.path.join(RESULTS, "s3_summary.csv"), index=False)
    print(tab.round(3).to_string(index=False))
