"""S1 -- Is there spatial structure left in boosting residuals, and does a latent
field on top of a cross-fitted XGBoost offset recover it and improve forecasts?

Truth: default SimConfig (Matern field sd 1, range 400 km; country effects; lags).
Train on weeks 0-103, evaluate on weeks 104-155 (same cells), as in the paper.
"""
from common import *  # noqa: F401,F403

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit

from sc import boosting as bo, diagnostics as dg
from sc.laplace import prob_from_eta
from sc.simulate import SimConfig, simulate

N_REPS = int(os.environ.get("N_REPS", 20))


def one(seed, return_objects=False):
    sim = simulate(SimConfig(seed=seed))
    df, n = sim.df, sim.lattice.n
    tr = df["train"].to_numpy(); te = ~tr
    y_te = df.loc[te, "y"].to_numpy()
    X = df[XGB_FEATURES].to_numpy(float)
    out = {"seed": seed}

    # (a) paper setting: one XGBoost on all training rows
    full = bo.fit_xgb(X[tr], df.loc[tr, "y"].to_numpy(), seed=seed)
    lf = bo.predict_logit(full, X)
    # (b) spatially cross-fitted offset (5 deg blocks)
    blk = bo.spatial_blocks(df["lon"], df["lat"], 5.0)
    off = bo.crossfit_offset(df, XGB_FEATURES, tr, blk, seed=seed)

    preds = {"oracle": df.loc[te, "p_true"].to_numpy(),
             "xgb_full": expit(lf[te]), "xgb_crossfit": expit(off[te])}
    for nm, l in [("xgb_full", lf), ("xgb_crossfit", off)]:
        r = dg.cell_pearson(df[tr], expit(l[tr])).to_numpy()
        out[f"moran_{nm}"] = dg.morans_i(r, sim.lattice.adjacency, n_perm=99)["I"]

    d_tr, d_te = df[tr], df[te]
    ones_tr, ones_te = np.ones((tr.sum(), 1)), np.ones((te.sum(), 1))
    # (c) cross-fitted offset + Matern field (+ country)
    for nm, country in [("xgb+field", False), ("xgb+field+country", True)]:
        m = glmm(d_tr, sim, [], field=True, country=country, offset=off[tr])
        D = [ones_te, de.index_design(d_te["cell"], n)]
        if country:
            D.append(de.index_design(d_te["country"], sim.cfg.n_countries))
        eta, sd = m.predict_eta(D, off[te], n_samples=100)
        preds[nm] = prob_from_eta(eta, sd)
        k = block_index(m, "field")
        out[f"corr_w_{nm}"] = float(np.corrcoef(m.block_mean(k), sim.cells["w"])[0, 1])
        out[f"range_{nm}"] = m.hyper()["field.range_km"]
        out[f"sd_{nm}"] = m.hyper()["field.sd"]
        if not country:
            m_field = m
    # (d) fully parametric spatial GLMM
    cols = ["pop", "log_lag4", "temp_anom", "prec_anom", "Z0"]
    m = glmm(d_tr, sim, cols, field=True, country=True)
    Xte, _ = de.fixed_design(d_te, cols)
    eta, sd = m.predict_eta([Xte, de.index_design(d_te["cell"], n),
                             de.index_design(d_te["country"], sim.cfg.n_countries)],
                            None, n_samples=100)
    preds["glmm_field+country"] = prob_from_eta(eta, sd)

    for nm, p in preds.items():
        for k, v in dg.scores(y_te, p).items():
            out[f"{nm}.{k}"] = v
    out["persistence.auc_pr"] = dg.scores(y_te, d_te["lag4"].to_numpy()
                                          + 1e-6 * np.random.default_rng(0).random(te.sum()))["auc_pr"]
    if return_objects:
        return out, sim, off, m_field
    return out


def figure(seed=1):
    out, sim, off, m = one(seed, return_objects=True)
    df, L = sim.df, sim.lattice
    tr = df["train"].to_numpy()
    r = dg.cell_pearson(df[tr], expit(off[tr])).to_numpy()

    def grid(v):
        g = np.full(L.mask.shape, np.nan); g[L.row, L.col] = v; return g

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    ext = [L.lon_c[0], L.lon_c[-1], L.lat_c[0], L.lat_c[-1]]
    for a, v, t in [(ax[0], sim.cells["w"], "true residual field w(s)"),
                    (ax[1], r, "cross-fitted XGBoost: cell Pearson residual"),
                    (ax[2], m.block_mean(1), "estimated field (XGB offset + Matern)")]:
        im = a.imshow(grid(v), origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-np.nanmax(np.abs(v)), vmax=np.nanmax(np.abs(v)))
        a.set_title(t, fontsize=10); plt.colorbar(im, ax=a, shrink=0.8)
        # country borders
        a.contour(L.lon_c, L.lat_c, grid(sim.cells["country"]), levels=np.arange(20) + .5,
                  colors="k", linewidths=0.4)
    vg = dg.variogram(r, L.lon, L.lat, np.arange(0, 1600, 100))
    ax[3].plot(vg.dist_km, vg.gamma, "o-")
    ax[3].set_xlabel("great-circle distance (km)"); ax[3].set_ylabel("semivariance")
    ax[3].set_title("variogram of XGBoost cell residuals", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "s1_maps.png"), dpi=130)


if __name__ == "__main__":
    res = pd.DataFrame(run_parallel(one, range(1, N_REPS + 1), checkpoint=os.path.join(RESULTS, "s1_ckpt.jsonl")))
    res.to_csv(os.path.join(RESULTS, "s1_raw.csv"), index=False)
    models = ["oracle", "xgb_full", "xgb_crossfit", "xgb+field", "xgb+field+country",
              "glmm_field+country"]
    tab = pd.DataFrame({m: {k: f"{res[f'{m}.{k}'].mean():.4f} ({res[f'{m}.{k}'].std():.4f})"
                            for k in ["log_score", "brier", "auc_pr", "auc_roc"]} for m in models}).T
    tab.loc["persistence", "auc_pr"] = f"{res['persistence.auc_pr'].mean():.4f}"
    tab.to_csv(os.path.join(RESULTS, "s1_scores.csv"))
    extra = res[[c for c in res if c.startswith(("moran", "corr_w", "range", "sd_"))]]
    extra.describe().loc[["mean", "std"]].T.to_csv(os.path.join(RESULTS, "s1_field.csv"))
    print(tab.to_string()); print(extra.describe().loc[["mean", "std"]].T.to_string())
    figure()
