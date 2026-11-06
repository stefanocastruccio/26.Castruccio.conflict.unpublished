"""S1 -- Residual spatial structure after boosting, and what modelling it buys.

Truth: default SimConfig (fields drawn from the exact continuous Matern process, not
from the fitted lattice; country effects; lags). Train on weeks 0-103 (the last 26
training weeks serve for early stopping), evaluate on weeks 104-155 at the same cells.

Predictors compared on the test period:
  oracle              true probabilities
  xgb_original        XGBoost as in the original analysis (sqrt class weight, early
                      stopping on the training data), prior-corrected log-odds
  xgb_te              same + the cell's leave-one-period-out training event rate
                      (a location-aware boosting baseline)
  xgb_crossfit        unweighted XGBoost, out-of-fold with whole countries as folds
  hybrid_iid          xgb_crossfit offset + country effects + iid cell effects
  hybrid_field        xgb_crossfit offset + country effects + Matern field
  glmm                parametric model with country effects + field
  persistence         ranking by the past 4-week event count
"""
from common import *  # noqa: F401,F403

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import average_precision_score

from sc import boosting as bo, diagnostics as dg
from sc.laplace import prob_from_eta
from sc.simulate import SimConfig, simulate

CKPT = "s1_ckpt.jsonl"
N_REPS = int(os.environ.get("N_REPS", 20))


def latent_fit(d_tr, d_te, sim, off_tr, off_te, kind):
    n, nC = sim.lattice.n, sim.cfg.n_countries
    blocks = [FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country")]
    D_tr = [np.ones((len(d_tr), 1)), de.index_design(d_tr["country"], nC)]
    D_te = [np.ones((len(d_te), 1)), de.index_design(d_te["country"], nC)]
    if kind == "field":
        blocks.append(MaternBlock(sim.lattice, name="field"))
    else:
        blocks.append(IIDBlock(n, name="cell"))
    D_tr.append(de.index_design(d_tr["cell"], n)); D_te.append(de.index_design(d_te["cell"], n))
    m = LatentGaussianModel(d_tr["y"].to_numpy(), blocks, D_tr, offset=off_tr).fit()
    eta, sd = m.predict_eta(D_te, off_te, n_samples=100)
    return m, prob_from_eta(eta, sd)


def one(seed, return_objects=False):
    sim = simulate(SimConfig(seed=seed))
    df, n = sim.df, sim.lattice.n
    tr = df["train"].to_numpy(); te = ~tr
    val = (df["week"] >= sim.cfg.T_train - 26).to_numpy() & tr
    y = df["y"].to_numpy()
    y_te = y[te]
    X = df[XGB_FEATURES].to_numpy(float)
    period = (df["week"] // 52).to_numpy()
    out = {"seed": seed, "prevalence_test": float(y_te.mean())}
    preds = {"oracle": df.loc[te, "p_true"].to_numpy()}

    m, spw = bo.fit_xgb_like_original(X[tr], y[tr], val[tr], seed=seed)
    lo = bo.predict_logit(m, X) - np.log(spw)
    preds["xgb_original"] = expit(lo[te])
    te_feat = bo.cell_rate_feature(df["cell"].to_numpy(), period, y, tr)
    Xte = np.c_[X, te_feat]
    m2, spw2 = bo.fit_xgb_like_original(Xte[tr], y[tr], val[tr], seed=seed)
    preds["xgb_te"] = expit(bo.predict_logit(m2, Xte[te]) - np.log(spw2))
    off = bo.crossfit_offset_groups(X, y, df["country"].to_numpy(), tr, val, seed=seed)
    preds["xgb_crossfit"] = expit(off[te])

    for nm, l in [("xgb_original", lo), ("xgb_crossfit", off)]:
        r = dg.cell_pearson(df[tr], expit(l[tr])).to_numpy()
        mi = dg.morans_i(r, sim.lattice.adjacency, n_perm=99)
        out[f"moran_{nm}"] = mi["I"]

    d_tr, d_te = df[tr], df[te]
    for kind in ("iid", "field"):
        m3, p = latent_fit(d_tr, d_te, sim, off[tr], off[te], kind)
        preds[f"hybrid_{kind}"] = p
        if kind == "field":
            k = block_index(m3, "field")
            out["corr_w"] = float(np.corrcoef(m3.block_mean(k), sim.cells["w"])[0, 1])
            out["range"] = m3.hyper()["field.range_km"]; out["sd"] = m3.hyper()["field.sd"]
            m_field = m3
    cols = ["pop", "log_lag4", "temp_anom", "prec_anom", "Z0"]
    mg = glmm(d_tr, sim, cols, field=True, country=True)
    Xg, _ = de.fixed_design(d_te, cols)
    eta, sd = mg.predict_eta([Xg, de.index_design(d_te["cell"], n),
                              de.index_design(d_te["country"], sim.cfg.n_countries)], None, n_samples=100)
    preds["glmm"] = prob_from_eta(eta, sd)

    ll = lambda p: -(y_te * np.log(np.clip(p, 1e-12, 1)) + (1 - y_te) * np.log(np.clip(1 - p, 1e-12, 1)))
    for nm, p in preds.items():
        for k, v in dg.scores(y_te, p).items():
            out[f"{nm}.{k}"] = v
        out[f"{nm}.logscore_minus_xgb_original"] = float(np.mean(ll(p) - ll(preds["xgb_original"])))
    out["persistence.auc_pr"] = float(average_precision_score(y_te, d_te["lag4"].to_numpy()))
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
    ext = [L.lon_c[0] - 0.25, L.lon_c[-1] + 0.25, L.lat_c[0] - 0.25, L.lat_c[-1] + 0.25]
    cg = grid(sim.cells["country"].to_numpy().astype(float))
    # country borders: segments between horizontally / vertically adjacent cells of different countries
    segs = []
    for i in range(cg.shape[0]):
        for j in range(cg.shape[1]):
            x, y = L.lon_c[j], L.lat_c[i]
            if j + 1 < cg.shape[1] and cg[i, j] != cg[i, j + 1]:
                segs.append([(x + 0.25, y - 0.25), (x + 0.25, y + 0.25)])
            if i + 1 < cg.shape[0] and cg[i, j] != cg[i + 1, j]:
                segs.append([(x - 0.25, y + 0.25), (x + 0.25, y + 0.25)])
    from matplotlib.collections import LineCollection
    for a, v, t in [(ax[0], sim.cells["w"].to_numpy(), "true field $w(s)$"),
                    (ax[1], r, "cross-fitted XGBoost: cell residual"),
                    (ax[2], m.block_mean(2), "posterior mean of $w(s)$, hybrid model")]:
        lim = np.nanquantile(np.abs(v), 0.98)
        im = a.imshow(grid(v), origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set_title(t, fontsize=10); plt.colorbar(im, ax=a, shrink=0.8)
        a.add_collection(LineCollection(segs, colors="k", linewidths=0.6))
        a.set_xlabel("longitude"); a.set_ylabel("latitude")
    vg = dg.variogram(r, L.lon, L.lat, np.arange(0, 1600, 100))
    ax[3].plot(vg.dist_km, vg.gamma, "o-")
    ax[3].set_xlabel("great-circle distance (km)"); ax[3].set_ylabel("semivariance")
    ax[3].set_title("variogram of XGBoost cell residuals", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "s1_maps.png"), dpi=130)


if __name__ == "__main__":
    res = pd.DataFrame(run_parallel(one, range(1, N_REPS + 1), checkpoint=os.path.join(RESULTS, CKPT)))
    if os.environ.get("SHARD"):
        sys.exit(0)
    res.to_csv(os.path.join(RESULTS, "s1_raw.csv"), index=False)
    models = ["oracle", "xgb_original", "xgb_te", "xgb_crossfit", "hybrid_iid", "hybrid_field", "glmm"]
    tab = pd.DataFrame({m: {k: f"{res[f'{m}.{k}'].mean():.4f} ({res[f'{m}.{k}'].std():.4f})"
                            for k in ["log_score", "brier", "auc_pr", "auc_roc"]} for m in models}).T
    tab.loc["persistence", "auc_pr"] = f"{res['persistence.auc_pr'].mean():.4f}"
    tab.to_csv(os.path.join(RESULTS, "s1_scores.csv"))
    print(tab.to_string())
    figure()
