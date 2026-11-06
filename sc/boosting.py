"""Gradient boosting pieces: spatially cross-fitted XGBoost offsets, exact TreeSHAP
summaries, and a GPBoost (trees + Gaussian process) wrapper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# hyperparameters reported in the paper (Materials and Methods)
PAPER_PARAMS = dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=30, reg_lambda=5.0, objective="binary:logistic",
                    tree_method="hist", eval_metric="aucpr")


def spatial_blocks(lon, lat, size_deg=5.0):
    """Integer id of the size_deg x size_deg block containing each point."""
    bx = np.floor(np.asarray(lon) / size_deg).astype(int)
    by = np.floor(np.asarray(lat) / size_deg).astype(int)
    _, ids = np.unique(np.c_[bx, by], axis=0, return_inverse=True)
    return ids.ravel()


def fit_xgb(X, y, n_rounds=300, params=None, weight=None, nthread=1, seed=0):
    p = dict(PAPER_PARAMS, **(params or {}))
    p.update(nthread=nthread, seed=seed)
    d = xgb.DMatrix(X, label=y, weight=weight)
    return xgb.train(p, d, num_boost_round=n_rounds)


def predict_logit(model, X):
    return model.predict(xgb.DMatrix(X), output_margin=True)


def shap_values(model, X):
    """Exact TreeSHAP (log-odds scale); last column is the bias term."""
    return model.predict(xgb.DMatrix(X), pred_contribs=True)


def crossfit_offset(df: pd.DataFrame, features, train_mask, block_ids, n_folds=5,
                    n_rounds=300, params=None, nthread=1, seed=0):
    """Out-of-fold XGBoost logits for every row.

    Spatial blocks are assigned to folds; each fold's model is trained on the
    *training-period* rows of the other folds and predicts all rows (train and
    test period) of its own blocks. The offset therefore never sees the
    location it is predicting, so spatially structured residual risk is left
    for the latent field rather than absorbed (and double counted) by trees.
    """
    rng = np.random.default_rng(seed)
    ub = np.unique(block_ids)
    fold_of_block = dict(zip(ub, rng.permutation(len(ub)) % n_folds))
    fold = np.vectorize(fold_of_block.get)(block_ids)
    X = df[features].to_numpy(float); y = df["y"].to_numpy()
    out = np.zeros(len(df))
    for k in range(n_folds):
        tr = train_mask & (fold != k)
        m = fit_xgb(X[tr], y[tr], n_rounds, params, nthread=nthread, seed=seed)
        out[fold == k] = predict_logit(m, X[fold == k])
    return out


def shap_slope_by_group(shap_col, x, groups):
    """Least-squares slope of a feature's SHAP values on the feature, per group.
    A model-agnostic 'effect' summary comparable to a regression coefficient."""
    res = {}
    for g in np.unique(groups):
        m = groups == g
        xc = x[m] - x[m].mean()
        res[g] = float(np.sum(xc * shap_col[m]) / np.sum(xc ** 2))
    return res


def fit_gpboost(df, features, train_mask, coords_cols=("lon", "lat"), group_col="country",
                num_neighbors=20, n_rounds=200, params=None):
    """Joint tree boosting + Vecchia GP + grouped country effect, Bernoulli-logit
    (Sigrist 2022). Coordinates are in degrees here; for global data project to
    3-d Cartesian coordinates first (see README)."""
    import gpboost as gpb
    tr = np.asarray(train_mask)
    X = df.loc[tr, features].to_numpy(float)
    y = df.loc[tr, "y"].to_numpy()
    gp = gpb.GPModel(gp_coords=df.loc[tr, list(coords_cols)].to_numpy(float),
                     group_data=df.loc[tr, group_col].to_numpy(),
                     cov_function="matern", cov_fct_shape=1.5,
                     gp_approx="vecchia", num_neighbors=num_neighbors,
                     likelihood="bernoulli_logit")
    data = gpb.Dataset(X, y)
    p = dict(learning_rate=0.05, max_depth=5, min_data_in_leaf=30, lambda_l2=5.0,
             verbose=0, **(params or {}))
    booster = gpb.train(params=p, train_set=data, gp_model=gp, num_boost_round=n_rounds)
    return booster, gp


# ---------------------------------------------------------------------------
# Helpers mirroring the real-data pipeline (realdata/reproduce_xgb.py, fit_acled.py)
# ---------------------------------------------------------------------------
def fit_xgb_like_original(X, y, val_mask, nthread=1, seed=0, max_rounds=2000):
    """XGBoost as in the original analysis: positives weighted by sqrt(n0/n1), number
    of trees by early stopping on a held-out validation part of the TRAINING data
    (here val_mask), then refitted on all training rows. Returns (model, spw); the
    model's log-odds must be prior-corrected by log(spw) before scoring."""
    p = dict(PAPER_PARAMS, nthread=nthread, seed=seed)
    fit = ~val_mask
    spw = float(np.sqrt((len(y[fit]) - y[fit].sum()) / max(y[fit].sum(), 1)))
    p["scale_pos_weight"] = spw
    m0 = xgb.train(p, xgb.DMatrix(X[fit], label=y[fit]), max_rounds,
                   evals=[(xgb.DMatrix(X[val_mask], label=y[val_mask]), "val")],
                   early_stopping_rounds=100, verbose_eval=False)
    p["scale_pos_weight"] = float(np.sqrt((len(y) - y.sum()) / max(y.sum(), 1)))
    return xgb.train(p, xgb.DMatrix(X, label=y), m0.best_iteration + 1), p["scale_pos_weight"]


def crossfit_offset_groups(X, y, groups, train_mask, val_mask, folds=5, seed=0, nthread=1):
    """Out-of-fold, unweighted XGBoost logits with whole groups (countries) as fold
    units; trees chosen by early stopping on val_mask (a training sub-period)."""
    rng = np.random.default_rng(seed)
    gs = np.unique(groups); folds = min(folds, len(gs))
    fold_of = dict(zip(gs, rng.permutation(len(gs)) % folds))
    fold = np.array([fold_of[g] for g in groups])
    p = dict(PAPER_PARAMS, nthread=nthread, seed=seed)
    off = np.zeros(len(y))
    for k in range(folds):
        out = fold != k
        fit, val = out & train_mask & ~val_mask, out & val_mask
        m0 = xgb.train(p, xgb.DMatrix(X[fit], label=y[fit]), 2000,
                       evals=[(xgb.DMatrix(X[val], label=y[val]), "val")],
                       early_stopping_rounds=100, verbose_eval=False)
        tr = out & train_mask
        m = xgb.train(p, xgb.DMatrix(X[tr], label=y[tr]), m0.best_iteration + 1)
        off[fold == k] = np.ravel(m.predict(xgb.DMatrix(X[fold == k]), output_margin=True))
    return off


def cell_rate_feature(cell, period, y, train_mask):
    """Leave-one-period-out training event rate of each cell (location-aware feature):
    training rows use the other training periods; test rows use all training periods."""
    d = pd.DataFrame({"c": cell, "p": period, "y": y})[train_mask]
    tot = d.groupby("c")["y"].agg(["sum", "count"])
    byp = d.groupby(["c", "p"])["y"].agg(["sum", "count"])
    s_all = tot["sum"].reindex(cell).to_numpy(); n_all = tot["count"].reindex(cell).to_numpy()
    idx = pd.MultiIndex.from_arrays([cell, period])
    s_p = np.where(train_mask, byp["sum"].reindex(idx).fillna(0).to_numpy(), 0)
    n_p = np.where(train_mask, byp["count"].reindex(idx).fillna(0).to_numpy(), 0)
    return (s_all - s_p) / np.maximum(n_all - n_p, 1)


def shap_full_credit(model, X, j):
    """Contribution of feature j with every pairwise interaction involving j credited
    fully to j (TreeSHAP interaction values): Phi_jj + 2 * sum_{k != j} Phi_jk.
    For an additive effect this equals the SHAP value; for f = g(b) t it recovers
    g(b)(t - E t), whereas the ordinary SHAP value of t gets only half of it."""
    I = model.predict(xgb.DMatrix(X), pred_interactions=True)[:, :-1, :-1]
    return I[:, j, j] + 2 * (I[:, j, :].sum(axis=1) - I[:, j, j])
