"""Residual diagnostics for binary cell x week predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from .gmrf import EARTH_RADIUS_KM


# ---------------------------------------------------------------- scores
def scores(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return {
        "log_score": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((y - p) ** 2)),
        "auc_pr": float(average_precision_score(y, p)),
        "auc_roc": float(roc_auc_score(y, p)),
        "prevalence": float(np.mean(y)),
    }


def persistence_baseline(lag4):
    """'Event if any event in the previous 4 weeks' -- the benchmark the paper lacks.
    Returned as a score (number of recent events) so AUC-PR can be computed."""
    return np.asarray(lag4, float)


def prior_correction(logit, scale_pos_weight):
    """Undo the log-odds shift induced by weighting positives by `scale_pos_weight`."""
    return logit - np.log(scale_pos_weight)


# ---------------------------------------------------------------- residuals
def rqr_bernoulli(y, p, rng=None):
    """Randomised quantile residuals (Dunn & Smyth 1996)."""
    rng = rng or np.random.default_rng(0)
    lo = np.where(y == 1, 1 - p, 0.0)
    hi = np.where(y == 1, 1.0, 1 - p)
    return norm.ppf(np.clip(lo + rng.random(len(y)) * (hi - lo), 1e-12, 1 - 1e-12))


def cell_pearson(df: pd.DataFrame, p, cell_col="cell", y_col="y"):
    """Standardised cell residual: sum_t (y - p) / sqrt(sum_t p (1 - p)).
    Approximately N(0, 1) per cell when the model is correct and there is no
    residual dependence; aggregating over time makes spatial structure visible."""
    g = pd.DataFrame({"c": df[cell_col].to_numpy(), "r": df[y_col].to_numpy() - p,
                      "v": p * (1 - p)}).groupby("c").sum()
    return (g["r"] / np.sqrt(g["v"])).rename("resid")


# ---------------------------------------------------------------- spatial stats
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def morans_i(values, W: sp.spmatrix, n_perm=999, rng=None):
    """Moran's I with a permutation p-value (one-sided, positive autocorrelation)."""
    rng = rng or np.random.default_rng(0)
    z = np.asarray(values, float) - np.mean(values)
    W = sp.csr_matrix(W)
    s0 = W.sum()

    def I(z):
        return len(z) / s0 * (z @ (W @ z)) / (z @ z)

    obs = I(z)
    perm = np.array([I(rng.permutation(z)) for _ in range(n_perm)])
    return {"I": float(obs), "p_value": float((1 + np.sum(perm >= obs)) / (n_perm + 1)),
            "E_I": -1 / (len(z) - 1)}


def variogram(values, lon, lat, bins_km, max_pairs=2_000_000, rng=None):
    """Empirical semivariogram with great-circle distances."""
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, float); n = len(v)
    if n * (n - 1) / 2 > max_pairs:
        i = rng.integers(0, n, max_pairs); j = rng.integers(0, n, max_pairs)
        keep = i != j; i, j = i[keep], j[keep]
    else:
        i, j = np.triu_indices(n, 1)
    d = haversine(lon[i], lat[i], lon[j], lat[j])
    g = 0.5 * (v[i] - v[j]) ** 2
    b = np.digitize(d, bins_km)
    out = [(0.5 * (bins_km[k - 1] + bins_km[k]), g[b == k].mean(), int((b == k).sum()))
           for k in range(1, len(bins_km)) if np.any(b == k)]
    return pd.DataFrame(out, columns=["dist_km", "gamma", "n_pairs"])


def residual_acf(df: pd.DataFrame, resid, max_lag=8):
    """Pooled within-cell temporal autocorrelation of observation-level residuals."""
    d = pd.DataFrame({"c": df["cell"].to_numpy(), "t": df["week"].to_numpy(), "r": resid})
    wide = d.pivot(index="t", columns="c", values="r").to_numpy()
    wide = wide - np.nanmean(wide, 0)
    den = np.nansum(wide ** 2)
    return pd.Series([np.nansum(wide[k:] * wide[:len(wide) - k]) / den
                      for k in range(max_lag + 1)], name="acf")


def border_contrast(cell_resid, adjacency: sp.spmatrix, country):
    """Correlation of residuals between rook neighbours in the same country vs.
    neighbours across a border. A big gap points at country-level effects
    (national covariates mis-specified / missing country random effect)."""
    A = sp.triu(sp.csr_matrix(adjacency), 1).tocoo()
    r = np.asarray(cell_resid, float); c = np.asarray(country)
    same = c[A.row] == c[A.col]
    corr = lambda m: float(np.corrcoef(r[A.row[m]], r[A.col[m]])[0, 1])
    return {"within_country": corr(same), "across_border": corr(~same),
            "n_within": int(same.sum()), "n_across": int((~same).sum())}
