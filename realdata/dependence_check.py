"""Dependence diagnostic on existing 2023-24 joint-prediction parquets (no refitting).

For each region and model (biv, uni_static, uni_st):
 (a) observed P(both) = mean(y_c*y_u) against mean(pc*pu) and biv's mean P(both);
 (b) logit(y_c) ~ a + b*logit(pc_model) + g*y_unrest  -> odds ratio exp(g), cluster-robust SE
     (by country); same with last week's y_unrest in the same cell (first test week dropped);
 (c) the same odds ratios within the Afghanistan and Maghreb windows (val2022_windows.csv)
     and within Yemen and Sudan (clustered by cell there, since one country = one cluster).

usage: python realdata/dependence_check.py WORKDIR LOOKUP WINDOWS_CSV OUT_JSON
"""
import json
import sys

import numpy as np
import pandas as pd

MODELS = ["biv", "uni_static", "uni_st"]
REGIONS = {"africa_me": "Africa-Middle East", "west_sahel": "West Africa and Sahel"}


def logit_fit(X, y, groups, maxit=50):
    """Logistic regression by Newton-Raphson; cluster-robust (CR1, statsmodels-style) SEs."""
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    w = p * (1 - p)
    Hinv = np.linalg.inv(X.T @ (X * w[:, None]))
    sc = X * (y - p)[:, None]
    codes, gi = np.unique(groups, return_inverse=True)
    S = np.zeros((len(codes), X.shape[1]))
    np.add.at(S, gi, sc)
    G, N, K = len(codes), len(y), X.shape[1]
    c = G / (G - 1) * (N - 1) / (N - K)
    V = c * Hinv @ (S.T @ S) @ Hinv
    return b, np.sqrt(np.diag(V)), G


def or_fit(d, model, xvar, cluster):
    d = d.dropna(subset=[xvar])
    y = d.y_conflict.to_numpy(float)
    x = d[xvar].to_numpy(float)
    out = dict(n=int(len(d)), n_conflict=int(y.sum()), n_x=int(x.sum()), n_both=int((y * x).sum()),
               cluster=cluster)
    if y.sum() < 5 or x.sum() < 5 or (y * x).sum() < 1 or len(np.unique(d[cluster])) < 2:
        out["note"] = "too few events or clusters"
        return out
    pc = np.clip(d[f"p_conflict_{model}"].to_numpy(float), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.ones(len(d)), np.log(pc / (1 - pc)), x])
    try:
        b, se, G = logit_fit(X, y, d[cluster].to_numpy())
    except np.linalg.LinAlgError as e:
        out["note"] = f"fit failed: {e}"
        return out
    out.update(n_clusters=int(G), slope_logit_p=float(b[1]), slope_se=float(se[1]),
               log_or=float(b[2]), log_or_se=float(se[2]), odds_ratio=float(np.exp(b[2])),
               or_ci95=[float(np.exp(b[2] - 1.96 * se[2])), float(np.exp(b[2] + 1.96 * se[2]))],
               z=float(b[2] / se[2]))
    return out


def rates(d):
    r = dict(n=int(len(d)), mean_y_conflict=float(d.y_conflict.mean()), mean_y_unrest=float(d.y_unrest.mean()),
             obs_both=float((d.y_conflict * d.y_unrest).mean()),
             biv_mean_pboth=float(d.pboth_biv.astype(float).mean()))
    for m in MODELS:
        pc = d[f"p_conflict_{m}"].astype(float); pu = d[f"p_unrest_{m}"].astype(float)
        r[f"{m}_mean_pc"] = float(pc.mean()); r[f"{m}_mean_pu"] = float(pu.mean())
        r[f"{m}_mean_pc_pu"] = float((pc * pu).mean())
    r["obs_over_biv_pboth"] = r["obs_both"] / r["biv_mean_pboth"]
    for m in MODELS:
        r[f"obs_over_{m}_pc_pu"] = r["obs_both"] / r[f"{m}_mean_pc_pu"]
    return r


def block(d, cluster):
    res = dict(rates=rates(d), odds_ratio={})
    for m in MODELS:
        res["odds_ratio"][m] = {"y_unrest_same_week": or_fit(d, m, "y_unrest", cluster),
                                "y_unrest_last_week": or_fit(d, m, "y_unrest_lag1", cluster)}
    return res


def main():
    work, lookup, wins, out = sys.argv[1:5]
    lk = pd.read_parquet(lookup, columns=["grid_id", "iso", "lon", "lat"]).drop_duplicates("grid_id")
    w = pd.read_csv(wins)
    win = {"afghanistan": w.iloc[0], "maghreb": w[w.countries.str.startswith("DZA")].iloc[0]}
    res = dict(description=__doc__.split("\n\n")[0], test_period="2023-2024", models=MODELS,
               windows={k: dict(lon=[float(v.lon0), float(v.lon1)], lat=[float(v.lat0), float(v.lat1)],
                                countries=v.countries) for k, v in win.items()},
               notes=["y_unrest_lag1 = y_unrest of the same cell in the previous week; the first test week "
                      "(2023-01-03) has no lag inside the test file and is dropped from that regression.",
                      "west_sahel P(both) and pc*pu come from its Monte Carlo rescoring (1708150.1); "
                      "africa_me from 1708150.2.",
                      "Country-clustered SEs for regions and windows; single-country subsets are clustered by cell."],
               regions={})
    for reg, name in REGIONS.items():
        d = pd.read_parquet(f"{work}/bivariate/{reg}/test_pred_jointpred.parquet")
        d = d.merge(lk, on="grid_id", how="left")
        assert d.iso.notna().all()
        d = d.sort_values(["grid_id", "week_start"]).reset_index(drop=True)
        prev = d.groupby("grid_id").week_start.shift(1)
        lag = d.groupby("grid_id").y_unrest.shift(1)
        d["y_unrest_lag1"] = lag.where(d.week_start - prev == pd.Timedelta(days=7))
        r = dict(name=name, overall=block(d, "iso"), subsets={})
        for k, v in win.items():
            s = d[d.lon.between(v.lon0, v.lon1) & d.lat.between(v.lat0, v.lat1)]
            if len(s):
                r["subsets"][f"window_{k}"] = block(s, "iso")
        for iso, nm in [("YEM", "yemen"), ("SDN", "sudan")]:
            s = d[d.iso == iso]
            if len(s):
                r["subsets"][nm] = block(s, "grid_id")
        res["regions"][reg] = r
        print(reg, "done", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
