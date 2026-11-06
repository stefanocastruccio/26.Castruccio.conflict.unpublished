"""Parametric latent Gaussian models on the clean ACLED table (paper Sec. 3).

Linear predictor (logit), training on 2018-2022, test 2023-2024:
  delta * log1p(own-outcome events, past 4 weeks) + log1p(other-outcome events)
  + cell covariates (population, GDP per capita, electricity, malaria, IPC 3+, rivers, lakes)
  + climate anomalies (4-week means of tmax, tmin, ppt, standardised within cell and
    calendar month over 2018-2022)
  + national indicators (EPI, GHG, material and ecological footprint, V-Dem)
  [+ country effects u] [+ barrier Matern field w] [+ biome-pooled SVC for tmax]
  [+ AR(1) x Matern space-time field v]

Models (--models, comma separated):
  glm          fixed effects only
  country      + iid country effects
  field        + country effects + spatial field
  spatialplus  as field, national indicators residualised on the 20 smoothest
               eigenvectors of the lattice (Dupont et al. 2022)
  svc          as field, tmax effect = biome mean + Matern deviation eta(s)
  st           on --st-region only: glm, field and field + space-time field
               (time steps of --st-step weeks), to compare delta

Outputs in --out: models_<outcome>.json (coefficients with SEs, hyperparameters,
marginal likelihood, test scores) and coefficients_<outcome>.csv.

Usage: python fit_glmm.py clean.parquet --outcome conflict --region africa_me
       [--models glm,country,field,spatialplus,svc,st] [--st-region sahel_horn] [--st-step 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from sc import design as de, diagnostics as dg  # noqa: E402
from sc.gmrf import FixedBlock, IIDBlock, MaternBlock, SpaceTimeAR1Block  # noqa: E402
from sc.laplace import LatentGaussianModel, prob_from_eta  # noqa: E402
from fit_acled import REGIONS, build_lattice  # noqa: E402

REGIONS = dict(REGIONS, sahel_horn=(-18, 52, 0, 22))
OWN = {"conflict": ["Battles_rollsum_4", "ExplosionsRemote_violence_rollsum_4",
                    "Violence_against_civilians_rollsum_4"],
       "unrest": ["Riots_rollsum_4", "Protests_rollsum_4"]}
YCOL = {"conflict": "Conflict", "unrest": "Unrest"}
NATIONAL = ["EPI_index", "GHG_pc_t", "MF_pc_t", "ECO_pc_gha", "v2x_libdem"]
CLIM = {"tmax_z": "tmax_rollmean_4", "tmin_z": "tmin_rollmean_4", "ppt_z": "ppt_rollmean_4"}


def prepare(df, outcome):
    other = "unrest" if outcome == "conflict" else "conflict"
    df = df.copy()
    df["y"] = df[YCOL[outcome]].astype(int)
    df["lag_own"] = np.log1p(df[OWN[outcome]].sum(axis=1).fillna(0))
    df["lag_other"] = np.log1p(df[OWN[other]].sum(axis=1).fillna(0))
    df["log_pop"] = np.log1p(df["population"])
    df["log_gdp_pc"] = np.log1p(df["gdp"] / df["population"].clip(lower=1))
    df["log_ipc3p"] = np.log1p(df[[c for c in ["phase_3", "phase_4", "phase_5"] if c in df]].sum(axis=1))
    df["log_river"] = np.log1p(df["river_length"]); df["log_lake"] = np.log1p(df["lake_area"])
    df["train"] = df["week_start"].dt.year <= 2022
    df["month"] = df["week_start"].dt.month
    for z, c in CLIM.items():   # anomaly within cell x calendar month, 2018-2022 climatology
        g = df[df.train].groupby(["grid_id", "month"])[c].agg(["mean", "std"])
        st = df[["grid_id", "month"]].merge(g, left_on=["grid_id", "month"], right_index=True, how="left")
        df[z] = (df[c].to_numpy() - st["mean"].to_numpy()) / st["std"].clip(lower=1e-6).to_numpy()
    cov = ["lag_own", "lag_other", "log_pop", "log_gdp_pc", "electricity_access", "malaria",
           "log_ipc3p", "log_river", "log_lake"] + list(CLIM) + NATIONAL
    tr = df.train
    for c in cov[2:]:   # standardise (lags kept on their natural log scale); missing -> mean
        mu, sd = df.loc[tr, c].mean(), df.loc[tr, c].std()
        df[c] = ((df[c] - mu) / (sd if sd > 0 else 1)).fillna(0.0)
    df["lag_own"] = df["lag_own"].fillna(0); df["lag_other"] = df["lag_other"].fillna(0)
    df["biome"] = pd.factorize(df["BIOME_NAME"])[0]
    return df, cov


def spatial_plus(df, cov_nat, L, cell, k=20):
    E = L.smooth_basis(k)[cell]                     # rows: observations
    cells, first = np.unique(cell, return_index=True)
    Ec = E[first]
    for c in cov_nat:
        cm = pd.Series(df[c].to_numpy()).groupby(cell).mean().reindex(cells).to_numpy()
        beta = np.linalg.lstsq(Ec, cm, rcond=None)[0]
        df[c] = df[c].to_numpy() - E @ beta
    return df


def fit(d, L, cell, nC, cov, *, country=False, field=False, svc=False, st=None, barrier=None):
    factors = [("biome", "tmax_z")] if svc else []
    cols = [c for c in cov if not (svc and c == "tmax_z")]
    X, names = de.fixed_design(d, cols, factors=factors)
    blocks, D = [FixedBlock(X.shape[1], names)], [X]
    if country:
        blocks.append(IIDBlock(nC, name="country")); D.append(de.index_design(d["country_id"], nC))
    if field:
        blocks.append(MaternBlock(L, name="field", range_init=300.0, barrier=barrier))
        D.append(de.index_design(cell, L.n))
    if svc:
        blocks.append(MaternBlock(L, name="svc", range_init=800.0, sd_init=0.05,
                                  pc_sd=(0.5, 0.01), barrier=barrier))
        D.append(de.index_design(cell, L.n, d["tmax_z"].to_numpy(float)))
    if st is not None:
        step, T = st
        blocks.append(SpaceTimeAR1Block(L, T, name="st", range_init=300.0, sd_init=0.3))
        D.append(de.spacetime_design(cell, d["tstep"].to_numpy(), L.n, T))
    m = LatentGaussianModel(d["y"].to_numpy(), blocks, D).fit()
    return m, names


def summarise(m, names, d_te=None, D_te=None):
    fs = m.fixed_summary()
    out = {"coef": {k: {"est": float(v[0]), "se": float(v[1])} for k, v in fs.items()},
           "hyper": m.hyper(), "log_ml": float(m.log_ml), "seconds": float(m.fit_seconds)}
    if d_te is not None:
        eta, sd = m.predict_eta(D_te, None, n_samples=100)
        out["test"] = dg.scores(d_te["y"].to_numpy(), prob_from_eta(eta, sd))
    return out


def test_designs(m, d_te, L, cell_te, nC, cov, svc=False):
    factors = [("biome", "tmax_z")] if svc else []
    cols = [c for c in cov if not (svc and c == "tmax_z")]
    X, _ = de.fixed_design(d_te, cols, factors=factors)
    D = [X]
    names = [b.name for b in m.blocks]
    if "country" in names:
        D.append(de.index_design(d_te["country_id"], nC))
    if "field" in names:
        D.append(de.index_design(cell_te, L.n))
    if "svc" in names:
        D.append(de.index_design(cell_te, L.n, d_te["tmax_z"].to_numpy(float)))
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--outcome", default="conflict")
    ap.add_argument("--region", default="africa_me")
    ap.add_argument("--models", default="glm,country,field,spatialplus,svc,st")
    ap.add_argument("--st-region", default="sahel_horn"); ap.add_argument("--st-step", type=int, default=8)
    ap.add_argument("--out", default="results_real/glmm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = a.models.split(",")
    t0 = time.time()

    df = pd.read_parquet(a.table)
    df["week_start"] = pd.to_datetime(df["week_start"])
    if a.region and a.region != "world":
        x0, x1, y0, y1 = REGIONS[a.region]
        df = df[df.lon.between(x0, x1) & df.lat.between(y0, y1)]
    df, cov = prepare(df, a.outcome)
    df["country_id"] = pd.factorize(df["ISO_A3"])[0]
    nC = int(df["country_id"].max() + 1)
    L, cell, observed = build_lattice(df.lon.to_numpy(), df.lat.to_numpy())
    df["cell"] = cell
    tr, te = df[df.train], df[~df.train]
    print(f"{len(df):,} rows, {observed.sum():,} cells, {nC} countries, prevalence {tr.y.mean():.4f}", flush=True)

    res = {"outcome": a.outcome, "region": a.region, "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_cells": int(observed.sum()), "n_countries": nC, "models": {}}
    spec = {"glm": {}, "country": dict(country=True), "field": dict(country=True, field=True),
            "svc": dict(country=True, field=True, svc=True)}
    for name in [m for m in models if m in spec]:
        m, names = fit(tr, L, tr["cell"].to_numpy(), nC, cov, barrier=~observed, **spec[name])
        res["models"][name] = summarise(m, names, te, test_designs(m, te, L, te["cell"].to_numpy(),
                                                                   nC, cov, svc=(name == "svc")))
        if name == "svc":
            k = [b.name for b in m.blocks].index("svc")
            pd.DataFrame({"lon": L.lon, "lat": L.lat, "observed": observed,
                          "eta_tmax": m.block_mean(k)}).to_csv(
                os.path.join(a.out, f"svc_tmax_{a.outcome}.csv"), index=False)
        print(name, json.dumps(res["models"][name].get("test", {})), flush=True)
    if "spatialplus" in models:
        d2 = spatial_plus(df.copy(), NATIONAL, L, df["cell"].to_numpy())
        tr2, te2 = d2[d2.train], d2[~d2.train]
        m, names = fit(tr2, L, tr2["cell"].to_numpy(), nC, cov, country=True, field=True, barrier=~observed)
        res["models"]["spatialplus"] = summarise(m, names, te2, test_designs(
            m, te2, L, te2["cell"].to_numpy(), nC, cov))
        print("spatialplus", json.dumps(res["models"]["spatialplus"]["test"]), flush=True)
    if "st" in models:
        x0, x1, y0, y1 = REGIONS[a.st_region] if a.st_region in REGIONS else (-180, 180, -90, 90)
        ds = tr[tr.lon.between(x0, x1) & tr.lat.between(y0, y1)].copy()
        Ls, cs, obs_s = build_lattice(ds.lon.to_numpy(), ds.lat.to_numpy())
        wk = (ds["week_start"] - ds["week_start"].min()).dt.days // 7
        ds["tstep"] = (wk // a.st_step).to_numpy()
        T = int(ds["tstep"].max() + 1)
        ds["country_id"] = pd.factorize(ds["ISO_A3"])[0]; nCs = int(ds["country_id"].max() + 1)
        st_res = {"region": a.st_region, "n_rows": int(len(ds)), "n_cells": int(obs_s.sum()),
                  "time_steps": T, "step_weeks": a.st_step}
        for name, kw in [("glm", {}), ("field", dict(country=True, field=True)),
                         ("field+st", dict(country=True, field=True, st=(a.st_step, T)))]:
            m, names = fit(ds, Ls, cs, nCs, cov, barrier=~obs_s, **kw)
            st_res[name] = summarise(m, names)
            print("st/" + name, "delta", st_res[name]["coef"]["lag_own"], flush=True)
        res["models"]["st"] = st_res
    res["seconds_total"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, f"models_{a.outcome}.json"), "w"), indent=1, default=float)
    rows = []
    for name, r in res["models"].items():
        parts = {name: r} if name != "st" else {f"st/{k}": v for k, v in r.items() if isinstance(v, dict)}
        for nm, rr in parts.items():
            for k, v in rr["coef"].items():
                rows.append(dict(model=nm, term=k, est=v["est"], se=v["se"]))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"coefficients_{a.outcome}.csv"), index=False)
    print(f"done in {res['seconds_total'] / 60:.1f} min")


if __name__ == "__main__":
    main()
