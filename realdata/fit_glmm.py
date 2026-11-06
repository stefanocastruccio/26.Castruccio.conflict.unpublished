"""Parametric latent Gaussian models on the clean ACLED table (paper Sec. 3).

Linear predictor (logit), training on 2018-2022, test 2023-2024:
  delta * log1p(own-outcome events, past 4 weeks) + log1p(other-outcome events)
  + cell covariates (population, malaria, IPC 3+, rivers, lakes)
  + climate: anomalies of the lagged 4-week means of tmax, tmin, ppt, drought and PM2.5,
    standardised within cell and calendar month with the 2018-2022 (training-period)
    climatology, plus each cell's climatology (training-period mean) as a covariate.
    The climatological SD is floored at the 10th percentile of the positive cell-month
    SDs of the variable, and z = 0 where that SD is 0; precipitation anomalies are
    computed on log1p(ppt)
  + biome effects and year effects
  + national indicators (log national GDP per capita, electricity access, EPI, GHG,
    material and ecological footprint, V-Dem), as between-country and within-country
    components
  Missing covariates stay missing through their construction (no IPC analysis is
  missing, not zero people in phase 3+), are set to the training mean after
  standardising, and missingness indicators are added: one per national indicator
  (identical ones merged) and one for IPC.
  [+ country effects u] [+ barrier Matern field w + iid cell effect e (nugget)]
  [+ biome-pooled SVC for tmax] [+ AR(1) x Matern space-time field v]

Models (--models, comma separated):
  glm          fixed effects only
  country      + iid country effects
  field        + country effects + spatial field + iid cell effect (--no-nugget omits it)
  spatialplus  as field, national indicators residualised on the 20 smoothest
               eigenvectors of the lattice (Dupont et al. 2022)
  svc          as field, tmax effect = biome mean + Matern deviation eta(s)
  st           on --st-region only: glm, field and field + space-time field
               (time steps of --st-step weeks), to compare delta. w and the nugget
               use the same lattice as the static field (observed cells of the box +
               6-ring buffer, sea barrier with range fraction 0.5); the space-time
               field v lives on the observed cells of the box only, so it cannot link
               cells across the sea.
The barrier rule is --land (sea cells are barriers); --land is required unless
--barrier-unobserved is given. The rule is recorded in the output.
Rows whose 4-week history is incomplete (the first weeks of each cell, and the first
weeks after a country's ACLED coverage starts) are dropped.

Outputs in --out: models_<outcome>.json (coefficients with SEs, hyperparameters,
marginal likelihood, test scores, preprocessing summary), coefficients_<outcome>.csv and
test_pred_<outcome>.parquet (test-period predictions p_<model>, for compare.py).

Usage: python fit_glmm.py clean.parquet --outcome conflict --region africa_me
       --land ne_10m_admin_0_countries.shp
       [--models glm,country,field,spatialplus,svc,st] [--st-region central_sahel] [--st-step 8]
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
from fit_acled import REGIONS, build_lattice, sea_cells  # noqa: E402

REGIONS = dict(REGIONS, sahel_horn=(-18, 52, 0, 22), central_sahel=(-6, 16, 9, 20))
OWN = {"conflict": ["Battles_rollsum_4", "ExplosionsRemote_violence_rollsum_4",
                    "Violence_against_civilians_rollsum_4"],
       "unrest": ["Riots_rollsum_4", "Protests_rollsum_4"]}
YCOL = {"conflict": "Conflict", "unrest": "Unrest"}
NATIONAL = ["log_gdp_pc", "electricity_access", "EPI_index", "GHG_pc_t", "MF_pc_t", "ECO_pc_gha",
            "v2x_libdem"]
CLIM = {"tmax_z": "tmax_rollmean_4", "tmin_z": "tmin_rollmean_4", "ppt_z": "ppt_rollmean_4",
        "drought_z": "drought_rollmean_4", "pm25_z": "pm25_rollmean_4"}


def climate_anomaly(df, c, log=False, q=0.10):
    """Anomaly of column c within cell x calendar month with the training-period
    climatology. The climatological SD is floored at the q-quantile of the positive
    cell-month SDs of the variable, and z = 0 where the SD is exactly 0 (a variable that
    was constant in that cell-month in every training year). With log=True the anomaly
    is computed on log1p(max(x, 0)) (precipitation). Returns (z, clim, info)."""
    x = df[c].astype(float)
    if log:
        x = np.log1p(x.clip(lower=0))
    key = [df["grid_id"].to_numpy(), df["month"].to_numpy()]
    g = x[df.train].groupby([df.loc[df.train, "grid_id"], df.loc[df.train, "month"]]).agg(["mean", "std"])
    idx = pd.MultiIndex.from_arrays(key)
    mu = g["mean"].reindex(idx).to_numpy(); sd = g["std"].reindex(idx).to_numpy()
    pos = g["std"][g["std"] > 0]
    floor = float(pos.quantile(q)) if len(pos) else 1.0
    z = (x.to_numpy() - mu) / np.maximum(sd, floor)
    z = np.where(sd == 0, 0.0, z)
    clim = df["grid_id"].map(x[df.train].groupby(df.loc[df.train, "grid_id"]).mean()).to_numpy()
    info = {"transform": "log1p" if log else "none", "sd_floor": floor, "floor_quantile": q,
            "cell_months": int(len(g)), "cell_months_sd_zero": int((g["std"] == 0).sum()),
            "cell_months_sd_below_floor": int(((g["std"] > 0) & (g["std"] < floor)).sum()),
            "cell_months_sd_undefined": int(g["std"].isna().sum())}
    return z, clim, info


def prepare(df, outcome):
    """Covariates of the latent models. Returns (df, cov, info).

    Missing values: covariates stay NaN through their construction (a missing IPC
    analysis is NaN, not zero people in phase 3+); after standardisation with the
    training mean and SD they are set to 0, the training mean. Missingness indicators
    (not standardised) are added for each national indicator (identical indicators are
    merged into one column, recorded in info) and one for IPC."""
    other = "unrest" if outcome == "conflict" else "conflict"
    info = {}
    df = df.copy()
    df["y"] = df[YCOL[outcome]].astype(int)
    df = df[df[OWN[outcome] + OWN[other]].notna().all(axis=1)].copy()   # complete 4-week history
    df["lag_own"] = np.log1p(df[OWN[outcome]].sum(axis=1))
    df["lag_other"] = np.log1p(df[OWN[other]].sum(axis=1))
    df["log_pop"] = np.log1p(df["population"])
    if "gdp_pc" in df.columns:       # national GDP per capita (load_partitions)
        df["log_gdp_pc"] = np.log(df["gdp_pc"].where(df["gdp_pc"] > 0))
        info["gdp_pc"] = "log(national GDP / national population)"
    else:
        print("WARNING: no gdp_pc column (old table); using per-cell gdp / population")
        df["log_gdp_pc"] = np.log1p(df["gdp"] / df["population"].clip(lower=1))
        info["gdp_pc"] = "log1p(per-cell gdp / population): old table, not national"
    ph = [c for c in ["phase_3", "phase_4", "phase_5"] if c in df]
    df["log_ipc3p"] = np.log1p(df[ph].sum(axis=1, min_count=1)) if ph else np.nan
    ipc_missing = df["log_ipc3p"].isna().astype(float)
    df["log_river"] = np.log1p(df["river_length"]); df["log_lake"] = np.log1p(df["lake_area"])
    df["train"] = df["week_start"].dt.year <= 2022
    df["month"] = df["week_start"].dt.month
    clim = {z: c for z, c in CLIM.items() if c in df.columns}
    info["climate"] = {}
    for z, c in clim.items():   # anomaly within cell x calendar month, 2018-2022 climatology
        df[z], df[z.replace("_z", "_clim")], info["climate"][z] = climate_anomaly(df, c, log=(z == "ppt_z"))
    # initial conditions (Wooldridge 2005): each cell's history at its first usable week
    df = df.sort_values(["grid_id", "week_start"])
    df["h_init"] = df.groupby("grid_id")["lag_own"].transform("first")
    # national indicators: between-country (country mean over the training years) and
    # within-country (year-to-year deviation) components, Mundlak style
    nat_parts, nat_missing = [], {}
    for v in NATIONAL:
        nat_missing[v] = df[v].isna().astype(float)
        cm = df[df.train].groupby("ISO_A3")[v].mean()
        df[f"{v}_between"] = df["ISO_A3"].map(cm)
        df[f"{v}_within"] = df[v] - df[f"{v}_between"]
        nat_parts += [f"{v}_between", f"{v}_within"]
    cov = (["lag_own", "lag_other", "h_init", "log_pop", "malaria", "log_ipc3p", "log_river", "log_lake"]
           + list(clim) + [z.replace("_z", "_clim") for z in clim] + nat_parts)
    tr = df.train
    info["share_missing_train"] = {c: float(df.loc[tr, c].isna().mean()) for c in cov[3:]}
    for c in cov[3:]:   # standardise (lags kept on their natural log scale); missing -> mean
        mu, sd = df.loc[tr, c].mean(), df.loc[tr, c].std()
        df[c] = ((df[c] - mu) / (sd if sd > 0 else 1)).fillna(0.0)
    df["lag_own"] = df["lag_own"].fillna(0); df["lag_other"] = df["lag_other"].fillna(0)
    # missingness indicators (0/1, not standardised); constant or duplicated columns dropped
    ind, kept, aliases = {"ipc_missing": ipc_missing}, [], {}
    ind.update({f"{v}_missing": m for v, m in nat_missing.items()})
    for name, col in ind.items():
        x = col.reindex(df.index).to_numpy()   # df was re-sorted after col was built
        xt = x[tr.to_numpy()]
        if xt.min() == xt.max():
            aliases[name] = "constant in training (dropped)"
            continue
        dup = next((k for k in kept if np.array_equal(df[k].to_numpy(), x)), None)
        if dup is not None:
            aliases[name] = f"identical to {dup} (merged)"
            continue
        df[name] = x; kept.append(name)
    info["missing_indicators"] = {"kept": kept, "merged_or_dropped": aliases}
    cov = cov + kept
    # indicators built from the training levels (test years 2023-2024 take the 2022 effect)
    biomes = sorted(df.loc[df.train, "BIOME_NAME"].unique())
    df["biome"] = df["BIOME_NAME"].map({b: i for i, b in enumerate(biomes)}).fillna(0).astype(int)
    for i in range(1, len(biomes)):
        df[f"biome_{i}"] = (df["biome"] == i).astype(float)
    yr = df["week_start"].dt.year.clip(upper=2022)
    years = sorted(yr[df.train].unique())
    for y in years[1:]:
        df[f"year_{y}"] = (yr == y).astype(float)
    for i in range(len(biomes)):
        df[f"tmax_z_x_biome_{i}"] = df["tmax_z"] * (df["biome"] == i)
    cov = cov + [f"biome_{i}" for i in range(1, len(biomes))] + [f"year_{y}" for y in years[1:]]
    return df, cov, info


def svc_cols(df):
    return sorted([c for c in df.columns if c.startswith("tmax_z_x_biome_")], key=lambda c: int(c.rsplit("_", 1)[1]))


def spatial_plus(df, cov_nat, lon, lat, m):
    """spatial+ (Dupont et al. 2022): replace each between-country national component
    by its residual after projection, over the observed cells, on the m smoothest
    eigenvectors of G e = lambda C e computed on the observed land cells only."""
    Lo, cell_o, _ = build_lattice(lon, lat, buffer=0)
    E = Lo.smooth_basis(m)
    W = Lo.area / Lo.area.sum()
    cells = np.unique(cell_o)
    for c in cov_nat:
        cm = pd.Series(df[c].to_numpy()).groupby(cell_o).mean().reindex(cells).to_numpy()
        Ec = E[cells]
        sw = np.sqrt(W[cells])
        beta = np.linalg.lstsq(Ec * sw[:, None], cm * sw, rcond=None)[0]
        df[c] = df[c].to_numpy() - E[cell_o] @ beta
    return df


def n_eigen_below(lon, lat, rho_km, kmax=400):
    """Number of eigenvalues of G e = lambda C e (observed cells) below kappa^2 = 8 / rho^2:
    the eigenvectors that vary on scales longer than the field's range."""
    from scipy.sparse.linalg import eigsh
    Lo, _, _ = build_lattice(lon, lat, buffer=0)
    k = min(kmax, Lo.n - 2)
    vals = eigsh(Lo.G, k=k, M=Lo.C, sigma=-1e-8, which="LM", return_eigenvectors=False)
    return int(max(5, (vals < 8.0 / rho_km ** 2).sum()))


def fit(d, L, cell, nC, cov, *, country=False, field=False, svc=False, st=None, barrier=None,
        nugget=None):
    """Latent model. Blocks in order: fixed, country u, field w, cell nugget e, svc, st v.
    nugget = (obs_index, n_obs): iid cell effect on the observed cells (added whenever
    the field is), so that heterogeneity at the cell scale is not forced into delta or
    w. st = (Lv, cell_v, T): AR(1) x Matern space-time field on its own lattice Lv."""
    cols = [c for c in cov if not (svc and c == "tmax_z")] + (svc_cols(d) if svc else [])
    X, names = de.fixed_design(d, cols)
    blocks, D = [FixedBlock(X.shape[1], names)], [X]
    if country:
        blocks.append(IIDBlock(nC, name="country")); D.append(de.index_design(d["country_id"], nC))
    if field:
        blocks.append(MaternBlock(L, name="field", range_init=300.0, barrier=barrier))
        D.append(de.index_design(cell, L.n))
        if nugget is not None:
            obs_index, n_obs = nugget
            blocks.append(IIDBlock(n_obs, name="cell", sd_init=0.3))
            D.append(de.index_design(obs_index[cell], n_obs))
    if svc:
        blocks.append(MaternBlock(L, name="svc", range_init=800.0, sd_init=0.05,
                                  pc_sd=(0.5, 0.01), barrier=barrier))
        D.append(de.index_design(cell, L.n, d["tmax_z"].to_numpy(float)))
    if st is not None:
        Lv, cell_v, T = st
        blocks.append(SpaceTimeAR1Block(Lv, T, name="st", range_init=300.0, sd_init=0.3))
        D.append(de.spacetime_design(cell_v, d["tstep"].to_numpy(), Lv.n, T))
    m = LatentGaussianModel(d["y"].to_numpy(), blocks, D).fit()
    return m, names


def summarise(m, names, d_te=None, D_te=None, preds=None, key=None):
    fs = m.fixed_summary()
    out = {"coef": {k: {"est": float(v[0]), "se": float(v[1])} for k, v in fs.items()},
           "hyper": m.hyper(), "log_ml": float(m.log_ml), "seconds": float(m.fit_seconds)}
    if d_te is not None:
        eta, sd = m.predict_eta(D_te, None, n_samples=100)
        p = prob_from_eta(eta, sd)
        out["test"] = dg.scores(d_te["y"].to_numpy(), p)
        if preds is not None:
            preds[f"p_{key}"] = p.astype("float32")
    return out


def test_designs(m, d_te, L, cell_te, nC, cov, svc=False, nugget=None):
    cols = [c for c in cov if not (svc and c == "tmax_z")] + (svc_cols(d_te) if svc else [])
    X, _ = de.fixed_design(d_te, cols)
    D = [X]
    names = [b.name for b in m.blocks]
    if "country" in names:
        D.append(de.index_design(d_te["country_id"], nC))
    if "field" in names:
        D.append(de.index_design(cell_te, L.n))
    if "cell" in names:
        obs_index, n_obs = nugget
        D.append(de.index_design(obs_index[cell_te], n_obs))
    if "svc" in names:
        D.append(de.index_design(cell_te, L.n, d_te["tmax_z"].to_numpy(float)))
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--outcome", default="conflict")
    ap.add_argument("--region", default="africa_me")
    ap.add_argument("--models", default="glm,country,field,spatialplus,svc,st")
    ap.add_argument("--st-region", default="central_sahel"); ap.add_argument("--st-step", type=int, default=8)
    ap.add_argument("--out", default="results_real/glmm")
    ap.add_argument("--land", default=None, help="country shapefile; sea cells become barriers")
    ap.add_argument("--barrier-unobserved", action="store_true",
                    help="without --land: every lattice cell without data is a barrier")
    ap.add_argument("--no-nugget", action="store_true", help="omit the iid cell effect (sensitivity)")
    a = ap.parse_args()
    if not a.land and not a.barrier_unobserved:
        ap.error("--land SHAPEFILE is required (or pass --barrier-unobserved explicitly)")
    os.makedirs(a.out, exist_ok=True)
    models = a.models.split(",")
    t0 = time.time()

    df = pd.read_parquet(a.table)
    df["week_start"] = pd.to_datetime(df["week_start"])
    if a.region and a.region != "world":
        x0, x1, y0, y1 = REGIONS[a.region]
        df = df[df.lon.between(x0, x1) & df.lat.between(y0, y1)]
    df, cov, prep_info = prepare(df, a.outcome)
    df["country_id"] = pd.factorize(df["ISO_A3"])[0]
    nC = int(df["country_id"].max() + 1)
    L, cell, observed = build_lattice(df.lon.to_numpy(), df.lat.to_numpy())
    df["cell"] = cell
    barrier = sea_cells(L, a.land) if a.land else ~observed
    barrier_rule = "sea cells from country shapefile" if a.land else "all cells without data"
    nug = None if a.no_nugget else (np.cumsum(observed) - 1, int(observed.sum()))
    tr, te = df[df.train], df[~df.train]
    print(f"{len(df):,} rows, {observed.sum():,} cells, {nC} countries, prevalence {tr.y.mean():.4f}", flush=True)

    res = {"outcome": a.outcome, "region": a.region, "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_cells": int(observed.sum()), "n_countries": nC, "barrier_rule": barrier_rule,
           "barrier_cells": int(barrier.sum()), "lattice": "observed cells + 6-ring buffer",
           "cell_nugget": nug is not None, "prep": prep_info, "covariates": cov, "models": {}}
    preds = {}
    spec = {"glm": {}, "country": dict(country=True), "field": dict(country=True, field=True),
            "svc": dict(country=True, field=True, svc=True)}
    for name in [m for m in models if m in spec]:
        m, names = fit(tr, L, tr["cell"].to_numpy(), nC, cov, barrier=barrier, nugget=nug, **spec[name])
        res["models"][name] = summarise(m, names, te, test_designs(m, te, L, te["cell"].to_numpy(),
                                                                   nC, cov, svc=(name == "svc"), nugget=nug),
                                        preds, name)
        if name == "svc":
            k = [b.name for b in m.blocks].index("svc")
            pd.DataFrame({"lon": L.lon, "lat": L.lat, "observed": observed,
                          "eta_tmax": m.block_mean(k)}).to_csv(
                os.path.join(a.out, f"svc_tmax_{a.outcome}.csv"), index=False)
        print(name, json.dumps(res["models"][name].get("test", {})), flush=True)
    if "spatialplus" in models:
        rho = res["models"].get("field", {}).get("hyper", {}).get("field.range_km", 300.0)
        m_star = n_eigen_below(df.lon.to_numpy(), df.lat.to_numpy(), rho)
        between = [f"{v}_between" for v in NATIONAL]
        res["spatialplus_m"] = {"rho_km": float(rho), "m_star": m_star}
        for mm in sorted({10, 20, m_star}):
            d2 = spatial_plus(df.copy(), between, df.lon.to_numpy(), df.lat.to_numpy(), mm)
            tr2, te2 = d2[d2.train], d2[~d2.train]
            m, names = fit(tr2, L, tr2["cell"].to_numpy(), nC, cov, country=True, field=True,
                           barrier=barrier, nugget=nug)
            key = f"spatialplus_m{mm}"
            res["models"][key] = summarise(m, names, te2, test_designs(
                m, te2, L, te2["cell"].to_numpy(), nC, cov, nugget=nug), preds, key)
            print(key, json.dumps(res["models"][key]["test"]), flush=True)
    if "st" in models:
        x0, x1, y0, y1 = REGIONS[a.st_region] if a.st_region in REGIONS else (-180, 180, -90, 90)
        ds = tr[tr.lon.between(x0, x1) & tr.lat.between(y0, y1)].copy()
        # w (and the nugget): same construction as the static field (6-ring buffer, sea
        # barrier); v: AR(1) x Matern field on the observed cells of the box only
        Lw, cw, obs_w = build_lattice(ds.lon.to_numpy(), ds.lat.to_numpy())
        barrier_w = sea_cells(Lw, a.land) if a.land else ~obs_w
        nug_w = None if a.no_nugget else (np.cumsum(obs_w) - 1, int(obs_w.sum()))
        Lv, cv, obs_v = build_lattice(ds.lon.to_numpy(), ds.lat.to_numpy(), buffer=0)
        wk = (ds["week_start"] - ds["week_start"].min()).dt.days // 7
        ds["tstep"] = (wk // a.st_step).to_numpy()
        T = int(ds["tstep"].max() + 1)
        ds["country_id"] = pd.factorize(ds["ISO_A3"])[0]; nCs = int(ds["country_id"].max() + 1)
        st_res = {"region": a.st_region, "n_rows": int(len(ds)), "n_cells": int(obs_v.sum()),
                  "time_steps": T, "step_weeks": a.st_step,
                  "w_lattice": f"observed cells + 6-ring buffer; barrier: {barrier_rule}",
                  "v_lattice": "observed cells of the box only (reflecting boundary)",
                  "cell_nugget": nug_w is not None}
        for name, kw in [("glm", {}), ("field", dict(country=True, field=True)),
                         ("field+st", dict(country=True, field=True, st=(Lv, cv, T)))]:
            m, names = fit(ds, Lw, cw, nCs, cov, barrier=barrier_w, nugget=nug_w, **kw)
            st_res[name] = summarise(m, names)
            print("st/" + name, "delta", st_res[name]["coef"]["lag_own"], flush=True)
        res["models"]["st"] = st_res
    if preds:
        pd.DataFrame({"grid_id": te["grid_id"].to_numpy(), "week_start": te["week_start"].to_numpy(),
                      "y": te["y"].to_numpy().astype("int8"), **preds}).to_parquet(
            os.path.join(a.out, f"test_pred_{a.outcome}.parquet"), index=False)
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
