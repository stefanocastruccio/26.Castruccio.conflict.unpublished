"""Hybrid models on the real grid-cell x week table (paper Sec. 3.7 and 5.3).

Input: the clean table written by load_partitions.py. Steps:

  1. Spatially cross-fitted XGBoost offset. Same rows, features and hyperparameters
     as the corrected boosting model (reproduce_xgb.features(mode="corrected"): no
     current-week climate, log national GDP per capita, global calendar-month
     precipitation anomaly) and, like it, no class weight, so the offset is on the
     log-odds scale. Unlike the corrected boosting model, which is trained on the
     whole world, the offset is trained only on the rows of --region (Africa + Middle
     East in the paper), with no subsampling; the outputs record this
     (offset_training_region). Folds are groups of whole
     COUNTRIES (K = 5), so the out-of-fold model never sees the country, and hence
     the national indicators' values, of the cells it predicts. The number of trees
     is chosen by early stopping on 2022 within each fold (training on 2018-2021),
     then the fold model is refitted on 2018-2022.
  2. Residual diagnostics of the offset: Moran's I, great-circle variogram, within- vs
     across-border neighbour correlation (across-border pairs also split into pairs
     predicted by the same fold model and by different fold models). The TEST-period
     (2023-2024) diagnostics are the ones reported ("diagnostics", variogram.csv); the
     training-period ones are kept for reference only ("diagnostics_train",
     variogram_train.csv).
  3. Latent models with the offset, fitted on 2018-2022:
       field : intercept + iid country effects + barrier Matern field (sea = barrier)
       iid   : intercept + iid country effects + iid cell effects (location-aware,
               non-spatial alternative)
  4. Test-period (2023-2024) scores and predictions (test_pred.parquet, kept on the
     cluster) for compare.py, with the cross-fitting fold, ISO_A3, the past 4-week count
     (lag_total) and events_52w (weeks with an event in the previous 52 weeks; 0 marks
     the onset rows); persistence baseline on the same rows.

Barrier rule: --land (country shapefile; sea cells are barriers, land cells without
data keep the range) is required. --barrier-unobserved instead makes every lattice
cell without data a barrier (old default, not the rule of the paper).

Usage:  python fit_acled.py clean.parquet --outcome conflict [--region africa_me]
        --land ne_10m_admin_0_countries.shp [--folds 5] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import average_precision_score

from sc import design as de, diagnostics as dg
from sc.boosting import PAPER_PARAMS
from sc.gmrf import FixedBlock, IIDBlock, MaternBlock, SphereLattice
from sc.laplace import LatentGaussianModel, prob_from_eta
from reproduce_xgb import features, history_52w, LAGS

YCOL = {"conflict": "Conflict", "unrest": "Unrest"}
# named boxes (lon0, lon1, lat0, lat1) for posterior summaries of the field, fixed in
# advance of the fits (not chosen from the fitted field)
BOXES = {"eastern_DRC": (27, 30.5, -5, 2), "northern_Mozambique": (38, 41, -14, -10),
         "Syria_Iraq": (36, 46, 32, 38), "Yemen": (43, 49, 13, 17), "Lake_Chad": (12, 16, 10, 14),
         "central_Sahel": (-5, 5, 11, 17), "Horn_of_Africa": (38, 48, 3, 10), "Sudan": (23, 35, 10, 16),
         "Arabian_interior": (42, 55, 18, 27), "western_Congo_basin": (14, 24, -5, 3),
         "southern_Africa": (18, 32, -34, -22)}
REGIONS = {"africa": (-20, 55, -36, 38), "mena": (-20, 65, 10, 45), "africa_me": (-20, 65, -36, 45),
           "south_asia": (60, 100, 0, 40), "latam": (-120, -30, -56, 33)}


def build_lattice(lon, lat, res=0.5, periodic=None, buffer=6):
    """Lattice over the observed cells plus `buffer` rings of cells without data."""
    if periodic is None:
        periodic = (lon.max() - lon.min() + res) >= 359.9
    pad_x = 0 if periodic else buffer
    lon_c = np.arange(lon.min() - pad_x * res, lon.max() + pad_x * res + res / 2, res)
    lat_c = np.arange(max(lat.min() - buffer * res, -89.75),
                      min(lat.max() + buffer * res, 89.75) + res / 2, res)
    ii = np.rint((lon - lon_c[0]) / res).astype(int)
    jj = np.rint((lat - lat_c[0]) / res).astype(int)
    obs = np.zeros((len(lat_c), len(lon_c)), bool); obs[jj, ii] = True
    mask = obs.copy()
    if buffer > 0:
        from scipy.ndimage import binary_dilation
        if periodic:   # dilate on a longitude-wrapped copy
            ext = np.concatenate([obs[:, -buffer:], obs, obs[:, :buffer]], axis=1)
            mask = binary_dilation(ext, iterations=buffer)[:, buffer:-buffer]
        else:
            mask = binary_dilation(obs, iterations=buffer)
    L = SphereLattice(lon_c, lat_c, mask=mask, periodic_lon=periodic)
    observed = np.zeros(L.n, bool); observed[L.index[jj, ii]] = True
    return L, L.index[jj, ii], observed


def sea_cells(L, shp, res=0.5):
    """True for lattice cells that do not intersect any country polygon (sea)."""
    import geopandas as gpd
    from shapely.geometry import box
    land = gpd.read_file(shp).to_crs(4326)
    cells = gpd.GeoDataFrame(geometry=[box(x - res / 2, y - res / 2, x + res / 2, y + res / 2)
                                       for x, y in zip(L.lon, L.lat)], crs=4326)
    hit = gpd.sjoin(cells, land[["geometry"]], how="inner", predicate="intersects").index
    sea = np.ones(L.n, bool); sea[np.unique(hit)] = False
    return sea


def crossfit_offset(df, X, y, country, folds=5, seed=0, nthread=4):
    """Out-of-fold XGBoost logits with whole countries as fold units."""
    rng = np.random.default_rng(seed)
    cs = np.unique(country)
    folds = min(folds, len(cs))
    fold_of = dict(zip(cs, rng.permutation(len(cs)) % folds))
    fold = np.array([fold_of[c] for c in country])
    year = df["week_start"].dt.year.to_numpy()
    params = dict(PAPER_PARAMS, nthread=nthread, seed=seed)
    off, rounds = np.zeros(len(df)), []
    for k in range(folds):
        out = fold != k
        fit, val = out & (year <= 2021), out & (year == 2022)
        m0 = xgb.train(params, xgb.DMatrix(X[fit], label=y[fit]), 2000,
                       evals=[(xgb.DMatrix(X[val], label=y[val]), "val2022")],
                       early_stopping_rounds=100, verbose_eval=False)
        r = m0.best_iteration + 1
        tr = out & (year <= 2022)
        m = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr]), r)
        off[fold == k] = np.ravel(m.predict(xgb.DMatrix(X[fold == k]), output_margin=True))
        rounds.append(int(r))
    return off, fold, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--outcome", default="conflict")
    ap.add_argument("--region", default=None, help="key of REGIONS; omit for the whole world")
    ap.add_argument("--land", default=None, help="country shapefile; sea cells become barriers")
    ap.add_argument("--barrier-unobserved", action="store_true",
                    help="without --land: every lattice cell without data is a barrier")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_real"); ap.add_argument("--nthread", type=int, default=4)
    a = ap.parse_args()
    if not a.land and not a.barrier_unobserved:
        ap.error("--land SHAPEFILE is required (or pass --barrier-unobserved explicitly)")
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_parquet(a.table)
    df["week_start"] = pd.to_datetime(df["week_start"])
    if a.region:
        x0, x1, y0, y1 = REGIONS[a.region]
        df = df[df.lon.between(x0, x1) & df.lat.between(y0, y1)].copy()
    ycol = YCOL[a.outcome]
    df["events_52w"] = history_52w(df, ycol).astype("float32")   # before any row is dropped
    df, feats = features(df, "corrected")       # same rows/features as reproduce_xgb.py
    df = df.reset_index(drop=True)
    y = df[ycol].to_numpy().astype(int)
    df["y"] = y
    df["lag_total"] = df[LAGS[ycol]].sum(axis=1).fillna(0)
    tr = (df["week_start"].dt.year <= 2022).to_numpy(); te = ~tr
    ppt = [c for c in feats if c.startswith("ppt")]
    month_mean = df[tr].groupby("month")[ppt].mean()
    X = df[feats].copy()
    X[ppt] = X[ppt].to_numpy() - month_mean.loc[df["month"].to_numpy(), ppt].to_numpy()
    X = X.to_numpy(np.float32)

    L, cell, observed = build_lattice(df.lon.to_numpy(), df.lat.to_numpy())
    df["cell"] = cell
    df["country_id"] = pd.factorize(df["ISO_A3"])[0]
    nC = int(df["country_id"].max() + 1)
    barrier = sea_cells(L, a.land) if a.land else ~observed
    print(f"{len(df):,} rows, {observed.sum():,} cells (+{(~observed).sum():,} without data, "
          f"{barrier.sum():,} barrier), {nC} countries, {len(feats)} features, "
          f"prevalence {y.mean():.4f}", flush=True)

    # 1. offset
    off, fold, rounds = crossfit_offset(df, X, y, df["ISO_A3"].to_numpy(), a.folds, a.seed, a.nthread)
    np.save(os.path.join(a.out, "offset.npy"), off)

    # 2. residual diagnostics of the offset: test period (reported) and training period
    d_tr, d_te = df[tr], df[te]
    fold_cell = pd.Series(fold).groupby(df["cell"].to_numpy()).first()
    country_cell_all = df.groupby("cell")["country_id"].first()
    diags = {}
    for per, d_p, o_p, fn in [("test", d_te, off[te], "variogram.csv"),
                              ("train", d_tr, off[tr], "variogram_train.csv")]:
        r_cell = dg.cell_pearson(d_p, expit(o_p))
        r_full = np.full(L.n, np.nan); r_full[r_cell.index] = r_cell.to_numpy()
        ok = ~np.isnan(r_full)
        adj_ok = L.adjacency[ok][:, ok]
        idx = np.flatnonzero(ok)
        diags[per] = {"period": "2023-2024" if per == "test" else "2018-2022",
                      "moran": dg.morans_i(r_full[ok], adj_ok, n_perm=999),
                      "border": dg.border_contrast(r_full[ok], adj_ok, country_cell_all.reindex(idx).to_numpy(),
                                                   fold=fold_cell.reindex(idx).to_numpy())}
        dg.variogram(r_full[ok], L.lon[ok], L.lat[ok], np.arange(0, 3000, 100)).to_csv(
            os.path.join(a.out, fn), index=False)

    # 3. latent models
    ones_tr, ones_te = np.ones((tr.sum(), 1)), np.ones((te.sum(), 1))
    Dc_tr = de.index_design(d_tr["country_id"], nC); Dc_te = de.index_design(d_te["country_id"], nC)
    Dw_tr = de.index_design(d_tr["cell"], L.n); Dw_te = de.index_design(d_te["cell"], L.n)
    specs = {
        "field": ([FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country"),
                   MaternBlock(L, name="field", range_init=300.0, barrier=barrier)],
                  [ones_tr, Dc_tr, Dw_tr], [ones_te, Dc_te, Dw_te]),
        "iid": ([FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country"),
                 IIDBlock(L.n, name="cell", sd_init=0.5)],
                [ones_tr, Dc_tr, Dw_tr], [ones_te, Dc_te, Dw_te]),
    }
    y_te = d_te["y"].to_numpy()
    res = {"outcome": a.outcome, "region": a.region, "rows": int(len(df)), "test_rows": int(te.sum()),
           "cells": int(observed.sum()), "barrier_cells": int(barrier.sum()),
           "barrier_rule": "sea cells from country shapefile" if a.land else "all cells without data",
           "folds": "countries", "offset_rounds_per_fold": rounds,
           "offset_training_region": {"region": a.region or "world",
                                      "box": list(REGIONS[a.region]) if a.region else None,
                                      "note": "cross-fitted offset trained on the rows of this region only"},
           "offset_class_weight": None,
           "diagnostics": diags["test"], "diagnostics_train": diags["train"],
           "xgb_crossfit": dg.scores(y_te, expit(off[te])),
           "persistence_auc_pr": float(average_precision_score(y_te, d_te["lag_total"].to_numpy())),
           "models": {}}
    preds = {"p_xgb_crossfit": expit(off[te]).astype("float32")}
    for name, (blocks, D_tr, D_te) in specs.items():
        m = LatentGaussianModel(d_tr["y"].to_numpy(), blocks, D_tr, offset=off[tr]).fit()
        eta, sd = m.predict_eta(D_te, off[te], n_samples=100)
        p = prob_from_eta(eta, sd)
        res["models"][name] = {"scores": dg.scores(y_te, p), "hyper": m.hyper(),
                               "log_ml": float(m.log_ml), "seconds": float(m.fit_seconds)}
        preds[f"p_{name}"] = p.astype("float32")
        if name == "field":
            S = m.sample(1000, np.random.default_rng(a.seed))[m.slices[2]]
            pd.DataFrame({"lon": L.lon, "lat": L.lat, "observed": observed, "barrier": barrier,
                          "field": m.block_mean(2), "field_sd": S.std(axis=1),
                          "p_positive": (S > 0).mean(axis=1)}).to_csv(
                os.path.join(a.out, "field.csv"), index=False)
            boxes = {}
            for bn, (x0, x1, y0, y1) in BOXES.items():
                sel = observed & (L.lon >= x0) & (L.lon <= x1) & (L.lat >= y0) & (L.lat <= y1)
                if sel.sum() == 0:
                    continue
                avg = S[sel].mean(axis=0)
                boxes[bn] = {"box": [x0, x1, y0, y1], "cells": int(sel.sum()), "mean": float(avg.mean()),
                             "sd": float(avg.std()), "p_positive": float((avg > 0).mean())}
            res["models"][name]["boxes"] = boxes
            # summary not based on chosen boxes: share of observed cells whose posterior
            # P(w > 0) exceeds 0.95 or is below 0.05
            pp = (S[observed] > 0).mean(axis=1)
            res["models"][name]["cells_p_positive"] = {
                "n_cells": int(observed.sum()), "share_above_0.95": float((pp > 0.95).mean()),
                "share_below_0.05": float((pp < 0.05).mean())}
        print(name, json.dumps(res["models"][name]["scores"]), flush=True)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    pd.DataFrame({"grid_id": d_te["grid_id"].to_numpy(), "week_start": d_te["week_start"].to_numpy(),
                  "lon": d_te["lon"].to_numpy(), "lat": d_te["lat"].to_numpy(),
                  "ISO_A3": d_te["ISO_A3"].astype(str).to_numpy(),
                  "y": y_te.astype("int8"), "lag_total": d_te["lag_total"].to_numpy(dtype="float32"),
                  "events_52w": d_te["events_52w"].to_numpy(dtype="float32"),
                  "fold": fold[te].astype("int16"),
                  **preds}).to_parquet(os.path.join(a.out, "test_pred.parquet"), index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=1, default=float))


if __name__ == "__main__":
    main()
