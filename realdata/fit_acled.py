"""Apply the residual-field pipeline to the real grid-cell x week table.

Expected input: one Parquet file with one row per (0.5 deg cell, week) holding the
outcome, the paper's predictors and the cell centre. Edit COLUMNS to match the
student's column names. Steps:

  1. cross-fitted XGBoost offset (5 deg spatial blocks, paper hyperparameters,
     NO class weighting so probabilities stay calibrated)
  2. residual diagnostics: Moran's I, variogram, temporal ACF, within/across border
  3. latent model: logit p = XGB offset + intercept + country effect + Matern field
     on the sphere (global 0.5 deg land lattice, periodic in longitude)
  4. held-out (2023-2024) scores for XGB vs XGB + field, and the persistence baseline

Usage:  python fit_acled.py data.parquet --outcome conflict [--region africa]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.special import expit

from sc import boosting as bo, design as de, diagnostics as dg
from sc.gmrf import FixedBlock, IIDBlock, MaternBlock, SphereLattice
from sc.laplace import LatentGaussianModel, prob_from_eta

# ---- column names of the table written by load_partitions.py -----------------
COLUMNS = dict(lon="lon", lat="lat", week="week_start", country="ISO_A3",
               outcomes={"conflict": "Conflict", "unrest": "Unrest"},
               lag_total={"conflict": ["Battles_rollsum_4", "ExplosionsRemote_violence_rollsum_4",
                                       "Violence_against_civilians_rollsum_4"],
                          "unrest": ["Riots_rollsum_4", "Protests_rollsum_4"]})
ID_COLUMNS = {"grid_id", "cell_id", "t", "country_is_proxy", "n_copies", "Strategic_developments"}
FEATURES = None   # None = every numeric column except ids/outcomes
REGIONS = {"africa": (-20, 55, -36, 38), "mena": (-20, 65, 10, 45), "africa_me": (-20, 65, -36, 45),
           "south_asia": (60, 100, 0, 40), "latam": (-120, -30, -56, 33)}
TRAIN_END = "2023-01-01"


def build_lattice(lon, lat, res=0.5, periodic=None, buffer=6):
    """Lattice over the observed cells plus `buffer` rings of unobserved cells
    (sea). The unobserved cells carry no data and are used as *barrier* cells
    (short range, Bakka et al. 2019): coastal cells do not get the inflated
    variance of a reflecting boundary, and correlation does not leak across
    straits (paper, Remark 1)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("--outcome", default="conflict")
    ap.add_argument("--region", default=None, help="key of REGIONS; omit for the whole world")
    ap.add_argument("--out", default="results_real"); ap.add_argument("--nthread", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C = COLUMNS
    df = pd.read_parquet(a.parquet)
    if a.region:
        x0, x1, y0, y1 = REGIONS[a.region]
        df = df[df[C["lon"]].between(x0, x1) & df[C["lat"]].between(y0, y1)].copy()
    df["y"] = (df[C["outcomes"][a.outcome]] > 0).astype(int)
    df["lag_total"] = df[C["lag_total"][a.outcome]].sum(axis=1)
    t = pd.to_datetime(df[C["week"]])
    df["train"] = t < pd.Timestamp(TRAIN_END)

    L, cell, observed = build_lattice(df[C["lon"]].to_numpy(), df[C["lat"]].to_numpy())
    df["cell"] = cell
    df["country_id"] = pd.factorize(df[C["country"]])[0]
    drop = {C["lon"], C["lat"], C["week"], C["country"], "y", "train", "cell",
            "country_id", *C["outcomes"].values(), *ID_COLUMNS}
    feats = FEATURES or [c for c in df.select_dtypes("number").columns if c not in drop]
    print(f"{len(df):,} rows, {observed.sum():,} observed + {(~observed).sum():,} buffer cells, {len(feats)} features, prevalence {df.y.mean():.4f}")

    # 1. cross-fitted offset
    blk = bo.spatial_blocks(df[C["lon"]], df[C["lat"]], 5.0)
    off = bo.crossfit_offset(df, feats, df["train"].to_numpy(), blk, nthread=a.nthread)
    np.save(os.path.join(a.out, "offset.npy"), off)

    # 2. diagnostics on the training period
    tr = df["train"].to_numpy(); te = ~tr
    d_tr = df[tr]
    r_cell = dg.cell_pearson(d_tr, expit(off[tr]))
    r_full = np.full(L.n, np.nan); r_full[r_cell.index] = r_cell.to_numpy()
    ok = ~np.isnan(r_full)
    adj_ok = L.adjacency[ok][:, ok]
    country_cell = df.groupby("cell")["country_id"].first().reindex(np.flatnonzero(ok)).to_numpy()
    diag = {"moran": dg.morans_i(r_full[ok], adj_ok, n_perm=199),
            "border": dg.border_contrast(r_full[ok], adj_ok, country_cell)}
    dg.variogram(r_full[ok], L.lon[ok], L.lat[ok], np.arange(0, 3000, 100)).to_csv(
        os.path.join(a.out, "variogram.csv"), index=False)
    print(json.dumps(diag, indent=1))

    # 3. latent field on top of the offset
    nC = df["country_id"].max() + 1
    m = LatentGaussianModel(
        d_tr["y"].to_numpy(),
        [FixedBlock(1, ["intercept"]), IIDBlock(nC, name="country"),
         MaternBlock(L, name="field", range_init=500.0, barrier=~observed)],
        [np.ones((tr.sum(), 1)), de.index_design(d_tr["country_id"], nC),
         de.index_design(d_tr["cell"], L.n)],
        offset=off[tr]).fit(verbose=True)
    print(m.hyper())
    pd.DataFrame({"lon": L.lon, "lat": L.lat, "observed": observed, "field": m.block_mean(2),
                  "field_sd": m.block_sd(2, n=200)}).to_csv(os.path.join(a.out, "field.csv"), index=False)

    # 4. held-out scores
    d_te = df[te]
    eta, sd = m.predict_eta([np.ones((te.sum(), 1)), de.index_design(d_te["country_id"], nC),
                             de.index_design(d_te["cell"], L.n)], off[te], n_samples=100)
    sc = {"xgb_crossfit": dg.scores(d_te["y"].to_numpy(), expit(off[te])),
          "xgb+field": dg.scores(d_te["y"].to_numpy(), prob_from_eta(eta, sd)),
          "persistence_auc_pr": dg.scores(d_te["y"].to_numpy(), d_te["lag_total"].to_numpy()
                                          + 1e-6 * np.random.default_rng(0).random(te.sum()))["auc_pr"],
          "hyper": m.hyper(), "diagnostics": diag}
    json.dump(sc, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    # test-period predictions (stay on the cluster; used by compare.py)
    pd.DataFrame({"grid_id": d_te["grid_id"].to_numpy(), "week_start": d_te[C["week"]].to_numpy(),
                  "y": d_te["y"].to_numpy().astype("int8"),
                  "lag_total": d_te["lag_total"].to_numpy(dtype="float32"),
                  "p_xgb_crossfit": expit(off[te]).astype("float32"),
                  "p_hybrid": prob_from_eta(eta, sd).astype("float32")}
                 ).to_parquet(os.path.join(a.out, "test_pred.parquet"), index=False)
    print(json.dumps(sc, indent=1, default=float))


if __name__ == "__main__":
    main()
