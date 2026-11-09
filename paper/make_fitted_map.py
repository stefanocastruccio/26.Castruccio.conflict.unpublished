"""Application figure: fitted intensity and latent episodes of the extended joint model (biv2)
for one time step of the shared space-time field.

(a), (b) fitted probability of a conflict / unrest event in a cell-week, averaged over the weeks
         of the step (plug-in posterior mode, training period), with the cells that recorded at
         least one event in the step marked by dots;
(c)      shared space-time field v at that step (posterior mean, logit scale, 1-degree cells);
(d)      unrest-specific space-time field at that step.

Runs on the cluster (it reads the event table). Usage:
  python paper/make_fitted_map.py CLEAN.parquet FITTED.parquet FIELDS_ST.csv.gz STEP OUT.png [--region ame]
FITTED.parquet and FIELDS_ST.csv.gz are written by realdata/fit_bivariate.py
(fitted_biv2.parquet in --out, fields_biv2_st.csv.gz).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm

BORDERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ne_50m_countries.geojson")
RES = 0.5


def raster(lon, lat, val, box, res):
    x0, x1, y0, y1 = box
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    g = np.full((ny, nx), np.nan)
    i = np.floor((np.asarray(lat) - y0) / res).astype(int).clip(0, ny - 1)
    j = np.floor((np.asarray(lon) - x0) / res).astype(int).clip(0, nx - 1)
    g[i, j] = np.asarray(val)
    return g


def style(ax, world, box, title):
    world.boundary.plot(ax=ax, color="#3D3D3D", linewidth=0.3, zorder=3)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")


def main(clean, fitted, fields, step, out):
    step = int(step)
    f = pd.read_parquet(fitted)
    f = f[f.step == step]
    weeks = f.week_start.unique()
    fx = f.groupby("grid_id").agg(lon=("lon", "first"), lat=("lat", "first"),
                                  p_conflict=("p_conflict", "mean"), p_unrest=("p_unrest", "mean"))
    ev = pd.read_parquet(clean, columns=["grid_id", "week_start", "Conflict", "Unrest"])
    ev["week_start"] = pd.to_datetime(ev["week_start"])
    ev = ev[ev.week_start.isin(pd.to_datetime(weeks)) & ev.grid_id.isin(fx.index)]
    obs = ev.groupby("grid_id")[["Conflict", "Unrest"]].max().reindex(fx.index).fillna(0)
    st = pd.read_csv(fields)
    st = st[st.step == step]
    start, end = pd.to_datetime(weeks).min(), pd.to_datetime(weeks).max() + pd.Timedelta(days=6)
    box = (np.floor(fx.lon.min() - 1), np.ceil(fx.lon.max() + 1), np.floor(fx.lat.min() - 1), np.ceil(fx.lat.max() + 1))
    world = gpd.read_file(BORDERS)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), constrained_layout=True)
    for ax, o, lab, k in [(axes[0, 0], "conflict", "Violent conflict", "a"), (axes[0, 1], "unrest", "Social unrest", "b")]:
        world.plot(ax=ax, color="#F2F1EE", edgecolor="none", zorder=0)
        g = raster(fx.lon, fx.lat, fx[f"p_{o}"], box, RES)
        im = ax.imshow(g, origin="lower", extent=box, cmap="magma_r", norm=LogNorm(vmin=1e-3, vmax=0.5),
                       interpolation="nearest", zorder=1)
        hit = obs[o.capitalize()] > 0
        ax.scatter(fx.lon[hit], fx.lat[hit], s=1.5, c="#00A0B0", linewidths=0, zorder=4)
        style(ax, world, box, f"({k}) {lab}: fitted probability per cell-week")
    cb = fig.colorbar(im, ax=axes[0, :], shrink=0.8, pad=0.01)
    cb.set_ticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3]); cb.set_ticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    cb.minorticks_off(); cb.ax.tick_params(labelsize=8)
    vmax = float(np.nanquantile(np.abs(np.r_[st.shared_st, st.get("st_unrest", st.shared_st)]), 0.99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for ax, col, lab, k in [(axes[1, 0], "shared_st", "Shared episodes $v$", "c"),
                            (axes[1, 1], "st_unrest", "Unrest-only episodes", "d")]:
        world.plot(ax=ax, color="#F2F1EE", edgecolor="none", zorder=0)
        g = raster(st.lon - 0.5, st.lat - 0.5, st[col], box, 1.0)   # 1-degree cells: centres -> lower-left
        im2 = ax.imshow(g, origin="lower", extent=box, cmap="RdBu_r", norm=norm, interpolation="nearest", zorder=1)
        style(ax, world, box, f"({k}) {lab} (logit scale)")
    cb2 = fig.colorbar(im2, ax=axes[1, :], shrink=0.8, pad=0.01)
    cb2.ax.tick_params(labelsize=8)
    fig.suptitle(f"{start:%d %b %Y} to {end:%d %b %Y}", fontsize=10, x=0.02, ha="left")
    fig.savefig(out, dpi=200)
    print("step", step, start.date(), end.date(), "cells", len(fx), "conflict cells", int(obs.Conflict.sum()),
          "unrest cells", int(obs.Unrest.sum()))


if __name__ == "__main__":
    main(*sys.argv[1:6])
