"""Application figure: observed against predicted, test period 2023-2024, Africa and
the Middle East.

Rows: conflict, unrest. Columns: (left) observed share of test weeks with at least one
event in each cell; (right) mean predicted probability of the hybrid model with the
Matern field (p_field) over the same weeks. Same logarithmic colour scale in each row.
A third column shows calibration: observed frequency against predicted probability in
20 quantile bins, for the hybrid model and the cross-fitted XGBoost offset.

Usage: python paper/make_fitted_figure.py CONFLICT_test_pred.parquet UNREST_test_pred.parquet OUT.png
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
from matplotlib.colors import ListedColormap, LogNorm

BOX = (-20.0, 65.0, -36.0, 45.0)
RES = 0.5
BORDERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ne_50m_countries.geojson")
MUTED, GRID, INK = "#6b6b6b", "#e4e3df", "#1a1a1a"


def grid(cells, col):
    x0, x1, y0, y1 = BOX
    nx, ny = int((x1 - x0) / RES), int((y1 - y0) / RES)
    g = np.full((ny, nx), np.nan)
    i = np.floor((cells.lat - y0) / RES).astype(int).clip(0, ny - 1)
    j = np.floor((cells.lon - x0) / RES).astype(int).clip(0, nx - 1)
    g[i, j] = cells[col].to_numpy()
    return g


def calib(ax, d, col, colour, label):
    q = pd.qcut(d[col].rank(method="first"), 20, labels=False)
    b = d.groupby(q).agg(p=(col, "mean"), o=("y", "mean"))
    ax.plot(b.p, b.o, "o-", color=colour, ms=3.5, lw=1.2, label=label)


def main(conf, unrest, out):
    world = gpd.read_file(BORDERS)
    x0, x1, y0, y1 = BOX
    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.035, 0.32, 0.85], wspace=0.06, hspace=0.28)
    for r, (path, name) in enumerate([(conf, "Violent conflict"), (unrest, "Social unrest")]):
        d = pd.read_parquet(path, columns=["grid_id", "lon", "lat", "y", "p_field", "p_xgb_crossfit"])
        cells = d.groupby("grid_id").agg(lon=("lon", "first"), lat=("lat", "first"),
                                         obs=("y", "mean"), pred=("p_field", "mean"))
        norm = LogNorm(vmin=1 / 400, vmax=1)
        for c, (col, title) in enumerate([("obs", "observed"), ("pred", "predicted, hybrid model")]):
            ax = fig.add_subplot(gs[r, c])
            g = grid(cells, col)
            world.plot(ax=ax, color="#F2F1EE", edgecolor="none", zorder=0)
            if col == "obs":
                ax.imshow(np.where(g == 0, 1.0, np.nan), origin="lower", extent=[x0, x1, y0, y1],
                          cmap=ListedColormap(["#D9D9D9"]), interpolation="nearest", zorder=1)
            im = ax.imshow(np.where(g > 0, g, np.nan), origin="lower", extent=[x0, x1, y0, y1],
                           cmap="magma_r", norm=norm, interpolation="nearest", zorder=2)
            world.boundary.plot(ax=ax, color="#3D3D3D", linewidth=0.3, zorder=3)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"({'abcdef'[3 * r + c]}) {name}: {title}", fontsize=9, loc="left")
        cax = fig.add_subplot(gs[r, 2])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0.003, 0.01, 0.03, 0.1, 0.3, 1]); cb.minorticks_off()
        cb.set_ticklabels(["0.3%", "1%", "3%", "10%", "30%", "100%"]); cb.ax.tick_params(labelsize=8)
        cb.set_label("share of test weeks" if r else "share of test weeks", fontsize=8)
        ax = fig.add_subplot(gs[r, 4])
        calib(ax, d, "p_xgb_crossfit", "#b7321f", "XGBoost, cross-fitted")
        calib(ax, d, "p_field", "#1f5fae", "hybrid model")
        ax.plot([1e-4, 1], [1e-4, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        lo = max(1e-4, min(d.p_field.quantile(0.01), d.p_xgb_crossfit.quantile(0.01)))
        ax.set_xlim(lo, 1); ax.set_ylim(lo, 1)
        ax.set_xlabel("predicted probability (bin mean)", fontsize=8)
        ax.set_ylabel("observed frequency", fontsize=8)
        ax.set_title(f"({'abcdef'[3 * r + 2]}) {name}: calibration", fontsize=9, loc="left")
        ax.grid(color=GRID, lw=0.6); ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:4])
