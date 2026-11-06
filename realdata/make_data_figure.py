"""Data figure for Section 2: where and when events are recorded.

(a), (b) share of covered weeks with at least one event in each 0.5-degree cell,
         2018-2024, for conflict and unrest (log colour scale; cells with no event
         week in light grey);
(c)      weekly share of covered cells with at least one event, both outcomes.

Also writes OUT.json with the summary numbers quoted in the text.

Usage: python make_data_figure.py CLEAN.parquet BORDERS OUT.png
BORDERS: any country polygon file readable by geopandas, e.g. the repository's
paper/data/ne_50m_countries.geojson (Natural Earth 1:50m, public domain).
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

OUTCOMES = [("Conflict", "Violent conflict", "#B2182B"), ("Unrest", "Social unrest", "#2166AC")]
RES = 0.5


def grid(cells, col):
    lon0, lat0 = -180 + RES / 2, -90 + RES / 2
    nx, ny = int(360 / RES), int(180 / RES)
    g = np.full((ny, nx), np.nan)
    i = np.round((cells.lat - lat0) / RES).astype(int).clip(0, ny - 1)
    j = np.round((cells.lon - lon0) / RES).astype(int).clip(0, nx - 1)
    g[i, j] = cells[col].to_numpy()
    return g


def main(path, shp, out):
    df = pd.read_parquet(path, columns=["grid_id", "lon", "lat", "week_start", "Conflict", "Unrest"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    cells = df.groupby("grid_id").agg(lon=("lon", "first"), lat=("lat", "first"),
                                      Conflict=("Conflict", "mean"), Unrest=("Unrest", "mean"),
                                      weeks=("Conflict", "size"))
    weekly = df.groupby("week_start")[["Conflict", "Unrest"]].mean()
    summary = {"rows": int(len(df)), "cells": int(len(cells)), "weeks": int(df.week_start.nunique())}
    for o, _, _ in OUTCOMES:
        s = cells[o]
        summary[o] = {"overall_share_event_weeks": float(df[o].mean()),
                      "share_cells_any_event": float((s > 0).mean()),
                      "share_cells_event_in_half_of_weeks": float((s >= 0.5).mean()),
                      "top1pct_cells_share_of_event_weeks": float(
                          np.sort((s * cells.weeks).to_numpy())[::-1][:max(1, len(s) // 100)].sum()
                          / (s * cells.weeks).sum()),
                      "weekly_share_min": float(weekly[o].min()), "weekly_share_max": float(weekly[o].max())}

    import geopandas as gpd
    world = gpd.read_file(shp)          # borders are required: they give the maps their context

    fig = plt.figure(figsize=(12, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.62], width_ratios=[1, 0.025], hspace=0.28, wspace=0.03)
    ext = [-180, 180, -90, 90]
    lat_lo = max(-58, cells.lat.min() - 2)
    for k, (o, lab, _) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[k, 0])
        world.plot(ax=ax, color="#F2F1EE", edgecolor="none", zorder=0)   # land without data
        g = grid(cells, o)
        zero = np.where(g == 0, 1.0, np.nan)
        ax.imshow(zero, origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap(["#D9D9D9"]),
                  interpolation="nearest", zorder=1)
        pos = np.where(g > 0, g, np.nan)
        im = ax.imshow(pos, origin="lower", extent=ext, cmap="magma_r", interpolation="nearest", zorder=2,
                       norm=LogNorm(vmin=1 / 400, vmax=1))
        world.boundary.plot(ax=ax, color="#3D3D3D", linewidth=0.3, zorder=3)
        ax.set_xlim(max(-180, cells.lon.min() - 2), min(180, cells.lon.max() + 2))
        ax.set_ylim(lat_lo, min(84, cells.lat.max() + 2))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({'ab'[k]}) {lab}: share of weeks with at least one event, 2018–2024",
                     fontsize=10, loc="left")
        cax = fig.add_subplot(gs[k, 1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0.003, 0.01, 0.03, 0.1, 0.3, 1]); cb.set_ticklabels(["0.3%", "1%", "3%", "10%", "30%", "100%"])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=8)
    ax = fig.add_subplot(gs[2, 0])
    for o, lab, col in OUTCOMES:
        ax.plot(weekly.index, 100 * weekly[o], color=col, lw=1.2, label=lab)
    ax.axvline(pd.Timestamp("2023-01-01"), color="#4D4D4D", lw=0.8, ls="--")
    ax.text(pd.Timestamp("2023-01-15"), ax.get_ylim()[1] * 0.92, "test period", fontsize=8, color="#4D4D4D")
    ax.set_ylabel("cells with an event (%)", fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              borderaxespad=0.1)
    ax.set_title("(c) Weekly share of covered cells with at least one event", fontsize=10, loc="left", pad=22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6); ax.tick_params(labelsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    json.dump(summary, open(out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
