"""Data figure (Section 2): where and when ACLED events are recorded, in the analysis
region (Africa and the Middle East, lon -20..65, lat -36..45).

(a), (b) share of covered weeks with at least one event in each 0.5-degree cell,
         2018-2024, for violent conflict and social unrest (log colour scale; grey:
         cells with no event week);
(c)      weekly share of covered cells in the region with at least one event.

Also writes OUT.json with summary numbers for the region and for the whole grid, and
lists the weeks whose share falls far below the median (for data checks).

Usage: python paper/make_data_figure.py CLEAN.parquet OUT.png
       (country borders: paper/data/ne_50m_countries.geojson, Natural Earth 1:50m)
"""
from __future__ import annotations

import json
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
BORDERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ne_50m_countries.geojson")
OUTCOMES = [("Conflict", "Violent conflict", "#B2182B"), ("Unrest", "Social unrest", "#2166AC")]
RES = 0.5


def summarise(df, cells):
    out = {"rows": int(len(df)), "cells": int(len(cells)), "weeks": int(df.week_start.nunique())}
    for o, _, _ in OUTCOMES:
        s, n = cells[o], cells["weeks"]
        ev = np.sort((s * n).to_numpy())[::-1]
        out[o] = {"share_event_weeks": float(df[o].mean()),
                  "share_cells_any_event": float((s > 0).mean()),
                  "share_cells_event_in_half_of_weeks": float((s >= 0.5).mean()),
                  "top1pct_cells_share_of_event_weeks": float(ev[:max(1, len(ev) // 100)].sum() / ev.sum())}
    return out


def main(path, out):
    df = pd.read_parquet(path, columns=["grid_id", "lon", "lat", "week_start", "Conflict", "Unrest"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    agg = dict(lon=("lon", "first"), lat=("lat", "first"), Conflict=("Conflict", "mean"),
               Unrest=("Unrest", "mean"), weeks=("Conflict", "size"))
    summary = {"global": summarise(df, df.groupby("grid_id").agg(**agg))}
    x0, x1, y0, y1 = BOX
    df = df[df.lon.between(x0, x1) & df.lat.between(y0, y1)]
    cells = df.groupby("grid_id").agg(**agg)
    weekly = df.groupby("week_start")[["Conflict", "Unrest"]].mean()
    ncell = df.groupby("week_start").size()
    summary["africa_me"] = summarise(df, cells)
    for o, _, _ in OUTCOMES:
        w = weekly[o]
        summary["africa_me"][o].update(weekly_share_min=float(w.min()), weekly_share_max=float(w.max()),
                                       weekly_share_median=float(w.median()))
        low = w[w < 0.6 * w.median()]
        summary["africa_me"][o]["weeks_below_60pct_of_median"] = {
            str(k.date()): {"share": float(v), "cells_in_week": int(ncell[k])} for k, v in low.items()}

    world = gpd.read_file(BORDERS)
    nx, ny = int((x1 - x0) / RES), int((y1 - y0) / RES)
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.42], width_ratios=[1, 1, 0.035], hspace=0.22, wspace=0.06)
    for k, (o, lab, _) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[0, k])
        g = np.full((ny, nx), np.nan)
        i = np.floor((cells.lat - y0) / RES).astype(int).clip(0, ny - 1)
        j = np.floor((cells.lon - x0) / RES).astype(int).clip(0, nx - 1)
        g[i, j] = cells[o].to_numpy()
        world.plot(ax=ax, color="#F2F1EE", edgecolor="none", zorder=0)
        ax.imshow(np.where(g == 0, 1.0, np.nan), origin="lower", extent=[x0, x1, y0, y1],
                  cmap=ListedColormap(["#D9D9D9"]), interpolation="nearest", zorder=1)
        im = ax.imshow(np.where(g > 0, g, np.nan), origin="lower", extent=[x0, x1, y0, y1], cmap="magma_r",
                       norm=LogNorm(vmin=1 / 400, vmax=1), interpolation="nearest", zorder=2)
        world.boundary.plot(ax=ax, color="#3D3D3D", linewidth=0.35, zorder=3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({'ab'[k]}) {lab}", fontsize=10, loc="left")
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.003, 0.01, 0.03, 0.1, 0.3, 1]); cb.minorticks_off()
    cb.set_ticklabels(["0.3%", "1%", "3%", "10%", "30%", "100%"])
    cb.set_label("share of weeks with at least one event", fontsize=9); cb.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[1, :2])
    for o, lab, col in OUTCOMES:
        ax.plot(weekly.index, 100 * weekly[o], color=col, lw=1.1, label=lab)
    ax.axvline(pd.Timestamp("2023-01-01"), color="#4D4D4D", lw=0.8, ls="--")
    ax.text(pd.Timestamp("2023-01-20"), 0.93, "test period", fontsize=8, color="#4D4D4D",
            transform=ax.get_xaxis_transform())
    ax.set_ylabel("cells with an event (%)", fontsize=9)
    ax.set_title("(c) Weekly share of covered cells with at least one event", fontsize=10, loc="left", pad=20)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6); ax.tick_params(labelsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    json.dump(summary, open(out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
