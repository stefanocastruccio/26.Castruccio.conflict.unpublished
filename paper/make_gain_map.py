"""Map of the 2023-2024 gain in joint log score of the joint model over the product of
the univariate static models, by country (AME region), with the windows selected on 2022.

Input: results_real/bivariate/selection/test2324_by_country.csv, val2022_windows.csv
Output: figures/real_gain_map.png
"""
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "..", "results_real", "bivariate", "selection")
INK, MUTED = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED})


def main():
    c = pd.read_csv(os.path.join(SEL, "test2324_by_country.csv"))
    w = pd.read_csv(os.path.join(SEL, "val2022_windows.csv")).head(5)
    world = gpd.read_file(os.path.join(HERE, "data", "ne_50m_countries.geojson"))
    key = np.where(world.ISO_A3.isin(c.iso), world.ISO_A3, world.ADM0_A3)
    world = world.assign(iso=key).merge(c, on="iso", how="left")
    world["g"] = 1e3 * world.gain_mean
    lim = 30.0
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    world[world.g.isna()].plot(ax=ax, color="#f2f1ee", edgecolor="#c9c7c1", lw=0.3)
    world[world.g.notna()].plot(ax=ax, column="g", cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim),
                                edgecolor="#8a8883", lw=0.3, legend=True,
                                legend_kwds={"label": r"gain in joint log score per cell-week ($\times 10^{-3}$)",
                                             "shrink": 0.6, "extend": "both"})
    for k, r in enumerate(w.itertuples()):
        ax.add_patch(Rectangle((r.lon0, r.lat0), r.lon1 - r.lon0, r.lat1 - r.lat0, fill=False,
                               ec=INK, lw=1.4 if k == 0 else 0.7, ls="-" if k == 0 else "--"))
    ax.set_xlim(-20, 75); ax.set_ylim(-36, 45)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(os.path.join(HERE, "figures", "real_gain_map.png"), dpi=200)


if __name__ == "__main__":
    main()
