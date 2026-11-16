"""Map and interval plot of the country escalation level kappa_c = kappa + u_c (unrest -> conflict) from the two-stage
country-effects fit (Task 50, results_real/bivariate/<region>/ce2s_v2svstterr.json).

  python realdata/plot_country_kappa.py <countries.shp> [region] [direction]

Left: map of the posterior mean of kappa_c, diverging around the regional kappa (blue below, red above, gray = the
regional value). Right: every country's 95% interval, sorted; the dashed line is the regional kappa, the solid one 0.
Countries whose interval excludes the regional kappa are drawn darker; the interval also carries the uncertainty of
kappa itself, so this is conservative for the deviation u_c.
"""
import json
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

shp = sys.argv[1]
region = sys.argv[2] if len(sys.argv) > 2 else "ame"
direction = sys.argv[3] if len(sys.argv) > 3 else "conflict"
root = os.path.join(os.path.dirname(__file__), "..")
d = json.load(open(os.path.join(root, f"results_real/bivariate/{region}/ce2s_v2svstterr.json")))["models"]["biv3_cs"]
kap = d["fixed_ci95"][f"{direction}:lag_other"]
tau = d["random"][f"k_{direction}_u"][f"k_{direction}_u.sd"]
p = d["tau_test"][f"k_{direction}_u"]["p_boundary"]
rows = [(iso, v[f"{direction}_u"]) for iso, v in d["countries"].items()]
iso = np.array([r[0] for r in rows]); mean = np.array([r[1]["mean"] for r in rows])
lo = np.array([r[1]["lo"] for r in rows]); hi = np.array([r[1]["hi"] for r in rows])
k0 = kap["est"]
off = (lo > k0) | (hi < k0)

world = gpd.read_file(shp)
code = "ADM0_A3" if "ADM0_A3" in world else "ISO_A3"
ALIAS = {"SAH": "ESH", "PSX": "PSE", "SDS": "SSD", "KAB": "BRT"}   # Natural Earth ADM0_A3 -> codes in the table
world[code] = world[code].replace(ALIAS)
reg = world[world[code].isin(iso)].copy()
reg["kc"] = reg[code].map(dict(zip(iso, mean)))
missing = sorted(set(iso) - set(reg[code]))

span = max(abs(mean - k0).max(), 1e-3)
cmap = LinearSegmentedColormap.from_list("div", ["#2166ac", "#92c5de", "#f0efec", "#f4a582", "#b2182b"])
norm = TwoSlopeNorm(vcenter=k0, vmin=k0 - span, vmax=k0 + span)
lab = "unrest → conflict" if direction == "conflict" else "conflict → unrest"

fig = plt.figure(figsize=(12, 7.2))
ax = fig.add_axes([0.02, 0.08, 0.56, 0.86])
x0, y0, x1, y1 = reg.total_bounds
world.cx[x0 - 5:x1 + 5, y0 - 5:y1 + 5].plot(ax=ax, color="#e6e6e3", edgecolor="white", linewidth=0.3)
reg.plot(ax=ax, column="kc", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.4)
reg[reg[code].isin(iso[off])].boundary.plot(ax=ax, color="#222222", linewidth=1.0)
ax.set_xlim(x0 - 1, x1 + 1); ax.set_ylim(y0 - 1, y1 + 1); ax.set_axis_off()
ax.set_title(f"Country escalation level $\\kappa_c=\\kappa+u_c$ ({lab})", fontsize=11, loc="left")
cax = fig.add_axes([0.08, 0.06, 0.42, 0.022])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_label(f"posterior mean; gray = regional $\\kappa$ = {k0:.3f}; black outline: 95% interval excludes it", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

ax2 = fig.add_axes([0.66, 0.08, 0.32, 0.86])
o = np.argsort(mean); y = np.arange(len(o))
for j, i in enumerate(o):
    c = cmap(norm(mean[i]))
    ax2.plot([lo[i], hi[i]], [j, j], color=c if off[i] else "#b5b5b0", lw=2 if off[i] else 1.2, solid_capstyle="round")
    ax2.plot(mean[i], j, "o", ms=4, color=c if off[i] else "#8a8a85", mec="none")
ax2.axvline(k0, color="#555555", lw=1, ls="--"); ax2.axvline(0, color="#222222", lw=0.8)
ax2.axvspan(kap["lo"], kap["hi"], color="#d9d9d5", alpha=0.5, lw=0)
ax2.set_yticks(y); ax2.set_yticklabels(iso[o], fontsize=5.5)
ax2.set_ylim(-1, len(o)); ax2.tick_params(axis="x", labelsize=8)
ax2.set_xlabel("$\\kappa_c$, 95% interval", fontsize=9)
ax2.set_title(f"$\\tau_u$ = {tau['est']:.3f} [{tau['lo']:.3f}, {tau['hi']:.3f}], boundary p = {p:.3f}", fontsize=9, loc="left")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
out = os.path.join(root, f"paper/notes/fig_country_kappa_{region}_{direction}.png")
fig.savefig(out, dpi=200)
print(out, "countries", len(iso), "on map", len(reg), "missing from shapefile", missing,
      "interval excludes regional kappa:", sorted(zip(iso[off], np.round(mean[off], 3))))
