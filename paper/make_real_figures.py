"""Figures and tables for the application section from results_real/.
Run from the repository root:  python paper/make_real_figures.py"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results_real")
FIG = os.path.join(os.path.dirname(__file__), "figures")
TAB = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a1a", "#6b6b6b", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#1f5fae", "#8fb6e6", "#f0efec", "#ef9a86", "#b7321f"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10})


def field_maps():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), constrained_layout=True)
    fields = {o: pd.read_csv(os.path.join(RES, "spatial", o, "field.csv")) for o in ("conflict", "unrest")}
    vmax = max(np.nanpercentile(np.abs(f.loc[f.observed, "field"]), 99) for f in fields.values())
    for ax, (o, f) in zip(axes, fields.items()):
        f = f[f.observed]
        im = ax.scatter(f.lon, f.lat, c=f.field, s=2.6, marker="s", linewidths=0,
                        cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), rasterized=True)
        ax.set_title(f"{o.capitalize()}: residual field $\\hat w(s)$", loc="left", color=INK)
        ax.set_aspect("equal"); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, label="posterior mean (logit scale)")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIG, "real_field.png"), dpi=200)


def variograms():
    fig, ax = plt.subplots(figsize=(5.5, 3.4), constrained_layout=True)
    for o, col in (("conflict", BLUE), ("unrest", ORANGE)):
        v = pd.read_csv(os.path.join(RES, "spatial", o, "variogram.csv"))
        ax.plot(v.dist_km, v.gamma, color=col, lw=2, marker="o", ms=4)
        ax.annotate(o.capitalize(), (v.dist_km.iloc[-1], v.gamma.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK)
    ax.set_xlabel("great-circle distance (km)"); ax.set_ylabel("semivariance of cell residuals")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(FIG, "real_variogram.png"), dpi=200)


def f(x, d=3):
    return f"{x:.{d}f}"


def tables():
    m = {(j["mode"], j["outcome"]): j for j in
         (json.load(open(p)) for p in glob.glob(os.path.join(RES, "xgb", "metrics_*.json")))}
    rows = []
    for mode, lab in (("original", "As published"), ("corrected", "Corrected")):
        for o in ("Conflict", "Unrest"):
            j = m[(mode, o)]
            rows.append(f"{lab} & {o} & {j['n_test_rows']/1e6:.2f}M & {100*j['test_prevalence']:.2f}\\% & "
                        f"{f(j['auc_pr'])} & {f(j['auc_pr_border_cells'])} & {f(j['auc_pr_interior_cells'])} & "
                        f"{f(j['persistence_auc_pr'])} \\\\")
    body = ("\\begin{tabular}{@{}llrrcccc@{}}\n\\toprule\n"
            "Pipeline & Outcome & Test rows & Prevalence & AUC-PR & Border cells & Interior cells & Persistence \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(TAB, "real_xgb.tex"), "w").write(body)

    rows = []
    for o in ("conflict", "unrest"):
        s = json.load(open(os.path.join(RES, "spatial", o, "summary.json")))
        for k, lab in (("xgb_crossfit", "Cross-fitted XGBoost"), ("xgb+field", "Cross-fitted XGBoost + $u + w$")):
            v = s[k]
            rows.append(f"{o.capitalize()} & {lab} & {f(v['log_score'], 4)} & {f(v['brier'], 5)} & "
                        f"{f(v['auc_pr'])} & {f(v['auc_roc'])} \\\\")
        rows.append(f"{o.capitalize()} & Persistence & -- & -- & {f(s['persistence_auc_pr'])} & -- \\\\")
    body = ("\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
            "Outcome & Model & Log score & Brier & AUC-PR & AUC-ROC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(TAB, "real_hybrid.tex"), "w").write(body)

    rows = []
    for o in ("conflict", "unrest"):
        s = json.load(open(os.path.join(RES, "spatial", o, "summary.json")))
        h, d = s["hyper"], s["diagnostics"]
        rows.append(f"{o.capitalize()} & {f(d['moran']['I'])} & {f(d['border']['within_country'])} & "
                    f"{f(d['border']['across_border'])} & {h['field.range_km']:.0f} & {f(h['field.sd'], 2)} & "
                    f"{f(h['country.sd'], 2)} \\\\")
    body = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Outcome & Moran's $I$ & \\multicolumn{2}{c}{Neighbour residual corr.} & $\\hat\\rho_w$ (km) & "
            "$\\hat\\sigma_w$ & $\\hat\\sigma_u$ \\\\\n & & within country & across border & & & \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(TAB, "real_field.tex"), "w").write(body)


if __name__ == "__main__":
    field_maps(); variograms(); tables()
    print("written:", sorted(os.listdir(FIG)), sorted(os.listdir(TAB)))
