"""Dot-and-interval figures for S2 (national indicators), S4 (biome climate effects) and
S5 (escalation or shared episodes), replacing the corresponding tables in the main text (the full tables go to the
supplement). Run from the repository root:  python paper/make_sim_figures.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e3df"
LATENT, BOOST = "#1f5fae", "#b7321f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10})


def dots(ax, rows, truth, xlim=None):
    """rows: list of (label, estimates array, coverage or None, colour)."""
    for k, (lab, est, cov, col) in enumerate(rows):
        y = len(rows) - 1 - k
        m, lo, hi = np.mean(est), np.quantile(est, 0.05), np.quantile(est, 0.95)
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(m, y, "o", color=col, ms=5, mec="white", mew=0.8, zorder=3)
        if cov is not None:
            ax.text(hi, y + 0.28, f"{cov:.2f}", va="center", ha="right", fontsize=7.5, color=MUTED)
    ax.axvline(truth, color=INK, lw=0.9, ls="--")
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xlim:
        ax.set_xlim(*xlim)


def s2():
    r = pd.read_csv(os.path.join(RES, "s2_raw.csv"))
    r["covers"] = (r.lo <= r.truth) & (r.truth <= r.hi)
    labels = [("glm", "Logistic regression"), ("country", "+ country effects $u$"),
              ("field", "+ spatial field $w$"), ("field+country", "+ $u$ + $w$"),
              ("spatial+", "spatial+"), ("xgb_shap", "XGBoost SHAP slope")]
    scen = [("B_noconf_null", r"(a) No confounding, $\alpha = 0$"),
            ("A_conf_null", r"(b) Confounding, $\alpha = 0$"),
            ("C_conf_effect", r"(c) Confounding, $\alpha = -0.3$")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharey=True, constrained_layout=True)
    for ax, (s, title) in zip(axes, scen):
        d = r[r.scenario == s]
        rows = [(lab, d[d.model == m].est.to_numpy(), d[d.model == m].covers.mean(),
                 BOOST if m == "xgb_shap" else LATENT) for m, lab in labels]
        dots(ax, rows, d.truth.iloc[0], xlim=(-0.55, 0.75))
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"estimate of $\alpha$")
    fig.savefig(os.path.join(FIG, "s2_estimates.png"), dpi=200)


def s4():
    r = pd.read_csv(os.path.join(RES, "s4_raw.csv"))
    r["covers"] = (r.lo <= r.truth) & (r.truth <= r.hi)
    labels = [("biome_nofield", "Biome slopes", LATENT), ("biome_field", "Biome slopes + $u$ + $w$", LATENT),
              ("svc", "SVC (biome means)", LATENT), ("xgb_slope", "SHAP slope", BOOST),
              ("xgb_slope_full", "SHAP slope, interactions credited", BOOST),
              ("xgb_median_raw", "Median SHAP, raw temperature", BOOST)]
    gam = sorted(r.truth.unique())
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True, constrained_layout=True)
    for b, ax in enumerate(axes):
        d = r[r.biome == b]
        rows = [(lab, d[d.model == m].est.to_numpy(),
                 None if m == "xgb_median_raw" else d[d.model == m].covers.mean(), col) for m, lab, col in labels]
        dots(ax, rows, gam[b], xlim=(-0.5, 0.5))
        ax.set_title(f"Biome {b + 1} ($\\gamma = {gam[b]:+.1f}$)", loc="left")
        ax.set_xlabel("estimated temperature effect")
    fig.savefig(os.path.join(FIG, "s4_estimates.png"), dpi=200)


def s5():
    r = pd.read_csv(os.path.join(RES, "s5_raw.csv"))
    r["covers"] = (r.est - 1.96 * r.se <= r.truth) & (r.truth <= r.est + 1.96 * r.se)
    labels = [("glm", "Logistic regression", MUTED), ("uni_field", "+ static field", MUTED),
              ("uni_st", "+ static + space-time field", MUTED),
              ("biv_st", "Joint model, shared space-time field", LATENT)]
    scen = [("no_escalation", r"(a) No escalation, $\kappa = 0$"),
            ("escalation", r"(b) Escalation, $\kappa = 0.4$")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, (s, title) in zip(axes[:2], scen):
        d = r[r.scenario == s]
        rows = [(lab, d[d.model == m].est.to_numpy(), d[d.model == m].covers.mean(), col)
                for m, lab, col in labels]
        dots(ax, rows, d.truth.iloc[0], xlim=(-0.35, 1.35))
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"estimate of $\kappa$")
    axes[1].set_yticklabels([])
    b = r[r.model == "biv_st"]
    rows = [(lab, b[b.scenario == s].loading.to_numpy(), None, LATENT)
            for s, lab in [("no_escalation", r"$\kappa = 0$"), ("escalation", r"$\kappa = 0.4$")]]
    dots(axes[2], rows, 0.8, xlim=(0.4, 1.2))
    axes[2].set_title(r"(c) Loading $\lambda$, joint model", loc="left")
    axes[2].set_xlabel(r"estimate of $\lambda$")
    fig.savefig(os.path.join(FIG, "s5_estimates.png"), dpi=200)


if __name__ == "__main__":
    s2(); s4(); s5()
