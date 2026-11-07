"""Write LaTeX tables for the simulation section from results/*.csv.
Run from the repository root:  python paper/make_tables.py"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def placeholder(*names):
    for n in names:
        write(n, r"\begin{tabular}{l}\toprule Results pending (simulation still running). \\ \bottomrule\end{tabular}" "\n")


def write(name, body):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(body)
    print("wrote", name)


def s0():
    raw = pd.read_csv(os.path.join(RES, "s0_raw.csv"))
    v = pd.read_csv(os.path.join(RES, "s0_variance.csv"), index_col=0).iloc[:, 0]
    if "config" not in raw:
        raw["config"] = "common"
    cfgs = [c for c in ["common", "rare"] if c in set(raw.config)]
    g = raw.groupby("config")
    lines = [(r"Prevalence of event weeks", "prevalence", 3),
             (r"$\hat\rho_w$ (km) [truth 400]", "range_hat", 0),
             (r"$\hat\sigma_w$ [truth 1]", "sd_hat", 3),
             (r"$\hat b_1$ [truth 0.5]", "b1_hat", 3),
             (r"Coverage of 95\% interval for $b_1$", "b1_cover", 2),
             (r"Coverage of 95\% intervals for $b_0 + w(s)$ (share of cells)", "field_cover", 3),
             (r"$\mathrm{cor}(\hat w, w)$", "field_corr", 3),
             (r"Fit time (s)", "seconds", 1)]
    rows = []
    for lab, k, d in lines:
        if k not in raw:
            continue
        if k == "b1_cover":   # a proportion: report its Monte Carlo standard error
            cells = [f"{g[k].mean()[c]:.{d}f} ({np.sqrt(g[k].mean()[c] * (1 - g[k].mean()[c]) / g.size()[c]):.{d}f})"
                     for c in cfgs]
        else:
            cells = [f"{g[k].mean()[c]:.{d}f} ({g[k].std()[c]:.{d}f})" for c in cfgs]
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    rows.append(f"Marginal variance of $w$, interior cells [nominal 1] & " + " & ".join([f"{v.iloc[0]:.2f}"] + ["--"] * (len(cfgs) - 1)) + r" \\")
    rows.append(f"Marginal variance of $w$, edge cells & " + " & ".join([f"{v.iloc[1]:.2f}"] + ["--"] * (len(cfgs) - 1)) + r" \\")
    n = g.size()
    head = " & ".join([f"{c} ($b_0={-2 if c == 'common' else -4.5}$, {n[c]} reps)" for c in cfgs])
    body = ("\\begin{tabular}{@{}l" + "c" * len(cfgs) + "@{}}\n\\toprule\nQuantity & " + head + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("s0.tex", body)


def s1():
    raw = pd.read_csv(os.path.join(RES, "s1_raw.csv"))
    models = [("oracle", "True probabilities"),
              ("xgb_original", "XGBoost as in the original analysis"),
              ("xgb_te", "XGBoost + cell training rate"),
              ("xgb_crossfit", "XGBoost, cross-fitted by country"),
              ("hybrid_iid", "Cross-fitted XGBoost + $u$ + iid cell effects"),
              ("hybrid_field", "Cross-fitted XGBoost + $u + w$"),
              ("glmm", "Parametric model with $u + w$")]
    rows = []
    for key, lab in models:
        if f"{key}.log_score" not in raw:
            continue
        cells = []
        for m, d in [("log_score", 4), ("brier", 4), ("auc_pr", 3), ("auc_roc", 3)]:
            v = raw[f"{key}.{m}"]
            cells.append(f"{v.mean():.{d}f} ({v.std() / np.sqrt(len(v)):.{d}f})")
        dl = raw.get(f"{key}.logscore_minus_xgb_original")
        cells.append("--" if dl is None or key == "xgb_original" else
                     f"{(dl < 0).sum()}/{len(dl)}")
        dt = None if dl is None else dl - raw["xgb_te.logscore_minus_xgb_original"]
        cells.append("--" if dt is None or key == "xgb_te" else f"{(dt < 0).sum()}/{len(dt)}")
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    v = raw["persistence.auc_pr"]
    rows.append(f"Persistence baseline & -- & -- & {v.mean():.3f} ({v.std() / np.sqrt(len(v)):.3f}) & -- & -- & -- \\\\")
    body = (r"\begin{tabular}{@{}lcccccc@{}}" "\n" r"\toprule" "\n"
            r"Model & Log score & Brier & AUC-PR & AUC-ROC & $<$ (a) & $<$ (b) \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s1_scores.tex", body)
    lines = [
        ("Moran's $I$ of cell residuals, XGBoost as in the original", "moran_xgb_original"),
        ("Moran's $I$ of cell residuals, cross-fitted XGBoost", "moran_xgb_crossfit"),
        (r"$\mathrm{cor}(\hat w, w)$, hybrid with $u + w$", "corr_w"),
        (r"$\hat\rho_w$ (km) [truth 400]", "range"),
        (r"$\hat\sigma_w$ [truth 1]", "sd"),
    ]
    dg = lambda k: 0 if k == "range" else 3   # ranges in km as integers
    body = (r"\begin{tabular}{@{}lcc@{}}" "\n" r"\toprule" "\n"
            r"Quantity & Mean & SD \\" "\n" r"\midrule" "\n"
            + "\n".join(f"{a} & {raw[k].mean():.{dg(k)}f} & {raw[k].std():.{dg(k)}f} \\\\" for a, k in lines if k in raw)
            + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s1_field.tex", body)


def s2():
    p = os.path.join(RES, "s2_summary.csv")
    if not os.path.exists(p):
        return placeholder("s2.tex")
    t = pd.read_csv(p)
    lab = {"glm": "Logistic regression", "country": "+ country effects $u$",
           "field": "+ spatial field $w$", "field+country": "+ $u$ + $w$",
           "spatial+": "spatial+: $u + w$, $Z_0$ projected$^\\ast$", "xgb_shap": "XGBoost SHAP slope$^\\dagger$"}
    scen = {"B_noconf_null": r"No confounding, $\alpha = 0$",
            "A_conf_null": r"Confounding, $\alpha = 0$",
            "C_conf_effect": r"Confounding, $\alpha = -0.3$"}
    rows = []
    for s, sl in scen.items():
        sub = t[t.scenario == s]
        rows.append(r"\multicolumn{6}{@{}l}{\emph{" + sl + r"}} \\")
        for m in lab:
            r = sub[sub.model == m]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append(f"\\quad {lab[m]} & {f(r.mean_est)} & {f(r.emp_sd)} & {f(r.mean_se)} & "
                        f"{f(r.coverage95, 2)} & {f(r.reject_zero, 2)} \\\\")
    body = (r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n"
            r"Estimator & Mean & Emp.\ SD & Mean SE & Coverage & Reject $\alpha=0$ \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s2.tex", body)


def s3():
    p = os.path.join(RES, "s3_summary.csv")
    if not os.path.exists(p):
        return placeholder("s3.tex")
    t = pd.read_csv(p)
    lab = {"lag_only": "Lags only", "lag+field": "Lags + $w$",
           "lag+field+st": "Lags + $w$ + $v$ (4-week steps)",
           "lag+field+st1": "Lags + $w$ + $v$ (weekly steps)"}
    tr = {"state_dep": r"State dependence ($\delta = 1$)",
          "heterogeneity": r"Heterogeneity ($\delta = 0$)",
          "mixed": r"Mixed ($\delta = 0.5$)"}
    rows = []
    for k, tl in tr.items():
        sub = t[t.truth == k]
        rows.append(r"\multicolumn{5}{@{}l}{\emph{" + tl + r"}} \\")
        if len(sub) == 0:
            continue
        for m, ml in lab.items():
            if (sub.model == m).sum() == 0:
                continue
            r = sub[sub.model == m].iloc[0]
            rows.append(f"\\quad {ml} & {f(r.mean_est)} & {f(r.emp_sd)} & {f(r.mean_se)} & {f(r.coverage95, 2)} \\\\")
        r = sub[sub.model == "xgb_lag_shap_share"].iloc[0]
        rows.append(f"\\quad XGBoost: share of mean $|$SHAP$|$ for lags & {f(r.mean_est)} & {f(r.emp_sd)} & -- & -- \\\\")
    body = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Estimator & Mean $\hat\delta$ & Emp.\ SD & Mean SE & Coverage \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s3.tex", body)

    # Supplement: every truth and every step length, with the SD/SE ratio and fit times
    lab_full = {"lag_only": "Lags only", "lag+field": "Lags + $w$",
                "lag+field+st8": "Lags + $w$ + $v$ (8-week steps)",
                "lag+field+st": "Lags + $w$ + $v$ (4-week steps)",
                "lag+field+st1": "Lags + $w$ + $v$ (weekly steps)"}
    tr_full = dict(tr)
    tr_full.update({"heterogeneity_slow": r"Heterogeneity, slow $v$ ($\delta = 0$, $\phi = 0.97$ per week)",
                    "mixed_slow": r"Mixed, slow $v$ ($\delta = 0.5$, $\phi = 0.97$ per week)"})
    rows = []
    for k, tl in tr_full.items():
        sub = t[t.truth == k]
        if len(sub) == 0:
            continue
        if rows:
            rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{7}{@{}l}{\emph{" + tl + r"}} \\")
        for m, ml in lab_full.items():
            if (sub.model == m).sum() == 0:
                continue
            r = sub[sub.model == m].iloc[0]
            ratio = r.emp_sd / r.mean_se if "sd_se_ratio" not in r or pd.isna(r.sd_se_ratio) else r.sd_se_ratio
            rows.append(f"\\quad {ml} & {f(r.mean_est)} & {f(r.emp_sd)} & {f(r.mean_se)} & {f(ratio, 2)} & "
                        f"{f(r.coverage95, 2)} & {f(r.mean_seconds, 1)} \\\\")
    body = (r"\begin{tabular}{@{}lcccccc@{}}" "\n" r"\toprule" "\n"
            r"Estimator & Mean $\hat\delta$ & Emp.\ SD & Mean SE & SD/SE & Coverage & Time (s) \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s3_full.tex", body)


def s4(suffix=""):
    """suffix '' : symmetric biome effects (s4.tex, s4_extra.tex);
    suffix '_asym': asymmetric run (s4_asym.tex, s4_extra_asym.tex, Supplement)."""
    p = os.path.join(RES, f"s4_summary{suffix}.csv")
    if not os.path.exists(p):
        return placeholder(f"s4{suffix}.tex", f"s4_extra{suffix}.tex")
    t = pd.read_csv(p)
    lab = {"biome_nofield": "Biome slopes, no latent effects",
           "biome_field": "Biome slopes + $u$ + $w$",
           "svc": "SVC: biome means + $\\eta$ + $u$ + $w$",
           "xgb_slope": "XGBoost SHAP slope (targets $\\gamma_b/2$)",
           "xgb_slope_full": "XGBoost SHAP slope, interactions credited",
           "xgb_median_raw": "XGBoost median SHAP, raw temperature$^\\ddagger$"}
    if suffix:   # with a non-zero cell-weighted mean of gamma the SHAP slope targets (gbar + gamma_b) / 2
        lab["xgb_slope"] = "XGBoost SHAP slope (targets $(\\bar\\gamma + \\gamma_b)/2$)"
    rows = []
    for m, ml in lab.items():
        sub = t[t.model == m].sort_values("biome")
        if len(sub) == 0:
            continue
        rows.append(r"\multicolumn{5}{@{}l}{\emph{" + ml + r"}} \\")
        for _, r in sub.iterrows():
            rows.append(f"\\quad biome {int(r.biome) + 1} ($\\gamma = {r.truth:+.1f}$) & {f(r.mean_est)} & "
                        f"{f(r.coverage95, 2)} & {f(r.coverage95_realized, 2)} & {f(r.detects, 2)} \\\\")
    body = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Estimator / biome & Mean est. & Cov.\ $\gamma_b$ & Cov.\ $\bar\beta_b$ & Detects sign \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(f"s4{suffix}.tex", body)
    pe = os.path.join(RES, f"s4_extra{suffix}.csv")
    if not os.path.exists(pe):
        return placeholder(f"s4_extra{suffix}.tex")
    e = pd.read_csv(pe)
    lines = [(r"$\mathrm{cor}\{\hat\beta(s), \beta(s)\}$, SVC", e.svc_corr_beta_s),
             (r"RMSE of $\hat\beta(s)$, SVC", e.svc_rmse_beta_s),
             (r"RMSE of $\hat\beta(s)$, biome slopes only", e.biome_only_rmse_beta_s),
             (r"$\hat\sigma_\eta$ [truth 0.15]", e.svc_sd_hat),
             (r"$\hat\rho_\eta$ (km) [truth 600]", e.svc_range_hat)]
    dg = lambda a: 0 if "(km)" in a else 3   # ranges in km as integers
    body = (r"\begin{tabular}{@{}lcc@{}}" "\n" r"\toprule" "\n" r"Quantity & Mean & SD \\" "\n"
            r"\midrule" "\n" + "\n".join(f"{a} & {b.mean():.{dg(a)}f} & {b.std():.{dg(a)}f} \\\\" for a, b in lines)
            + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(f"s4_extra{suffix}.tex", body)


def s5():
    """S11 of the supplement: full S5 results by scenario and estimator."""
    p = os.path.join(RES, "s5_summary.csv")
    if not os.path.exists(p):
        return placeholder("s5.tex")
    t = pd.read_csv(p)
    rp = os.path.join(RES, "s5_raw.csv")
    raw = pd.read_csv(rp) if os.path.exists(rp) else None
    lab = {"glm": "Logistic regression", "uni_field": "Static field $w_c$",
           "uni_st": "Static field $w_c$ + space-time field", "biv_st": "Joint model, shared space-time field",
           "biv_st2": "Joint model, shared + unrest-specific space-time fields"}
    scen = {"no_escalation": r"No escalation, $\kappa = 0$", "escalation": r"Escalation, $\kappa = 0.4$",
            "no_escalation_uST": r"No escalation, $\kappa = 0$, with unrest-only episodes",
            "escalation_uST": r"Escalation, $\kappa = 0.4$, with unrest-only episodes"}
    base = ("no_escalation", "escalation")
    scen = {k: v for k, v in scen.items() if k in base or k in set(t.scenario)}
    fm = lambda x, d=3: (lambda v: f"${v}$" if v.startswith("-") and v != "--" else v)(f(x, d))
    rows = []
    for s_, sl in scen.items():
        sub = t[t.scenario == s_]
        if rows:
            rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{11}{@{}l}{\emph{" + sl + r"}} \\")
        if len(sub) == 0:
            rows.append(r"\multicolumn{11}{@{}l}{\todo{missing}} \\")
            continue
        for m, ml in lab.items():
            r = sub[sub.model == m]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            lam = "--"
            if pd.notna(r.loading):
                sd = None
                if raw is not None and "loading" in raw:
                    sd = raw[(raw.scenario == s_) & (raw.model == m)].loading.std()
                lam = f"{r.loading:.2f}" + ("" if sd is None or pd.isna(sd) else f" ({sd:.2f})")
            cov = f"{r.coverage95:.2f}" + ("" if "mc_se_coverage" not in r or pd.isna(r.mc_se_coverage)
                                           else f" ({r.mc_se_coverage:.2f})")
            rows.append(f"\\quad {ml} & {fm(r.mean_est)} & {f(r.emp_sd)} & {f(r.mean_se)} & {cov} & "
                        f"{f(r.reject_zero, 2)} & {f(r.d_c, 2)} & {lam} & {f(r.log_score, 4)} & "
                        f"{f(r.auc_pr)} & {f(r.onset_auc_pr)} \\\\")
    body = (r"\begin{tabular}{@{}lcccccccccc@{}}" "\n" r"\toprule" "\n"
            r"Estimator & Mean $\hat\kappa$ & Emp.\ SD & Mean SE & Coverage & Reject & Mean & Mean $\hat\lambda$ & "
            r"Log & AUC-PR & Onset \\" "\n"
            r" & & & & (MC SE) & $\kappa = 0$ & $\hat d_c$ & (SD) & score & & AUC-PR \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s5.tex", body)


def barrier():
    r = pd.read_csv(os.path.join(RES, "barrier_check.csv"))
    tr = ["land only", "sea = land", "barrier 0.1", "barrier 0.2", "barrier 0.5"]
    head = ["Land only", "Sea = land", "$f=0.1$", "$f=0.2$", "$f=0.5$"]
    body = r"\begin{tabular}{@{}ll" + "r" * len(tr) + "@{}}\n\\toprule\n"
    body += "Strait & & " + " & ".join(head) + r" \\" + "\n\\midrule\n"
    for w in sorted(r.width_cells.unique()):
        d = r[r.width_cells == w].set_index("treatment")
        for k, (lab, col, dg) in enumerate([("var.\\ coast", "var_coast", 2), ("var.\\ inland", "var_inland", 2),
                                             ("corr.\\ across", "corr_across", 2)]):
            first = f"{w} cells" if k == 0 else ""
            body += f"{first} & {lab} & " + " & ".join(f(d.loc[t, col], dg) for t in tr) + r" \\" + "\n"
        body += "\\addlinespace\n" if w != r.width_cells.max() else ""
    body += "\\bottomrule\n\\end{tabular}\n"
    write("barrier.tex", body)


if __name__ == "__main__":
    s0(); s1(); s2(); s3(); s4(); s4("_asym"); s5(); barrier()
