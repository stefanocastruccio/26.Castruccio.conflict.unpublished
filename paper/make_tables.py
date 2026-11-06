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
    t = pd.read_csv(os.path.join(RES, "s0_summary.csv"), index_col=0)
    v = pd.read_csv(os.path.join(RES, "s0_variance.csv"), index_col=0).iloc[:, 0]
    lines = [(r"$\hat\rho_w$ (km) [truth 400]", "range_hat", 1),
             (r"$\hat\sigma_w$ [truth 1]", "sd_hat", 3),
             (r"$\hat b_1$ [truth 0.5]", "b1_hat", 3),
             (r"Coverage of 95\% interval for $b_1$", "b1_cover", 3),
             (r"Coverage of 95\% intervals for $b_0 + w(s)$ (share of cells)", "field_cover", 3),
             (r"$\mathrm{cor}(\hat w, w)$", "field_corr", 3),
             (r"Fit time (s)", "seconds", 1)]
    rows = [f"{a} & {t.loc[k, 'mean']:.{d}f} & {t.loc[k, 'std']:.{d}f} \\\\" for a, k, d in lines]
    rows.append(f"Marginal variance of $w$, interior cells [nominal 1] & {v.iloc[0]:.2f} & -- \\\\")
    rows.append(f"Marginal variance of $w$, edge cells & {v.iloc[1]:.2f} & -- \\\\")
    body = (r"\begin{tabular}{@{}lcc@{}}" "\n" r"\toprule" "\n" r"Quantity & Mean & SD \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s0.tex", body)


def s1():
    raw = pd.read_csv(os.path.join(RES, "s1_raw.csv"))
    models = [("oracle", "True probabilities"),
              ("xgb_full", "XGBoost (as in original analysis)"),
              ("xgb_crossfit", "XGBoost, spatially cross-fitted"),
              ("xgb+field", "Cross-fitted XGBoost + $w$"),
              ("xgb+field+country", "Cross-fitted XGBoost + $u + w$"),
              ("glmm_field+country", "Parametric model with $u + w$")]
    rows = []
    for key, lab in models:
        cells = []
        for m, d in [("log_score", 4), ("brier", 4), ("auc_pr", 3), ("auc_roc", 3)]:
            v = raw[f"{key}.{m}"]
            cells.append(f"{v.mean():.{d}f} ({v.std() / np.sqrt(len(v)):.{d}f})")
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    v = raw["persistence.auc_pr"]
    rows.append(f"Persistence baseline & -- & -- & {v.mean():.3f} ({v.std() / np.sqrt(len(v)):.3f}) & -- \\\\")
    body = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Model & Log score & Brier & AUC-PR & AUC-ROC \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s1_scores.tex", body)
    fs = raw
    lines = [
        ("Moran's $I$ of cell residuals, XGBoost (original)", fs["moran_xgb_full"]),
        ("Moran's $I$ of cell residuals, XGBoost cross-fitted", fs["moran_xgb_crossfit"]),
        (r"$\mathrm{cor}(\hat w, w)$, XGBoost + $w$", fs["corr_w_xgb+field"]),
        (r"$\mathrm{cor}(\hat w, w)$, XGBoost + $u + w$", fs["corr_w_xgb+field+country"]),
        (r"$\hat\rho_w$ (km), XGBoost + $u + w$ [truth 400]", fs["range_xgb+field+country"]),
        (r"$\hat\sigma_w$, XGBoost + $u + w$ [truth 1]", fs["sd_xgb+field+country"]),
    ]
    body = (r"\begin{tabular}{@{}lcc@{}}" "\n" r"\toprule" "\n"
            r"Quantity & Mean & SD \\" "\n" r"\midrule" "\n"
            + "\n".join(f"{a} & {b.mean():.3f} & {b.std():.3f} \\\\" for a, b in lines)
            + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s1_field.tex", body)


def s2():
    p = os.path.join(RES, "s2_summary.csv")
    if not os.path.exists(p):
        return placeholder("s2.tex")
    t = pd.read_csv(p)
    lab = {"glm": "Logistic regression", "country": "+ country effects $u$",
           "field": "+ spatial field $w$", "field+country": "+ $u$ + $w$",
           "spatial+": "spatial+ with $u$ + $w$", "xgb_shap": "XGBoost SHAP slope$^\\dagger$"}
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
           "lag+field+st": "Lags + $w$ + $v$"}
    tr = {"state_dep": r"State dependence ($\delta = 1$)",
          "heterogeneity": r"Heterogeneity ($\delta = 0$)"}
    rows = []
    for k, tl in tr.items():
        sub = t[t.truth == k]
        rows.append(r"\multicolumn{5}{@{}l}{\emph{" + tl + r"}} \\")
        for m, ml in lab.items():
            r = sub[sub.model == m].iloc[0]
            rows.append(f"\\quad {ml} & {f(r.mean_est)} & {f(r.emp_sd)} & {f(r.mean_se)} & {f(r.coverage95, 2)} \\\\")
        r = sub[sub.model == "xgb_lag_shap_share"].iloc[0]
        rows.append(f"\\quad XGBoost: share of mean $|$SHAP$|$ for lags & {f(r.mean_est)} & {f(r.emp_sd)} & -- & -- \\\\")
    body = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Estimator & Mean $\hat\delta$ & Emp.\ SD & Mean SE & Coverage \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s3.tex", body)


def s4():
    p = os.path.join(RES, "s4_summary.csv")
    if not os.path.exists(p):
        return placeholder("s4.tex", "s4_extra.tex")
    t = pd.read_csv(p)
    lab = {"biome_nofield": "Biome slopes, no latent effects",
           "biome_field": "Biome slopes + $u$ + $w$",
           "svc": "SVC: biome means + $\\eta$ + $u$ + $w$",
           "xgb_slope": "XGBoost SHAP slope",
           "xgb_median": "XGBoost median SHAP$^\\ddagger$"}
    rows = []
    for m, ml in lab.items():
        sub = t[t.model == m].sort_values("biome")
        rows.append(r"\multicolumn{5}{@{}l}{\emph{" + ml + r"}} \\")
        for _, r in sub.iterrows():
            rows.append(f"\\quad biome {int(r.biome) + 1} ($\\gamma = {r.truth:+.1f}$) & {f(r.mean_est)} & "
                        f"{f(r.coverage95, 2)} & {f(r.coverage95_realized, 2)} & {f(r.detects, 2)} \\\\")
    body = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Estimator / biome & Mean est. & Cov.\ $\gamma_b$ & Cov.\ $\bar\beta_b$ & Detects sign \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s4.tex", body)
    e = pd.read_csv(os.path.join(RES, "s4_extra.csv"))
    lines = [(r"$\mathrm{cor}\{\hat\beta(s), \beta(s)\}$, SVC", e.svc_corr_beta_s),
             (r"RMSE of $\hat\beta(s)$, SVC", e.svc_rmse_beta_s),
             (r"RMSE of $\hat\beta(s)$, biome slopes only", e.biome_only_rmse_beta_s),
             (r"$\hat\sigma_\eta$ [truth 0.15]", e.svc_sd_hat),
             (r"$\hat\rho_\eta$ (km) [truth 600]", e.svc_range_hat)]
    body = (r"\begin{tabular}{@{}lcc@{}}" "\n" r"\toprule" "\n" r"Quantity & Mean & SD \\" "\n"
            r"\midrule" "\n" + "\n".join(f"{a} & {b.mean():.3f} & {b.std():.3f} \\\\" for a, b in lines)
            + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("s4_extra.tex", body)


if __name__ == "__main__":
    s0(); s1(); s2(); s3(); s4()
