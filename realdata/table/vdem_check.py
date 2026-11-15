"""Task 43 checks: V-Dem state capacity (v2stfisccap, v2svstterr) in the global table.

  python realdata/table/vdem_check.py

1. share of cell-weeks with each indicator missing, by ACLED region (all years 1990-2025);
2. correlation with v2x_libdem (Pearson and Spearman over cell-weeks, and over country-years);
3. five example countries in 2000, 2015 and 2023 (country-year values).
Writes results_real/vdem_state_capacity_check.json (aggregates only).
"""
import glob, json, os
import pandas as pd

T = "/groups/scastruc/conflict_spatial_work/table_global/clean_global"
V = ["v2stfisccap", "v2svstterr"]
cols = ["region", "ISO_A3", "year", "v2x_libdem"] + V
miss, n, cy = {}, {}, []
acc = []
for f in sorted(glob.glob(os.path.join(T, "year=*.parquet"))):
    d = pd.read_parquet(f, columns=cols)
    g = d.groupby("region", dropna=False)
    for r, x in g:
        n[r] = n.get(r, 0) + len(x)
        for v in V:
            miss.setdefault(v, {}); miss[v][r] = miss[v].get(r, 0) + int(x[v].isna().sum())
    acc.append(d[["v2x_libdem"] + V].dropna().sample(frac=0.05, random_state=0))
    cy.append(d.drop(columns="region").drop_duplicates(["ISO_A3", "year"]))
out = {"missing_share_by_region": {v: {str(r): round(miss[v][r] / n[r], 4) for r in sorted(n, key=str)} for v in V},
       "cell_weeks_by_region": {str(r): int(n[r]) for r in sorted(n, key=str)}}
out["missing_share_all"] = {v: round(sum(miss[v].values()) / sum(n.values()), 4) for v in V}
s = pd.concat(acc); c = pd.concat(cy).dropna(subset=["v2x_libdem"] + V)
out["corr_with_libdem"] = {v: {"cell_weeks_pearson_5pct_sample": round(s.v2x_libdem.corr(s[v]), 3),
                               "cell_weeks_spearman_5pct_sample": round(s.v2x_libdem.corr(s[v], method="spearman"), 3),
                               "country_years_pearson": round(c.v2x_libdem.corr(c[v]), 3),
                               "country_years_spearman": round(c.v2x_libdem.corr(c[v], method="spearman"), 3),
                               "country_years_n": int(len(c))} for v in V}
out["corr_fisccap_terr_country_years"] = round(c.v2stfisccap.corr(c.v2svstterr), 3)
ex = pd.concat(cy).drop_duplicates(["ISO_A3", "year"])
ex = ex[ex.ISO_A3.isin(["CHN", "MLI", "MMR", "COL", "DEU"]) & ex.year.isin([2000, 2015, 2023])].sort_values(["ISO_A3", "year"])
out["examples"] = ex[["ISO_A3", "year", "v2x_libdem"] + V].round(3).to_dict("records")
print(json.dumps(out, indent=1))
od = os.path.join(os.path.dirname(__file__), "..", "..", "results_real")
json.dump(out, open(os.path.join(od, "vdem_state_capacity_check.json"), "w"), indent=1)
