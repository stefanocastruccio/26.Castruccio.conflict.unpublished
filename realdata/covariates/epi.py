"""Yale Environmental Performance Index per country-year (task 35, Luca's EPI_index), from the open 2026 edition.

  python realdata/covariates/epi.py     # network (login node); writes grid/national/epi_1990_2025.parquet

The 2026 EPI (https://epi.yale.edu/2026/downloads, epi2026results*.xlsx, CC BY-NC-SA 4.0) gives per country the score on the
most recent data (EPI.new) and its change against data at most 10 years earlier (EPI.chg). We place EPI.new at 2024 and
EPI.new - EPI.chg at 2014 (the technical appendix details the actual years per indicator), interpolate linearly in
between and hold the end values outside (flag `epi_extrapolated`). Older editions (2006-2022) are on Dataverse behind a
guestbook form (name, email), so they are not used; filling that form would give a proper edition-by-edition series.
Output columns: iso3, year, epi, epi_extrapolated.
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

URL = "https://epi.yale.edu/sites/default/files/2026-09/epi2026results2026-07-07.xlsx"
Y1, Y0 = 2024, 2014


DV = os.path.join(C.RAW, "epi", "dataverse")
OLDER = [  # (label, workbook, sheet, current col, change col or None, current year, baseline year); most recent first
    ("2022", "d/2022-epi-xlsx/2022-epi.xlsx", "3_EPI_Results", "EPI.new", "EPI.change", 2020, 2010),
    ("2020", "a/2020-epi-xlsx/2020-epi.xlsx", "3_EPI_Results", "EPI.new", "EPI.change", 2018, 2008),
    ("2018", "b/2018-epi-xlsx/2018-epi.xlsx", "2018EPI_ScoresCurrent", "EPI.current", None, 2016, None),
]


def series(e1, e0, y1, y0):
    out = {}
    for y in C.YEARS:
        if y0 is None or e0 is None or not np.isfinite(e0):
            out[y] = (e1, y != y1)
        else:
            f = min(max((y - y0) / (y1 - y0), 0.0), 1.0); out[y] = (e0 + f * (e1 - e0), not (y0 <= y <= y1))
    return out


def main():
    raw = os.path.join(C.RAW, "epi", "epi2026results.xlsx")
    if not os.path.exists(raw):
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        open(raw, "wb").write(urllib.request.urlopen(URL, timeout=300).read())
    d = pd.read_excel(raw, sheet_name="data")[["iso", "EPI.new", "EPI.chg"]].dropna(subset=["EPI.new"])
    rows = []
    for _, r in d.iterrows():
        e1 = float(r["EPI.new"]); e0 = e1 - float(r["EPI.chg"]) if pd.notna(r["EPI.chg"]) else e1
        for y in C.YEARS:
            f = min(max((y - Y0) / (Y1 - Y0), 0.0), 1.0)
            rows.append((r["iso"], y, e0 + f * (e1 - e0), not (Y0 <= y <= Y1)))
    out = pd.DataFrame(rows, columns=["iso3", "year", "epi", "epi_extrapolated"]); out["epi_edition"] = "2026"
    # Countries absent from the 2026 edition: older editions (owner downloaded them from Dataverse, 2026-11-10), shifted onto
    # the 2026 scale by the mean difference over the countries both editions score, at the older edition's current year.
    for lab, wb, sh, cur, chg, y1, y0 in OLDER:
        f = os.path.join(DV, wb)
        if not os.path.exists(f):
            continue
        o = pd.read_excel(f, sheet_name=sh)
        o = o[["iso", cur] + ([chg] if chg else [])].dropna(subset=[cur])
        ref = out[out.year == y1].set_index("iso3").epi
        common = o[o.iso.isin(ref.index)]
        shift = float((ref.reindex(common.iso).to_numpy() - common[cur].to_numpy()).mean())
        add = []
        for _, r in o[~o.iso.isin(out.iso3)].iterrows():
            e1 = float(r[cur]) + shift; e0 = (e1 - float(r[chg])) if chg and pd.notna(r[chg]) else None
            for y, (v, ex) in series(e1, e0, y1, y0).items():
                add.append((r["iso"], y, v, ex, lab))
        if add:
            out = pd.concat([out, pd.DataFrame(add, columns=["iso3", "year", "epi", "epi_extrapolated", "epi_edition"])])
            print(f"edition {lab}: shift {shift:+.2f} on {len(common)} common countries; added {sorted({a[0] for a in add})}")
    # Countries scored by no edition (2018, 2020, 2022, 2026): SOM, SSD, SYR, YEM, PSE. Imputed as the yearly mean of their
    # EPI region (regions from the 2022 edition's 4_EPI_Results_Regions), flagged epi_edition = 'imputed_region_mean'.
    rg = os.path.join(DV, "d/2022-epi-xlsx/2022-epi.xlsx")
    if os.path.exists(rg):
        reg = pd.read_excel(rg, sheet_name="4_EPI_Results_Regions")[["iso", "region"]].set_index("iso").region
        unscored = {"SOM": "Sub-Saharan Africa", "SSD": "Sub-Saharan Africa", "SYR": "Greater Middle East",
                    "YEM": "Greater Middle East", "PSE": "Greater Middle East"}
        base = out[out.epi_edition == "2026"].assign(region=lambda x: x.iso3.map(reg))
        rmean = base.groupby(["region", "year"]).epi.mean()
        imp = [(c, y, float(rmean.loc[(r, y)]), True, "imputed_region_mean") for c, r in unscored.items()
               if c not in set(out.iso3) for y in C.YEARS]
        out = pd.concat([out, pd.DataFrame(imp, columns=["iso3", "year", "epi", "epi_extrapolated", "epi_edition"])])
        print("imputed (region mean):", sorted({i[0] for i in imp}))
    terr = {"SOL": "SOM", "CYN": "CYP", "BRT": "EGY", "ESH": "MAR", "KAS": "IND", "KAB": "KAZ", "ALA": "FIN", "KOS": "SRB"}
    codes = {str(c) for c in C.cells().ISO_A3.dropna()}
    extra = []
    for t, par in terr.items():   # territories take their parent country's score (flag epi_edition = '<ed>:parent')
        if t in codes and t not in set(out.iso3) and par in set(out.iso3):
            x = out[out.iso3 == par].copy(); x["iso3"] = t; x["epi_edition"] = x.epi_edition + ":parent"; extra.append(x)
    if extra:
        out = pd.concat([out] + extra)
    od = os.path.join(C.GRID, "national"); os.makedirs(od, exist_ok=True)
    out.to_parquet(os.path.join(od, "epi_1990_2025.parquet"), index=False)
    print(f"EPI: {out.iso3.nunique()} codes ({d.iso.nunique()} from the 2026 edition); cell codes without EPI: {sorted(codes - set(out.iso3))}")


if __name__ == "__main__":
    main()
