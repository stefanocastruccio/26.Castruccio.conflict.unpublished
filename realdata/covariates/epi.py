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
    out = pd.DataFrame(rows, columns=["iso3", "year", "epi", "epi_extrapolated"])
    od = os.path.join(C.GRID, "national"); os.makedirs(od, exist_ok=True)
    out.to_parquet(os.path.join(od, "epi_1990_2025.parquet"), index=False)
    codes = set(C.cells().ISO_A3.dropna())
    print(f"EPI: {d.iso.nunique()} countries; cell codes without EPI: {sorted(codes - set(d.iso))}")


if __name__ == "__main__":
    main()
