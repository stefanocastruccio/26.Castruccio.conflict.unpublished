"""Ecological footprint of consumption per person (global hectares), Global Footprint Network API (Luca's ECO_pc_gha).

  python realdata/covariates/gfn.py        # York University workbook (preferred; free, no form, 234 countries)
  python realdata/covariates/gfn.py api    # GFN API (needs ~/.gfn_key; serves only ~126 high-quality countries)

API: https://api.footprintnetwork.org/v1/data/all/<year>/EFCpc (HTTP basic auth: user, API key). Credentials are read
from ~/.gfn_key and never printed or written anywhere. Countries come as ISO alpha-2 (isoa2) and are mapped to ISO3 with
the World Bank country list (plus XK -> XKX). Years without a published value are filled by carrying the last available
year forward (flag eco_pc_carried). Output: grid/national/gfn_eco_1990_2025.parquet (iso3, year, eco_pc_gha, eco_pc_carried).
"""
from __future__ import annotations

import json
import os
import time
import stat
import sys
import urllib.request

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

B = "https://api.footprintnetwork.org/v1"


def creds(path=os.path.expanduser("~/.gfn_key")):
    if os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        sys.exit(f"{path} is readable by others; chmod 600 {path}")
    kv = {}
    for line in open(path):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1); kv[k.strip()] = v.strip().strip('"').strip("'")
    return kv["GFN_USER"], kv["GFN_KEY"]


def iso2_to_iso3():
    j = json.loads(urllib.request.urlopen("https://api.worldbank.org/v2/country?format=json&per_page=400", timeout=120).read())
    m = {c["iso2Code"]: c["id"] for c in j[1] if c.get("iso2Code")}
    m.update({"XK": "XKX"})
    return m


def main():
    auth = creds(); m = iso2_to_iso3(); rows = []
    years = [d["year"] for d in requests.get(B + "/years", auth=auth, timeout=60).json()]
    for y in sorted(y for y in years if 1990 <= y <= 2025):
        for k in range(8):   # the API rate-limits (HTTP 429): pause between years, back off on 429
            r = requests.get(f"{B}/data/all/{y}/EFCpc", auth=auth, timeout=180)
            if r.status_code != 429:
                break
            wait = int(r.headers.get("Retry-After", 30 * (k + 1))); print(f"  {y}: 429, waiting {wait} s", flush=True); time.sleep(wait)
        r.raise_for_status()
        time.sleep(5)
        for d in r.json():
            rows.append((d.get("isoa2"), y, d.get("value")))
        print(y, len(r.json()), "countries", flush=True)
    df = pd.DataFrame(rows, columns=["isoa2", "year", "eco_pc_gha"]).dropna(subset=["isoa2", "eco_pc_gha"])
    df["iso3"] = df.isoa2.map(m)
    unm = sorted(df.loc[df.iso3.isna(), "isoa2"].unique())
    df = df.dropna(subset=["iso3"])
    full = pd.MultiIndex.from_product([sorted(df.iso3.unique()), list(C.YEARS)], names=["iso3", "year"]).to_frame(index=False)
    full = full.merge(df[["iso3", "year", "eco_pc_gha"]], how="left")
    full["eco_pc_carried"] = full.eco_pc_gha.isna()
    full["eco_pc_gha"] = full.groupby("iso3").eco_pc_gha.ffill()
    full["eco_pc_carried"] &= full.eco_pc_gha.notna()
    od = os.path.join(C.GRID, "national"); os.makedirs(od, exist_ok=True)
    full.to_parquet(os.path.join(od, "gfn_eco_1990_2025.parquet"), index=False)
    last = df.groupby("iso3").year.max()
    print(f"GFN EFCpc: {df.iso3.nunique()} countries, years {df.year.min()}-{df.year.max()}; last year per country: "
          f"{last.value_counts().sort_index().to_dict()}; unmapped isoa2: {unm}")


YORK = os.path.join(C.RAW, "gfn", "NEFBA_2026.xlsx")
YORK_URL = ("https://footprint.info.yorku.ca/files/2026/04/"
            "NEFBA_Data_global_national_ecological_footprint_biocapacity.xlsx")
TERR = {"SOL": "SOM", "CYN": "CYP", "BRT": "EGY", "ESH": "MAR", "KAS": "IND", "KAB": "KAZ", "ALA": "FIN"}


def from_york():
    """Preferred source (2026-11-10): the free York University / FoDaFo workbook of the National Ecological Footprint and
    Biocapacity Accounts, 2026 Edition (1961-2025, 234 countries; no form). eco_pc_gha = efc_total_percapita (consumption
    footprint per person) where published (flags +++, ++, E); else efp_total_percapita (production footprint per person,
    flag '+') with eco_pc_source = 'production'. Not published (flag '-', 'X') -> NaN. Territories take their parent
    country (TERR). Years after a country's last published year carry the last value forward (eco_pc_carried)."""
    if not os.path.exists(YORK):
        os.makedirs(os.path.dirname(YORK), exist_ok=True)
        open(YORK, "wb").write(urllib.request.urlopen(YORK_URL, timeout=300).read())
    d = pd.read_excel(YORK, "national_data", usecols=["country_ISO3", "year", "flag", "efc_total_percapita", "efp_total_percapita"])
    d = d[(d.year >= C.YEARS[0]) & (d.year <= C.YEARS[-1])]
    d["eco_pc_gha"] = d.efc_total_percapita.where(d.efc_total_percapita.notna(), d.efp_total_percapita)
    d["eco_pc_source"] = np.where(d.efc_total_percapita.notna(), "consumption",
                                  np.where(d.efp_total_percapita.notna(), "production", None))
    codes = sorted({str(x) for x in set(C.cells().ISO_A3.dropna()) | set(d.country_ISO3.dropna())})
    full = pd.MultiIndex.from_product([codes, list(C.YEARS)], names=["iso3", "year"]).to_frame(index=False)
    full["src"] = full.iso3.map(lambda c: TERR.get(c, c))
    full = full.merge(d.rename(columns={"country_ISO3": "src"})[["src", "year", "flag", "eco_pc_gha", "eco_pc_source"]], how="left")
    full["eco_pc_carried"] = full.eco_pc_gha.isna()
    full[["eco_pc_gha", "eco_pc_source"]] = full.groupby("iso3")[["eco_pc_gha", "eco_pc_source"]].ffill()
    full["eco_pc_carried"] &= full.eco_pc_gha.notna()
    od = os.path.join(C.GRID, "national"); os.makedirs(od, exist_ok=True)
    full.rename(columns={"src": "source_iso3"}).to_parquet(os.path.join(od, "gfn_eco_1990_2025.parquet"), index=False)
    y = full[full.year == 2020]
    print(f"York NEFBA 2026: {d.country_ISO3.nunique()} countries; 2020: consumption {int((y.eco_pc_source == 'consumption').sum())}, "
          f"production fallback {int((y.eco_pc_source == 'production').sum())}, none {int(y.eco_pc_gha.isna().sum())} "
          f"(of {len(y)} codes); cell codes without any value: "
          f"{sorted(set(C.cells().ISO_A3.dropna()) - set(full.dropna(subset=['eco_pc_gha']).iso3))}")


if __name__ == "__main__":
    from_york() if (len(sys.argv) < 2 or sys.argv[1] != "api") else main()
