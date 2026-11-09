"""National covariates per country-year 1990-2025 (task 35, item 4), keyed by the ISO_A3 codes of the grid cells.

  python realdata/covariates/national.py      # network (login node); writes $WORK/covariates/grid/national/

Sources (no login):
  World Bank API v2: NY.GDP.MKTP.CD (GDP, current US$), NY.GDP.PCAP.CD (GDP per capita, current US$),
                     EG.ELC.ACCS.ZS (access to electricity, % of population)
  V-Dem v2x_libdem (liberal democracy index): v-dem.net requires a form, so it is taken from Our World in Data's
                     republication of the V-Dem estimate (grapher "liberal-democracy-index", column libdem_vdem__estimate_best)
  OWID CO2 and GHG data (github owid/co2-data): ghg_per_capita (t CO2-eq per person, incl. land use)
Cell codes that are not ISO countries are mapped to a source country (MAP below); codes still unmatched are logged.
Output: national_1990_2025.parquet (iso3 = cell code, source_iso3, year, gdp, gdp_pc, elec_access, libdem, ghg_pc)
        and national_unmatched.csv.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

WB = "https://api.worldbank.org/v2/country/all/indicator/{ind}?date=1990:2025&format=json&per_page=20000"
IND = {"gdp": "NY.GDP.MKTP.CD", "gdp_pc": "NY.GDP.PCAP.CD", "elec_access": "EG.ELC.ACCS.ZS"}
LIBDEM = "https://ourworldindata.org/grapher/liberal-democracy-index.csv?v=1&csvType=full&useColumnShortNames=true"
OWID = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
# cell code -> source code (World Bank / OWID use XKX for Kosovo; disputed territories take their administering country)
MAP = {"KOS": "XKX", "SOL": "SOM", "CYN": "CYP", "ESH": "MAR", "BRT": "EGY"}
MAP_OWID = {"XKX": "OWID_KOS"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-covariates"}), timeout=300) as r:
        return r.read()


def worldbank():
    out = []
    for name, ind in IND.items():
        j = json.loads(get(WB.format(ind=ind)))
        rows = [(d["countryiso3code"], int(d["date"]), d["value"]) for d in j[1] if d["countryiso3code"]]
        out.append(pd.DataFrame(rows, columns=["source_iso3", "year", name]).set_index(["source_iso3", "year"]))
        print(name, ind, len(rows), "rows; pages", j[0]["pages"], flush=True)
    return pd.concat(out, axis=1)


def main():
    outd = os.path.join(C.GRID, "national"); os.makedirs(outd, exist_ok=True)
    codes = sorted(C.cells().ISO_A3.dropna().unique())
    wb = worldbank()
    lib = pd.read_csv(io.BytesIO(get(LIBDEM)))
    lib = lib.rename(columns={"code": "source_iso3", "libdem_vdem__estimate_best": "libdem"})[["source_iso3", "year", "libdem"]]
    ow = pd.read_csv(io.BytesIO(get(OWID)), usecols=["iso_code", "year", "ghg_per_capita"])
    ow = ow.rename(columns={"iso_code": "source_iso3", "ghg_per_capita": "ghg_pc"})
    grid = pd.MultiIndex.from_product([codes, list(C.YEARS)], names=["iso3", "year"]).to_frame(index=False)
    grid["source_iso3"] = grid.iso3.map(lambda c: MAP.get(c, c))
    grid["owid_iso3"] = grid.source_iso3.map(lambda c: MAP_OWID.get(c, c))
    df = grid.merge(wb.reset_index(), on=["source_iso3", "year"], how="left")
    df = df.merge(lib.rename(columns={"source_iso3": "owid_iso3"}), on=["owid_iso3", "year"], how="left")
    df = df.merge(ow.rename(columns={"source_iso3": "owid_iso3"}), on=["owid_iso3", "year"], how="left")
    df.drop(columns="owid_iso3").to_parquet(os.path.join(outd, "national_1990_2025.parquet"), index=False)
    vars_ = ["gdp", "gdp_pc", "elec_access", "libdem", "ghg_pc"]
    um = df.groupby("iso3")[vars_].apply(lambda g: g.notna().any()).reset_index()
    um = um[~um[vars_].all(axis=1)]
    um.to_csv(os.path.join(outd, "national_unmatched.csv"), index=False)
    print("codes", len(codes), "| country-years", len(df))
    print("missing share per variable:", df[vars_].isna().mean().round(3).to_dict())
    for v in vars_:
        print(f"  {v}: codes with no data at all:", sorted(um.loc[~um[v], "iso3"]))


if __name__ == "__main__":
    main()
