"""Cell -> region lookup for the covariate outputs (task 35 scope, decided 2026-11-09).

  python realdata/covariates/cell_region.py      # writes $WORK/covariates/grid/cell_region.parquet

Each cell of cells() (the $CLEAN cells, 40S-72N) gets the country of the nearest Natural Earth polygon (ADM0_A3, as in
fit_bivariate.region_rows), and a region from that country:
  africa      UN region Africa (incl. Madagascar and the island states; Egypt here)
  middle_east TUR SYR LBN ISR PSE JOR IRQ IRN SAU YEM OMN ARE QAT BHR KWT CYP (+ CYN, Northern Cyprus; PSX = Palestine in Natural Earth)
  south_asia  AFG PAK IND NPL BTN BGD LKA MDV MMR THA LAO KHM VNM MYS SGP IDN PHL TLS BRN (+ KAS, Siachen/Kashmir)
  latam       World Bank region Latin America & Caribbean (Mexico, Central America, Caribbean, South America)
  europe      continent Europe minus RUS, UKR, BLR (EU, UK, NOR, CHE, ISL, Western Balkans incl. KOS, MDA, microstates)
Everything else (North America, East and Central Asia, Russia, Oceania, ...) is region None and dropped from outputs.
Output columns: grid_id, lon, lat, ISO_A3 (from $CLEAN), adm0 (nearest polygon), region.
"""
from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

ME = {"TUR", "SYR", "LBN", "ISR", "PSE", "JOR", "IRQ", "IRN", "SAU", "YEM", "OMN", "ARE", "QAT", "BHR", "KWT", "CYP", "CYN", "PSX"}
SA = {"AFG", "PAK", "IND", "NPL", "BTN", "BGD", "LKA", "MDV", "MMR", "THA", "LAO", "KHM", "VNM", "MYS", "SGP", "IDN", "PHL",
      "TLS", "BRN", "KAS"}
EU_EXCL = {"RUS", "UKR", "BLR"}
BOXES = {"africa": (-20, 55, -36, 38), "south_asia": (60, 142, -12, 38), "latam": (-118, -34, -56, 33)}


def lookup():
    import geopandas as gpd
    cl = C.cells()
    land = gpd.read_file(os.environ.get("COUNTRIES", os.path.join(C.WORK, "inputs/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"))).to_crs(4326)
    meta = land[["ADM0_A3", "CONTINENT", "REGION_UN", "REGION_WB"]].drop_duplicates("ADM0_A3").set_index("ADM0_A3")
    pts = gpd.GeoDataFrame(cl[["grid_id"]], geometry=gpd.points_from_xy(cl.lon, cl.lat), crs=4326)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        near = gpd.sjoin_nearest(pts, land[["ADM0_A3", "geometry"]], how="left")
    near = near[~near.index.duplicated()]
    cl = cl.assign(adm0=near["ADM0_A3"].to_numpy())

    def region(a):
        if a in ME:
            return "middle_east"
        if a in SA:
            return "south_asia"
        m = meta.loc[a] if a in meta.index else None
        if m is None:
            return None
        if m.REGION_UN == "Africa":
            return "africa"
        if m.REGION_WB == "Latin America & Caribbean":
            return "latam"
        if m.CONTINENT == "Europe" and a not in EU_EXCL:
            return "europe"
        return None
    cl["region"] = cl.adm0.map(region)
    out = cl[["grid_id", "lon", "lat", "ISO_A3", "adm0", "region"]]
    os.makedirs(C.GRID, exist_ok=True)
    out.to_parquet(os.path.join(C.GRID, "cell_region.parquet"), index=False)
    print("cells per region:", out.region.value_counts(dropna=False).to_dict())
    for r, (x0, x1, y0, y1) in BOXES.items():
        inb = out[out.lon.between(x0, x1) & out.lat.between(y0, y1) & out.region.isna()]
        print(f"unassigned in the {r} box: {len(inb)} cells; countries {inb.adm0.value_counts().to_dict()}")
    print("ISO_A3 (CLEAN) vs nearest-polygon ADM0 disagree for", int((out.ISO_A3 != out.adm0).sum()), "cells")
    return out


def restrict(df, key="grid_id"):
    """Keep only rows of cells in the five regions."""
    cr = pd.read_parquet(os.path.join(C.GRID, "cell_region.parquet"))
    keep = cr.loc[cr.region.notna(), "grid_id"]
    return df[df[key].isin(keep)]


if __name__ == "__main__":
    lookup()
