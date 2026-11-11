"""IPC acute food insecurity (Luca's phase_1..phase_5) per 0.5-degree cell-week, from IPC's open data on HDX (task 35).

  python realdata/covariates/ipc.py      # network (login node) for the download; writes grid/ipc/ipc_<year>.parquet

Source: HDX dataset "Global: Acute Food Insecurity Country Data" (publisher IPC, CC0, no key):
  ipc_global_national_wide.csv (all analyses; 50 countries; current periods Oct 2020 - 2026).
Method (step 1, `ipc_method = 'national'`): each analysis's CURRENT-period national numbers of people in Phase 1..5 are
spread over the country's cells in proportion to the cell's GHSL population of that year (grid/pop/pop_1990_2025.parquet);
phase shares (phase<k>_share, phase3p_share) are the national shares. A week gets the analysis whose current period
contains its week_start (latest analysis if several); weeks after a current period carry the latest analysis forward
for up to 26 weeks (ipc_carried = True); otherwise missing. Countries are the cells' ISO_A3 (as in $CLEAN).
Step 2 (area-level matching to IPC boundaries) is not implemented yet.
Columns: grid_id, week_start, phase1..phase5 (people), phase1_share..phase5_share, phase3p_share, ipc_carried, ipc_analysis.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

PKG = "https://data.humdata.org/api/3/action/package_show?id=global-acute-food-insecurity-country-data"
RAWD = os.path.join(C.RAW, "ipc")
CARRY_WEEKS = 26


def download():
    os.makedirs(RAWD, exist_ok=True)
    f = os.path.join(RAWD, "ipc_global_national_wide.csv")
    if not os.path.exists(f):
        rs = json.loads(urllib.request.urlopen(PKG, timeout=120).read())["result"]["resources"]
        url = [r["url"] for r in rs if r["name"] == "ipc_global_national_wide.csv"][0]
        open(f, "wb").write(urllib.request.urlopen(url, timeout=300).read())
    return f


def main():
    nat = pd.read_csv(download(), low_memory=False)
    nat["cf"] = pd.to_datetime(nat["Current from"], errors="coerce"); nat["ct"] = pd.to_datetime(nat["Current to"], errors="coerce")
    nat = nat.dropna(subset=["cf", "ct"]).sort_values(["Country", "cf"])
    cl = C.cells()[["grid_id", "ISO_A3"]]
    pop = pd.read_parquet(os.path.join(C.GRID, "pop", "pop_1990_2025.parquet"))
    wd = C.weekday(); out = []
    for iso, g in nat.groupby("Country"):
        cells = cl[cl.ISO_A3 == iso].grid_id.to_numpy()
        if not len(cells):
            print("no cells for", iso); continue
        weeks = pd.DatetimeIndex([w for y in C.YEARS for w in C.weeks(y, wd)])
        weeks = weeks[(weeks >= g.cf.min()) & (weeks <= g.ct.max() + pd.Timedelta(weeks=CARRY_WEEKS))]
        rows = []
        for w in weeks:
            cur = g[(g.cf <= w) & (g.ct >= w)]
            if len(cur):
                a, carried = cur.iloc[-1], False
            else:
                prev = g[g.ct < w]
                if not len(prev) or (w - prev.ct.iloc[-1]).days > 7 * CARRY_WEEKS:
                    continue
                a, carried = prev.iloc[-1], True
            rows.append((w, a, carried))
        if not rows:
            continue
        p = pop[pop.grid_id.isin(cells)].pivot(index="grid_id", columns="year", values="value").reindex(cells).fillna(0)
        for w, a, carried in rows:
            y = min(max(w.year, C.YEARS[0]), C.YEARS[-1]); wgt = p[y].to_numpy(float)
            wgt = wgt / wgt.sum() if wgt.sum() > 0 else np.full(len(cells), 1 / len(cells))
            rec = {"grid_id": cells.astype("int32"), "week_start": w, "ipc_carried": carried,
                   "ipc_analysis": str(a["Date of analysis"])}
            for k in range(1, 6):
                n = a.get(f"Phase {k} number current"); s = a.get(f"Phase {k} percentage current")
                rec[f"phase{k}"] = (wgt * (n if pd.notna(n) else np.nan)).astype("float32")
                rec[f"phase{k}_share"] = np.float32(s) if pd.notna(s) else np.float32(np.nan)
            s3 = a.get("Phase 3+ percentage current"); rec["phase3p_share"] = np.float32(s3) if pd.notna(s3) else np.float32(np.nan)
            out.append(pd.DataFrame(rec))
    df = pd.concat(out, ignore_index=True)
    od = os.path.join(C.GRID, "ipc"); os.makedirs(od, exist_ok=True)
    for y, g in df.groupby(df.week_start.dt.year):
        g.to_parquet(os.path.join(od, f"ipc_{y}.parquet"), index=False)
    print(f"IPC (national, population-weighted): {df.grid_id.nunique()} cells in {nat.Country.nunique()} countries, "
          f"weeks {df.week_start.min().date()}..{df.week_start.max().date()}, rows {len(df):,}, carried share {df.ipc_carried.mean():.3f}")


if __name__ == "__main__":
    main()
