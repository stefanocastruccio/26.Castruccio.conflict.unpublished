"""IPC acute food insecurity (Luca's phase_1..phase_5) per 0.5-degree cell-week, from IPC's open data on HDX (task 35).

  python realdata/covariates/ipc.py                 # HDX source (task 35); writes grid/ipc/ipc_<year>.parquet
  python realdata/covariates/ipc.py --source api    # IPC API (task 51): analyses from 2017 on; the HDX files are kept as
                                                    # grid/ipc/ipc_<year>.hdx.parquet; prints the HDX/API overlap check

Source: HDX dataset "Global: Acute Food Insecurity Country Data" (publisher IPC, CC0, no key):
  ipc_global_national_wide.csv (all analyses; 50 countries; current periods Oct 2020 - 2026).
Method (step 1, `ipc_method = 'national'`): each analysis's CURRENT-period national numbers of people in Phase 1..5 are
spread over the country's cells in proportion to the cell's GHSL population of that year (grid/pop/pop_1990_2025.parquet);
phase shares (phase<k>_share, phase3p_share) are the national shares. A week gets the analysis whose current period
contains its week_start (latest analysis if several); weeks after a current period carry the latest analysis forward
for up to 26 weeks (ipc_carried = True); otherwise missing. Countries are the cells' ISO_A3 (as in $CLEAN).
Task 51 (`--source api`): https://api.ipcinfo.org/population (key from ~/.ipc_key only; never printed, logged or saved: the
raw responses in raw/ipc_api/population_<year>.json are scrubbed of it). One call per year start=end=<year>, 2016-2026, all
countries; acute analyses (condition "A"), deduplicated by analysis id; national current-period numbers; "Mon YYYY - Mon YYYY"
current periods become first-of-month .. end-of-month dates (as in the HDX file); ISO2 -> ISO3 from Natural Earth ($COUNTRIES).
The key has no access to Cadre Harmonise countries (HTTP 403), so the API source is the UNION: every HDX analysis plus the
API analyses that HDX does not have (matched on country and current period); HDX values are kept where both exist.
The rest of the method is unchanged.
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


API = "https://api.ipcinfo.org"
RAWA = os.path.join(C.RAW, "ipc_api")
API_YEARS = range(2016, 2027)


def _api_get(path, **q):
    """GET from the IPC API with the key from ~/.ipc_key; returns text with the key removed. The URL is never printed."""
    import urllib.parse
    key = open(os.path.expanduser("~/.ipc_key")).read().strip()
    url = API + path + "?" + urllib.parse.urlencode({**q, "key": key})
    try:
        txt = urllib.request.urlopen(url, timeout=600).read().decode()
    except Exception as e:  # report without the URL
        raise RuntimeError(f"IPC API {path} {q}: {type(e).__name__} {getattr(e, 'code', '')}") from None
    return txt.replace(key, "<KEY>")


def download_api():
    os.makedirs(RAWA, exist_ok=True)
    recs = []
    for y in API_YEARS:
        f = os.path.join(RAWA, f"population_{y}.json")
        if not os.path.exists(f):
            open(f + ".part", "w").write(_api_get("/population", start=y, end=y)); os.replace(f + ".part", f)
        recs += json.load(open(f))
    return recs


def _iso2to3():
    import geopandas as gpd
    ne = gpd.read_file(os.environ.get("COUNTRIES", os.path.join(C.WORK, "inputs", "ne_10m_admin_0_countries",
                                      "ne_10m_admin_0_countries.shp")), ignore_geometry=True)
    m = {}
    for a2c, a3c in (("ISO_A2", "ISO_A3"), ("ISO_A2_EH", "ISO_A3_EH")):
        if a2c in ne and a3c in ne:
            for a2, a3 in zip(ne[a2c], ne[a3c]):
                if isinstance(a2, str) and len(a2) == 2 and isinstance(a3, str) and len(a3) == 3 and a2 not in m:
                    m[a2] = a3
    return m


def national_api():
    """API analyses as a table with the HDX column names used below."""
    recs = {r["id"]: r for r in download_api() if r.get("condition") == "A"}
    m = _iso2to3(); rows = []; miss = set()
    for r in recs.values():
        cur = (r.get("current_period_dates") or "").split(" - ")
        if len(cur) != 2 or r.get("country") not in m:
            miss.add(r.get("country")) if r.get("country") not in m else None
            continue
        cf = pd.to_datetime(cur[0], format="%b %Y", errors="coerce"); ct = pd.to_datetime(cur[1], format="%b %Y", errors="coerce")
        row = {"Country": m[r["country"]], "Date of analysis": r.get("analysis_date"), "Current from": cf,
               "Current to": ct + pd.offsets.MonthEnd(0), "Population analyzed current": r.get("estimated_population"),
               "Phase 3+ number current": r.get("p3plus"), "Phase 3+ percentage current": r.get("p3plus_percentage"),
               "ipc_api_id": r["id"]}
        for k in range(1, 6):
            row[f"Phase {k} number current"] = r.get(f"phase{k}_population")
            row[f"Phase {k} percentage current"] = r.get(f"phase{k}_percentage")
        rows.append(row)
    if miss:
        print("API countries without an ISO3 match (dropped):", sorted(x for x in miss if x), flush=True)
    nat = pd.DataFrame(rows)
    return nat[nat["Current to"] >= "2017-01-01"]


def overlap(api, hdx):
    """Analyses in both sources (same country and current period): count and largest relative difference in phase 3+ people."""
    k = ["Country", "cf", "ct"]
    a = api.assign(cf=pd.to_datetime(api["Current from"]), ct=pd.to_datetime(api["Current to"]))
    h = hdx.assign(cf=pd.to_datetime(hdx["Current from"], errors="coerce"), ct=pd.to_datetime(hdx["Current to"], errors="coerce"))
    j = a.merge(h, on=k, suffixes=("_api", "_hdx"))
    x, y = j["Phase 3+ number current_api"].astype(float), j["Phase 3+ number current_hdx"].astype(float)
    rel = (x - y).abs() / y.where(y > 0)
    i = rel.idxmax() if rel.notna().any() else None
    print(f"overlap: {len(j)} analyses in both sources (of {len(h)} HDX, {len(a)} API); largest relative difference in phase 3+ "
          f"people {rel.max():.4f}" + (f" ({j.loc[i, 'Country']} {j.loc[i, 'cf'].date()}: API {x[i]:.0f}, HDX {y[i]:.0f})" if i is not None else "")
          + f"; median {rel.median():.4f}; share with difference > 1%: {(rel > 0.01).mean():.3f}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--source", choices=["hdx", "api"], default="hdx")
    src = ap.parse_args().source
    if src == "api":
        api, hdx = national_api(), pd.read_csv(download(), low_memory=False)
        overlap(api, hdx)
        # The key has no access to Cadre Harmonise countries (403), which HDX covers from 2021: keep every HDX analysis and add
        # the API analyses HDX does not have (same country and current period), so the existing 2021-2025 values do not change.
        hk = set(zip(hdx.Country, pd.to_datetime(hdx["Current from"], errors="coerce"), pd.to_datetime(hdx["Current to"], errors="coerce")))
        add = api[[k not in hk for k in zip(api.Country, api["Current from"], api["Current to"])]]
        nat = pd.concat([hdx, add.assign(**{"Current from": add["Current from"].dt.strftime("%Y-%m-%d"),
                                            "Current to": add["Current to"].dt.strftime("%Y-%m-%d")})], ignore_index=True)
        print(f"combined: {len(hdx)} HDX analyses + {len(add)} API-only analyses", flush=True)
        cf = pd.to_datetime(nat["Current from"], errors="coerce")
        for y in range(2017, 2027):
            for lab, t in (("HDX", hdx), ("combined", nat)):
                c = pd.to_datetime(t["Current from"], errors="coerce").dt.year == y
                print(f"{y} {lab}: {c.sum()} analyses in {t[c].Country.nunique()} countries", flush=True)
    else:
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
    if src == "api":  # keep the HDX outputs once, as ipc_<year>.hdx.parquet
        for f in sorted(os.listdir(od)):
            if f.endswith(".parquet") and not f.endswith(".hdx.parquet"):
                h = os.path.join(od, f.replace(".parquet", ".hdx.parquet"))
                if not os.path.exists(h):
                    os.replace(os.path.join(od, f), h)
    for y, g in df.groupby(df.week_start.dt.year):
        g.to_parquet(os.path.join(od, f"ipc_{y}.parquet"), index=False)
    print(f"IPC (national, population-weighted): {df.grid_id.nunique()} cells in {nat.Country.nunique()} countries, "
          f"weeks {df.week_start.min().date()}..{df.week_start.max().date()}, rows {len(df):,}, carried share {df.ipc_carried.mean():.3f}")


if __name__ == "__main__":
    main()
