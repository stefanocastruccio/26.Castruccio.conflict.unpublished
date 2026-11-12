"""Task 40, part A: the GLOBAL cell-week CORE table from the raw ACLED events.

  python realdata/table/core.py        # batch job (crc/jobs/40_core.sh); writes $WORK/table_global/core/year=<y>/part.parquet

Cells: the 56,894 task-35 cells (covariates/cells_60S.parquet: grid_id, lon, lat, ISO_A3) + region (grid/cell_region.parquet).
Weeks: week_start on the weekday of $CLEAN's week_start; from the first such day in December 1996 to the last week that ends
(week_start + 6) before 2025-11-10, the end of the account's ACLED data.
Events -> cells: the 0.5-degree cell containing (latitude, longitude). Events outside every cell (sea, coast): the nearest of the
8 neighbouring cells, of ANY country (coordinator decision 2026-11-12: an event belongs to the place where it happened; the cell's
country only drives the country effect and the national covariates); dropped if no neighbouring cell exists or if the event's
country has no cell at all in the grid (e.g. Bahrain). Reassignments across a country border are counted in the report. The event's ISO3 is ACLED's numeric `iso` mapped to Natural
Earth ADM0_A3 (the cells' ISO_A3 codes) by ISO_N3, with a country-name fallback.
Counts per cell-week: the six ACLED event types (columns as in $CLEAN: Battles, ExplosionsRemote_violence,
Violence_against_civilians, Protests, Riots, Strategic_developments); n_conflict = Battles + ExplosionsRemote_violence +
Violence_against_civilians; n_unrest = Protests + Riots; the same two with geo_precision 1 (*_gp1) and time_precision 1 (*_tp1);
fatalities. Conflict = n_conflict > 0, Unrest = n_unrest > 0.
Coverage: start per country from $WORK/inputs/acled_coverage.csv (the table $CLEAN used). Countries not in it: the first event date of
their ACLED region (the region most of the country's events carry); countries with neither events nor a table entry: the latest of the
regions' first event dates (conservative), flagged coverage_source = 'latest_region'. A country's first week is the first week whose
week_start is on or after its coverage start; earlier rows are not created. hist_incomplete marks the first 4 covered weeks.
Histories: <Type>_rollsum_4 = sum over weeks t-4..t-1 for the 5 types of $CLEAN (NaN in the first 4 covered weeks).
Report: $WORK/table_global/core_report.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "covariates"))
import common as C  # noqa: E402

WORK = C.WORK
OUT = os.path.join(WORK, "table_global")
END = pd.Timestamp("2025-11-10")                       # first day not in the account's data
TYPES = {"Battles": "Battles", "Explosions/Remote violence": "ExplosionsRemote_violence",
         "Violence against civilians": "Violence_against_civilians", "Protests": "Protests", "Riots": "Riots",
         "Strategic developments": "Strategic_developments"}
CONF = ["Battles", "ExplosionsRemote_violence", "Violence_against_civilians"]
UNR = ["Protests", "Riots"]
HIST = CONF + ["Riots", "Protests"]


def week_grid():
    wd = C.weekday()
    d = pd.date_range("1996-12-01", END - pd.Timedelta(days=1), freq="D")
    w = d[d.dayofweek == wd]
    return w[w + pd.Timedelta(days=6) < END]


def iso_map():
    import pyogrio
    ne = pyogrio.read_dataframe(os.path.join(WORK, "inputs", "ne_10m_admin_0_countries"), read_geometry=False)
    n3 = pd.to_numeric(ne["ISO_N3"], errors="coerce")
    m = {int(k): v for k, v in zip(n3, ne["ADM0_A3"]) if pd.notna(k) and k > 0}
    names = {str(n).lower(): a for col in ("ADMIN", "NAME", "NAME_LONG", "SOVEREIGNT") for n, a in zip(ne[col], ne["ADM0_A3"])}
    return m, names


def main():
    t0 = time.time(); rep = {}
    os.makedirs(OUT, exist_ok=True)
    cells = C.cells()[["grid_id", "lon", "lat", "ISO_A3", "row", "col"]].copy()
    cr = pd.read_parquet(os.path.join(C.GRID, "cell_region.parquet"))[["grid_id", "region"]]
    cells = cells.merge(cr, on="grid_id", how="left")
    lut = np.full(360 * 720, -1, np.int64); lut[cells.row * 720 + cells.col] = np.arange(len(cells))
    W = week_grid(); W0 = W[0]; nW = len(W)
    rep["weeks"] = [str(W[0].date()), str(W[-1].date()), nW]
    # ---- events
    cols = ["event_date", "event_type", "latitude", "longitude", "geo_precision", "time_precision", "fatalities", "iso", "country", "region"]
    ev = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(WORK, "acled", "raw", "acled_*.parquet")))],
                   ignore_index=True)
    ev["event_date"] = pd.to_datetime(ev.event_date)
    for c in ("latitude", "longitude", "geo_precision", "time_precision", "fatalities", "iso"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    rep["events_raw"] = int(len(ev))
    m3, names = iso_map()
    ev["iso3"] = ev.iso.map(m3)
    miss = ev.iso3.isna()
    ev.loc[miss, "iso3"] = ev.loc[miss, "country"].str.lower().map(names)
    r = np.floor((ev.latitude.to_numpy() + 90) / 0.5).astype(int); c = np.floor((ev.longitude.to_numpy() + 180) / 0.5).astype(int)
    ok = (r >= 0) & (r < 360) & (c >= 0) & (c < 720)
    k = np.full(len(ev), -1, np.int64); k[ok] = lut[r[ok] * 720 + c[ok] % 720]
    # outside every cell: nearest neighbouring cell of the same ISO3
    out = np.where(k < 0)[0]; moved = 0; cross = {}
    has_cell = set(cells.ISO_A3)
    iso_arr = cells.ISO_A3.to_numpy(); clon = cells.lon.to_numpy(); clat = cells.lat.to_numpy()
    e_iso = ev.iso3.to_numpy(); e_lat = ev.latitude.to_numpy(); e_lon = ev.longitude.to_numpy()
    for i in out:
        if not ok[i] or e_iso[i] not in has_cell:
            continue
        best, bd = -1, 9e9
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r[i] + dr, (c[i] + dc) % 720
                if 0 <= rr < 360:
                    j = lut[rr * 720 + cc]
                    if j >= 0:
                        d = (clat[j] - e_lat[i]) ** 2 + ((clon[j] - e_lon[i] + 180) % 360 - 180) ** 2
                        if d < bd:
                            best, bd = j, d
        if best >= 0:
            k[i] = best; moved += 1
            if iso_arr[best] != e_iso[i]:
                key = f"{e_iso[i]}->{iso_arr[best]}"; cross[key] = cross.get(key, 0) + 1
    ev["cell"] = k
    ev["widx"] = ((ev.event_date - W0).dt.days // 7).astype(int)
    inweek = (ev.widx >= 0) & (ev.widx < nW)
    rep["events_outside_week_range"] = int((~inweek).sum())
    rep["events_moved_to_neighbour_cell"] = int(moved)
    rep["events_moved_across_border"] = int(sum(cross.values()))
    rep["moved_across_border_top"] = dict(sorted(cross.items(), key=lambda x: -x[1])[:25])
    drop = (ev.cell < 0) & inweek
    rep["events_dropped_no_cell"] = int(drop.sum())
    rep["dropped_share_by_acled_region"] = (ev[inweek].assign(d=drop[inweek]).groupby("region").d.mean().round(5).to_dict())
    rep["dropped_by_country_top"] = ev[drop].groupby("country").size().sort_values(ascending=False).head(25).to_dict()
    rep["dropped_countries_without_cell"] = sorted(set(ev.loc[drop & ~ev.iso3.isin(has_cell), "country"]))
    ev = ev[inweek & (ev.cell >= 0)].copy()
    ev["type"] = ev.event_type.map(TYPES)
    rep["events_unknown_type"] = int(ev.type.isna().sum())
    ev = ev[ev.type.notna()]
    rep["events_used"] = int(len(ev))
    rep["geo_precision1_share_by_cell_region"] = (ev.assign(g=ev.geo_precision == 1, reg=cells.region.to_numpy()[ev.cell])
                                                  .groupby("reg", dropna=False).g.mean().round(4).rename(index=lambda x: str(x)).to_dict())
    # ---- coverage per country
    cov = pd.read_csv(os.path.join(WORK, "inputs", "acled_coverage.csv"), parse_dates=["start"]).set_index("ISO_A3").start
    ev["cell_iso"] = iso_arr[ev.cell]
    reg_of_iso = ev.groupby("cell_iso").region.agg(lambda s: s.value_counts().index[0])
    reg_start = ev.groupby("region").event_date.min()
    start, src = {}, {}
    for iso in pd.unique(iso_arr):
        if iso in cov.index:
            start[iso], src[iso] = cov[iso], "acled_coverage.csv"
        elif iso in reg_of_iso.index:
            start[iso], src[iso] = reg_start[reg_of_iso[iso]], f"region_first_event:{reg_of_iso[iso]}"
        else:
            start[iso], src[iso] = reg_start.max(), "latest_region"
    rep["coverage_source_counts"] = pd.Series(src).str.split(":").str[0].value_counts().to_dict()
    rep["coverage_not_in_table"] = {k_: [str(start[k_].date()), v] for k_, v in src.items() if v != "acled_coverage.csv"}
    s_idx = {iso: int(np.searchsorted(W, pd.Timestamp(st))) for iso, st in start.items()}
    cstart = np.array([s_idx[i] for i in iso_arr])
    # ---- sparse counts per cell-week
    ev["one"] = 1
    piv = ev.pivot_table(index=["cell", "widx"], columns="type", values="one", aggfunc="sum", fill_value=0)
    for t in TYPES.values():
        if t not in piv:
            piv[t] = 0
    piv = piv[list(TYPES.values())].astype("int32")
    g = ev.groupby(["cell", "widx"])
    piv["fatalities"] = g.fatalities.sum().astype("int32")
    for tag, msk in (("gp1", ev.geo_precision == 1), ("tp1", ev.time_precision == 1)):
        e2 = ev[msk]
        piv[f"n_conflict_{tag}"] = e2[e2.type.isin(CONF)].groupby(["cell", "widx"]).size().reindex(piv.index, fill_value=0).astype("int32")
        piv[f"n_unrest_{tag}"] = e2[e2.type.isin(UNR)].groupby(["cell", "widx"]).size().reindex(piv.index, fill_value=0).astype("int32")
    piv = piv.reset_index()
    # history contributions: an event week w adds to rows w+1..w+4
    hist = pd.concat([piv[["cell", "widx"] + HIST].assign(widx=piv.widx + s) for s in range(1, 5)])
    hist = hist.groupby(["cell", "widx"])[HIST].sum().add_suffix("_rollsum_4").reset_index()
    # ---- dense rows per year
    years = sorted(set(W.year)); stats = []
    gid = cells.grid_id.to_numpy(); reg = cells.region.to_numpy()
    for y in years:
        wi = np.where(W.year == y)[0]
        cc, ww = np.meshgrid(np.arange(len(cells)), wi, indexing="ij"); cc, ww = cc.ravel(), ww.ravel()
        keep = ww >= cstart[cc]; cc, ww = cc[keep], ww[keep]
        df = pd.DataFrame({"cell": cc, "widx": ww})
        df = df.merge(piv, on=["cell", "widx"], how="left").merge(hist, on=["cell", "widx"], how="left")
        cnt = [c_ for c_ in piv.columns if c_ not in ("cell", "widx")]
        df[cnt] = df[cnt].fillna(0).astype("int32")
        hc = [f"{h}_rollsum_4" for h in HIST]
        df[hc] = df[hc].fillna(0).astype("float32")
        inc = df.widx.to_numpy() < cstart[df.cell.to_numpy()] + 4
        df.loc[inc, hc] = np.nan
        df["hist_incomplete"] = inc
        df["n_conflict"] = df[CONF].sum(1).astype("int32"); df["n_unrest"] = df[UNR].sum(1).astype("int32")
        df["Conflict"] = (df.n_conflict > 0).astype("int8"); df["Unrest"] = (df.n_unrest > 0).astype("int8")
        df.insert(0, "grid_id", gid[df.cell].astype("int32")); df.insert(1, "week_start", W[df.widx])
        df["ISO_A3"] = iso_arr[df.cell]; df["region"] = reg[df.cell]
        df["lon"] = clon[df.cell].astype("float32"); df["lat"] = clat[df.cell].astype("float32")
        df = df.drop(columns=["cell", "widx"]).sort_values(["week_start", "grid_id"])
        od = os.path.join(OUT, "core", f"year={y}"); os.makedirs(od, exist_ok=True)
        df.to_parquet(os.path.join(od, "part.parquet"), index=False)
        st = df.groupby(df.region.fillna("other")).agg(rows=("grid_id", "size"), conflict_cw=("Conflict", "sum"), unrest_cw=("Unrest", "sum"))
        stats.append(st.assign(year=y).reset_index())
        print(y, len(df), f"{time.time() - t0:.0f}s", flush=True)
    st = pd.concat(stats)
    rep["by_region_year"] = st.to_dict(orient="records")
    rep["rows_total"] = int(st.rows.sum())
    json.dump(rep, open(os.path.join(OUT, "core_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k_: v for k_, v in rep.items() if k_ != "by_region_year"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
