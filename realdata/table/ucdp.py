"""Task 47: UCDP Georeferenced Event Dataset (GED) on the same cells and weeks as the ACLED table (no analysis).

  python realdata/table/ucdp.py <ged.csv>     # batch job (crc/jobs/47_ucdp.sh); writes $WORK/table_global/ucdp/year=<y>.parquet

Input: the global GED csv (one row per event; UCDP's columns id, year, type_of_violence, date_start, date_end, date_prec,
latitude, longitude, where_prec, country, best, high, low, ...).
Space: the 0.5-degree cell containing (latitude, longitude), the same 56,894 cells as core.py. Events outside every cell (sea,
coast): the nearest of the 8 neighbouring cells of any country, as core.py; dropped if there is none or if the event's country
has no cell. where_prec 6 (country only) and 7 (international waters or airspace) have no meaningful location: they are dropped
and counted.
Time: the week (core.py's weekday anchor, extended back to 1989) that contains date_start. UCDP events can span several days or
weeks (date_end > date_start, or date_prec 4-5: known to the month or worse); they are kept in the week of date_start and
counted in n_multiweek / n_tp_coarse, so they can be excluded later.
Columns per cell-week (sparse: only cell-weeks with an event or a non-zero history; a missing row means zero):
  ucdp_n (events), ucdp_n_sb / ucdp_n_ns / ucdp_n_os (state-based, non-state, one-sided violence), ucdp_deaths (sum of best),
  ucdp_n_gp1 (where_prec 1-2: exact or within 25 km), ucdp_n_tp1 (date_prec 1: exact day), ucdp_n_multiweek, ucdp_n_tp_coarse,
  UCDP = ucdp_n > 0, and histories ucdp_n_rollsum_4, ucdp_n_sb/ns/os_rollsum_4 (weeks t-4..t-1).
Report: $WORK/table_global/ucdp_report.json (version, events read / used / dropped by reason, by precision, by region and year).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import core  # noqa: E402

START = pd.Timestamp("1988-12-01")
TYPES = {1: "sb", 2: "ns", 3: "os"}


def week_grid_1989():
    wd = core.C.weekday()
    d = pd.date_range(START, core.END - pd.Timedelta(days=1), freq="D")
    w = d[d.dayofweek == wd]
    return w[w + pd.Timedelta(days=6) < core.END]


def main(path):
    t0 = time.time(); rep = {"source_file": os.path.basename(path)}
    cells = core.C.cells()[["grid_id", "lon", "lat", "ISO_A3", "row", "col"]].copy()
    lut = np.full(360 * 720, -1, np.int64); lut[cells.row * 720 + cells.col] = np.arange(len(cells))
    W = week_grid_1989(); W0 = W[0]; nW = len(W)
    rep["weeks"] = [str(W[0].date()), str(W[-1].date()), nW]
    ev = pd.read_csv(path, low_memory=False)
    rep["events_raw"] = int(len(ev)); rep["years"] = [int(ev.year.min()), int(ev.year.max())]
    for c in ("latitude", "longitude", "where_prec", "date_prec", "best", "type_of_violence"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev["date_start"] = pd.to_datetime(ev.date_start, errors="coerce"); ev["date_end"] = pd.to_datetime(ev.date_end, errors="coerce")
    rep["where_prec"] = ev.where_prec.value_counts().sort_index().to_dict()
    rep["date_prec"] = ev.date_prec.value_counts().sort_index().to_dict()
    noloc = ev.where_prec.isin([6, 7])
    rep["dropped_where_prec_6_7"] = int(noloc.sum())
    ev = ev[~noloc].copy()
    # country -> the cells' ISO_A3 (names, as core.iso_map's fallback)
    _, names = core.iso_map(set(cells.ISO_A3))
    base = ev.country.astype(str).str.replace(r"\s*\(.*\)\s*$", "", regex=True).str.lower()
    ev["iso3"] = base.map(names)
    rep["countries_unmatched"] = sorted(set(ev.loc[ev.iso3.isna(), "country"].astype(str)))
    # cell, as core.load_events
    r = np.floor((ev.latitude.to_numpy() + 90) / 0.5).astype(int); c = np.floor((ev.longitude.to_numpy() + 180) / 0.5).astype(int)
    ok = (r >= 0) & (r < 360) & (c >= 0) & (c < 720)
    k = np.full(len(ev), -1, np.int64); k[ok] = lut[r[ok] * 720 + c[ok] % 720]
    has_cell = set(cells.ISO_A3); clon = cells.lon.to_numpy(); clat = cells.lat.to_numpy()
    e_iso = ev.iso3.to_numpy(); e_lat = ev.latitude.to_numpy(); e_lon = ev.longitude.to_numpy()
    moved = 0
    for i in np.where(k < 0)[0]:
        if not ok[i] or (e_iso[i] not in has_cell and isinstance(e_iso[i], str)):
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
    ev["cell"] = k
    ev["widx"] = ((ev.date_start - W0).dt.days // 7)
    inweek = ev.widx.between(0, nW - 1)
    rep["events_moved_to_neighbour_cell"] = int(moved)
    rep["dropped_outside_week_range"] = int((~inweek).sum())
    rep["dropped_no_cell"] = int(((ev.cell < 0) & inweek).sum())
    rep["dropped_no_cell_by_country_top"] = ev[(ev.cell < 0) & inweek].country.value_counts().head(25).to_dict()
    ev = ev[inweek & (ev.cell >= 0)].copy(); ev["widx"] = ev.widx.astype(int)
    rep["events_used"] = int(len(ev))
    ev["multiweek"] = ((ev.date_end - ev.date_start).dt.days >= 7) | ((ev.date_end.notna()) & (((ev.date_end - W0).dt.days // 7) != ev.widx))
    rep["share_multiweek"] = float(ev.multiweek.mean()); rep["share_date_prec_4_5"] = float(ev.date_prec.isin([4, 5]).mean())
    rep["events_by_type"] = ev.type_of_violence.map(TYPES).value_counts().to_dict()
    rep["events_by_region_year"] = ev.groupby(["region", "year"]).size().unstack(fill_value=0).to_dict(orient="index") if "region" in ev else {}
    # counts per cell-week
    g = ev.groupby(["cell", "widx"])
    piv = pd.DataFrame({"ucdp_n": g.size(), "ucdp_deaths": g.best.sum(),
                        "ucdp_n_gp1": g.where_prec.apply(lambda x: int(x.isin([1, 2]).sum())),
                        "ucdp_n_tp1": g.date_prec.apply(lambda x: int((x == 1).sum())),
                        "ucdp_n_multiweek": g.multiweek.sum(), "ucdp_n_tp_coarse": g.date_prec.apply(lambda x: int(x.isin([4, 5]).sum()))})
    for code, tag in TYPES.items():
        piv[f"ucdp_n_{tag}"] = ev[ev.type_of_violence == code].groupby(["cell", "widx"]).size().reindex(piv.index, fill_value=0)
    piv = piv.fillna(0).astype("int64").reset_index()
    hc = ["ucdp_n", "ucdp_n_sb", "ucdp_n_ns", "ucdp_n_os"]
    hist = pd.concat([piv[["cell", "widx"] + hc].assign(widx=piv.widx + s) for s in range(1, 5)])
    hist = hist.groupby(["cell", "widx"])[hc].sum().add_suffix("_rollsum_4").reset_index()
    df = piv.merge(hist, on=["cell", "widx"], how="outer")
    df = df[df.widx < nW]
    cnt = [c_ for c_ in piv.columns if c_ not in ("cell", "widx")]
    df[cnt] = df[cnt].fillna(0).astype("int32")
    df[[f"{h}_rollsum_4" for h in hc]] = df[[f"{h}_rollsum_4" for h in hc]].fillna(0).astype("float32")
    df["UCDP"] = (df.ucdp_n > 0).astype("int8")
    df.insert(0, "grid_id", cells.grid_id.to_numpy()[df.cell].astype("int32")); df.insert(1, "week_start", W[df.widx])
    df = df.drop(columns=["cell", "widx"])
    od = os.path.join(core.OUT, "ucdp"); os.makedirs(od, exist_ok=True)
    for y, d in df.groupby(df.week_start.dt.year):
        d.sort_values(["week_start", "grid_id"]).to_parquet(os.path.join(od, f"year={y}.parquet"), index=False)
    rep["rows"] = int(len(df)); rep["cell_weeks_with_event"] = int(df.UCDP.sum()); rep["seconds"] = time.time() - t0
    json.dump(rep, open(os.path.join(core.OUT, "ucdp_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k_: v for k_, v in rep.items() if k_ != "events_by_region_year"}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
